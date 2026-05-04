"""
Lazy singleton'ы для retriever и LLM-клиентов.

Каждый запрос переиспользует:
- один ``CodeRetriever`` (Neo4j + Weaviate + embeddings — дорого создавать).
- один primary OpenAI-совместимый клиент.
- один fallback-клиент (если в конфиге задан другой ``api_base``/ключ).

``build_llm_role(config, role)`` собирает ``LLMRole`` под конкретную роль
(quality / rewrite / answer / cypher) с failover.
"""

import os
from typing import Any, Dict, Optional

from src.infra.logger import get_logger
from ..services._llm import LLMRole, ModelRoute

logger = get_logger(__name__)


_retriever: Optional[Any] = None
_repo_paths: Optional[Any] = None
_primary_client: Optional[Any] = None
_fallback_client: Optional[Any] = None


def get_retriever() -> Any:
    global _retriever, _repo_paths
    if _retriever is not None:
        return _retriever

    from pathlib import Path

    from src.core.graph import Neo4jClient, WeaviateIndexer
    from src.core.graph.repo_paths import RepoPathResolver
    from src.infra.config.agent import AgentConfig
    from src.search.retrieval import CodeRetriever

    cfg = AgentConfig()

    logger.info("[INIT] Creating Neo4j client...")
    neo4j = Neo4jClient(
        uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        user=os.getenv("NEO4J_USER", "neo4j"),
        password=os.getenv("NEO4J_PASSWORD", ""),
    )

    # Per-repo resolver: читает RepositoryNode.local_path из Neo4j и
    # кеширует. Поддерживает 40+ независимых репо без хардкода единого
    # RAG_REPOS_DIR (тот остаётся override верхнего уровня).
    logger.info("[INIT] Creating RepoPathResolver...")
    _repo_paths = RepoPathResolver(neo4j)

    logger.info("[INIT] Creating Weaviate client...")
    # Legacy fallback: если в env есть RAG_REPOS_DIR — он же передаётся в
    # WeaviateIndexer (для read_code при индексации). Per-repo пути уже
    # покрыты через RepoPathResolver, но WeaviateIndexer пока работает
    # с одним общим корнем — миграция отдельной задачей.
    repos_dir = Path(cfg.repos_dir) if cfg.repos_dir else None
    if repos_dir:
        logger.info(f"[INIT] env RAG_REPOS_DIR override = {repos_dir}")
    weaviate = WeaviateIndexer(
        weaviate_url=os.getenv("WEAVIATE_URL", "http://localhost:8080"),
        repos_dir=repos_dir,
    )
    logger.info("[INIT] Creating CodeRetriever...")
    _retriever = CodeRetriever(neo4j, weaviate)
    return _retriever


def get_repo_paths() -> Optional[Any]:
    """
    Singleton ``RepoPathResolver``. Возвращает None, если ``get_retriever()``
    ещё не звался (резолвер инициализируется внутри). Безопасно использовать
    после первого ``get_retriever()``.
    """
    return _repo_paths


def _make_openai_client(api_key: Optional[str], api_base: str) -> Any:
    from openai import OpenAI
    return OpenAI(api_key=api_key, base_url=api_base)


def get_primary_client(config) -> Any:
    """Один primary client на процесс (cached)."""
    global _primary_client
    if _primary_client is None:
        _primary_client = _make_openai_client(config.api_key, config.api_base)
        logger.info(f"[INIT] primary LLM endpoint: {config.api_base}")
    return _primary_client


def get_fallback_client(config) -> Optional[Any]:
    """
    Fallback client — если задан другой ``api_base_fallback`` или
    ``api_key_fallback``. Если оба отсутствуют — возвращает None
    (failover отключён).
    """
    global _fallback_client
    if _fallback_client is not None:
        return _fallback_client

    fb_base = config.api_base_fallback
    fb_key = config.api_key_fallback

    # Случай 1: задан явно отдельный endpoint → создаём отдельный клиент.
    if fb_base:
        _fallback_client = _make_openai_client(fb_key, fb_base)
        logger.info(f"[INIT] fallback LLM endpoint: {fb_base}")
        return _fallback_client

    # Случай 2: только другой ключ → используем primary base, но другой ключ.
    if fb_key and fb_key != config.api_key:
        _fallback_client = _make_openai_client(fb_key, config.api_base)
        logger.info(f"[INIT] fallback LLM uses primary base, separate key")
        return _fallback_client

    # Случай 3: ничего — fallback на уровне endpoint'а отключён.
    return None


def build_llm_role(config, role: str) -> LLMRole:
    """
    Построить ``LLMRole`` для конкретной роли pipeline.

    role ∈ {"quality", "rewrite", "answer", "cypher"}.

    Семантика fallback:
    - Если ``rag_<role>_model_fallback`` задан И у нас есть отдельный
      fallback‑клиент → fallback на другом endpoint+модели.
    - Если ``rag_<role>_model_fallback`` задан, но fallback‑клиента нет →
      fallback на primary endpoint, но с другой моделью.
    - Если fallback‑модель НЕ задана, но fallback‑клиент есть → fallback на
      другом endpoint той же моделью.
    - Иначе fallback не настраивается (только primary).
    """
    primary_model_attr = f"rag_{role}_model"
    fallback_model_attr = f"rag_{role}_model_fallback"

    primary_model = getattr(config, primary_model_attr, "") or ""
    fallback_model = getattr(config, fallback_model_attr, "") or ""

    # cypher_model по умолчанию использует answer_model (см. AgentConfig).
    if role == "cypher" and not primary_model:
        primary_model = config.rag_answer_model
        if not fallback_model:
            fallback_model = config.rag_answer_model_fallback or ""

    primary_client = get_primary_client(config)
    fallback_client = get_fallback_client(config)

    primary_route = ModelRoute(client=primary_client, model=primary_model)

    fallback_route: Optional[ModelRoute] = None
    if fallback_model and fallback_client is not None:
        fallback_route = ModelRoute(client=fallback_client, model=fallback_model)
    elif fallback_model:
        # та же endpoint, другая модель
        fallback_route = ModelRoute(client=primary_client, model=fallback_model)
    elif fallback_client is not None:
        # другая endpoint, та же модель (для случая когда основной провайдер
        # упал, но та же model name доступна через резервный — например, OpenRouter↔OpenAI)
        fallback_route = ModelRoute(client=fallback_client, model=primary_model)

    return LLMRole(primary=primary_route, fallback=fallback_route)
