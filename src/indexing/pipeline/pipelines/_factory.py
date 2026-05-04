"""
Общая фабрика IndexingExecutor для сценариев pipelines/*.py.

Выделена сюда, чтобы каждый сценарий оставался максимально тонким —
как ``risk_full_by_book.py`` в твоей api.bo: только перечень тасок и
комментарии "зачем".
"""

from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional

from src.core.graph import (
    APILinker,
    GraphBuilder,
    Neo4jClient,
    WeaviateIndexer,
)
from src.core.repo_loader import RepositoryLoader
from src.infra.logger import get_logger
from ..base.executor import IndexingExecutor

logger = get_logger(__name__)


@contextmanager
def make_executor(
    *,
    filters: Optional[Dict[str, Any]] = None,
    neo4j_uri: str = "bolt://localhost:7687",
    neo4j_user: str = "neo4j",
    neo4j_password: str = "password",
    weaviate_url: str = "http://localhost:8080",
    embedding_model: str = "BAAI/bge-m3",
) -> Iterator[IndexingExecutor]:
    """
    Создаёт executor с DI-инжекцией всех технических компонентов.
    Корректно закрывает соединения по выходу.
    """
    neo4j_client = Neo4jClient(uri=neo4j_uri, user=neo4j_user, password=neo4j_password)
    graph_builder = GraphBuilder(neo4j_client)
    api_linker = APILinker(use_openapi=True)
    weaviate_indexer = WeaviateIndexer(
        weaviate_url=weaviate_url,
        embedding_model=embedding_model,
        neo4j_client=neo4j_client,
    )
    repo_loader = RepositoryLoader()

    executor = IndexingExecutor(
        repo_loader=repo_loader,
        graph_builder=graph_builder,
        neo4j_client=neo4j_client,
        api_linker=api_linker,
        weaviate_indexer=weaviate_indexer,
        filters=filters or {},
    )

    try:
        yield executor
    finally:
        try:
            neo4j_client.close()
        except Exception:
            pass
        try:
            weaviate_indexer.close()
        except Exception:
            pass


def build_stats(executor: IndexingExecutor) -> Dict[str, Any]:
    """Сводная статистика. Формат идентичен старому GraphPipeline."""
    ctx = executor.context
    graph_stats = ctx.get("graph_stats") or {}
    repo_info = ctx.get("repo_info")

    stats = {
        "repository": getattr(repo_info, "name", None) or executor.filters.get("source"),
        "files_parsed": ctx.get("files_parsed", 0),
        "entities_found": ctx.get("entities_found", 0),
        "nodes_created": graph_stats.get("nodes_created", 0),
        "relationships_created": graph_stats.get("relationships_created", 0),
        "api_links_created": ctx.get("api_links_created", 0),
        "nodes_indexed": ctx.get("nodes_indexed", 0),
    }

    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)
    for k, v in stats.items():
        logger.info(f"{k}: {v}")
    logger.info("=" * 60)

    return stats
