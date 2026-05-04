"""
Iterative RAG search с quality feedback loop.

Сценарий тонкий: собирает ``LLMRole``-и (с поддержкой failover) из
``AgentConfig``, создаёт ``SearchExecutor``, запускает ``GenerateAnswerTask``.

Failover: если в конфиге заданы ``rag_<role>_model_fallback`` и/или
``api_base_fallback`` / ``api_key_fallback`` — при ошибке primary модели
автоматически переключаемся на fallback. См. ``_singletons.build_llm_role``.
"""

from contextlib import contextmanager
from typing import Any, Dict, Optional

from src.infra.config.agent import AgentConfig
from ..base.executor import SearchExecutor
from ..hooks import ExecutorHooks, LangfuseHooks
from ..services._llm import LLMRole
from ..tasks import GenerateAnswerTask
from . import _singletons


def run(
    query: str,
    *,
    max_iterations: int = 3,
    quality_threshold: float = 0.6,
    retriever: Optional[Any] = None,
    quality_llm: Optional[LLMRole] = None,
    rewrite_llm: Optional[LLMRole] = None,
    answer_llm: Optional[LLMRole] = None,
    cypher_llm: Optional[LLMRole] = None,
    hooks: Optional[ExecutorHooks] = None,
    config: Optional[AgentConfig] = None,
) -> Dict[str, Any]:
    """
    Запускает iterative-RAG для ``query``.

    DI-параметры опциональны:
    - retriever: lazy singleton CodeRetriever.
    - <role>_llm: LLMRole; если не задан — собирается из ``config`` через
      ``_singletons.build_llm_role(config, role)`` (с автоматическим failover).
    - hooks: по умолчанию ``LangfuseHooks`` (silent no-op без ключей).
    - config: ``AgentConfig`` — если не задан, берётся дефолтный.

    Возвращает dict с answer/sources/iterations/quality_score/quality_feedback.
    """

    cfg = config or AgentConfig()
    retriever = retriever if retriever is not None else _singletons.get_retriever()
    # RepoPathResolver инициализируется внутри get_retriever — берём его
    # после, чтобы не дублировать создание Neo4j-клиента.
    repo_paths = _singletons.get_repo_paths()

    quality_llm = quality_llm or _singletons.build_llm_role(cfg, "quality")
    rewrite_llm = rewrite_llm or _singletons.build_llm_role(cfg, "rewrite")
    answer_llm = answer_llm or _singletons.build_llm_role(cfg, "answer")
    cypher_llm = cypher_llm or _singletons.build_llm_role(cfg, "cypher")

    langfuse_hooks = hooks if hooks is not None else LangfuseHooks(root_span_name="rag_pipeline")

    executor = SearchExecutor(
        retriever=retriever,
        quality_llm=quality_llm,
        rewrite_llm=rewrite_llm,
        answer_llm=answer_llm,
        cypher_llm=cypher_llm,
        repo_paths=repo_paths,
        filters={
            "query": query,
            "max_iterations": max_iterations,
            "quality_threshold": quality_threshold,
        },
        hooks=langfuse_hooks,
    )

    root_cm = (
        langfuse_hooks.root(query=query)
        if isinstance(langfuse_hooks, LangfuseHooks)
        else _no_op_cm()
    )

    with root_cm:
        # Резолв deps:
        #   GenerateAnswerTask
        #     ← RagControllerTask (loop с repeatable‑телами:
        #         CollectContext → GrepEnrich → CypherEnrich → CheckQuality → Rewrite)
        #         ← DetectStrategyTask
        executor.run(GenerateAnswerTask)

        # Прокидываем финальный ответ в trace.output (через root span).
        # Без этого Langfuse показывает trace.output=None.
        if isinstance(langfuse_hooks, LangfuseHooks):
            answer = executor.context.get("answer") or ""
            iters = executor.context.get("iterations", 0)
            score = executor.context.get("quality_score", 0.0)
            # Компактный output: сам ответ + критические метрики на root.
            langfuse_hooks.finalize(
                f"[iters={iters} score={score:.2f}]\n\n{answer}"
            )

    return _build_result(executor.context)


def _build_result(ctx: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "answer": ctx.get("answer", ""),
        "sources": ctx.get("sources", []),
        "iterations": ctx.get("iterations", 0),
        "quality_score": ctx.get("quality_score", 0.0),
        "quality_feedback": ctx.get("quality_feedback", ""),
        "final_query": ctx.get("current_query", ""),
        "cypher_query": ctx.get("cypher_query", ""),
        "cypher_rows": ctx.get("cypher_rows", 0),
    }


@contextmanager
def _no_op_cm():
    yield
