from typing import Any, Dict, Optional, Set, Type

from src.infra.logger import get_logger
from ..hooks import ExecutorHooks, NoOpHooks
from ..services._llm import LLMRole
from .task import SearchTask

logger = get_logger(__name__)


class SearchExecutor:
    """
    Оркестратор поискового пайплайна.

    Отличия от IndexingExecutor:
    - ``run_again(TaskClass)`` — повторный запуск таска без проверки executed-set.
      Используется внутри RagControllerTask для тел loop'а.
    - hooks (``ExecutorHooks``) обёрнут вокруг каждого вызова таска через
      ``around_run`` (context manager). Tracing/метрики/прочее делаются здесь,
      а не размазываются по таскам.

    Каждая роль pipeline (quality / rewrite / answer / cypher) передаётся
    как ``LLMRole`` — это пара (primary, optional fallback). Failover между
    primary и fallback моделями прозрачен для сервисов.
    """

    def __init__(
        self,
        *,
        retriever: Any,
        quality_llm: LLMRole,
        rewrite_llm: LLMRole,
        answer_llm: LLMRole,
        cypher_llm: Optional[LLMRole] = None,
        filters: Optional[Dict[str, Any]] = None,
        hooks: Optional[ExecutorHooks] = None,
        repo_paths: Optional[Any] = None,
    ):
        # DI
        self.retriever = retriever
        self.quality_llm = quality_llm
        self.rewrite_llm = rewrite_llm
        self.answer_llm = answer_llm
        # cypher_llm fallback'ает на answer_llm если не задан явно.
        self.cypher_llm: LLMRole = cypher_llm or answer_llm

        # RepoPathResolver — резолвинг абсолютных путей репо для
        # GrepEnrichTask и read_code. Опциональный: если None, GrepEnrichTask
        # деградирует до legacy-источников (env RAG_REPOS_DIR /
        # weaviate.repos_dir). Тип Any чтобы не тянуть импорт core.graph
        # в pipeline/base.
        self.repo_paths: Optional[Any] = repo_paths

        # Бизнес-параметры запуска
        self.filters: Dict[str, Any] = filters or {}

        # Состояние
        self.context: Dict[str, Any] = {}
        self.executed: Set[Type[SearchTask]] = set()

        # Tracing / metrics / etc.
        self.hooks: ExecutorHooks = hooks or NoOpHooks()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, task_class: Type[SearchTask]) -> None:
        """One-shot: дедуп + резолв deps."""
        if task_class in self.executed:
            return
        for dep in task_class.dependencies:
            self.run(dep)
        self._invoke(task_class)
        self.executed.add(task_class)

    def run_again(self, task_class: Type[SearchTask]) -> None:
        """
        Повторный запуск для тела loop'а: игнорирует executed-set.

        Deps по-прежнему резолвятся через обычный ``run`` (one-shot).
        Сам таск в executed НЕ добавляется — следующий ``run_again`` снова
        его выполнит.
        """
        for dep in task_class.dependencies:
            self.run(dep)
        self._invoke(task_class)

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _invoke(self, task_class: Type[SearchTask]) -> None:
        logger.info(f"[RUN] {task_class.__name__}")
        task = task_class(self)
        with self.hooks.around_run(task_class, self.context, self.filters):
            task.run(self.filters)
