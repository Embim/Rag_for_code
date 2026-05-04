from typing import Any, Dict, Optional, Set, Type

from src.infra.logger import get_logger
from .task import IndexingTask

logger = get_logger(__name__)


class IndexingExecutor:
    """
    Оркестратор индексации.

    Держит общий context (dict) и набор уже выполненных тасок.
    При run(TaskClass) рекурсивно резолвит TaskClass.dependencies,
    повторно один и тот же таск-класс не запускается.

    Технические компоненты передаются через DI в конструктор и доступны
    тасками как self.executor.<component>. Бизнес-параметры передаются
    в filters dict и прокидываются в task.run(filters).
    """

    def __init__(
        self,
        *,
        repo_loader: Any,
        graph_builder: Any,
        neo4j_client: Any,
        api_linker: Any,
        weaviate_indexer: Any,
        filters: Optional[Dict[str, Any]] = None,
    ):
        # Компоненты — DI
        self.repo_loader = repo_loader
        self.graph_builder = graph_builder
        self.neo4j_client = neo4j_client
        self.api_linker = api_linker
        self.weaviate_indexer = weaviate_indexer

        # Бизнес-параметры запуска
        self.filters: Dict[str, Any] = filters or {}

        # Состояние пайплайна
        self.context: Dict[str, Any] = {}
        self.executed: Set[Type[IndexingTask]] = set()

    def run(self, task_class: Type[IndexingTask]) -> None:
        if task_class in self.executed:
            return

        # Зависимости — рекурсивно
        for dependency in task_class.dependencies:
            self.run(dependency)

        logger.info(f"[RUN] {task_class.__name__}")

        task = task_class(self)
        task.run(self.filters)

        self.executed.add(task_class)
