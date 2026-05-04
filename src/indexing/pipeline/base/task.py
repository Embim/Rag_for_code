from typing import List, Type, TYPE_CHECKING, Dict, Any

if TYPE_CHECKING:
    from .executor import IndexingExecutor


class IndexingTask:
    """
    Базовый класс для шагов индексации.

    Каждая конкретная таска описывает:
    - dependencies: список тасок, которые должны выполниться раньше;
    - run(filters): чтение нужных входов из self.context, вызов сервиса,
      запись результата обратно в self.context.

    Сами компоненты (Neo4jClient, GraphBuilder, ...) доступны через
    self.executor — DI-инжектятся в IndexingExecutor.
    """

    dependencies: List[Type["IndexingTask"]] = []

    def __init__(self, executor: "IndexingExecutor"):
        self.executor = executor
        self.context: Dict[str, Any] = executor.context

    def run(self, filters: Dict[str, Any]) -> None:
        raise NotImplementedError("Define run() in your task")
