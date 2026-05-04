from typing import Any, Dict, List, Optional, Type, TYPE_CHECKING

if TYPE_CHECKING:
    from .executor import SearchExecutor


class SearchTask:
    """
    Базовый таск поискового пайплайна.

    Контракт:
    - dependencies: one-shot предусловия (резолвятся ``executor.run``,
      повторно один и тот же класс не запускается).
    - run(filters): достать из ``self.context``, вызвать сервис, записать назад.

    Опциональные tracing-hooks (используются ExecutorHooks):
    - trace_input(context, filters) → dict | None
    - trace_output(context) → dict | None

    Repeatable-семантика — на стороне вызывающего: одни таски запускаются
    через ``executor.run`` (one-shot), другие через ``executor.run_again``
    (повтор в loop'е). Сам класс таска про это не знает.
    """

    dependencies: List[Type["SearchTask"]] = []

    def __init__(self, executor: "SearchExecutor"):
        self.executor = executor
        self.context: Dict[str, Any] = executor.context

    def run(self, filters: Dict[str, Any]) -> None:
        raise NotImplementedError("Define run() in your task")

    # ---- Tracing hooks (optional override on subclasses) ----

    @staticmethod
    def trace_input(
        context: Dict[str, Any], filters: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        return None

    @staticmethod
    def trace_output(context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        return None
