from contextlib import contextmanager
from typing import Any, Dict, Iterator, Protocol, Type, runtime_checkable


@runtime_checkable
class ExecutorHooks(Protocol):
    """
    Интерфейс хуков SearchExecutor.

    around_run — context manager, оборачивающий вызов task.run(). Внутри
    можно открыть Langfuse-span, лог-таймер, прокинуть ошибки в Sentry,
    и т.п. Сам таск ничего об этом не знает.
    """

    @contextmanager
    def around_run(
        self,
        task_class: Type[Any],
        context: Dict[str, Any],
        filters: Dict[str, Any],
    ) -> Iterator[None]:
        ...


class NoOpHooks:
    """Дефолтная реализация — ничего не делает."""

    @contextmanager
    def around_run(
        self,
        task_class: Type[Any],
        context: Dict[str, Any],
        filters: Dict[str, Any],
    ) -> Iterator[None]:
        yield
