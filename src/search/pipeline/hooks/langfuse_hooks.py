from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional, Type

from src.infra.logger import get_logger

logger = get_logger(__name__)


class LangfuseHooks:
    """
    Tracing-хук на Langfuse v3.

    Для каждого вызова таска открывает span с именем класса. Если у таска
    определены ``trace_input``/``trace_output`` — их результат записывается
    в span как input/output.

    Если Langfuse не установлен или клиент недоступен — деградирует до
    no-op без падения пайплайна.
    """

    def __init__(self, root_span_name: str = "rag_pipeline") -> None:
        self.root_span_name = root_span_name
        self._client = self._safe_get_client()

    @staticmethod
    def _safe_get_client() -> Optional[Any]:
        # Если ключи Langfuse не сконфигурены — отключаем tracing бесшумно.
        import os

        if not os.getenv("LANGFUSE_PUBLIC_KEY") and not os.getenv("LANGFUSE_SECRET_KEY"):
            return None

        try:
            from langfuse import get_client

            return get_client()
        except Exception as e:
            logger.warning(f"[Langfuse] Disabled: {e}")
            return None

    def _start_observation_cm(self, *, name: str, input_: Any):
        """
        Совместимость v3 (``start_as_current_span``) и v4
        (``start_as_current_observation``).

        У некоторых версий v4 первый аргумент — ``as_type='span'``, в v3 этого
        параметра нет. Пробуем v4 → v3.
        """
        # Сначала пробуем v4 (новый API с as_type)
        v4 = getattr(self._client, 'start_as_current_observation', None)
        if v4 is not None:
            try:
                return v4(name=name, as_type='span', input=input_)
            except TypeError:
                # v4-без as_type или иной мисмэтч сигнатуры
                try:
                    return v4(name=name, input=input_)
                except TypeError:
                    return v4(name=name)

        # Fallback к v3
        v3 = getattr(self._client, 'start_as_current_span', None)
        if v3 is not None:
            try:
                return v3(name=name, input=input_)
            except TypeError:
                return v3(name=name)
        return None

    def _set_trace_io(self, *, input_: Any = None, output: Any = None) -> None:
        """v4: ``set_current_trace_io``. v3: ``update_current_trace``."""
        kwargs: Dict[str, Any] = {}
        if input_ is not None:
            kwargs['input'] = input_
        if output is not None:
            kwargs['output'] = output
        if not kwargs:
            return
        for method_name in ('set_current_trace_io', 'update_current_trace'):
            fn = getattr(self._client, method_name, None)
            if fn is None:
                continue
            try:
                fn(**kwargs)
                return
            except Exception as e:
                logger.warning(f"[Langfuse] {method_name} failed: {e}")
        # Если ни один не сработал — тихо пропускаем, не ломаем pipeline.

    @contextmanager
    def around_run(
        self,
        task_class: Type[Any],
        context: Dict[str, Any],
        filters: Dict[str, Any],
    ) -> Iterator[None]:
        if self._client is None:
            yield
            return

        name = getattr(task_class, "span_name", None) or task_class.__name__

        try:
            trace_input = task_class.trace_input(context, filters)
        except Exception as e:
            logger.warning(f"[Langfuse] {name}.trace_input() failed: {e}")
            trace_input = None

        cm = self._start_observation_cm(name=name, input_=trace_input)
        if cm is None:
            # SDK не предоставляет ни v3, ни v4 API — деградируем
            yield
            return

        with cm:
            yield

            try:
                trace_output = task_class.trace_output(context)
            except Exception as e:
                logger.warning(f"[Langfuse] {name}.trace_output() failed: {e}")
                trace_output = None
            if trace_output is not None:
                fn = getattr(self._client, 'update_current_span', None)
                if fn is not None:
                    try:
                        fn(output=trace_output)
                    except Exception as e:
                        logger.warning(f"[Langfuse] {name} update_current_span failed: {e}")

    # ------------------------------------------------------------------
    # Опционально: открыть/закрыть корневой span на весь сценарий
    # ------------------------------------------------------------------

    @contextmanager
    def root(self, *, query: str, tags: Optional[list] = None) -> Iterator[None]:
        """
        Корневой span на весь сценарий. Дёргается из pipeline-сценария.

        После выхода из ``yield`` пытаемся прокинуть финальный ответ из
        ``context['answer']`` в ``trace.output`` (если он там есть) и
        вызываем ``flush()`` — без этого Langfuse v3 батчит события и они
        могут не долетать до сервера в долгоиграющем процессе (uvicorn).
        """
        if self._client is None:
            yield
            return

        # Holder для финальной информации, которую сценарий установит после
        # завершения pipeline (см. ``finalize`` ниже).
        self._last_root_answer: Optional[str] = None

        cm = self._start_observation_cm(
            name=self.root_span_name, input_={"query": query}
        )
        if cm is None:
            yield
            return

        with cm:
            self._set_trace_io(input_={"query": query})

            try:
                yield
            finally:
                output = self._last_root_answer
                if output is not None:
                    self._set_trace_io(output=output)
                    fn = getattr(self._client, 'update_current_span', None)
                    if fn is not None:
                        try:
                            fn(output=output)
                        except Exception as e:
                            logger.warning(
                                f"[Langfuse] update_current_span (root) failed: {e}"
                            )

                try:
                    self._client.flush()
                except Exception as e:
                    logger.warning(f"[Langfuse] flush() failed: {e}")

    def finalize(self, output: Any) -> None:
        """
        Сценарий зовёт по окончании pipeline, чтобы прокинуть финальный
        ответ в ``trace.output``. ``output`` обычно строка-ответ или dict
        с answer + sources + iterations.

        Безопасно для NoOp-конфига (если Langfuse не сконфигурен — ничего
        не делает).
        """
        if self._client is None:
            return
        self._last_root_answer = output
