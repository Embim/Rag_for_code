"""
LLM helper с поддержкой failover.

Архитектура:
- ``ModelRoute(client, model)`` — одна "точка": конкретный OpenAI-совместимый
  клиент + имя модели.
- ``LLMRole(primary, fallback=None)`` — роль (quality / rewrite / answer / cypher)
  с failover: при ошибке primary автоматически дёргает fallback.
- ``LLMRole.call(prompt, ...)`` — единый вход, прозрачно проксирует и трассирует
  через Langfuse (если он сконфигурирован).

Сервисы получают ``LLMRole``, не знают ничего про fallback‑логику.
"""

from dataclasses import dataclass
from typing import Any, Optional

from src.infra.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ModelRoute:
    """
    Конкретная LLM-точка: client (OpenAI-compat) + имя модели.

    OpenRouter и OpenAI-direct обе совместимы с этим интерфейсом.
    """
    client: Any
    model: str

    def call(
        self,
        prompt: str,
        *,
        max_tokens: int = 2048,
        temperature: float = 0.1,
    ) -> str:
        """
        Один HTTP-вызов. Поддерживает reasoning-модели (gpt-5*, o1, o3, gpt-oss):

        - Если в ответе ``content=""``, fallback на ``reasoning``/``reasoning_content``.
        - Для reasoning-моделей по умолчанию передаём ``reasoning_effort=low``,
          чтобы они не тратили большую часть output-tokens на скрытое
          "думание". Для нашего use case — генерация ответа по уже подобранному
          контекстом — глубокий reasoning не нужен; вся "работа мозгов"
          сделана retrieval'ом.
        - Управляется env'ом ``RAG_REASONING_EFFORT`` (low/medium/high/none).
          ``none`` — не передавать параметр (для не-reasoning-моделей или
          когда хочется дефолт провайдера).
        """
        kwargs: dict = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        # Аргументы reasoning-моделей. OpenAI/OpenRouter принимают через
        # extra_body или непосредственно как kwarg — клиенты openai>=1.40
        # умеют ``reasoning_effort`` напрямую. Используем extra_body чтобы
        # быть совместимым с любым форматом провайдера.
        effort = _reasoning_effort_for(self.model)
        if effort:
            # OpenRouter: ``reasoning: { effort: "low" }``.
            # OpenAI Responses-API: ``reasoning_effort`` как top-level.
            # Передаём оба варианта через extra_body — клиент проигнорирует
            # тот, который провайдер не знает.
            kwargs["extra_body"] = {
                "reasoning": {"effort": effort},
                "reasoning_effort": effort,
            }

        try:
            response = self.client.chat.completions.create(**kwargs)
        except TypeError:
            # Старый openai SDK не понимает extra_body — повторяем без него.
            kwargs.pop("extra_body", None)
            response = self.client.chat.completions.create(**kwargs)

        msg = response.choices[0].message
        content = msg.content
        if not content:
            content = (
                getattr(msg, "reasoning", None)
                or getattr(msg, "reasoning_content", None)
                or ""
            )
        return (content or "").strip()


# Reasoning-модели (по имени) — для них стоит ограничить effort.
_REASONING_MODEL_PREFIXES = (
    "openai/gpt-5",
    "openai/o1",
    "openai/o3",
    "openai/o4",
    "openai/gpt-oss",
    "deepseek/deepseek-r1",
    "anthropic/claude-3.7-sonnet:thinking",
)


def _reasoning_effort_for(model: str) -> Optional[str]:
    """
    Какой ``reasoning_effort`` передавать для конкретной модели.

    - Если в env ``RAG_REASONING_EFFORT=none`` — не передаём ничего.
    - Если задан (low/medium/high) — используем для всех reasoning-моделей.
    - По умолчанию для нашего use case — ``low`` (минимум скрытого reasoning,
      максимум видимого ответа).
    - Для не-reasoning моделей возвращаем None, чтобы не отправлять лишний
      параметр (некоторые провайдеры могут на нём упасть).
    """
    import os
    effort = (os.getenv("RAG_REASONING_EFFORT") or "low").strip().lower()
    if effort == "none" or not effort:
        return None
    if effort not in ("low", "medium", "high"):
        return None
    if not any(model.startswith(p) for p in _REASONING_MODEL_PREFIXES):
        return None
    return effort


@dataclass
class LLMRole:
    """
    Роль pipeline-а с опциональным failover.

    При ошибке primary (любой Exception) логирует WARNING и пробует fallback.
    Если fallback нет или он тоже упал — пробрасывает последнее исключение.
    """
    primary: ModelRoute
    fallback: Optional[ModelRoute] = None

    @classmethod
    def single(cls, client: Any, model: str) -> "LLMRole":
        """Удобный конструктор для случая без failover."""
        return cls(primary=ModelRoute(client=client, model=model))

    def call(
        self,
        prompt: str,
        *,
        name: str = "llm_call",
        max_tokens: int = 2048,
        temperature: float = 0.1,
    ) -> str:
        """
        Вызов с трассировкой Langfuse + failover.

        ``name`` — имя observation в Langfuse.
        """
        langfuse = _safe_langfuse_client()

        if langfuse is None:
            return self._call_with_failover(prompt, max_tokens, temperature)

        with langfuse.start_as_current_observation(name=name, as_type="generation"):
            try:
                langfuse.update_current_generation(input=prompt)
            except Exception:
                pass

            result = self._call_with_failover(
                prompt, max_tokens, temperature, langfuse=langfuse
            )

            try:
                langfuse.update_current_generation(output=result)
            except Exception:
                pass
            return result

    def _call_with_failover(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        *,
        langfuse: Optional[Any] = None,
    ) -> str:
        """Failover: пытаемся primary; при ошибке — fallback."""
        try:
            self._update_model_meta(langfuse, self.primary.model)
            return self.primary.call(
                prompt, max_tokens=max_tokens, temperature=temperature
            )
        except Exception as e:
            if self.fallback is None:
                logger.error(
                    f"[llm] primary {self.primary.model!r} failed "
                    f"and no fallback configured: {e}"
                )
                raise
            logger.warning(
                f"[llm] primary {self.primary.model!r} failed ({e}); "
                f"falling back to {self.fallback.model!r}"
            )
            self._update_model_meta(
                langfuse, self.fallback.model, fallback=True
            )
            return self.fallback.call(
                prompt, max_tokens=max_tokens, temperature=temperature
            )

    @staticmethod
    def _update_model_meta(
        langfuse: Optional[Any], model: str, *, fallback: bool = False
    ) -> None:
        if langfuse is None:
            return
        try:
            langfuse.update_current_generation(
                model=model,
                metadata={"fallback": fallback},
            )
        except Exception:
            pass


def _safe_langfuse_client() -> Optional[Any]:
    import os

    if not os.getenv("LANGFUSE_PUBLIC_KEY") and not os.getenv("LANGFUSE_SECRET_KEY"):
        return None
    try:
        from langfuse import get_client
        return get_client()
    except Exception:
        return None
