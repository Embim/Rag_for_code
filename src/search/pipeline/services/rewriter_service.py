"""
RewriterService — LLM переписывает запрос **после** retrieval на основе
quality-feedback (используется в RagControllerTask внутри loop'а).

Не путать с :class:`src.search.preprocessing.reformulation.QueryReformulator`,
который работает **до** retrieval (preprocessing) с разными стратегиями
(simple/expanded/multi/rephrase/...) и дисковым кешем. Контракт другой:
``QueryReformulator`` принимает только ``query`` и ``method``;
``RewriterService.rewrite`` принимает ``(original, current, feedback)``
и нужен внутри RAG‑loop'а с обратной связью.
"""

from ._llm import LLMRole


PROMPT = """Ты - эксперт по поиску в кодовой базе.

Оригинальный запрос: {original}
Текущий запрос: {current}
Проблема с контекстом: {feedback}

Перепиши запрос чтобы найти более релевантный код.
Используй:
- Технические термины (названия функций, классов)
- Ключевые слова из домена
- Более узкий фокус на конкретную часть

Ответь ТОЛЬКО новым запросом, без пояснений."""


class RewriterService:
    def __init__(self, llm: LLMRole):
        self.llm = llm

    def rewrite(self, *, original: str, current: str, feedback: str) -> str:
        prompt = PROMPT.format(original=original, current=current, feedback=feedback)
        response = self.llm.call(
            prompt,
            name="query_rewrite",
            max_tokens=256,
        )
        new_query = response.strip().strip("\"'")
        return new_query or current
