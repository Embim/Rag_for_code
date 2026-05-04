"""
QualityService — LLM-оценка качества контекста.

Вход: query + список chunks. Выход: (score 0..1, feedback).
"""

from typing import Any, Dict, List, Tuple

from src.infra.logger import get_logger
from ._llm import LLMRole

logger = get_logger(__name__)


PROMPT = """Оцени релевантность контекста к вопросу. Кратко.

Вопрос: {query}

Контекст:
{context}

Учти что graph chunks (CONTAINS/CALLS/CALLS_TREE) — это методы класса
и цепочки вызовов; это сильный сигнал релевантности, не слабый.

Ответь СТРОГО:
SCORE: <0.0..1.0>
FEEDBACK: <одна строка — что не так или почему хорошо>"""


CODE_TRUNCATE = 800        # rerank уже сделал отбор top-20/15 — короткая
                           # цитата 25-30 LOC даёт достаточно для оценки.
MAX_CHUNKS_FOR_EVAL = 20   # после rerank top-20 покрывают релевантную часть
                           # (раньше 40 было нужно потому что rerank'а не было)


class QualityService:
    def __init__(self, llm: LLMRole):
        self.llm = llm

    def assess(self, *, query: str, chunks: List[Dict[str, Any]]) -> Tuple[float, str]:
        if not chunks:
            return 0.0, "Контекст не найден. Нужно переформулировать запрос."

        # Source-метка нужна quality LLM, чтобы понять что есть graph chunks.
        # Иначе модель видит только тип ноды (Method, Function, Class) и не
        # различает primary vs graph-expansion.
        context_str = "\n\n".join(
            f"[source={c.get('source') or '?'} type={c.get('type') or 'Unknown'}] "
            f"{c.get('name') or 'Unknown'} ({c.get('file') or ''})"
            f"\n```\n{str(c.get('code') or '')[:CODE_TRUNCATE]}\n```"
            for c in chunks[:MAX_CHUNKS_FOR_EVAL]
        )

        prompt = PROMPT.format(query=query, context=context_str)
        response = self.llm.call(
            prompt,
            name="quality_check",
            max_tokens=512,
        )
        return _parse(response)


def _parse(response: str) -> Tuple[float, str]:
    score = 0.5
    feedback = "Не удалось оценить качество"
    for line in response.splitlines():
        line = line.strip()
        if line.startswith("SCORE:"):
            try:
                score = max(0.0, min(1.0, float(line.replace("SCORE:", "").strip())))
            except ValueError:
                pass
        elif line.startswith("FEEDBACK:"):
            feedback = line.replace("FEEDBACK:", "").strip()
    return score, feedback
