"""
CypherEnrichTask — третий источник контекста для LLM (после vector‑search).

LLM генерирует Cypher по NL-вопросу пользователя, мы исполняем его, превращаем
строки результата в "chunk'и" с ``source='cypher'`` и добавляем к существующим
``context['context']``.

Repeatable — может крутиться внутри RagControllerTask на каждой итерации
вместе с CollectContextTask.
"""

from typing import Any, Dict, List

from src.infra.logger import get_logger
from ..base.task import SearchTask
from ..services import CypherGenerationService

logger = get_logger(__name__)


def _row_to_chunk(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Превращает одну строку cypher‑результата в chunk-структуру, совместимую
    с теми, что приходят из RetrieverService (для LLM-контекста).

    Cypher-результаты бывают разными: одиночное число (count), список имён,
    нода целиком. Превращаем в текстовое представление.
    """
    if isinstance(row, dict) and len(row) == 1:
        # одиночное значение: count, name, и т.п.
        (k, v), = row.items()
        text = f"{k}: {v}"
    else:
        text = ', '.join(f"{k}={v!r}" for k, v in row.items())

    return {
        'id': '',
        'name': text[:120],
        'type': 'CypherRow',
        'file': '',
        'code': text,
        'score': 1.0,
        'source': 'cypher',
        'raw': row,
    }


class CypherEnrichTask(SearchTask):
    """
    Дёргает LLM → cypher → выполнение → chunks.

    Если LLM вернул SKIP или сгенерил невалидный cypher — task завершается
    тихо без эффекта. Не блокирует pipeline.

    context['context'] ← extend with cypher-chunks (source='cypher').
    context['cypher_query'] ← последний сгенерированный cypher.
    context['cypher_rows']  ← число извлечённых строк.
    """

    dependencies = []

    def run(self, filters: Dict[str, Any]) -> None:
        # ВАЖНО: cypher генерим по ОРИГИНАЛЬНОМУ запросу пользователя
        # (не ``current_query`` после rewrite). Структура графа стабильна,
        # rewrite query семантически близок → cypher был бы тот же,
        # а LLM-вызов стоит ~5 секунд. Лучше прогнать cypher один раз
        # за весь loop, а не на каждой итерации.
        query = filters["query"]

        # Дедуп: query не меняется между итерациями, поэтому повторно
        # генерировать и исполнять cypher не нужно. Сохраняем уже
        # накопленный context как есть.
        if self.context.get("_cypher_last_query") == query:
            return
        self.context["_cypher_last_query"] = query

        service = CypherGenerationService(llm=self.executor.cypher_llm)

        cypher = service.generate(query)
        self.context["cypher_query"] = cypher or ""
        if not cypher:
            self.context["cypher_rows"] = 0
            return

        rows = service.execute(self.executor.retriever.neo4j, cypher)
        self.context["cypher_rows"] = len(rows)
        if not rows:
            return

        chunks = [_row_to_chunk(r) for r in rows]
        # Не дедупим по id — у cypher-row'ов id пустой.
        existing: List[Dict[str, Any]] = self.context.get("context") or []
        existing.extend(chunks)
        self.context["context"] = existing

        logger.info(
            f"[cypher] +{len(chunks)} chunks from cypher; "
            f"total context now {len(existing)}"
        )

    @staticmethod
    def trace_input(context, filters):
        # Cypher работает на оригинальном query (см. run()), отражаем то
        # же самое в трассе.
        return {"query": filters.get("query")}

    @staticmethod
    def trace_output(context):
        return {
            "cypher": context.get("cypher_query") or "",
            "rows": context.get("cypher_rows", 0),
        }
