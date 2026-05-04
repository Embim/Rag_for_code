from ..base.task import SearchTask
from ..services import RetrieverService


# Источники, которые накапливаются дешёвыми enrichment-задачами и
# дедупятся по оригинальному query — их НЕ нужно сбрасывать на новой
# итерации rewrite'а. Иначе grep/graph/cypher chunks, добавленные на
# iter 1, будут затёрты `self.context["context"] = ...` на iter 2,
# а сами enrichment-задачи дедуп пропустит → контекст потерян.
_PERSIST_SOURCES = ('grep', 'graph', 'cypher')


class CollectContextTask(SearchTask):
    """
    Repeatable (запускается через executor.run_again в RagControllerTask).

    Берёт context["current_query"] (или filters["query"] на первой итерации),
    делает поиск через RetrieverService, обновляет PRIMARY часть
    ``context["context"]`` — но сохраняет накопленные ``grep``/``graph``/
    ``cypher`` chunks, чтобы они не терялись при rewrite-итерациях.
    Инкрементит счётчик итераций.
    """

    dependencies = []

    def run(self, filters):
        query = self.context.get("current_query") or filters["query"]
        strategy = self.context["strategy"]

        service = RetrieverService(self.executor.retriever)
        primary_chunks = service.search(
            query=query,
            strategy=strategy,
            config_override=filters.get("retrieval_config"),
        )

        # Сохраняем enrichment-chunks от предыдущих итераций. Весь primary
        # обновляется свежим результатом (под новый rewrite query).
        existing = self.context.get("context") or []
        enrichment = [
            c for c in existing
            if (c.get("source") or "") in _PERSIST_SOURCES
        ]

        self.context["context"] = primary_chunks + enrichment
        self.context["iterations"] = self.context.get("iterations", 0) + 1

    @staticmethod
    def trace_input(context, filters):
        return {
            "query": context.get("current_query") or filters.get("query"),
            "iteration": context.get("iterations", 0),
        }

    @staticmethod
    def trace_output(context):
        chunks = context.get("context", [])
        return {
            "chunks": len(chunks),
            "primary": sum(1 for c in chunks if c.get("source") == "primary"),
            "grep": sum(1 for c in chunks if c.get("source") == "grep"),
            "graph": sum(1 for c in chunks if c.get("source") == "graph"),
            "cypher": sum(1 for c in chunks if c.get("source") == "cypher"),
            "node_names": [c.get("name") for c in chunks[:10]],
        }
