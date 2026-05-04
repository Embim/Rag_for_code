"""
RerankEnrichTask — пятый этап обогащения контекста: cross-encoder rerank
на ``grep`` и ``graph`` chunks.

Зачем нужен
-----------
Primary chunks из Weaviate уже отrerank'нуты cross-encoder'ом внутри
``_initial_search``. А вот ``grep`` и ``graph`` chunks добавляются "сырыми":

- ``grep_enrich`` берёт хиты по идентификаторам — они отсортированы по
  частоте/файлу, не по релевантности к query.
- ``graph_expand`` вытаскивает ноды по структурным паттернам (CONTAINS
  все методы класса; CALLS callees) — там вообще нет понятия score.

При большом классе вроде ``PositionByBookService`` (18 методов) это значит:
``__init__``, ``_get_market_data``, утилитарные хелперы попадают в LLM
наравне с ключевыми ``_get_position_*_by_book``. На 220 KB контекста
слабые модели (30B free) "тонут" — не доплавляют до конца.

Cross-encoder читает (query, chunk_text) парой и выдаёт реальный score
релевантности — отсортирует методы по тому, как часто и плотно они
связаны с темой запроса.

Эффект на запрос «как считается net_pos»:
- 60 graph chunks → top-15 по релевантности (~50 KB вместо 220 KB)
- 60 grep hits → top-20 по релевантности

Repeatable + дедуп по оригинальному query (как у grep/cypher/graph):
rerank — детерминистическая функция, повторно прогонять нет смысла,
если query не меняется.
"""

from typing import Any, Dict, List

from src.infra.logger import get_logger
from ..base.task import SearchTask

logger = get_logger(__name__)


class RerankEnrichTask(SearchTask):
    """
    Cross-encoder rerank поверх grep + graph chunks.

    Дедупится по оригинальному query.

    context['context']        ← обновляется (тот же primary + reranked grep/graph)
    context['rerank_grep']    ← кол-во grep chunks после rerank
    context['rerank_graph']   ← кол-во graph chunks после rerank
    """

    dependencies = []

    # Сколько оставлять после rerank каждого источника.
    # Подобрано эмпирически: при top-15 graph + top-20 grep промпт
    # падает с ~320 KB до ~140 KB — слабые модели уже доплавляют до конца.
    TOP_GREP = 20
    TOP_GRAPH = 15
    # Минимум исходных chunks чтобы вообще запускать rerank — нет смысла
    # тратить 200мс если их и так мало.
    MIN_CHUNKS_TO_RERANK = 8

    def run(self, filters: Dict[str, Any]) -> None:
        chunks: List[Dict[str, Any]] = self.context.get("context") or []

        # Дедуп по СОСТАВУ chunks, не по query: на iter 2+ GraphExpand
        # может добавить новые ноды от новых якорей (rewrite-query →
        # новые primary → новые якоря в графе). Если просто дедупить по
        # query, rerank скипнется и эти новые chunks влезут в LLM сырыми.
        # Вместо этого считаем хеш набора входных id — если изменился,
        # запускаем rerank ещё раз.
        ids_signature = frozenset(
            c.get("id") or "" for c in chunks
            if c.get("source") in ("grep", "graph")
        )
        if self.context.get("_rerank_last_ids") == ids_signature:
            return
        self.context["_rerank_last_ids"] = ids_signature
        if not chunks:
            self.context["rerank_grep"] = 0
            self.context["rerank_graph"] = 0
            return

        # Rerank работает на оригинальном query (не на rewrite): идентификаторы
        # стабильны, rewrite добавляет полусинонимы которые сбивают cross-encoder.
        original_query = filters["query"]

        # Разделение по источникам. primary уже отrerank'нут в _initial_search;
        # не трогаем (повторный rerank — лишняя работа без эффекта).
        # cypher — короткие факты, тоже не трогаем.
        primary: List[Dict[str, Any]] = []
        grep: List[Dict[str, Any]] = []
        graph: List[Dict[str, Any]] = []
        cypher: List[Dict[str, Any]] = []
        other: List[Dict[str, Any]] = []
        for c in chunks:
            src = c.get("source") or ""
            if src == "primary":
                primary.append(c)
            elif src == "grep":
                grep.append(c)
            elif src == "graph":
                graph.append(c)
            elif src == "cypher":
                cypher.append(c)
            else:
                other.append(c)

        reranker = self._get_reranker()
        if reranker is None:
            # Нет cross-encoder'а (модель не загружена / ранкер выключен) —
            # task в no-op. Контекст не трогаем.
            self.context["rerank_grep"] = len(grep)
            self.context["rerank_graph"] = len(graph)
            return

        new_grep = self._rerank_or_keep(reranker, original_query, grep, self.TOP_GREP, "grep")
        new_graph = self._rerank_or_keep(reranker, original_query, graph, self.TOP_GRAPH, "graph")

        # Сохраняем очерёдность: primary (топ ↑) → grep → graph → cypher → other.
        # primary остаётся в исходном порядке (rerank Weaviate его уже задал).
        self.context["context"] = primary + new_grep + new_graph + cypher + other
        self.context["rerank_grep"] = len(new_grep)
        self.context["rerank_graph"] = len(new_graph)

        logger.info(
            f"[rerank] grep {len(grep)}->{len(new_grep)}, "
            f"graph {len(graph)}->{len(new_graph)} "
            f"(total context {len(chunks)}->{len(primary) + len(new_grep) + len(new_graph) + len(cypher) + len(other)})"
        )

    def _get_reranker(self) -> Any:
        """
        Берём тот же singleton cross-encoder, что использует ``CodeRetriever``
        для primary rerank — модель уже загружена в GPU, дополнительной
        нагрузки нет.
        """
        # 1) Если ретривер хранит его — переиспользуем.
        try:
            r = self.executor.retriever.reranker
            if r is not None:
                return r
        except AttributeError:
            pass

        # 2) Иначе берём singleton из ranking-модуля.
        try:
            from src.search.ranking.cross_encoder import get_reranker
            return get_reranker()
        except Exception as e:
            logger.warning(f"[rerank] reranker unavailable: {e}")
            return None

    def _rerank_or_keep(
        self,
        reranker: Any,
        query: str,
        items: List[Dict[str, Any]],
        top_k: int,
        label: str,
    ) -> List[Dict[str, Any]]:
        """
        Если items мало — ничего не делаем (rerank не оправдан).
        Если items >= MIN — rerank через cross-encoder, top_k оставляем.
        """
        if len(items) <= self.MIN_CHUNKS_TO_RERANK:
            return items

        try:
            # rerank_with_scores возвращает список (doc, score) tuples,
            # отсортированный по score desc.
            scored = reranker.rerank_with_scores(query, items)
        except Exception as e:
            logger.warning(f"[rerank] {label} failed: {e}")
            return items

        kept: List[Dict[str, Any]] = []
        for doc, score in scored[:top_k]:
            # Положим rerank_score в chunk для трассы/отладки.
            doc_copy = dict(doc)
            doc_copy['rerank_score'] = float(score)
            kept.append(doc_copy)
        return kept

    @staticmethod
    def trace_input(context, filters):
        chunks = context.get("context", [])
        return {
            "query": filters.get("query"),
            "grep_in": sum(1 for c in chunks if c.get("source") == "grep"),
            "graph_in": sum(1 for c in chunks if c.get("source") == "graph"),
        }

    @staticmethod
    def trace_output(context):
        return {
            "rerank_grep": context.get("rerank_grep", 0),
            "rerank_graph": context.get("rerank_graph", 0),
        }
