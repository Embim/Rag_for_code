"""
RagControllerTask — управляющий таск iterative-RAG loop'а.

Внутри ``run`` крутит while:
    collect_context → cypher_enrich → check_quality
        → (rewrite_query → next iter) | break.

Все тела loop'а вызываются через ``executor.run_again`` — дедуп executed-set
для них отключён.

CypherEnrichTask дополняет ``context['context']`` строками из cypher‑результата
по NL‑вопросу пользователя. Если LLM посчитает вопрос не‑structural — task
тихо завершится без эффекта (см. CypherGenerationService).
"""

from src.infra.logger import get_logger
from ..base.task import SearchTask
from .check_quality import CheckQualityTask
from .collect_context import CollectContextTask
from .cypher_enrich import CypherEnrichTask
from .detect_strategy import DetectStrategyTask
from .graph_expand import GraphExpandTask
from .grep_enrich import GrepEnrichTask
from .rerank_enrich import RerankEnrichTask
from .rewrite_query import RewriteQueryTask

logger = get_logger(__name__)


DEFAULT_QUALITY_THRESHOLD = 0.6
DEFAULT_MAX_ITERATIONS = 3
# Early stopping: если score не вырос на эту дельту за итерацию — считаем
# плато. Защищает от прокрутки 6 итераций когда LLM-оценка стабильно низкая
# но contextually больше уже добавлять нечего.
_PLATEAU_DELTA = 0.05
# Сколько подряд итераций "плато" до выхода. 2 — щадящее (даём шанс на
# отскок одной итерации шумного quality_check).
_PLATEAU_STOP_COUNT = 2


class RagControllerTask(SearchTask):
    dependencies = [DetectStrategyTask]

    def run(self, filters):
        # Init iteration state (если RagController вызван несколько раз —
        # current_query сбросится на оригинальный).
        self.context["current_query"] = filters["query"]
        self.context["iterations"] = 0

        max_iterations = int(
            filters.get("max_iterations", DEFAULT_MAX_ITERATIONS)
        )
        threshold = float(
            filters.get("quality_threshold", DEFAULT_QUALITY_THRESHOLD)
        )

        prev_score: float = -1.0     # score от прошлой итерации (для плато)
        plateau_streak: int = 0       # подряд итераций без роста

        while True:
            # 1. Vector search через Weaviate (primary chunks).
            self.executor.run_again(CollectContextTask)
            # 2. Grep по локальным репозиториям — точный поиск идентификаторов
            #    (net_pos, getPosition, …). Дёшево (~50-200 мс), без LLM.
            self.executor.run_again(GrepEnrichTask)
            # 3. Graph expansion от primary-якорей: для Class -> CONTAINS методы,
            #    Method/Function -> CALLS callees + multi-hop tree, и т.п.
            self.executor.run_again(GraphExpandTask)
            # 4. NL→Cypher enrichment (структурные факты по графу).
            self.executor.run_again(CypherEnrichTask)
            # 5. Cross-encoder rerank на grep+graph chunks. Без него крупный
            #    Class (18 методов через CONTAINS) кладёт в LLM 220 KB сырья;
            #    после rerank остаётся ~50 KB top-15 наиболее релевантных
            #    методов. Слабые модели (30B free) после этого доплавляют
            #    до конца контекста.
            self.executor.run_again(RerankEnrichTask)
            # 6. LLM-оценка собранного контекста.
            self.executor.run_again(CheckQualityTask)

            score = self.context.get("quality_score", 0.0)
            iterations = self.context.get("iterations", 0)
            cypher_rows = self.context.get("cypher_rows", 0)
            grep_hits = self.context.get("grep_hits", 0)
            graph_chunks = self.context.get("graph_chunks", 0)

            # Плато-детектор: считаем сколько итераций подряд score
            # практически не меняется.
            if prev_score >= 0 and (score - prev_score) < _PLATEAU_DELTA:
                plateau_streak += 1
            else:
                plateau_streak = 0
            prev_score = score

            logger.info(
                f"[RAG] iter={iterations} score={score:.2f} "
                f"threshold={threshold} max={max_iterations} "
                f"plateau={plateau_streak}/{_PLATEAU_STOP_COUNT} "
                f"cypher_rows={cypher_rows} grep_hits={grep_hits} "
                f"graph_chunks={graph_chunks}"
            )

            if score >= threshold:
                break
            if iterations >= max_iterations:
                break
            if plateau_streak >= _PLATEAU_STOP_COUNT:
                logger.info(
                    f"[RAG] early stop: plateau {plateau_streak} iters at "
                    f"score≈{score:.2f}, нет смысла rewriter'ить дальше"
                )
                break

            self.executor.run_again(RewriteQueryTask)

    @staticmethod
    def trace_input(context, filters):
        return {
            "query": filters.get("query"),
            "max_iterations": filters.get("max_iterations"),
            "quality_threshold": filters.get("quality_threshold"),
        }

    @staticmethod
    def trace_output(context):
        return {
            "iterations": context.get("iterations"),
            "final_query": context.get("current_query"),
            "quality_score": context.get("quality_score"),
        }
