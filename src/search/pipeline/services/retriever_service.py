"""
RetrieverService — фасад над CodeRetriever, делает поиск + dedup + нормализацию
в плоский список chunks (как ожидает quality/answer-генерация).
"""

from typing import Any, Dict, List

from src.search.retrieval import CodeRetriever, SearchStrategy
from src.infra.logger import get_logger

logger = get_logger(__name__)


# Дефолтный retrieval-конфиг.
DEFAULT_RETRIEVAL_CONFIG: Dict[str, Any] = {
    "top_k_vector": 80,
    "top_k_final": 40,
    "expand_results": True,
    "hybrid_alpha": 0.7,
}

PRIMARY_LIMIT = 40
GRAPH_LIMIT = 30
CODE_TRUNCATE = 2000


class RetrieverService:
    def __init__(self, retriever: CodeRetriever):
        self.retriever = retriever

    def search(
        self,
        *,
        query: str,
        strategy: SearchStrategy,
        config_override: Dict[str, Any] = None,
    ) -> List[Dict[str, Any]]:
        config = config_override or DEFAULT_RETRIEVAL_CONFIG

        result = self.retriever.search(
            query=query,
            strategy=strategy,
            config_override=config,
        )

        logger.info(
            f"[search] strategy={strategy.value} primary={len(result.primary_nodes)} "
            f"expanded={len(result.expanded_nodes)}"
        )

        return self._dedup_and_format(result)

    # ------------------------------------------------------------------

    @staticmethod
    def _dedup_and_format(result: Any) -> List[Dict[str, Any]]:
        seen: set = set()
        chunks: List[Dict[str, Any]] = []

        def _key(node: Dict[str, Any]) -> str:
            node_id = node.get("node_id") or node.get("id") or ""
            if node_id:
                return node_id
            return f"{node.get('name', '')}::{node.get('file_path') or node.get('file', '')}"

        for node in result.primary_nodes[:PRIMARY_LIMIT]:
            k = _key(node)
            if k in seen:
                continue
            seen.add(k)
            chunks.append(_node_to_chunk(node, source="primary"))

        for node in result.expanded_nodes[:GRAPH_LIMIT]:
            k = _key(node)
            if k in seen:
                continue
            seen.add(k)
            chunks.append(_node_to_chunk(node, source="graph"))

        return chunks


def _node_to_chunk(node: Dict[str, Any], *, source: str) -> Dict[str, Any]:
    chunk = {
        "id": node.get("node_id") or node.get("id", ""),
        "name": node.get("name", "Unknown"),
        "type": node.get("node_type") or node.get("type", "Unknown"),
        "file": node.get("file_path") or node.get("file", ""),
        "code": (node.get("code") or node.get("content", ""))[:CODE_TRUNCATE],
        "score": node.get("score", 0.0),
        "source": source,
    }
    if source == "graph":
        chunk["relationship"] = node.get("relationship", "RELATED")
    return chunk
