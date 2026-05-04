from pathlib import Path
from typing import Any, Dict, List, Tuple

from src.core.graph import GraphBuilder, Neo4jClient
from src.core.parsers import ParseResult
from src.core.repo_loader import RepositoryInfo
from src.infra.logger import get_logger

logger = get_logger(__name__)


class GraphService:
    """Сборка knowledge-graph в Neo4j из распарсенных файлов."""

    def __init__(self, graph_builder: GraphBuilder, neo4j_client: Neo4jClient):
        self.graph_builder = graph_builder
        self.neo4j_client = neo4j_client

    def clear(self) -> None:
        logger.warning("Clearing existing graph data...")
        self.neo4j_client.clear_database()

    def build(
        self,
        repo_info: RepositoryInfo,
        parse_results: List[Tuple[Path, ParseResult]],
    ) -> Dict[str, int]:
        return self.graph_builder.build_graph(repo_info, parse_results)
