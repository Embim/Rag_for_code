from src.infra.logger import get_logger
from ..base.task import IndexingTask
from ..services import GraphService
from .parse_files import ParseFilesTask

logger = get_logger(__name__)


class BuildGraphTask(IndexingTask):
    """
    Сборка knowledge-graph в Neo4j из распарсенных файлов.

    filters: clear_existing — если True, перед записью чистит БД.
    context["graph_stats"] ← {nodes_created, relationships_created, ...}.
    """

    dependencies = [ParseFilesTask]

    def run(self, filters):
        service = GraphService(
            graph_builder=self.executor.graph_builder,
            neo4j_client=self.executor.neo4j_client,
        )

        if filters.get("clear_existing"):
            service.clear()

        stats = service.build(
            self.context["repo_info"],
            self.context["parse_results"],
        )

        self.context["graph_stats"] = stats
        logger.info(
            f"Graph built: {stats.get('nodes_created', 0)} nodes, "
            f"{stats.get('relationships_created', 0)} relationships"
        )
