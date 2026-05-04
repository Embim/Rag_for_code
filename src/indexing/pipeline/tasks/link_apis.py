from src.infra.logger import get_logger
from ..base.task import IndexingTask
from ..services import ApiLinkingService
from .build_graph import BuildGraphTask

logger = get_logger(__name__)


class LinkApisTask(IndexingTask):
    """
    Связывает frontend ↔ backend в графе (CALLS_ENDPOINT (через ApiCallNode)).

    context["api_links_created"] ← int.
    """

    dependencies = [BuildGraphTask]

    def run(self, filters):
        service = ApiLinkingService(
            neo4j_client=self.executor.neo4j_client,
            api_linker=self.executor.api_linker,
        )
        try:
            self.context["api_links_created"] = service.link()
        except Exception as e:
            logger.error(f"API linking failed: {e}")
            self.context["api_links_created"] = 0
