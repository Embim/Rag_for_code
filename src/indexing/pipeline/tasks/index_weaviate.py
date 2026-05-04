from src.infra.logger import get_logger
from ..base.task import IndexingTask
from ..services import WeaviateService
from .build_graph import BuildGraphTask

logger = get_logger(__name__)


class IndexWeaviateTask(IndexingTask):
    """
    Семантическое индексирование нод графа в Weaviate.

    context["nodes_indexed"] ← int.
    """

    dependencies = [BuildGraphTask]

    def run(self, filters):
        service = WeaviateService(weaviate_indexer=self.executor.weaviate_indexer)
        try:
            count = service.index_from_neo4j(batch_size=50)
            self.context["nodes_indexed"] = count
            logger.info(f"Indexed {count} nodes in Weaviate")
        except Exception as e:
            logger.error(f"Weaviate indexing failed: {e}")
            self.context["nodes_indexed"] = 0
