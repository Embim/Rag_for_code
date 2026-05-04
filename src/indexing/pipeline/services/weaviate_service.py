from src.core.graph import WeaviateIndexer
from src.infra.logger import get_logger

logger = get_logger(__name__)


class WeaviateService:
    """Создание схемы и индексация нод из Neo4j в Weaviate."""

    def __init__(self, weaviate_indexer: WeaviateIndexer):
        self.weaviate_indexer = weaviate_indexer

    def index_from_neo4j(self, batch_size: int = 50) -> int:
        self.weaviate_indexer.create_schema()
        return self.weaviate_indexer.index_from_neo4j(batch_size=batch_size)
