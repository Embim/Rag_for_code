"""
Document Retrieval System for SOP/Policy documents.

Searches in DocumentNode collection in Weaviate.
"""

import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from ..graph import WeaviateIndexer
from ...logger import get_logger


logger = get_logger(__name__)


@dataclass
class DocumentSearchConfig:
    """Configuration for document search."""

    top_k: int = 10  # Number of documents to return
    hybrid_alpha: float = 0.5  # 0.0 = keyword only, 1.0 = semantic only
    document_types: Optional[List[str]] = None  # Filter by type (SOP, Policy, etc.)


@dataclass
class DocumentSearchResult:
    """Result of document search."""

    documents: List[Dict[str, Any]] = field(default_factory=list)
    total_found: int = 0
    execution_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'documents': self.documents,
            'total_found': self.total_found,
            'execution_time_ms': self.execution_time_ms
        }


class DocumentRetriever:
    """
    Document retrieval system for SOP/Policy documents.

    Searches in Weaviate DocumentNode collection using:
    - Semantic search (vector similarity)
    - Keyword search (BM25)
    - Hybrid search (combination)
    """

    def __init__(
        self,
        weaviate_indexer: WeaviateIndexer,
        config: Optional[DocumentSearchConfig] = None
    ):
        """
        Initialize document retriever.

        Args:
            weaviate_indexer: Weaviate indexer instance
            config: Search configuration
        """
        self.weaviate = weaviate_indexer
        self.config = config or DocumentSearchConfig()

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        document_types: Optional[List[str]] = None
    ) -> DocumentSearchResult:
        """
        Search for documents.

        Args:
            query: Search query (natural language)
            top_k: Number of documents to return (overrides config)
            document_types: Filter by document types (overrides config)

        Returns:
            DocumentSearchResult with found documents
        """
        import time
        start_time = time.time()

        top_k = top_k or self.config.top_k
        document_types = document_types or self.config.document_types

        logger.info(f"Searching documents: '{query}', top_k={top_k}")

        try:
            # Get DocumentNode collection
            collection = self.weaviate.client.collections.get("DocumentNode")

            # Generate query embedding
            query_vector = self.weaviate.embedding_model.encode(
                query,
                convert_to_numpy=True
            ).tolist()

            # Build filters if needed
            where_filter = None
            if document_types:
                # Filter by document type
                from weaviate.classes.query import Filter
                where_filter = Filter.by_property("document_type").contains_any(document_types)

            # Perform hybrid search
            if where_filter:
                response = collection.query.hybrid(
                    query=query,
                    vector=query_vector,
                    alpha=self.config.hybrid_alpha,
                    limit=top_k,
                    filters=where_filter
                )
            else:
                response = collection.query.hybrid(
                    query=query,
                    vector=query_vector,
                    alpha=self.config.hybrid_alpha,
                    limit=top_k
                )

            # Format results
            documents = []
            for obj in response.objects:
                doc = {
                    'node_id': obj.properties.get('node_id'),
                    'name': obj.properties.get('name'),
                    'document_type': obj.properties.get('document_type'),
                    'file_path': obj.properties.get('file_path'),
                    'content': obj.properties.get('content'),
                    'author': obj.properties.get('author'),
                    'created_date': obj.properties.get('created_date'),
                    'modified_date': obj.properties.get('modified_date'),
                    'sections_count': obj.properties.get('sections_count', 0),
                    'images_count': obj.properties.get('images_count', 0),
                    'score': obj.metadata.score if hasattr(obj.metadata, 'score') else 0.0,
                    'metadata': json.loads(obj.properties.get('metadata', '{}'))
                }
                documents.append(doc)

            execution_time_ms = (time.time() - start_time) * 1000

            logger.info(f"Found {len(documents)} documents in {execution_time_ms:.1f}ms")

            return DocumentSearchResult(
                documents=documents,
                total_found=len(documents),
                execution_time_ms=execution_time_ms
            )

        except Exception as e:
            logger.error(f"Document search failed: {e}")
            return DocumentSearchResult(
                documents=[],
                total_found=0,
                execution_time_ms=(time.time() - start_time) * 1000
            )

    def get_document_by_id(self, node_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific document by ID.

        Args:
            node_id: Document node ID

        Returns:
            Document dict or None if not found
        """
        try:
            collection = self.weaviate.client.collections.get("DocumentNode")

            from weaviate.classes.query import Filter

            response = collection.query.fetch_objects(
                filters=Filter.by_property("node_id").equal(node_id),
                limit=1
            )

            if response.objects:
                obj = response.objects[0]
                return {
                    'node_id': obj.properties.get('node_id'),
                    'name': obj.properties.get('name'),
                    'document_type': obj.properties.get('document_type'),
                    'file_path': obj.properties.get('file_path'),
                    'content': obj.properties.get('content'),
                    'author': obj.properties.get('author'),
                    'created_date': obj.properties.get('created_date'),
                    'modified_date': obj.properties.get('modified_date'),
                    'sections_count': obj.properties.get('sections_count', 0),
                    'images_count': obj.properties.get('images_count', 0),
                    'metadata': json.loads(obj.properties.get('metadata', '{}'))
                }

            return None

        except Exception as e:
            logger.error(f"Failed to get document {node_id}: {e}")
            return None

    def list_document_types(self) -> List[str]:
        """
        Get list of all document types in the collection.

        Returns:
            List of unique document types
        """
        try:
            collection = self.weaviate.client.collections.get("DocumentNode")

            # Fetch all documents (just document_type property)
            response = collection.query.fetch_objects(
                limit=1000,  # Adjust if you have more documents
                return_properties=["document_type"]
            )

            # Extract unique types
            types = set()
            for obj in response.objects:
                doc_type = obj.properties.get('document_type')
                if doc_type:
                    types.add(doc_type)

            return sorted(list(types))

        except Exception as e:
            logger.error(f"Failed to list document types: {e}")
            return []

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about indexed documents.

        Returns:
            Dictionary with statistics
        """
        try:
            collection = self.weaviate.client.collections.get("DocumentNode")

            # Get total count
            aggregate_result = collection.aggregate.over_all()

            total_count = aggregate_result.total_count if aggregate_result else 0

            # Get document types
            doc_types = self.list_document_types()

            return {
                'total_documents': total_count,
                'document_types': doc_types,
                'types_count': len(doc_types)
            }

        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {
                'total_documents': 0,
                'document_types': [],
                'types_count': 0
            }
