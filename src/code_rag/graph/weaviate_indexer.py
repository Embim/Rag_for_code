"""
Weaviate indexer - indexes knowledge graph nodes for vector search.

This component bridges Neo4j knowledge graph and Weaviate vector search:
1. Takes nodes from Neo4j
2. Generates embeddings using BAAI/bge-m3
3. Indexes in Weaviate for semantic search
4. Enables hybrid search (graph structure + semantic similarity)
"""

import json
from typing import List, Dict, Any, Optional, Tuple
from sentence_transformers import SentenceTransformer
import weaviate
from weaviate.classes.config import Configure, Property, DataType
from weaviate.classes.query import Filter

from .models import GraphNode, NodeType
from .neo4j_client import Neo4jClient
from src.logger import get_logger


logger = get_logger(__name__)


class WeaviateIndexer:
    """
    Indexes knowledge graph nodes in Weaviate for vector search.

    Strategy:
    1. Create Weaviate schema for each node type
    2. Generate embeddings for node content
    3. Batch index nodes with embeddings
    4. Enable hybrid search (BM25 + vector)
    """

    def __init__(
        self,
        weaviate_url: str = "http://localhost:8080",
        embedding_model: str = "BAAI/bge-m3",
        neo4j_client: Optional[Neo4jClient] = None
    ):
        """
        Initialize Weaviate indexer.

        Args:
            weaviate_url: Weaviate connection URL
            embedding_model: Sentence transformer model name
            neo4j_client: Optional Neo4j client for fetching nodes
        """
        self.weaviate_url = weaviate_url
        self.neo4j_client = neo4j_client

        # Initialize Weaviate client
        try:
            self.client = weaviate.connect_to_local(
                host=weaviate_url.replace("http://", "").replace("https://", "").split(":")[0],
                port=int(weaviate_url.split(":")[-1]) if ":" in weaviate_url else 8080
            )
            logger.info(f"Connected to Weaviate at {weaviate_url}")
        except Exception as e:
            logger.error(f"Failed to connect to Weaviate: {e}")
            raise ConnectionError(f"Cannot connect to Weaviate at {weaviate_url}: {e}")

        # Initialize embedding model
        logger.info(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        logger.info(f"Embedding model loaded (dimension: {self.embedding_dim})")

    def close(self):
        """Close Weaviate connection."""
        if self.client:
            self.client.close()
            logger.info("Weaviate connection closed")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def create_schema(self):
        """
        Create Weaviate schema for code entities.

        Creates collections for each node type with appropriate properties.
        """
        # Define schema for CodeNode collection
        # This is a unified collection for all code entities
        try:
            # Check if collection exists
            if self.client.collections.exists("CodeNode"):
                logger.info("CodeNode collection already exists")
                return

            # Create collection with vectorizer
            self.client.collections.create(
                name="CodeNode",
                vectorizer_config=Configure.Vectorizer.none(),  # We provide our own vectors
                properties=[
                    Property(name="node_id", data_type=DataType.TEXT),
                    Property(name="node_type", data_type=DataType.TEXT),
                    Property(name="name", data_type=DataType.TEXT),
                    Property(name="file_path", data_type=DataType.TEXT),
                    Property(name="repository", data_type=DataType.TEXT),
                    Property(name="content", data_type=DataType.TEXT),  # Searchable content
                    Property(name="signature", data_type=DataType.TEXT),
                    Property(name="docstring", data_type=DataType.TEXT),
                    Property(name="start_line", data_type=DataType.INT),
                    Property(name="end_line", data_type=DataType.INT),
                    Property(name="metadata", data_type=DataType.TEXT),  # JSON string for additional data
                ]
            )

            logger.info("Created CodeNode collection in Weaviate")

        except Exception as e:
            logger.error(f"Failed to create Weaviate schema: {e}")
            raise

    def _build_searchable_content(self, node: GraphNode) -> str:
        """
        Build searchable text content from a node.

        Combines node name, signature, docstring, and other relevant text
        into a single searchable string for embedding generation.

        Args:
            node: Graph node

        Returns:
            Searchable text content
        """
        parts = []

        # Node name
        parts.append(f"Name: {node.name}")

        # Node type
        parts.append(f"Type: {node.type.value}")

        # Type-specific content
        if hasattr(node, 'signature') and node.properties.get('signature'):
            parts.append(f"Signature: {node.properties['signature']}")

        if hasattr(node, 'docstring') and node.properties.get('docstring'):
            docstring = node.properties['docstring'].strip()
            if docstring:
                parts.append(f"Documentation: {docstring}")

        # File path context
        if node.properties.get('file_path'):
            parts.append(f"File: {node.properties['file_path']}")

        # For endpoints, include HTTP method and path
        if node.type == NodeType.ENDPOINT:
            method = node.properties.get('http_method', '')
            path = node.properties.get('path', '')
            parts.append(f"Endpoint: {method} {path}")

        # For components, include props and hooks
        if node.type == NodeType.COMPONENT:
            props = node.properties.get('props_type', '')
            if props:
                parts.append(f"Props: {props}")
            hooks = node.properties.get('hooks_used', '')
            if hooks:
                parts.append(f"Hooks: {hooks}")

        # For models, include fields
        if node.type == NodeType.MODEL:
            model_type = node.properties.get('model_type', '')
            parts.append(f"Model type: {model_type}")

        return "\n".join(parts)

    def _extract_repository_name(self, node_id: str) -> str:
        """
        Extract repository name from node ID.

        Node IDs have format: repo:repository_name/file/path:entity

        Args:
            node_id: Node ID

        Returns:
            Repository name
        """
        if node_id.startswith("repo:"):
            parts = node_id[5:].split("/")
            return parts[0].split(":")[0]
        return "unknown"

    def index_nodes(self, nodes: List[GraphNode], batch_size: int = 100) -> int:
        """
        Index nodes in Weaviate with embeddings.

        Args:
            nodes: List of graph nodes to index
            batch_size: Batch size for indexing

        Returns:
            Number of nodes indexed
        """
        if not nodes:
            logger.warning("No nodes to index")
            return 0

        logger.info(f"Indexing {len(nodes)} nodes in Weaviate...")

        collection = self.client.collections.get("CodeNode")
        indexed_count = 0

        # Process in batches
        for i in range(0, len(nodes), batch_size):
            batch = nodes[i:i + batch_size]

            # Prepare data for batch
            batch_data = []
            batch_contents = []

            for node in batch:
                content = self._build_searchable_content(node)
                batch_contents.append(content)

                # Prepare properties
                properties = {
                    "node_id": node.id,
                    "node_type": node.type.value,
                    "name": node.name,
                    "file_path": node.properties.get('file_path', ''),
                    "repository": self._extract_repository_name(node.id),
                    "content": content,
                    "signature": node.properties.get('signature', ''),
                    "docstring": node.properties.get('docstring', ''),
                    "start_line": node.properties.get('start_line', 0),
                    "end_line": node.properties.get('end_line', 0),
                    "metadata": json.dumps(node.properties)
                }

                batch_data.append(properties)

            # Generate embeddings in batch
            try:
                embeddings = self.embedding_model.encode(
                    batch_contents,
                    show_progress_bar=False,
                    convert_to_numpy=True
                )

                # Index batch
                with collection.batch.dynamic() as batch_inserter:
                    for properties, vector in zip(batch_data, embeddings):
                        batch_inserter.add_object(
                            properties=properties,
                            vector=vector.tolist()
                        )

                indexed_count += len(batch)
                logger.info(f"Indexed batch {i // batch_size + 1}: {indexed_count}/{len(nodes)} nodes")

            except Exception as e:
                logger.error(f"Failed to index batch {i // batch_size + 1}: {e}")
                continue

        logger.info(f"Successfully indexed {indexed_count} nodes in Weaviate")
        return indexed_count

    def index_from_neo4j(
        self,
        node_types: Optional[List[NodeType]] = None,
        batch_size: int = 100
    ) -> int:
        """
        Fetch nodes from Neo4j and index them in Weaviate.

        Args:
            node_types: Optional list of node types to index (defaults to all)
            batch_size: Batch size for indexing

        Returns:
            Number of nodes indexed
        """
        if not self.neo4j_client:
            raise ValueError("Neo4j client not provided")

        # Default to all node types except Repository (too high-level)
        if node_types is None:
            node_types = [
                NodeType.FILE,
                NodeType.FUNCTION,
                NodeType.CLASS,
                NodeType.COMPONENT,
                NodeType.ENDPOINT,
                NodeType.MODEL,
                NodeType.ROUTE
            ]

        total_indexed = 0

        for node_type in node_types:
            logger.info(f"Fetching {node_type.value} nodes from Neo4j...")

            # Fetch nodes of this type
            node_dicts = self.neo4j_client.find_nodes(
                node_type=node_type,
                limit=10000  # Fetch in large batches
            )

            if not node_dicts:
                logger.info(f"No {node_type.value} nodes found")
                continue

            # Convert to GraphNode objects
            nodes = self._convert_to_graph_nodes(node_dicts, node_type)

            # Index in Weaviate
            count = self.index_nodes(nodes, batch_size=batch_size)
            total_indexed += count

        logger.info(f"Total nodes indexed from Neo4j: {total_indexed}")
        return total_indexed

    def _convert_to_graph_nodes(
        self,
        node_dicts: List[Dict[str, Any]],
        node_type: NodeType
    ) -> List[GraphNode]:
        """
        Convert Neo4j node dictionaries to GraphNode objects.

        Args:
            node_dicts: List of node dictionaries from Neo4j
            node_type: Type of nodes

        Returns:
            List of GraphNode objects
        """
        nodes = []

        for node_dict in node_dicts:
            # Create base GraphNode
            node = GraphNode(
                id=node_dict.get('id', ''),
                type=node_type,
                name=node_dict.get('name', ''),
                properties=node_dict
            )
            nodes.append(node)

        return nodes

    def search(
        self,
        query: str,
        node_types: Optional[List[str]] = None,
        limit: int = 10,
        alpha: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Hybrid search in Weaviate (BM25 + vector search).

        Args:
            query: Search query
            node_types: Optional filter by node types
            limit: Maximum number of results
            alpha: Hybrid search alpha (0.0 = BM25 only, 1.0 = vector only, 0.5 = balanced)

        Returns:
            List of search results with scores
        """
        collection = self.client.collections.get("CodeNode")

        # Generate query embedding
        query_vector = self.embedding_model.encode(
            query,
            show_progress_bar=False,
            convert_to_numpy=True
        ).tolist()

        # Build filter if node types specified
        where_filter = None
        if node_types:
            where_filter = Filter.by_property("node_type").contains_any(node_types)

        # Perform hybrid search
        try:
            if where_filter:
                response = collection.query.hybrid(
                    query=query,
                    vector=query_vector,
                    alpha=alpha,
                    limit=limit,
                    filters=where_filter
                )
            else:
                response = collection.query.hybrid(
                    query=query,
                    vector=query_vector,
                    alpha=alpha,
                    limit=limit
                )

            # Format results
            results = []
            for obj in response.objects:
                results.append({
                    'node_id': obj.properties.get('node_id'),
                    'node_type': obj.properties.get('node_type'),
                    'name': obj.properties.get('name'),
                    'file_path': obj.properties.get('file_path'),
                    'repository': obj.properties.get('repository'),  # ✅ Добавлено поле repository
                    'content': obj.properties.get('content'),
                    'score': obj.metadata.score if hasattr(obj.metadata, 'score') else 0.0,
                    'metadata': json.loads(obj.properties.get('metadata', '{}'))
                })

            return results

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get indexing statistics.

        Returns:
            Dictionary with statistics
        """
        try:
            collection = self.client.collections.get("CodeNode")

            # Get total count
            result = collection.aggregate.over_all(total_count=True)
            total_count = result.total_count

            # Count by node type
            type_counts = {}
            for node_type in NodeType:
                response = collection.aggregate.over_all(
                    filters=Filter.by_property("node_type").equal(node_type.value),
                    total_count=True
                )
                type_counts[node_type.value] = response.total_count

            return {
                'total_nodes': total_count,
                'nodes_by_type': type_counts,
                'embedding_model': str(self.embedding_model),
                'embedding_dimension': self.embedding_dim
            }

        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {
                'total_nodes': 0,
                'nodes_by_type': {},
                'error': str(e)
            }

    def delete_all(self):
        """
        Delete all indexed data.

        WARNING: This deletes everything! Use with caution.
        """
        try:
            if self.client.collections.exists("CodeNode"):
                self.client.collections.delete("CodeNode")
                logger.warning("Deleted CodeNode collection - all indexed data removed")

            # Recreate schema
            self.create_schema()

        except Exception as e:
            logger.error(f"Failed to delete collection: {e}")
            raise
