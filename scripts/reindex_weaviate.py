"""
Reindex Weaviate from Neo4j with updated metadata.

Reads all nodes from Neo4j and reindexes them in Weaviate with
proper repository and file_path fields.
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.code_rag.graph import Neo4jClient, WeaviateIndexer
from src.code_rag.graph.models import GraphNode, NodeType
from src.logger import get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)


def reindex_weaviate():
    """Reindex Weaviate from Neo4j nodes."""

    # Connect to Neo4j
    neo4j_uri = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
    neo4j_password = os.getenv('NEO4J_PASSWORD', 'password')

    logger.info(f"Connecting to Neo4j at {neo4j_uri}...")
    neo4j_client = Neo4jClient(
        uri=neo4j_uri,
        user='neo4j',
        password=neo4j_password
    )

    # Connect to Weaviate
    logger.info("Connecting to Weaviate at http://localhost:8080...")
    weaviate_indexer = WeaviateIndexer(
        weaviate_url='http://localhost:8080',
        embedding_model='BAAI/bge-m3'
    )

    try:
        # Clear and recreate schema
        logger.info("⚠️  Clearing Weaviate and recreating schema...")
        try:
            weaviate_indexer.client.collections.delete("CodeNode")
            logger.info("Deleted existing CodeNode collection")
        except Exception as e:
            logger.info(f"No existing collection to delete: {e}")

        weaviate_indexer.create_schema()

        # Fetch all nodes from Neo4j
        logger.info("Fetching nodes from Neo4j...")
        cypher = """
        MATCH (n:GraphNode)
        RETURN n
        LIMIT 10000
        """

        results = list(neo4j_client.execute_cypher(cypher))
        logger.info(f"Found {len(results)} nodes to index")

        # Convert to GraphNode objects
        nodes = []
        for record in results:
            node_data = dict(record['n'])

            # Map node_type to NodeType enum
            node_type_str = node_data.get('type', 'Unknown')
            try:
                node_type = NodeType(node_type_str) if node_type_str != 'Unknown' else None
            except ValueError:
                node_type = None

            # Create GraphNode
            node = GraphNode(
                id=node_data.get('id', ''),
                name=node_data.get('name', 'Unknown'),
                type=node_type,
                properties=node_data
            )

            nodes.append(node)

        # Index in Weaviate
        logger.info(f"Indexing {len(nodes)} nodes in Weaviate...")
        indexed_count = weaviate_indexer.index_nodes(nodes, batch_size=100)

        logger.info(f"✅ Successfully indexed {indexed_count} nodes")

        # Verify
        stats = weaviate_indexer.get_statistics()
        logger.info(f"Weaviate statistics: {stats}")

        # Test search
        logger.info("\nTesting search for 'book trade'...")
        results = weaviate_indexer.search('book trade', limit=3)
        logger.info(f"Found {len(results)} results:")
        for r in results[:3]:
            logger.info(f"  - {r.get('name')} in {r.get('file_path')} (repo: {r.get('metadata', {}).get('repository', 'N/A')})")

    except Exception as e:
        logger.error(f"❌ Reindexing failed: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        neo4j_client.close()
        weaviate_indexer.close()
        logger.info("Connections closed")


def main():
    """Main entry point."""
    logger.info("🚀 Starting Weaviate reindexing from Neo4j...")
    reindex_weaviate()
    logger.info("🎉 Weaviate reindexing complete!")


if __name__ == '__main__':
    main()
