"""
Update repository and file_path metadata for existing Neo4j nodes.

This script extracts repository and file_path from node IDs and updates
the properties without reindexing everything.
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.code_rag.graph import Neo4jClient
from src.logger import get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)


def extract_metadata_from_id(node_id: str):
    """
    Extract repository and file_path from node ID.

    ID format: repo:repository_name:file/path.py:ClassName.method_name
    Example: repo:api:app/backend/booking.py:TradeUploader.book_trade
    """
    if not node_id or not node_id.startswith('repo:'):
        return None, None

    # Split by colons, but preserve path
    parts = node_id.split(':', 3)

    if len(parts) >= 3:
        repository = parts[1]

        # Extract file path (part before last colon)
        if len(parts) >= 4:
            file_path = parts[2]
        else:
            file_path = parts[2]

        return repository, file_path

    return None, None


def update_node_metadata(neo4j_client: Neo4jClient):
    """Update metadata for all existing nodes."""

    logger.info("Fetching all nodes from Neo4j...")

    # Get all nodes with IDs
    cypher = """
    MATCH (n:GraphNode)
    WHERE n.id STARTS WITH 'repo:'
    RETURN elementId(n) as element_id, n.id as node_id
    """

    nodes = list(neo4j_client.execute_cypher(cypher))
    logger.info(f"Found {len(nodes)} nodes to update")

    # Update nodes in batches
    updated_count = 0
    batch_size = 1000

    for i in range(0, len(nodes), batch_size):
        batch = nodes[i:i + batch_size]

        # Prepare batch update
        updates = []
        for node in batch:
            node_id = node['node_id']
            element_id = node['element_id']

            repository, file_path = extract_metadata_from_id(node_id)

            if repository and file_path:
                updates.append({
                    'element_id': element_id,
                    'repository': repository,
                    'file_path': file_path
                })

        # Execute batch update
        if updates:
            update_cypher = """
            UNWIND $updates as update
            MATCH (n)
            WHERE elementId(n) = update.element_id
            SET n.repository = update.repository,
                n.file_path = update.file_path
            """

            neo4j_client.execute_cypher(update_cypher, parameters={'updates': updates})
            updated_count += len(updates)

            logger.info(f"Updated {updated_count}/{len(nodes)} nodes...")

    logger.info(f"✅ Successfully updated {updated_count} nodes with metadata")

    # Verify update
    verify_cypher = """
    MATCH (n:GraphNode)
    WHERE n.repository IS NOT NULL
    RETURN count(n) as nodes_with_repository
    """

    result = list(neo4j_client.execute_cypher(verify_cypher))
    if result:
        logger.info(f"Verification: {result[0]['nodes_with_repository']} nodes now have repository field")


def main():
    """Main entry point."""

    logger.info("🚀 Starting metadata update...")

    # Connect to Neo4j
    neo4j_uri = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
    neo4j_user = 'neo4j'
    neo4j_password = os.getenv('NEO4J_PASSWORD', 'password')

    logger.info(f"Connecting to Neo4j at {neo4j_uri}...")

    client = Neo4jClient(
        uri=neo4j_uri,
        user=neo4j_user,
        password=neo4j_password
    )

    try:
        # Update metadata
        update_node_metadata(client)

        logger.info("🎉 Metadata update complete!")

    except Exception as e:
        logger.error(f"❌ Update failed: {e}")
        raise
    finally:
        client.close()
        logger.info("Neo4j connection closed")


if __name__ == '__main__':
    main()
