"""
Neo4j client for knowledge graph operations.

Handles connection to Neo4j and CRUD operations for nodes and relationships.
"""

from typing import List, Dict, Any, Optional
from neo4j import GraphDatabase, Driver, Session
from neo4j.exceptions import ServiceUnavailable, AuthError

from .models import GraphNode, GraphRelationship, NodeType, RelationshipType
from ...logger import get_logger


logger = get_logger(__name__)


class Neo4jClient:
    """
    Client for Neo4j graph database.

    Handles connection management and graph operations.
    """

    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: str = "password",
        database: str = "neo4j"
    ):
        """
        Initialize Neo4j client.

        Args:
            uri: Neo4j connection URI
            user: Username
            password: Password
            database: Database name (default: neo4j)
        """
        self.uri = uri
        self.user = user
        self.database = database
        self._driver: Optional[Driver] = None

        try:
            self._driver = GraphDatabase.driver(uri, auth=(user, password))
            # Test connection
            self._driver.verify_connectivity()
            logger.info(f"Connected to Neo4j at {uri}")
        except (ServiceUnavailable, AuthError) as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise ConnectionError(f"Cannot connect to Neo4j at {uri}: {e}")

    def close(self):
        """Close Neo4j connection."""
        if self._driver:
            self._driver.close()
            logger.info("Neo4j connection closed")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def clear_database(self):
        """
        Clear all nodes and relationships in the database.

        WARNING: This deletes everything! Use with caution.
        """
        with self._driver.session(database=self.database) as session:
            session.run("MATCH (n) DETACH DELETE n")
            logger.warning("Database cleared - all nodes and relationships deleted")

    def create_indexes(self):
        """Create indexes for better query performance."""
        with self._driver.session(database=self.database) as session:
            # Index on node id
            session.run(
                "CREATE INDEX IF NOT EXISTS FOR (n:GraphNode) ON (n.id)"
            )

            # Index on node type
            session.run(
                "CREATE INDEX IF NOT EXISTS FOR (n:GraphNode) ON (n.type)"
            )

            # Index on node name
            session.run(
                "CREATE INDEX IF NOT EXISTS FOR (n:GraphNode) ON (n.name)"
            )

            # Index for files
            session.run(
                "CREATE INDEX IF NOT EXISTS FOR (n:File) ON (n.file_path)"
            )

            # Index for endpoints
            session.run(
                "CREATE INDEX IF NOT EXISTS FOR (n:Endpoint) ON (n.path)"
            )

            logger.info("Indexes created successfully")

    def create_node(self, node: GraphNode) -> bool:
        """
        Create a node in the graph.

        Args:
            node: GraphNode to create

        Returns:
            True if created successfully
        """
        with self._driver.session(database=self.database) as session:
            try:
                # Use node type as label + general GraphNode label
                labels = [node.type.value, "GraphNode"]
                labels_str = ':'.join(labels)

                query = f"""
                MERGE (n:{labels_str} {{id: $id}})
                SET n = $properties
                RETURN n
                """

                result = session.run(
                    query,
                    id=node.id,
                    properties=node.to_dict()
                )

                result.single()  # Consume result
                return True

            except Exception as e:
                logger.error(f"Failed to create node {node.id}: {e}")
                return False

    def create_nodes_batch(self, nodes: List[GraphNode]) -> int:
        """
        Create multiple nodes in a batch.

        More efficient than creating one by one.

        Args:
            nodes: List of nodes to create

        Returns:
            Number of nodes created
        """
        if not nodes:
            return 0

        with self._driver.session(database=self.database) as session:
            try:
                # Group nodes by type for efficiency
                nodes_by_type = {}
                for node in nodes:
                    node_type = node.type.value
                    if node_type not in nodes_by_type:
                        nodes_by_type[node_type] = []
                    nodes_by_type[node_type].append(node.to_dict())

                count = 0
                for node_type, node_dicts in nodes_by_type.items():
                    query = f"""
                    UNWIND $nodes AS node
                    MERGE (n:{node_type}:GraphNode {{id: node.id}})
                    SET n = node
                    """

                    result = session.run(query, nodes=node_dicts)
                    result.consume()
                    count += len(node_dicts)

                logger.info(f"Created {count} nodes in batch")
                return count

            except Exception as e:
                logger.error(f"Failed to create nodes in batch: {e}")
                return 0

    def create_relationship(self, relationship: GraphRelationship) -> bool:
        """
        Create a relationship between two nodes.

        Args:
            relationship: GraphRelationship to create

        Returns:
            True if created successfully
        """
        with self._driver.session(database=self.database) as session:
            try:
                rel_type = relationship.type.value

                query = f"""
                MATCH (source:GraphNode {{id: $source_id}})
                MATCH (target:GraphNode {{id: $target_id}})
                MERGE (source)-[r:{rel_type}]->(target)
                SET r = $properties
                RETURN r
                """

                result = session.run(
                    query,
                    source_id=relationship.source_id,
                    target_id=relationship.target_id,
                    properties=relationship.to_dict()
                )

                result.single()
                return True

            except Exception as e:
                logger.error(
                    f"Failed to create relationship {relationship.type.value} "
                    f"from {relationship.source_id} to {relationship.target_id}: {e}"
                )
                return False

    def create_relationships_batch(self, relationships: List[GraphRelationship]) -> int:
        """
        Create multiple relationships in a batch.

        Args:
            relationships: List of relationships to create

        Returns:
            Number of relationships created
        """
        if not relationships:
            return 0

        with self._driver.session(database=self.database) as session:
            try:
                # Group by relationship type
                rels_by_type = {}
                for rel in relationships:
                    rel_type = rel.type.value
                    if rel_type not in rels_by_type:
                        rels_by_type[rel_type] = []
                    rels_by_type[rel_type].append(rel.to_dict())

                count = 0
                for rel_type, rel_dicts in rels_by_type.items():
                    query = f"""
                    UNWIND $rels AS rel
                    MATCH (source:GraphNode {{id: rel.source_id}})
                    MATCH (target:GraphNode {{id: rel.target_id}})
                    MERGE (source)-[r:{rel_type}]->(target)
                    SET r = rel
                    """

                    result = session.run(query, rels=rel_dicts)
                    result.consume()
                    count += len(rel_dicts)

                logger.info(f"Created {count} relationships in batch")
                return count

            except Exception as e:
                logger.error(f"Failed to create relationships in batch: {e}")
                return 0

    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a node by ID.

        Args:
            node_id: Node ID

        Returns:
            Node properties or None if not found
        """
        with self._driver.session(database=self.database) as session:
            result = session.run(
                "MATCH (n:GraphNode {id: $id}) RETURN n",
                id=node_id
            )

            record = result.single()
            if record:
                return dict(record['n'])
            return None

    def find_nodes(
        self,
        node_type: Optional[NodeType] = None,
        properties: Optional[Dict[str, Any]] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Find nodes by type and/or properties.

        Args:
            node_type: Optional node type filter
            properties: Optional property filters
            limit: Maximum number of results

        Returns:
            List of matching nodes
        """
        with self._driver.session(database=self.database) as session:
            # Build query
            if node_type:
                match_clause = f"MATCH (n:{node_type.value})"
            else:
                match_clause = "MATCH (n:GraphNode)"

            # Add property filters
            where_clauses = []
            params = {'limit': limit}

            if properties:
                for key, value in properties.items():
                    where_clauses.append(f"n.{key} = ${key}")
                    params[key] = value

            where_clause = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""

            query = f"{match_clause} {where_clause} RETURN n LIMIT $limit"

            result = session.run(query, **params)

            return [dict(record['n']) for record in result]

    def get_relationships(
        self,
        source_id: Optional[str] = None,
        target_id: Optional[str] = None,
        rel_type: Optional[RelationshipType] = None
    ) -> List[Dict[str, Any]]:
        """
        Get relationships by source, target, and/or type.

        Args:
            source_id: Optional source node ID
            target_id: Optional target node ID
            rel_type: Optional relationship type

        Returns:
            List of relationships
        """
        with self._driver.session(database=self.database) as session:
            # Build query
            if source_id and target_id:
                match_clause = "MATCH (s:GraphNode {id: $source_id})-[r]->(t:GraphNode {id: $target_id})"
                params = {'source_id': source_id, 'target_id': target_id}
            elif source_id:
                match_clause = "MATCH (s:GraphNode {id: $source_id})-[r]->()"
                params = {'source_id': source_id}
            elif target_id:
                match_clause = "MATCH ()-[r]->(t:GraphNode {id: $target_id})"
                params = {'target_id': target_id}
            else:
                match_clause = "MATCH ()-[r]->()"
                params = {}

            # Add relationship type filter
            if rel_type:
                match_clause = match_clause.replace('-[r]->', f'-[r:{rel_type.value}]->')

            query = f"{match_clause} RETURN r"

            result = session.run(query, **params)

            return [dict(record['r']) for record in result]

    def execute_cypher(self, query: str, **params) -> List[Dict[str, Any]]:
        """
        Execute a raw Cypher query.

        Args:
            query: Cypher query string
            **params: Query parameters

        Returns:
            List of result records
        """
        with self._driver.session(database=self.database) as session:
            result = session.run(query, **params)
            return [dict(record) for record in result]

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get database statistics.

        Returns:
            Dictionary with statistics
        """
        with self._driver.session(database=self.database) as session:
            # Count nodes by type
            node_counts = {}
            for node_type in NodeType:
                result = session.run(
                    f"MATCH (n:{node_type.value}) RETURN count(n) as count"
                )
                record = result.single()
                node_counts[node_type.value] = record['count'] if record else 0

            # Count relationships by type
            rel_counts = {}
            for rel_type in RelationshipType:
                result = session.run(
                    f"MATCH ()-[r:{rel_type.value}]->() RETURN count(r) as count"
                )
                record = result.single()
                rel_counts[rel_type.value] = record['count'] if record else 0

            return {
                'nodes': node_counts,
                'relationships': rel_counts,
                'total_nodes': sum(node_counts.values()),
                'total_relationships': sum(rel_counts.values())
            }
