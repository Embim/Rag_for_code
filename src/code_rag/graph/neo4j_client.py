"""
Neo4j client for knowledge graph operations.

Handles connection to Neo4j and CRUD operations for nodes and relationships.
"""

from typing import List, Dict, Any, Optional
from neo4j import GraphDatabase, Driver, Session
from neo4j.exceptions import ServiceUnavailable, AuthError

from .models import GraphNode, GraphRelationship, NodeType, RelationshipType
from src.logger import get_logger


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

    def clear_database(self, batch_size: int = 10000):
        """
        Clear all nodes and relationships in the database.

        WARNING: This deletes everything! Use with caution.

        Args:
            batch_size: Number of nodes to delete per transaction
        """
        with self._driver.session(database=self.database) as session:
            # Use batched delete for better performance on large graphs
            query = """
            MATCH (n)
            CALL {
                WITH n
                DETACH DELETE n
            } IN TRANSACTIONS OF $batch_size ROWS
            """
            session.run(query, batch_size=batch_size)
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

    def create_nodes_batch(self, nodes: List[GraphNode], chunk_size: int = 1000) -> int:
        """
        Create multiple nodes in batches with chunking.

        More efficient than creating one by one, especially for large datasets.

        Args:
            nodes: List of nodes to create
            chunk_size: Number of nodes to process per chunk (default: 1000)

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

                total_count = 0
                for node_type, node_dicts in nodes_by_type.items():
                    # Process in chunks
                    for i in range(0, len(node_dicts), chunk_size):
                        chunk = node_dicts[i:i + chunk_size]

                        query = f"""
                        UNWIND $nodes AS node
                        MERGE (n:{node_type}:GraphNode {{id: node.id}})
                        SET n = node
                        """

                        result = session.run(query, nodes=chunk)
                        result.consume()
                        total_count += len(chunk)

                        if len(node_dicts) > chunk_size:
                            logger.info(f"Created {total_count}/{len(node_dicts)} {node_type} nodes")

                logger.info(f"✅ Created {total_count} nodes in total")
                return total_count

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

    def create_relationships_batch(self, relationships: List[GraphRelationship], chunk_size: int = 1000) -> int:
        """
        Create multiple relationships in batches with chunking.

        Args:
            relationships: List of relationships to create
            chunk_size: Number of relationships to process per chunk (default: 1000)

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

                total_count = 0
                for rel_type, rel_dicts in rels_by_type.items():
                    # Process in chunks
                    for i in range(0, len(rel_dicts), chunk_size):
                        chunk = rel_dicts[i:i + chunk_size]

                        query = f"""
                        UNWIND $rels AS rel
                        MATCH (source:GraphNode {{id: rel.source_id}})
                        MATCH (target:GraphNode {{id: rel.target_id}})
                        MERGE (source)-[r:{rel_type}]->(target)
                        SET r = rel
                        """

                        result = session.run(query, rels=chunk)
                        result.consume()
                        total_count += len(chunk)

                        if len(rel_dicts) > chunk_size:
                            logger.info(f"Created {total_count}/{len(rel_dicts)} {rel_type} relationships")

                logger.info(f"✅ Created {total_count} relationships in total")
                return total_count

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

    # ============== GRAPH TRAVERSAL METHODS ==============

    def get_related_nodes(
        self,
        node_ids: List[str],
        depth: int = 20,
        relationship_types: Optional[List[str]] = None,
        direction: str = "both",
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Find related nodes at a given depth from source nodes.

        Args:
            node_ids: List of source node IDs
            depth: How many hops to traverse (1-3 recommended)
            relationship_types: Optional list of relationship types to follow
            direction: "in", "out", or "both"
            limit: Maximum number of results

        Returns:
            List of dicts with node properties, relationship type, and distance
        """
        if not node_ids:
            return []

        with self._driver.session(database=self.database) as session:
            # Build relationship pattern based on direction
            if direction == "out":
                rel_pattern = "-[r*1..{}]->"
            elif direction == "in":
                rel_pattern = "<-[r*1..{}]-"
            else:  # both
                rel_pattern = "-[r*1..{}]-"

            rel_pattern = rel_pattern.format(depth)

            # Add relationship type filter if specified
            if relationship_types:
                types_str = "|".join(relationship_types)
                rel_pattern = rel_pattern.replace("[r*", f"[r:{types_str}*")

            query = f"""
            MATCH (source:GraphNode)
            WHERE source.id IN $node_ids
            MATCH (source){rel_pattern}(related:GraphNode)
            WHERE NOT (related.id IN $node_ids)
            WITH DISTINCT related,
                 [rel IN r | type(rel)] AS rel_types,
                 size(r) AS distance
            RETURN related, rel_types, distance
            ORDER BY distance, related.name
            LIMIT $limit
            """

            result = session.run(query, node_ids=node_ids, limit=limit)

            nodes = []
            for record in result:
                node_dict = dict(record['related'])
                node_dict['_relationship_types'] = record['rel_types']
                node_dict['_distance'] = record['distance']
                nodes.append(node_dict)

            logger.info(f"Found {len(nodes)} related nodes for {len(node_ids)} source nodes")
            return nodes

    def find_callers(
        self,
        node_id: str,
        depth: int = 2,
        limit: int = 30
    ) -> List[Dict[str, Any]]:
        """
        Find all functions/methods that call the given node.

        Args:
            node_id: ID of the target node (function/method)
            depth: How many levels of callers to find
            limit: Maximum number of results

        Returns:
            List of caller nodes with call chain info
        """
        with self._driver.session(database=self.database) as session:
            query = """
            MATCH (target:GraphNode {id: $node_id})
            MATCH path = (caller:GraphNode)-[:CALLS*1..{}]->(target)
            WHERE caller.id <> $node_id
            WITH DISTINCT caller, length(path) as depth,
                 [n IN nodes(path) | n.name] AS call_chain
            RETURN caller, depth, call_chain
            ORDER BY depth, caller.name
            LIMIT $limit
            """.format(depth)

            result = session.run(query, node_id=node_id, limit=limit)

            nodes = []
            for record in result:
                node_dict = dict(record['caller'])
                node_dict['_call_depth'] = record['depth']
                node_dict['_call_chain'] = record['call_chain']
                node_dict['_relationship'] = 'CALLS'
                nodes.append(node_dict)

            logger.info(f"Found {len(nodes)} callers for {node_id}")
            return nodes

    def find_callees(
        self,
        node_id: str,
        depth: int = 2,
        limit: int = 30
    ) -> List[Dict[str, Any]]:
        """
        Find all functions/methods that are called by the given node.

        Args:
            node_id: ID of the source node (function/method)
            depth: How many levels of callees to find
            limit: Maximum number of results

        Returns:
            List of callee nodes with call chain info
        """
        with self._driver.session(database=self.database) as session:
            query = """
            MATCH (source:GraphNode {id: $node_id})
            MATCH path = (source)-[:CALLS*1..{}]->(callee:GraphNode)
            WHERE callee.id <> $node_id
            WITH DISTINCT callee, length(path) as depth,
                 [n IN nodes(path) | n.name] AS call_chain
            RETURN callee, depth, call_chain
            ORDER BY depth, callee.name
            LIMIT $limit
            """.format(depth)

            result = session.run(query, node_id=node_id, limit=limit)

            nodes = []
            for record in result:
                node_dict = dict(record['callee'])
                node_dict['_call_depth'] = record['depth']
                node_dict['_call_chain'] = record['call_chain']
                node_dict['_relationship'] = 'CALLED_BY'
                nodes.append(node_dict)

            logger.info(f"Found {len(nodes)} callees for {node_id}")
            return nodes

    def trace_ui_to_database(
        self,
        start_node_id: str,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Trace path from UI component to database models.

        Path: Component → Endpoint → Function → Model

        Args:
            start_node_id: ID of starting node (usually Component or any UI element)
            limit: Maximum number of paths

        Returns:
            List of nodes along the path with relationship info
        """
        with self._driver.session(database=self.database) as session:
            # Try multiple path patterns to find UI→DB connections
            # Use general patterns and filter by relationship types to avoid warnings
            query = """
            MATCH (start:GraphNode {id: $node_id})
            OPTIONAL MATCH path1 = (start)-[r1]->(endpoint)
                                   -[r2]->(func)
                                   -[r3]->(model)
            OPTIONAL MATCH path2 = (start)-[r4*1..3]->(func2)
                                   -[r5]->(model2)
            OPTIONAL MATCH path3 = (start)-[r6]->(endpoint2)
                                   -[r7]->(model3)
            WITH start,
                 CASE WHEN r1 IS NOT NULL AND type(r1) = 'SENDS_REQUEST_TO' 
                      AND 'Endpoint' IN labels(endpoint) THEN endpoint ELSE NULL END AS endpoint,
                 CASE WHEN r2 IS NOT NULL AND type(r2) = 'HANDLES_REQUEST' THEN func ELSE NULL END AS func,
                 CASE WHEN r3 IS NOT NULL AND type(r3) = 'USES_MODEL' 
                      AND 'Model' IN labels(model) THEN model ELSE NULL END AS model,
                 CASE WHEN r4 IS NOT NULL AND ALL(rel IN r4 WHERE type(rel) = 'CALLS') 
                      THEN func2 ELSE NULL END AS func2,
                 CASE WHEN r5 IS NOT NULL AND type(r5) = 'USES_MODEL' 
                      AND 'Model' IN labels(model2) THEN model2 ELSE NULL END AS model2,
                 CASE WHEN r6 IS NOT NULL AND type(r6) = 'SENDS_REQUEST_TO' 
                      AND 'Endpoint' IN labels(endpoint2) THEN endpoint2 ELSE NULL END AS endpoint2,
                 CASE WHEN r7 IS NOT NULL AND type(r7) = 'USES_MODEL' 
                      AND 'Model' IN labels(model3) THEN model3 ELSE NULL END AS model3
            WITH collect(DISTINCT {node: endpoint, rel: 'SENDS_REQUEST_TO', order: 1}) +
                 collect(DISTINCT {node: func, rel: 'HANDLES_REQUEST', order: 2}) +
                 collect(DISTINCT {node: model, rel: 'USES_MODEL', order: 3}) +
                 collect(DISTINCT {node: func2, rel: 'CALLS', order: 2}) +
                 collect(DISTINCT {node: model2, rel: 'USES_MODEL', order: 3}) +
                 collect(DISTINCT {node: endpoint2, rel: 'SENDS_REQUEST_TO', order: 1}) +
                 collect(DISTINCT {node: model3, rel: 'USES_MODEL', order: 3}) AS all_nodes
            UNWIND all_nodes AS item
            WITH item.node AS node, item.rel AS rel, item.order AS ord
            WHERE node IS NOT NULL
            RETURN DISTINCT node, rel, ord
            ORDER BY ord
            LIMIT $limit
            """

            result = session.run(query, node_id=start_node_id, limit=limit)

            nodes = []
            for record in result:
                if record['node']:
                    node_dict = dict(record['node'])
                    node_dict['_relationship'] = record['rel']
                    node_dict['_path_order'] = record['ord']
                    nodes.append(node_dict)

            logger.info(f"Traced {len(nodes)} nodes in UI→DB path from {start_node_id}")
            return nodes

    def trace_database_to_ui(
        self,
        start_node_id: str,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Trace path from database model to UI components.

        Path: Model → Function → Endpoint → Component

        Args:
            start_node_id: ID of starting node (usually Model)
            limit: Maximum number of paths

        Returns:
            List of nodes along the path with relationship info
        """
        with self._driver.session(database=self.database) as session:
            # Use general patterns and filter by relationship types to avoid warnings
            query = """
            MATCH (start:GraphNode {id: $node_id})
            OPTIONAL MATCH path1 = (start)<-[r1]-(func)
                                   <-[r2]-(endpoint)
                                   <-[r3]-(component)
            OPTIONAL MATCH path2 = (start)<-[r4]-(func2)
                                   <-[r5*1..3]-(caller)
            OPTIONAL MATCH path3 = (start)<-[r6]-(endpoint2)
                                   <-[r7]-(component2)
            WITH start,
                 CASE WHEN r1 IS NOT NULL AND type(r1) = 'USES_MODEL' THEN func ELSE NULL END AS func,
                 CASE WHEN r2 IS NOT NULL AND type(r2) = 'HANDLES_REQUEST' 
                      AND 'Endpoint' IN labels(endpoint) THEN endpoint ELSE NULL END AS endpoint,
                 CASE WHEN r3 IS NOT NULL AND type(r3) = 'SENDS_REQUEST_TO' 
                      AND 'Component' IN labels(component) THEN component ELSE NULL END AS component,
                 CASE WHEN r4 IS NOT NULL AND type(r4) = 'USES_MODEL' THEN func2 ELSE NULL END AS func2,
                 CASE WHEN r5 IS NOT NULL AND ALL(rel IN r5 WHERE type(rel) = 'CALLS') 
                      THEN caller ELSE NULL END AS caller,
                 CASE WHEN r6 IS NOT NULL AND type(r6) = 'USES_MODEL' 
                      AND 'Endpoint' IN labels(endpoint2) THEN endpoint2 ELSE NULL END AS endpoint2,
                 CASE WHEN r7 IS NOT NULL AND type(r7) = 'SENDS_REQUEST_TO' 
                      AND 'Component' IN labels(component2) THEN component2 ELSE NULL END AS component2
            WITH collect(DISTINCT {node: func, rel: 'USES_MODEL', order: 1}) +
                 collect(DISTINCT {node: endpoint, rel: 'HANDLES_REQUEST', order: 2}) +
                 collect(DISTINCT {node: component, rel: 'SENDS_REQUEST_TO', order: 3}) +
                 collect(DISTINCT {node: func2, rel: 'USES_MODEL', order: 1}) +
                 collect(DISTINCT {node: caller, rel: 'CALLS', order: 2}) +
                 collect(DISTINCT {node: endpoint2, rel: 'USES_MODEL', order: 1}) +
                 collect(DISTINCT {node: component2, rel: 'SENDS_REQUEST_TO', order: 3}) AS all_nodes
            UNWIND all_nodes AS item
            WITH item.node AS node, item.rel AS rel, item.order AS ord
            WHERE node IS NOT NULL
            RETURN DISTINCT node, rel, ord
            ORDER BY ord
            LIMIT $limit
            """

            result = session.run(query, node_id=start_node_id, limit=limit)

            nodes = []
            for record in result:
                if record['node']:
                    node_dict = dict(record['node'])
                    node_dict['_relationship'] = record['rel']
                    node_dict['_path_order'] = record['ord']
                    nodes.append(node_dict)

            logger.info(f"Traced {len(nodes)} nodes in DB→UI path from {start_node_id}")
            return nodes

    def get_impact_analysis(
        self,
        node_id: str,
        depth: int = 2
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Analyze what would be impacted by changing the given node.

        Args:
            node_id: ID of the node to analyze
            depth: How deep to search for dependencies

        Returns:
            Dict with categorized impacted nodes:
            - callers: functions that call this node
            - importers: files that import this node
            - inheritors: classes that inherit from this node
            - model_users: functions/endpoints using this model
        """
        with self._driver.session(database=self.database) as session:
            results = {
                'callers': [],
                'importers': [],
                'inheritors': [],
                'model_users': [],
            }

            # Find callers
            caller_query = """
            MATCH (target:GraphNode {id: $node_id})
            MATCH (caller)-[:CALLS*1..{}]->(target)
            WHERE caller.id <> $node_id
            RETURN DISTINCT caller
            LIMIT 20
            """.format(depth)
            for record in session.run(caller_query, node_id=node_id):
                node_dict = dict(record['caller'])
                node_dict['_impact_type'] = 'caller'
                results['callers'].append(node_dict)

            # Find importers
            importer_query = """
            MATCH (target:GraphNode {id: $node_id})
            MATCH (importer)-[:IMPORTS]->(target)
            RETURN DISTINCT importer
            LIMIT 20
            """
            for record in session.run(importer_query, node_id=node_id):
                node_dict = dict(record['importer'])
                node_dict['_impact_type'] = 'importer'
                results['importers'].append(node_dict)

            # Find inheritors (for classes)
            inheritor_query = """
            MATCH (target:GraphNode {id: $node_id})
            MATCH (child)-[:INHERITS*1..{}]->(target)
            RETURN DISTINCT child
            LIMIT 20
            """.format(depth)
            for record in session.run(inheritor_query, node_id=node_id):
                node_dict = dict(record['child'])
                node_dict['_impact_type'] = 'inheritor'
                results['inheritors'].append(node_dict)

            # Find model users (for models)
            model_user_query = """
            MATCH (target:GraphNode {id: $node_id})
            MATCH (user)-[:USES_MODEL]->(target)
            RETURN DISTINCT user
            LIMIT 20
            """
            for record in session.run(model_user_query, node_id=node_id):
                node_dict = dict(record['user'])
                node_dict['_impact_type'] = 'model_user'
                results['model_users'].append(node_dict)

            total = sum(len(v) for v in results.values())
            logger.info(f"Impact analysis for {node_id}: {total} impacted nodes")
            return results
