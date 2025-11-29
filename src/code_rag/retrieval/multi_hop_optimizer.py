"""
Multi-hop Graph Traversal Optimizer.

Implements cost control strategies:
1. Early stopping when enough relevant results found
2. Breadth-first search with priority queue
3. Timeout and budget enforcement
4. Path caching for popular queries

Goal: Make multi-hop traversal fast and efficient even on large graphs.
"""

import time
import heapq
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

from ...logger import get_logger


logger = get_logger(__name__)


@dataclass
class TraversalBudget:
    """Budget constraints for graph traversal."""

    max_nodes: int = 1000  # Maximum nodes to visit
    max_hops: int = 4  # Maximum path depth
    timeout_seconds: float = 30.0  # Maximum execution time
    early_stop_threshold: int = 20  # Stop if found N relevant results

    # Current state
    nodes_visited: int = 0
    start_time: float = field(default_factory=time.time)

    def is_exhausted(self) -> bool:
        """Check if budget is exhausted."""
        if self.nodes_visited >= self.max_nodes:
            logger.warning(f"Budget exhausted: visited {self.nodes_visited} nodes")
            return True

        elapsed = time.time() - self.start_time
        if elapsed >= self.timeout_seconds:
            logger.warning(f"Budget exhausted: timeout after {elapsed:.1f}s")
            return True

        return False

    def can_visit(self) -> bool:
        """Check if we can visit more nodes."""
        return not self.is_exhausted()

    def visit_node(self):
        """Record that we visited a node."""
        self.nodes_visited += 1

    def elapsed_time(self) -> float:
        """Get elapsed time in seconds."""
        return time.time() - self.start_time


@dataclass
class PathNode:
    """Node in traversal path with priority."""

    node_id: str
    node_data: Dict[str, Any]
    depth: int  # Hop count from start
    score: float  # Priority score (higher = better)
    parent_path: List[str] = field(default_factory=list)  # IDs of nodes in path to this

    def __lt__(self, other):
        """For priority queue (higher score = higher priority)."""
        return self.score > other.score  # Reverse for max-heap behavior


class RelationshipPriority:
    """
    Assigns priority scores to relationships.

    Higher priority = more relevant for typical code queries.
    """

    # Relationship type to priority mapping
    PRIORITIES = {
        # Direct functional relationships (highest priority)
        'CALLS': 10,
        'SENDS_REQUEST_TO': 10,
        'HANDLES_REQUEST': 10,
        'USES_MODEL': 9,

        # Data relationships
        'FOREIGN_KEY': 8,
        'MANY_TO_MANY': 8,
        'ONE_TO_ONE': 8,

        # Structural relationships
        'CONTAINS': 7,
        'INHERITS': 7,

        # Import relationships (lower priority - less direct)
        'IMPORTS': 5,

        # Rendering relationships
        'RENDERS_AT': 6,

        # Default
        'DEFAULT': 3,
    }

    @classmethod
    def get_priority(cls, relationship_type: str) -> int:
        """Get priority for a relationship type."""
        return cls.PRIORITIES.get(relationship_type, cls.PRIORITIES['DEFAULT'])


class PathCache:
    """
    Cache for frequently traversed paths.

    Stores popular paths to avoid re-computing them.
    """

    def __init__(self, max_size: int = 1000):
        """
        Initialize path cache.

        Args:
            max_size: Maximum number of cached paths
        """
        self.max_size = max_size
        self.cache: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}
        self.hits: Dict[Tuple[str, str, str], int] = defaultdict(int)

    def _make_key(
        self,
        start_id: str,
        end_type: str,
        strategy: str
    ) -> Tuple[str, str, str]:
        """Create cache key."""
        return (start_id, end_type, strategy)

    def get(
        self,
        start_id: str,
        end_type: str,
        strategy: str
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Get cached path.

        Args:
            start_id: Starting node ID
            end_type: Target node type
            strategy: Search strategy

        Returns:
            Cached path nodes or None
        """
        key = self._make_key(start_id, end_type, strategy)

        if key in self.cache:
            self.hits[key] += 1
            logger.debug(f"Cache hit for {start_id} -> {end_type} (strategy={strategy})")
            return self.cache[key]

        return None

    def put(
        self,
        start_id: str,
        end_type: str,
        strategy: str,
        path_nodes: List[Dict[str, Any]]
    ):
        """
        Cache a path.

        Args:
            start_id: Starting node ID
            end_type: Target node type
            strategy: Search strategy
            path_nodes: Nodes in the path
        """
        # Check size limit
        if len(self.cache) >= self.max_size:
            # Remove least popular entry
            if self.hits:
                least_popular = min(self.hits.items(), key=lambda x: x[1])
                del self.cache[least_popular[0]]
                del self.hits[least_popular[0]]
                logger.debug(f"Evicted cache entry: {least_popular[0]}")

        key = self._make_key(start_id, end_type, strategy)
        self.cache[key] = path_nodes
        self.hits[key] = 0

        logger.debug(f"Cached path {start_id} -> {end_type} ({len(path_nodes)} nodes)")

    def clear(self):
        """Clear the cache."""
        self.cache.clear()
        self.hits.clear()

    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_hits = sum(self.hits.values())
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'total_hits': total_hits,
            'hit_rate': total_hits / max(1, len(self.cache)),
        }


class MultiHopOptimizer:
    """
    Optimized multi-hop graph traversal.

    Implements:
    - Priority-based BFS (explores most relevant paths first)
    - Early stopping when enough results found
    - Budget enforcement (time + node limits)
    - Path caching for popular queries
    """

    def __init__(self, neo4j_client, path_cache: Optional[PathCache] = None):
        """
        Initialize optimizer.

        Args:
            neo4j_client: Neo4j client for executing queries
            path_cache: Optional path cache (creates new if not provided)
        """
        self.neo4j = neo4j_client
        self.path_cache = path_cache or PathCache()

    def traverse(
        self,
        start_nodes: List[Dict[str, Any]],
        target_type: str,
        relationship_types: List[str],
        budget: TraversalBudget,
        strategy: str = "default"
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Perform optimized multi-hop traversal.

        Args:
            start_nodes: Starting nodes
            target_type: Target node type to find
            relationship_types: Allowed relationship types
            budget: Traversal budget
            strategy: Search strategy name (for caching)

        Returns:
            Tuple of (found_nodes, relationships)
        """
        found_nodes = []
        found_relationships = []
        visited = set()

        # Priority queue: (priority_score, PathNode)
        queue = []

        # Initialize with start nodes
        for node in start_nodes:
            # Support both 'node_id' (from Weaviate) and 'id' (from Neo4j)
            node_id = node.get('node_id') or node.get('id')
            if not node_id:
                continue

            # Check cache first
            cached = self.path_cache.get(node_id, target_type, strategy)
            if cached:
                found_nodes.extend(cached)
                continue

            # Add to queue with initial priority
            path_node = PathNode(
                node_id=node_id,
                node_data=node,
                depth=0,
                score=10.0,  # Initial score
                parent_path=[]
            )
            heapq.heappush(queue, path_node)
            visited.add(node_id)

        # BFS with priority
        while queue and budget.can_visit():
            # Get highest priority node
            current = heapq.heappop(queue)

            # Check if we've found enough results
            if len(found_nodes) >= budget.early_stop_threshold:
                logger.info(
                    f"Early stopping: found {len(found_nodes)} results "
                    f"(threshold={budget.early_stop_threshold})"
                )
                break

            # Check depth limit
            if current.depth >= budget.max_hops:
                continue

            # Get neighbors
            neighbors = self._get_neighbors(
                current.node_id,
                relationship_types,
                budget
            )

            for neighbor_data, rel_type in neighbors:
                # Support both 'node_id' (from Weaviate) and 'id' (from Neo4j)
                neighbor_id = neighbor_data.get('node_id') or neighbor_data.get('id')
                if not neighbor_id or neighbor_id in visited:
                    continue

                visited.add(neighbor_id)
                budget.visit_node()

                # Check if this is a target node
                neighbor_type = neighbor_data.get('type')
                if neighbor_type == target_type:
                    found_nodes.append(neighbor_data)

                    # Cache the path
                    if current.depth > 0:  # Only cache multi-hop paths
                        self._cache_path(
                            start_id=current.parent_path[0] if current.parent_path else current.node_id,
                            target_type=target_type,
                            strategy=strategy,
                            path_nodes=current.parent_path + [current.node_id, neighbor_id]
                        )

                    # Record relationship
                    found_relationships.append({
                        'type': rel_type,
                        'source': current.node_id,
                        'target': neighbor_id,
                    })

                # Add to queue if within depth limit
                if current.depth + 1 < budget.max_hops:
                    # Calculate priority score
                    score = self._calculate_score(
                        rel_type,
                        current.score,
                        current.depth + 1
                    )

                    path_node = PathNode(
                        node_id=neighbor_id,
                        node_data=neighbor_data,
                        depth=current.depth + 1,
                        score=score,
                        parent_path=current.parent_path + [current.node_id]
                    )

                    heapq.heappush(queue, path_node)

                # Check budget
                if not budget.can_visit():
                    logger.warning("Budget exhausted during traversal")
                    break

        logger.info(
            f"Traversal complete: found {len(found_nodes)} nodes, "
            f"visited {budget.nodes_visited} nodes in {budget.elapsed_time():.2f}s"
        )

        return found_nodes, found_relationships

    def _get_neighbors(
        self,
        node_id: str,
        relationship_types: List[str],
        budget: TraversalBudget
    ) -> List[Tuple[Dict[str, Any], str]]:
        """
        Get neighbors of a node.

        Returns:
            List of (neighbor_data, relationship_type) tuples
        """
        try:
            # Build relationship pattern
            rel_pattern = '|'.join(relationship_types)

            cypher = f"""
            MATCH (start {{id: $node_id}})-[r:{rel_pattern}]-(neighbor)
            RETURN neighbor, type(r) as rel_type
            LIMIT 50
            """

            result = self.neo4j.execute_cypher(
                cypher,
                parameters={'node_id': node_id}
            )

            neighbors = []
            for record in result:
                neighbor = dict(record['neighbor'])
                rel_type = record['rel_type']
                neighbors.append((neighbor, rel_type))

            return neighbors

        except Exception as e:
            logger.error(f"Failed to get neighbors for {node_id}: {e}")
            return []

    def _calculate_score(
        self,
        rel_type: str,
        parent_score: float,
        depth: int
    ) -> float:
        """
        Calculate priority score for a node.

        Higher score = higher priority in traversal.

        Factors:
        - Relationship type priority
        - Parent score (propagates high priority)
        - Depth (penalize deeper nodes)
        """
        # Base score from relationship type
        rel_priority = RelationshipPriority.get_priority(rel_type)

        # Depth penalty (deeper = lower priority)
        depth_penalty = 0.8 ** depth

        # Combine factors
        score = (rel_priority * 0.6 + parent_score * 0.4) * depth_penalty

        return score

    def _cache_path(
        self,
        start_id: str,
        target_type: str,
        strategy: str,
        path_nodes: List[str]
    ):
        """Cache a discovered path."""
        # For now, cache just the node IDs
        # Could be enhanced to include full node data
        path_data = [{'id': node_id} for node_id in path_nodes]
        self.path_cache.put(start_id, target_type, strategy, path_data)
