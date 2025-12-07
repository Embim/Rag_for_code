"""
Code Retrieval System.

Implements multi-hop graph traversal strategies for finding code entities:
- UI to Database: Trace from component to data model
- Database to UI: Find all UI that displays a model
- Impact Analysis: Find what breaks if entity changes
- Pattern Search: Find entities matching criteria
"""

from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from ..graph import Neo4jClient, WeaviateIndexer
from ..graph.models import NodeType, RelationshipType
from .multi_hop_optimizer import (
    MultiHopOptimizer,
    TraversalBudget,
    PathCache,
)
from ...logger import get_logger


logger = get_logger(__name__)


class SearchStrategy(str, Enum):
    """Available search strategies."""
    UI_TO_DATABASE = "ui_to_database"
    DATABASE_TO_UI = "database_to_ui"
    IMPACT_ANALYSIS = "impact_analysis"
    PATTERN_SEARCH = "pattern_search"
    SEMANTIC_ONLY = "semantic_only"  # Pure vector search


@dataclass
class SearchConfig:
    """Configuration for code search."""

    strategy: SearchStrategy = SearchStrategy.SEMANTIC_ONLY

    # Multi-hop configuration
    max_hops: int = 3
    expand_results: bool = True  # Expand with graph neighbors

    # Retrieval parameters
    top_k_vector: int = 15
    top_k_bm25: int = 25
    top_k_final: int = 10
    hybrid_alpha: float = 0.3  # 0.0 = BM25 only, 1.0 = vector only

    # Phase 4: Advanced search features
    enable_query_expansion: bool = False  # Disabled by default (can break queries)
    query_expansion_method: str = "synonyms"  # "synonyms", "llm", "hybrid"
    enable_query_reformulation: bool = False  # Disabled by default (breaks queries: 'book new trades' -> 'm')
    query_reformulation_method: str = "simple"  # "simple", "expanded", "multi", "rephrase", "decompose", "clarify", "all"
    enable_reranking: bool = True
    reranker_type: str = "cross_encoder"  # "cross_encoder", "llm", "none"
    enable_rrf: bool = False  # RRF for combining multiple search results
    rrf_k: int = 60  # RRF constant

    # Filters
    node_types: Optional[List[str]] = None
    repositories: Optional[List[str]] = None
    file_patterns: Optional[List[str]] = None

    # Strategy-specific config
    follow_calls: bool = True
    follow_imports: bool = True
    follow_inheritance: bool = True

    # Cost control
    max_nodes_to_visit: int = 1000
    timeout_seconds: int = 30


@dataclass
class SearchResult:
    """Result of code search."""

    # Primary results (from initial search)
    primary_nodes: List[Dict[str, Any]] = field(default_factory=list)

    # Expanded results (from graph traversal)
    expanded_nodes: List[Dict[str, Any]] = field(default_factory=list)

    # Relationships between results
    relationships: List[Dict[str, Any]] = field(default_factory=list)

    # Search metadata
    strategy_used: str = ""
    total_nodes_visited: int = 0
    execution_time_ms: float = 0.0

    def all_nodes(self) -> List[Dict[str, Any]]:
        """Get all nodes (primary + expanded)."""
        return self.primary_nodes + self.expanded_nodes

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'primary_nodes': self.primary_nodes,
            'expanded_nodes': self.expanded_nodes,
            'relationships': self.relationships,
            'metadata': {
                'strategy': self.strategy_used,
                'total_nodes': self.total_nodes_visited,
                'execution_time_ms': self.execution_time_ms,
            }
        }


class CodeRetriever:
    """
    Code retrieval system with multi-hop graph traversal.

    Combines:
    - Semantic search (Weaviate vector similarity)
    - Keyword search (BM25)
    - Graph traversal (Neo4j multi-hop)

    Implements various search strategies for different use cases.
    """

    def __init__(
        self,
        neo4j_client: Neo4jClient,
        weaviate_indexer: WeaviateIndexer,
        config: Optional[SearchConfig] = None
    ):
        """
        Initialize code retriever.

        Args:
            neo4j_client: Neo4j client for graph traversal
            weaviate_indexer: Weaviate client for vector search
            config: Search configuration
        """
        self.neo4j = neo4j_client
        self.weaviate = weaviate_indexer
        self.config = config or SearchConfig()

        # Initialize multi-hop optimizer with shared path cache
        self.path_cache = PathCache(max_size=1000)
        self.optimizer = MultiHopOptimizer(neo4j_client, self.path_cache)

        # Phase 4: Initialize advanced search components
        self.query_expander = None
        self.query_reformulator = None
        self.reranker = None
        self.rrf_fusion = None

        # Query Expansion
        if self.config.enable_query_expansion:
            try:
                from ...query import QueryExpander
                self.query_expander = QueryExpander()
                logger.info(f"Query Expansion enabled (method: {self.config.query_expansion_method})")
            except Exception as e:
                logger.warning(f"Query Expansion не загружен: {e}")

        # Query Reformulation
        if self.config.enable_query_reformulation:
            try:
                from ...query import QueryReformulator
                from ...config.agent import AgentConfig
                agent_config = AgentConfig.from_env()
                self.query_reformulator = QueryReformulator(api_key=agent_config.api_key)
                logger.info(f"Query Reformulation enabled (method: {self.config.query_reformulation_method})")
            except Exception as e:
                logger.warning(f"Query Reformulation не загружен: {e}")

        # Reranker
        if self.config.enable_reranking:
            try:
                if self.config.reranker_type == "cross_encoder":
                    from ...ranking import CrossEncoderReranker
                    self.reranker = CrossEncoderReranker()
                    logger.info("Cross-Encoder Reranker enabled")
            except Exception as e:
                logger.warning(f"Reranker не загружен: {e}")

        # RRF Fusion
        if self.config.enable_rrf:
            try:
                from ...ranking import ReciprocalRankFusion
                self.rrf_fusion = ReciprocalRankFusion(k=self.config.rrf_k)
                logger.info(f"RRF enabled (k={self.config.rrf_k})")
            except Exception as e:
                logger.warning(f"RRF не загружен: {e}")

    def search(
        self,
        query: str,
        strategy: Optional[SearchStrategy] = None,
        config_override: Optional[Dict[str, Any]] = None
    ) -> SearchResult:
        """
        Search for code entities.

        Args:
            query: Search query (natural language or keywords)
            strategy: Search strategy to use (overrides config)
            config_override: Override specific config values

        Returns:
            SearchResult with found nodes and relationships
        """
        import time
        start_time = time.time()

        # Apply overrides
        config = self._apply_config_overrides(config_override)
        strategy = strategy or config.strategy

        logger.info(f"Searching with strategy: {strategy.value}, query: '{query}'")

        # Phase 4: Query Reformulation (pre-processing)
        working_query = query
        if self.query_reformulator and config.enable_query_reformulation:
            try:
                reformulated_query = self.query_reformulator.reformulate(
                    query,
                    method=config.query_reformulation_method
                )
                # reformulate() returns a single string, not a list
                if reformulated_query and reformulated_query.strip():
                    working_query = reformulated_query
                    logger.info(f"Query reformulated: '{query}' -> '{working_query}'")
            except Exception as e:
                logger.warning(f"Query reformulation failed: {e}")

        # Step 1: Initial semantic/keyword search (with reformulated query)
        primary_nodes = self._initial_search(working_query, config)

        if not primary_nodes:
            logger.warning(f"No results found for query: '{query}' (working query: '{working_query}')")
            logger.warning(f"Search config: repositories={config.repositories}, node_types={config.node_types}")
            return SearchResult(
                strategy_used=strategy.value,
                execution_time_ms=(time.time() - start_time) * 1000
            )

        # Phase 4: Reranking
        if self.reranker and config.enable_reranking:
            try:
                # Ensure 'text' field exists in each node for reranker
                documents = []
                for node in primary_nodes:
                    doc = node.copy()
                    if 'text' not in doc:
                        doc['text'] = f"{doc.get('name', '')} {doc.get('docstring', '')} {doc.get('content', '')}"
                    documents.append(doc)

                # Rerank (documents is List[dict])
                reranked_df = self.reranker.rerank(
                    query=working_query,
                    documents=documents,
                    top_k=config.top_k_final
                )

                # Convert back to list of dicts
                primary_nodes = reranked_df.to_dict('records')
                logger.info(f"Reranked {len(primary_nodes)} nodes")
            except Exception as e:
                logger.warning(f"Reranking failed: {e}")

        # Step 2: Apply strategy-specific expansion
        if config.expand_results and strategy != SearchStrategy.SEMANTIC_ONLY:
            expanded, relationships = self._expand_results(
                primary_nodes,
                strategy,
                config
            )
        else:
            expanded = []
            relationships = []

        # Build result
        result = SearchResult(
            primary_nodes=primary_nodes[:config.top_k_final],
            expanded_nodes=expanded,
            relationships=relationships,
            strategy_used=strategy.value,
            total_nodes_visited=len(primary_nodes) + len(expanded),
            execution_time_ms=(time.time() - start_time) * 1000
        )

        logger.info(
            f"Search complete: {len(result.primary_nodes)} primary, "
            f"{len(result.expanded_nodes)} expanded in {result.execution_time_ms:.1f}ms"
        )

        return result

    def _apply_config_overrides(
        self,
        overrides: Optional[Dict[str, Any]]
    ) -> SearchConfig:
        """Apply configuration overrides."""
        if not overrides:
            return self.config

        # Create a copy of config with overrides
        import copy
        config = copy.copy(self.config)

        for key, value in overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)

        return config

    def _initial_search(
        self,
        query: str,
        config: SearchConfig
    ) -> List[Dict[str, Any]]:
        """
        Initial search using Weaviate (semantic + BM25).

        Phase 4: Includes Query Expansion and RRF fusion.

        Returns list of nodes sorted by relevance.
        """
        try:
            # Phase 4: Query Expansion
            queries = [query]
            if self.query_expander and config.enable_query_expansion:
                try:
                    expanded = self.query_expander.expand(
                        query,
                        max_expansions=5
                    )
                    # Take top 3 variants to balance quality vs speed
                    queries = expanded[:3]
                    if len(queries) > 1:
                        logger.info(f"Query expanded to {len(queries)} variants")
                except Exception as e:
                    logger.warning(f"Query expansion failed: {e}")
                    queries = [query]

            # Execute search for each query variant
            all_results = []
            for q in queries:
                try:
                    results = self.weaviate.search(
                        query=q,
                        node_types=config.node_types,
                        limit=max(config.top_k_vector, config.top_k_bm25),
                        alpha=config.hybrid_alpha
                    )

                    logger.info(f"Weaviate returned {len(results)} results for query '{q}'")

                    # Filter by repository if specified
                    if config.repositories:
                        # Flatten repositories list if nested (e.g., [['ui', 'api']] -> ['ui', 'api'])
                        repos = config.repositories
                        if repos and isinstance(repos[0], list):
                            repos = repos[0]
                            logger.warning(f"Flattened nested repositories list: {config.repositories} -> {repos}")

                        logger.info(f"Filtering by repositories: {repos}")
                        before_count = len(results)
                        results = [
                            r for r in results
                            if r.get('repository') in repos
                        ]
                        logger.info(f"After repository filter: {before_count} -> {len(results)} results")

                    all_results.append(results)
                except Exception as e:
                    logger.error(f"Search failed for query '{q}': {e}")
                    continue

            if not all_results:
                logger.warning("all_results is empty - no queries returned results")
                return []

            # Check if all result sets are empty
            total_results = sum(len(res) for res in all_results)
            logger.info(f"Total results across {len(all_results)} queries: {total_results}")

            if total_results == 0:
                logger.warning("All result sets are empty")
                return []

            # Phase 4: Combine results with RRF if enabled
            if self.rrf_fusion and config.enable_rrf and len(all_results) > 1:
                try:
                    import pandas as pd
                    # Convert to DataFrames
                    dfs = [pd.DataFrame(res) for res in all_results]

                    # Add retrieval_score if missing (use 1.0 as placeholder)
                    for df in dfs:
                        if 'retrieval_score' not in df.columns:
                            df['retrieval_score'] = 1.0

                    # Fuse results
                    if len(dfs) == 2:
                        fused_df = self.rrf_fusion.fuse_two_results(dfs[0], dfs[1])
                    else:
                        fused_df = self.rrf_fusion.fuse_multiple_results(dfs)

                    final_results = fused_df.to_dict('records')
                    logger.info(f"RRF fusion: combined {len(all_results)} result sets into {len(final_results)} nodes")
                    return final_results
                except Exception as e:
                    logger.warning(f"RRF fusion failed: {e}, using first result set")
                    return all_results[0]

            # No RRF: just return first result set (original query)
            return all_results[0] if all_results else []

        except Exception as e:
            logger.error(f"Initial search failed: {e}")
            return []

    def _expand_results(
        self,
        primary_nodes: List[Dict[str, Any]],
        strategy: SearchStrategy,
        config: SearchConfig
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Expand results using graph traversal based on strategy.

        Returns:
            Tuple of (expanded_nodes, relationships)
        """
        if strategy == SearchStrategy.UI_TO_DATABASE:
            return self._ui_to_database_expansion(primary_nodes, config)

        elif strategy == SearchStrategy.DATABASE_TO_UI:
            return self._database_to_ui_expansion(primary_nodes, config)

        elif strategy == SearchStrategy.IMPACT_ANALYSIS:
            return self._impact_analysis_expansion(primary_nodes, config)

        elif strategy == SearchStrategy.PATTERN_SEARCH:
            # Pattern search doesn't need expansion
            return [], []

        else:
            # Default: expand with immediate neighbors
            return self._default_expansion(primary_nodes, config)

    def _ui_to_database_expansion(
        self,
        nodes: List[Dict[str, Any]],
        config: SearchConfig
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Expand from UI components to database models.

        Traversal path:
        Component -> API Call -> Endpoint -> Function -> Model
        """
        expanded = []
        relationships = []
        visited = set()

        for node in nodes:
            # Support both 'node_id' (from Weaviate) and 'id' (from Neo4j)
            node_id = node.get('node_id') or node.get('id')
            if not node_id or node_id in visited:
                continue

            visited.add(node_id)

            # Query Neo4j for path to models
            cypher = """
            MATCH path = (start {id: $node_id})
                -[:SENDS_REQUEST_TO|HANDLES_REQUEST|CALLS|USES_MODEL*1..4]->
                (target:Model)
            RETURN
                nodes(path) as path_nodes,
                relationships(path) as path_rels
            LIMIT 10
            """

            try:
                result = self.neo4j.execute_cypher(
                    cypher,
                    parameters={'node_id': node_id}
                )

                for record in result:
                    path_nodes = record['path_nodes']
                    path_rels = record['path_rels']

                    # Add nodes to expanded (except first which is already in primary)
                    for pnode in path_nodes[1:]:
                        node_dict = dict(pnode)
                        if node_dict['id'] not in visited:
                            expanded.append(node_dict)
                            visited.add(node_dict['id'])

                    # Add relationships
                    for rel in path_rels:
                        relationships.append({
                            'type': rel.type,
                            'source': rel.start_node['id'],
                            'target': rel.end_node['id'],
                        })

            except Exception as e:
                logger.error(f"Graph traversal failed for node {node_id}: {e}")
                continue

        return expanded, relationships

    def _database_to_ui_expansion(
        self,
        nodes: List[Dict[str, Any]],
        config: SearchConfig
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Expand from database models to UI components.

        Traversal path (reverse of ui_to_database):
        Model -> Function -> Endpoint -> API Call -> Component
        """
        expanded = []
        relationships = []
        visited = set()

        for node in nodes:
            # Support both 'node_id' (from Weaviate) and 'id' (from Neo4j)
            node_id = node.get('node_id') or node.get('id')
            if not node_id or node_id in visited:
                continue

            visited.add(node_id)

            # Query Neo4j for path to components (reverse direction)
            cypher = """
            MATCH path = (start {id: $node_id})
                <-[:USES_MODEL|CALLS|HANDLES_REQUEST|SENDS_REQUEST_TO*1..4]-
                (target:Component)
            RETURN
                nodes(path) as path_nodes,
                relationships(path) as path_rels
            LIMIT 10
            """

            try:
                result = self.neo4j.execute_cypher(
                    cypher,
                    parameters={'node_id': node_id}
                )

                for record in result:
                    path_nodes = record['path_nodes']
                    path_rels = record['path_rels']

                    for pnode in path_nodes[1:]:
                        node_dict = dict(pnode)
                        if node_dict['id'] not in visited:
                            expanded.append(node_dict)
                            visited.add(node_dict['id'])

                    for rel in path_rels:
                        relationships.append({
                            'type': rel.type,
                            'source': rel.start_node['id'],
                            'target': rel.end_node['id'],
                        })

            except Exception as e:
                logger.error(f"Graph traversal failed for node {node_id}: {e}")
                continue

        return expanded, relationships

    def _impact_analysis_expansion(
        self,
        nodes: List[Dict[str, Any]],
        config: SearchConfig
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Find what will be affected if node changes.

        Looks for:
        - Functions/components that CALL this function
        - Classes that INHERIT from this class
        - Components that IMPORT this module
        """
        expanded = []
        relationships = []
        visited = set()

        for node in nodes:
            # Support both 'node_id' (from Weaviate) and 'id' (from Neo4j)
            node_id = node.get('node_id') or node.get('id')
            if not node_id or node_id in visited:
                continue

            visited.add(node_id)

            # Find all nodes that depend on this one
            cypher = """
            MATCH (start {id: $node_id})<-[r:CALLS|INHERITS|IMPORTS]-(dependent)
            RETURN dependent, r
            LIMIT 50
            """

            try:
                result = self.neo4j.execute_cypher(
                    cypher,
                    parameters={'node_id': node_id}
                )

                for record in result:
                    dependent = dict(record['dependent'])
                    rel = record['r']

                    if dependent['id'] not in visited:
                        expanded.append(dependent)
                        visited.add(dependent['id'])

                    relationships.append({
                        'type': rel.type,
                        'source': rel.start_node['id'],
                        'target': rel.end_node['id'],
                        'impact': 'breaking_change',  # Mark as potential breaking change
                    })

            except Exception as e:
                logger.error(f"Impact analysis failed for node {node_id}: {e}")
                continue

        return expanded, relationships

    def _default_expansion(
        self,
        nodes: List[Dict[str, Any]],
        config: SearchConfig
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Default expansion: get immediate neighbors.
        """
        expanded = []
        relationships = []
        visited = set()

        for node in nodes:
            # Support both 'node_id' (from Weaviate) and 'id' (from Neo4j)
            node_id = node.get('node_id') or node.get('id')
            if not node_id or node_id in visited:
                continue

            visited.add(node_id)

            # Get neighbors
            cypher = """
            MATCH (start {id: $node_id})-[r]-(neighbor)
            RETURN neighbor, r
            LIMIT 10
            """

            try:
                result = self.neo4j.execute_cypher(
                    cypher,
                    parameters={'node_id': node_id}
                )

                for record in result:
                    neighbor = dict(record['neighbor'])
                    rel = record['r']

                    if neighbor['id'] not in visited:
                        expanded.append(neighbor)
                        visited.add(neighbor['id'])

                    relationships.append({
                        'type': rel.type,
                        'source': rel.start_node['id'],
                        'target': rel.end_node['id'],
                    })

            except Exception as e:
                logger.error(f"Expansion failed for node {node_id}: {e}")
                continue

        return expanded, relationships
