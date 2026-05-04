"""
Code Retrieval System.

Pipeline:
1. Initial search via Weaviate (vector + BM25 + optional rerank).
2. Optional graph expansion: Component ↔ Endpoint ↔ Function ↔ Model
   через ``MAKES_CALL|CALLS_ENDPOINT|HANDLES_REQUEST|CALLS|USES_MODEL``.
   Поддерживается для двух стратегий:
     - SearchStrategy.UI_TO_DATABASE   — от UI к Model.
     - SearchStrategy.DATABASE_TO_UI   — обратно.
   Прочие стратегии (SEMANTIC_ONLY) — graph expansion не делается.

NL→Cypher (LLM генерирует cypher по запросу) — отдельная фича для следующей
итерации, см. memory/MEMORY.md.
"""

from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from pathlib import Path

from src.core.graph import Neo4jClient, WeaviateIndexer
from src.core.graph.models import NodeType, RelationshipType
from src.infra.logger import get_logger

# SearchStrategy is defined in src/config/search.py (single source of truth).
# We import from there to avoid a circular dependency:
#   code_retriever → graph → neo4j_client → logger → src.config → config.search
# Importing from config.search here is safe because config.search does not
# import from code_rag at module level.
from src.infra.config.search import SearchStrategy  # noqa: F401 — re-exported via __init__


logger = get_logger(__name__)


def _diversify_results(
    results: List[Dict[str, Any]],
    *,
    cap_per_file: Optional[int] = None,
    cap_per_name: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Diversity-фильтр для top-K Weaviate.

    Два независимых cap'а:

    - ``cap_per_file``: не более N результатов с одного ``file_path``.
      Защита от случая «один большой класс выдал 5 методов подряд».

    - ``cap_per_name``: не более N результатов с одинаковым ``name``.
      Защита от случая «8 одноимённых ``compute_valuation`` из 8
      разных файлов заняли весь top-K» — реально наблюдалось на запросе
      «как считается net_pos», embedding имени совпадал у всех 8 копий
      и выдавил из top-K единственную нужную ноду из ``position_by_book_service.py``.

    Сохраняет порядок (top-K по score). ``cap=0/None`` → без cap'а по этому критерию.
    """
    if not cap_per_file and not cap_per_name:
        return results

    by_file: Dict[str, int] = {}
    by_name: Dict[str, int] = {}
    capped: List[Dict[str, Any]] = []
    for r in results:
        if cap_per_file:
            file_key = r.get('file_path') or r.get('node_id') or ''
            if by_file.get(file_key, 0) >= cap_per_file:
                continue
        if cap_per_name:
            name_key = (r.get('name') or '').strip()
            if name_key and by_name.get(name_key, 0) >= cap_per_name:
                continue
        if cap_per_file:
            by_file[file_key] = by_file.get(file_key, 0) + 1
        if cap_per_name:
            name_key = (r.get('name') or '').strip()
            if name_key:
                by_name[name_key] = by_name.get(name_key, 0) + 1
        capped.append(r)
    return capped


@dataclass
class SearchConfig:
    """Configuration for code search."""

    strategy: SearchStrategy = SearchStrategy.SEMANTIC_ONLY

    # Graph expansion: вкл. только для UI_TO_DATABASE / DATABASE_TO_UI стратегий.
    expand_results: bool = True

    # Retrieval parameters
    # Initial pool (top_k_vector / top_k_bm25) — сколько Weaviate возвращает
    # ДО diversify-фильтра. После cap_per_name=2 + cap_per_file=2 пул
    # обычно сокращается на 60-80%, поэтому начинаем с большего числа,
    # чтобы после diversify осталось ≥ top_k_final разнообразных чанков.
    top_k_vector: int = 30
    top_k_bm25: int = 50
    top_k_final: int = 10
    hybrid_alpha: float = 0.3  # 0.0 = BM25 only, 1.0 = vector only

    # Phase 4: Advanced search features
    enable_query_expansion: bool = False  # Disabled by default (can break queries)
    query_expansion_method: str = "synonyms"  # "synonyms", "llm", "hybrid"
    enable_query_reformulation: bool = False  # Disabled by default (breaks queries: 'book new trades' -> 'm')
    query_reformulation_method: str = "simple"  # "simple", "expanded", "multi", "rephrase", "decompose", "clarify", "all"
    enable_reranking: bool = True
    reranker_type: str = "cross_encoder"  # "cross_encoder", "llm", "none"

    # Diversity для финальной выдачи Weaviate. Два независимых cap'а:
    #
    # - max_per_file: не более N чанков с одного ``file_path``. Защита от
    #   случая «класс выдал 5 методов подряд».
    # - max_per_name: не более N чанков с одинаковым ``name``. Защита от
    #   случая «8 одноимённых ``compute_valuation`` из 8 разных файлов»
    #   (наблюдалось на запросе «как считается net_pos» — embedding имени
    #   совпадал у всех 8 копий и они занимали весь top-K).
    #
    # 0/None — без cap'а по этому критерию.
    max_per_file: int = 2
    max_per_name: int = 2

    # Filters
    node_types: Optional[List[str]] = None
    repositories: Optional[List[str]] = None
    file_patterns: Optional[List[str]] = None

    # Cost control
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

        # Lazy advanced search components
        self.query_expander = None
        self.query_reformulator = None
        self.reranker = None

        # Query Expansion
        if self.config.enable_query_expansion:
            try:
                from src.search.preprocessing import QueryExpander
                self.query_expander = QueryExpander()
                logger.info(f"Query Expansion enabled (method: {self.config.query_expansion_method})")
            except Exception as e:
                logger.warning(f"Query Expansion не загружен: {e}")

        # Query Reformulation
        if self.config.enable_query_reformulation:
            try:
                from src.search.preprocessing import QueryReformulator
                from src.infra.config.agent import AgentConfig
                agent_config = AgentConfig()
                self.query_reformulator = QueryReformulator(api_key=agent_config.api_key)
                logger.info(f"Query Reformulation enabled (method: {self.config.query_reformulation_method})")
            except Exception as e:
                logger.warning(f"Query Reformulation не загружен: {e}")

        # Reranker
        if self.config.enable_reranking:
            try:
                if self.config.reranker_type == "cross_encoder":
                    from src.search.ranking import CrossEncoderReranker
                    self.reranker = CrossEncoderReranker()
                    logger.info("Cross-Encoder Reranker enabled")
            except Exception as e:
                logger.warning(f"Reranker не загружен: {e}")

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

        # Step 1: Initial semantic/keyword search (with reformulated query).
        # Weaviate возвращает уже всё что нужно (content, code, signature, docstring) —
        # отдельный Neo4j-enrichment больше не делаем: после refactor #1 в Neo4j нет
        # источника `code`, signature/docstring дублируются в Weaviate.
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

        # Step 2: Graph expansion — только для UI↔DB стратегий.
        # IMPACT_ANALYSIS / DEFAULT (cross-file CALLS) ушли — давали много шума
        # из-за неточного resolve вызовов в парсерах.
        expanded, relationships = ([], [])
        if config.expand_results and strategy in (
            SearchStrategy.UI_TO_DATABASE, SearchStrategy.DATABASE_TO_UI
        ):
            expanded, relationships = self._expand_ui_db(primary_nodes, strategy)

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

            # Multi-query (expansion/reformulation) даёт несколько result sets;
            # сейчас просто берём результат исходного запроса. Hybrid-fusion
            # (BM25 + vector) выполняется внутри Weaviate через hybrid_alpha.
            picked = all_results[0] if all_results else []
            picked = _diversify_results(
                picked,
                cap_per_file=config.max_per_file,
                cap_per_name=config.max_per_name,
            )
            return picked

        except Exception as e:
            logger.error(f"Initial search failed: {e}")
            return []

    def _expand_ui_db(
        self,
        nodes: List[Dict[str, Any]],
        strategy: SearchStrategy,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Graph expansion для UI↔DB: один cypher‑паттерн с переменной длины пути
        ``MAKES_CALL|CALLS_ENDPOINT|HANDLES_REQUEST|CALLS|USES_MODEL*1..6``.

        Направление зависит от стратегии:
        - UI_TO_DATABASE: Component → … → Model
        - DATABASE_TO_UI: Component ← … ← Model (start = Model)
        """
        if strategy == SearchStrategy.UI_TO_DATABASE:
            cypher = """
            MATCH path = (start {id: $node_id})
                -[:MAKES_CALL|CALLS_ENDPOINT|HANDLES_REQUEST|CALLS|USES_MODEL*1..6]->
                (target:Model)
            RETURN nodes(path) AS path_nodes, relationships(path) AS path_rels
            LIMIT 10
            """
        elif strategy == SearchStrategy.DATABASE_TO_UI:
            cypher = """
            MATCH path = (start {id: $node_id})
                <-[:USES_MODEL|CALLS|HANDLES_REQUEST|CALLS_ENDPOINT|MAKES_CALL*1..6]-
                (target:Component)
            RETURN nodes(path) AS path_nodes, relationships(path) AS path_rels
            LIMIT 10
            """
        else:
            return [], []

        expanded: List[Dict[str, Any]] = []
        relationships: List[Dict[str, Any]] = []
        visited: Set[str] = set()

        for node in nodes:
            node_id = node.get('node_id') or node.get('id')
            if not node_id or node_id in visited:
                continue
            visited.add(node_id)

            try:
                result = self.neo4j.execute_cypher(
                    cypher, parameters={'node_id': node_id}
                )
            except Exception as e:
                logger.error(f"Graph expansion failed for {node_id}: {e}")
                continue

            for record in result:
                for pnode in record['path_nodes'][1:]:
                    nd = dict(pnode)
                    if nd.get('id') and nd['id'] not in visited:
                        expanded.append(nd)
                        visited.add(nd['id'])
                for rel in record['path_rels']:
                    relationships.append({
                        'type': rel.type,
                        'source': rel.start_node['id'],
                        'target': rel.end_node['id'],
                    })

        return expanded, relationships
