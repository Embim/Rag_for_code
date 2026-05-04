"""
Search configuration for Code RAG.
"""

from dataclasses import dataclass
from typing import Optional, List, Literal
from enum import Enum

from .base import BaseConfig


class SearchStrategy(str, Enum):
    """Search strategies — single source of truth.

    Минимальный набор:
    - SEMANTIC_ONLY — Weaviate-only (vector + BM25 через hybrid_alpha).
    - UI_TO_DATABASE / DATABASE_TO_UI — graph expansion через
      ``MAKES_CALL|CALLS_ENDPOINT|HANDLES_REQUEST|CALLS|USES_MODEL``.

    Удалены: IMPACT_ANALYSIS, PATTERN_SEARCH, BM25_ONLY, HYBRID, MULTI_HOP —
    либо были основаны на неточном CALLS-графе, либо дублировали SEMANTIC_ONLY
    (через параметр ``hybrid_alpha``).
    """
    SEMANTIC_ONLY = "semantic_only"
    UI_TO_DATABASE = "ui_to_database"
    DATABASE_TO_UI = "database_to_ui"


@dataclass
class SearchConfig(BaseConfig):
    """
    Configuration for code search operations.

    Attributes:
        top_k: Number of results to return
        top_k_dense: Results from vector search
        top_k_bm25: Results from BM25 search
        hybrid_alpha: Balance between dense (1.0) and sparse (0.0)
        enable_reranking: Use cross-encoder reranking
        enable_query_expansion: Expand queries with synonyms
        enable_query_reformulation: Reformulate queries with LLM
        query_reformulation_method: Method for reformulation
        max_hops: Maximum hops for multi-hop search
        timeout_seconds: Search timeout
        scope_filter: Filter by scope (frontend/backend/shared)
    """
    # Basic settings
    top_k: int = 50
    top_k_dense: int = 100
    top_k_bm25: int = 100
    hybrid_alpha: float = 0.7

    # Reranking
    enable_reranking: bool = True

    # Query processing
    enable_query_expansion: bool = False
    enable_query_reformulation: bool = False
    query_reformulation_method: Literal[
        "simple", "expanded", "multi", "rephrase", "decompose", "clarify", "all"
    ] = "expanded"

    # Multi-hop
    max_hops: int = 3
    timeout_seconds: float = 30.0

    # Filters
    repositories: Optional[List[str]] = None
    node_types: Optional[List[str]] = None
    scope_filter: Optional[Literal["frontend", "backend", "shared"]] = None

    def for_strategy(self, strategy: SearchStrategy) -> 'SearchConfig':
        """Get config optimized for specific strategy."""
        config = SearchConfig(**self.to_dict())
        if strategy == SearchStrategy.SEMANTIC_ONLY:
            config.hybrid_alpha = 1.0
        elif strategy in (SearchStrategy.UI_TO_DATABASE, SearchStrategy.DATABASE_TO_UI):
            config.max_hops = 50
            config.enable_reranking = True
        return config
