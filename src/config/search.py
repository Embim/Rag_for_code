"""
Search configuration for Code RAG.
"""

import os
from dataclasses import dataclass, field
from typing import Optional, List, Literal
from enum import Enum

from .base import BaseConfig


class SearchStrategy(str, Enum):
    """Available search strategies."""
    SEMANTIC_ONLY = "semantic_only"
    BM25_ONLY = "bm25_only"
    HYBRID = "hybrid"
    MULTI_HOP = "multi_hop"
    UI_TO_DATABASE = "ui_to_database"
    DATABASE_TO_UI = "database_to_ui"
    IMPACT_ANALYSIS = "impact_analysis"


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
        rerank_top_k: Number of results after reranking
        enable_query_expansion: Expand queries with synonyms
        enable_query_reformulation: Reformulate queries with LLM
        query_reformulation_method: Method for reformulation
        enable_rrf: Use Reciprocal Rank Fusion
        rrf_k: RRF constant (default 60)
        max_hops: Maximum hops for multi-hop search
        timeout_seconds: Search timeout
        repository_filter: Filter by repository name
        file_type_filter: Filter by file extension
        scope_filter: Filter by scope (frontend/backend/shared)
    """
    # Basic settings - Increased for better agent exploration
    top_k: int = 50  # Increased from 10 to 30 for more comprehensive results
    top_k_dense: int = 100  # Increased from 50 to 100 for better recall
    top_k_bm25: int = 100  # Increased from 50 to 100 for better recall
    hybrid_alpha: float = 0.7

    # Reranking
    enable_reranking: bool = True
    rerank_top_k: int = 50  # Increased from 20 to 50 for better precision
    reranker_model: str = field(
        default_factory=lambda: os.getenv(
            "RERANKER_MODEL",
            "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"  # Multilingual by default
        )
    )
    
    # Query processing
    enable_query_expansion: bool = False
    enable_query_reformulation: bool = False
    query_reformulation_method: Literal[
        "simple", "expanded", "multi", "rephrase", "decompose", "clarify", "all"
    ] = "expanded"
    
    # Fusion
    enable_rrf: bool = True
    rrf_k: int = 60
    
    # Multi-hop
    max_hops: int = 3
    timeout_seconds: float = 30.0
    
    # Filters
    repository_filter: Optional[str] = None
    repositories: Optional[List[str]] = None  # Multi-repository filter (plural)
    node_types: Optional[List[str]] = None  # Node type filter
    file_type_filter: Optional[List[str]] = None
    scope_filter: Optional[Literal["frontend", "backend", "shared"]] = None
    
    # Strategy
    default_strategy: SearchStrategy = SearchStrategy.HYBRID
    
    def for_strategy(self, strategy: SearchStrategy) -> 'SearchConfig':
        """Get config optimized for specific strategy."""
        config = SearchConfig(**self.to_dict())
        
        if strategy == SearchStrategy.SEMANTIC_ONLY:
            config.hybrid_alpha = 1.0
            config.enable_rrf = False
        elif strategy == SearchStrategy.BM25_ONLY:
            config.hybrid_alpha = 0.0
            config.enable_rrf = False
        elif strategy == SearchStrategy.MULTI_HOP:
            config.max_hops = max(3, self.max_hops)
            config.enable_reranking = True
        elif strategy in (SearchStrategy.UI_TO_DATABASE, SearchStrategy.DATABASE_TO_UI):
            config.max_hops = 10
            config.enable_reranking = True
        elif strategy == SearchStrategy.IMPACT_ANALYSIS:
            config.max_hops = 10
            config.top_k = max(50, self.top_k)  # Increased from 20 to 50
        
        return config

