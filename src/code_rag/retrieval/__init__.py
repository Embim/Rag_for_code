"""
Code and Document retrieval module.

Provides advanced search capabilities for:
- Code repositories: multi-hop graph traversal, strategy-based search
- Documentation: SOP, policies, manuals
- Semantic + keyword search
"""

from .code_retriever import (
    CodeRetriever,
    SearchStrategy,
    SearchConfig,
    SearchResult,
)

from .document_retriever import (
    DocumentRetriever,
    DocumentSearchConfig,
    DocumentSearchResult,
)

from .scope_detector import (
    ScopeDetector,
    QueryScope,
    ScopeHint,
)


__all__ = [
    'CodeRetriever',
    'SearchStrategy',
    'SearchConfig',
    'SearchResult',
    'DocumentRetriever',
    'DocumentSearchConfig',
    'DocumentSearchResult',
    'ScopeDetector',
    'QueryScope',
    'ScopeHint',
]
