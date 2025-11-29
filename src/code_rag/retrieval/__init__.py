"""
Code retrieval module.

Provides advanced search capabilities for code repositories:
- Multi-hop graph traversal
- Strategy-based search (UI-to-DB, DB-to-UI, impact analysis)
- Scope detection (frontend/backend/hybrid)
- Semantic + keyword search
"""

from .code_retriever import (
    CodeRetriever,
    SearchStrategy,
    SearchConfig,
    SearchResult,
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
    'ScopeDetector',
    'QueryScope',
    'ScopeHint',
]
