"""
Query Scope Detector.

Determines whether a search query is related to:
- Frontend (UI, components, React)
- Backend (API, database, models)
- Hybrid (both)

This helps narrow down search to relevant parts of the codebase.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

from ...logger import get_logger


logger = get_logger(__name__)


class QueryScope(str, Enum):
    """Query scope classification."""
    FRONTEND = "frontend"
    BACKEND = "backend"
    HYBRID = "hybrid"  # Both frontend and backend
    UNKNOWN = "unknown"


@dataclass
class ScopeHint:
    """Hint about query scope with confidence."""
    scope: QueryScope
    confidence: float  # 0.0 to 1.0
    keywords_matched: List[str]
    reasoning: str


class ScopeDetector:
    """
    Detects the scope of a search query.

    Uses keyword matching and optional LLM classification.
    """

    # Keyword indicators for frontend
    FRONTEND_KEYWORDS = {
        # General UI
        'component', 'button', 'form', 'input', 'page', 'screen', 'view',
        'modal', 'dialog', 'popup', 'menu', 'navigation', 'navbar', 'sidebar',
        'header', 'footer', 'layout', 'template',

        # React-specific
        'react', 'jsx', 'tsx', 'hook', 'usestate', 'useeffect', 'usememo',
        'props', 'render', 'virtual dom',

        # Styling
        'css', 'style', 'styled', 'tailwind', 'bootstrap', 'theme',
        'color', 'font', 'responsive',

        # Frontend actions
        'click', 'hover', 'scroll', 'animation', 'transition',
        'display', 'show', 'hide', 'toggle',

        # State management
        'redux', 'zustand', 'context', 'state management',

        # Routing
        'route', 'router', 'navigation', 'redirect', 'link',

        # User interaction
        'ui', 'ux', 'interface', 'user interface', 'frontend',
    }

    # Keyword indicators for backend
    BACKEND_KEYWORDS = {
        # API
        'api', 'endpoint', 'rest', 'graphql', 'http', 'request', 'response',
        'get', 'post', 'put', 'delete', 'patch',

        # Database
        'database', 'db', 'query', 'model', 'table', 'schema',
        'migration', 'orm', 'sql', 'nosql', 'postgres', 'mysql', 'mongo',

        # Django
        'django', 'view', 'viewset', 'serializer', 'queryset',
        'admin', 'middleware',

        # FastAPI
        'fastapi', 'pydantic', 'depends', 'dependency',

        # Backend concepts
        'backend', 'server', 'service', 'controller', 'repository',
        'business logic', 'validation', 'authentication', 'authorization',
        'permission', 'jwt', 'token', 'session',

        # Data processing
        'process', 'calculate', 'transform', 'aggregate', 'filter',
        'pagination', 'cache', 'redis',
    }

    # Keywords that indicate both (hybrid)
    HYBRID_KEYWORDS = {
        'как работает',  # Russian: "how does it work"
        'поток данных',  # Russian: "data flow"
        'full flow', 'end to end', 'complete process',
        'integration', 'connect', 'связь',  # Russian: "connection"
        'from ui to database', 'from frontend to backend',
        'whole system', 'entire flow',
    }

    def __init__(self, use_llm: bool = False):
        """
        Initialize scope detector.

        Args:
            use_llm: Whether to use LLM for classification (more accurate but slower)
        """
        self.use_llm = use_llm

    def detect_scope(self, query: str) -> ScopeHint:
        """
        Detect the scope of a query.

        Args:
            query: Search query

        Returns:
            ScopeHint with detected scope and confidence
        """
        query_lower = query.lower()

        # Check for hybrid keywords first
        hybrid_matches = [kw for kw in self.HYBRID_KEYWORDS if kw in query_lower]
        if hybrid_matches:
            return ScopeHint(
                scope=QueryScope.HYBRID,
                confidence=0.9,
                keywords_matched=hybrid_matches,
                reasoning=f"Query contains hybrid keywords: {', '.join(hybrid_matches)}"
            )

        # Count frontend and backend keywords
        frontend_matches = [kw for kw in self.FRONTEND_KEYWORDS if kw in query_lower]
        backend_matches = [kw for kw in self.BACKEND_KEYWORDS if kw in query_lower]

        frontend_score = len(frontend_matches)
        backend_score = len(backend_matches)

        # Determine scope based on scores
        if frontend_score > 0 and backend_score > 0:
            # Both present - hybrid
            confidence = min(0.8, (frontend_score + backend_score) / 10)
            return ScopeHint(
                scope=QueryScope.HYBRID,
                confidence=confidence,
                keywords_matched=frontend_matches + backend_matches,
                reasoning=(
                    f"Query contains both frontend ({len(frontend_matches)}) "
                    f"and backend ({len(backend_matches)}) keywords"
                )
            )

        elif frontend_score > backend_score:
            # Frontend dominant
            confidence = min(0.9, frontend_score / 5)
            return ScopeHint(
                scope=QueryScope.FRONTEND,
                confidence=confidence,
                keywords_matched=frontend_matches,
                reasoning=f"Query contains {len(frontend_matches)} frontend keywords"
            )

        elif backend_score > frontend_score:
            # Backend dominant
            confidence = min(0.9, backend_score / 5)
            return ScopeHint(
                scope=QueryScope.BACKEND,
                confidence=confidence,
                keywords_matched=backend_matches,
                reasoning=f"Query contains {len(backend_matches)} backend keywords"
            )

        else:
            # No clear indicators
            if self.use_llm:
                return self._llm_classify(query)
            else:
                return ScopeHint(
                    scope=QueryScope.UNKNOWN,
                    confidence=0.0,
                    keywords_matched=[],
                    reasoning="No scope keywords detected"
                )

    def _llm_classify(self, query: str) -> ScopeHint:
        """
        Use LLM to classify query scope.

        TODO: Implement LLM-based classification for better accuracy.
        """
        # Placeholder for LLM classification
        logger.warning("LLM classification not yet implemented, returning UNKNOWN")

        return ScopeHint(
            scope=QueryScope.UNKNOWN,
            confidence=0.0,
            keywords_matched=[],
            reasoning="LLM classification not implemented"
        )

    def apply_scope_filter(
        self,
        scope: QueryScope,
        search_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Apply scope filter to search configuration.

        Args:
            scope: Detected scope
            search_config: Search configuration to modify

        Returns:
            Modified search configuration
        """
        if scope == QueryScope.FRONTEND:
            # Restrict to frontend node types
            search_config['node_types'] = ['Component', 'Route']
            logger.info("Applied frontend scope filter")

        elif scope == QueryScope.BACKEND:
            # Restrict to backend node types
            search_config['node_types'] = [
                'Endpoint', 'Function', 'Model', 'Class'
            ]
            logger.info("Applied backend scope filter")

        elif scope == QueryScope.HYBRID:
            # No restrictions - search everything
            search_config['node_types'] = None
            logger.info("Applied hybrid scope (no restrictions)")

        # UNKNOWN - also search everything but log it
        else:
            search_config['node_types'] = None
            logger.info("Scope unknown - searching all node types")

        return search_config
