"""
Search and Q&A endpoints.

Provides:
- POST /search - Quick code search
- POST /ask - Agent-powered deep exploration
"""

import time
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status

from ..models import (
    SearchRequest,
    SearchResponse,
    SearchResult,
    AskRequest,
    AskResponse,
    ErrorResponse,
)
from ..dependencies import (
    get_retriever,
    get_scope_detector,
    get_orchestrator,
    get_agent_cache,
    require_agents,
)
from ..auth import require_auth, APIKey
from ...code_rag.retrieval import CodeRetriever, SearchStrategy, SearchConfig
from ...code_rag.retrieval.scope_detector import ScopeDetector, QueryScope
from ...agents import QueryOrchestrator, AgentCache
from ...logger import get_logger


logger = get_logger(__name__)

router = APIRouter(prefix="/api", tags=["search"])


# ============================================================================
# Helper Functions
# ============================================================================

def _convert_to_search_result(node_dict: dict, score: float = 0.0) -> SearchResult:
    """Convert node dictionary to SearchResult model.

    Supports:
    - Weaviate nodes (node_id, node_type, file_path)
    - Neo4j nodes (id, type)
    - Code Explorer sources (id, type, file)
    """
    # Ensure score is always a valid float (never None)
    node_score = node_dict.get('score')
    final_score = node_score if node_score is not None else score

    # Extract entity_id (try different keys)
    entity_id = (node_dict.get('node_id') or
                 node_dict.get('entity_id') or
                 node_dict.get('id', ''))

    # Extract entity_type (try different keys)
    entity_type = (node_dict.get('node_type') or
                   node_dict.get('entity_type') or
                   node_dict.get('type', 'Unknown'))

    # Extract file_path (try different keys and extract from entity_id)
    file_path = node_dict.get('file_path') or node_dict.get('file', '')
    if not file_path and entity_id and ':' in entity_id:
        # Try to extract from entity_id format: "repo:api:path/to/file.py:Function"
        parts = entity_id.split(':')
        if len(parts) >= 3:
            file_path = parts[2]

    # Extract content/code (try different keys)
    content = (node_dict.get('content') or
               node_dict.get('body') or
               node_dict.get('code') or
               node_dict.get('code_snippet', ''))

    return SearchResult(
        entity_id=entity_id,
        entity_type=entity_type,
        name=node_dict.get('name', ''),
        file_path=file_path,
        content=content,
        score=final_score,
        metadata={
            'signature': node_dict.get('signature'),
            'start_line': node_dict.get('start_line') or node_dict.get('line'),
            'end_line': node_dict.get('end_line'),
            'docstring': node_dict.get('docstring'),
            'language': node_dict.get('language'),
        }
    )


# ============================================================================
# Endpoints
# ============================================================================

@router.post(
    "/search",
    response_model=SearchResponse,
    summary="Quick code search",
    description="Perform fast semantic, BM25, or hybrid search over the codebase"
)
async def search(
    request: SearchRequest,
    retriever: CodeRetriever = Depends(get_retriever),
    scope_detector: ScopeDetector = Depends(get_scope_detector),
    current_key: APIKey = Depends(require_auth),
):
    """
    Quick code search endpoint.

    Uses configured search strategy (semantic, BM25, hybrid, or multi-hop)
    to find relevant code entities.

    **Example:**
    ```json
    {
        "query": "how does authentication work?",
        "scope": "backend",
        "limit": 10,
        "strategy": "hybrid"
    }
    ```
    """
    start_time = time.time()

    try:
        # Detect scope if not provided
        detected_scope = request.scope
        if not detected_scope:
            scope_hint = scope_detector.detect_scope(request.query)
            detected_scope = scope_hint.scope.value  # ScopeHint is a dataclass
            logger.info(f"Auto-detected scope: {detected_scope} (confidence: {scope_hint.confidence:.2f})")

        # Map strategy name to enum - use correct SearchStrategy values
        # Include aliases for backward compatibility with documentation
        strategy_map = {
            # Primary strategies
            "semantic": SearchStrategy.SEMANTIC_ONLY,
            "semantic_only": SearchStrategy.SEMANTIC_ONLY,
            "ui_to_database": SearchStrategy.UI_TO_DATABASE,
            "database_to_ui": SearchStrategy.DATABASE_TO_UI,
            "impact_analysis": SearchStrategy.IMPACT_ANALYSIS,
            "pattern_search": SearchStrategy.PATTERN_SEARCH,
            # Aliases for convenience (map to semantic with different config)
            "hybrid": SearchStrategy.SEMANTIC_ONLY,  # Weaviate hybrid search via alpha
            "bm25": SearchStrategy.SEMANTIC_ONLY,    # Use alpha=0.0 for BM25-only
            "vector": SearchStrategy.SEMANTIC_ONLY,   # Use alpha=1.0 for vector-only
        }
        strategy = strategy_map.get(
            request.strategy or "semantic",
            SearchStrategy.SEMANTIC_ONLY
        )

        # Apply scope filter to config
        config_override = {
            'top_k_final': request.limit,
            'top_k_vector': request.limit * 2,  # Get more candidates for filtering
        }

        # Adjust hybrid_alpha based on strategy alias
        if request.strategy == "bm25":
            config_override['hybrid_alpha'] = 0.0  # BM25 only
        elif request.strategy == "vector":
            config_override['hybrid_alpha'] = 1.0  # Vector only
        elif request.strategy == "hybrid":
            config_override['hybrid_alpha'] = 0.5  # Balanced hybrid

        # Apply scope-based node type filter
        if detected_scope == "frontend":
            config_override['node_types'] = ['Component', 'Route']
        elif detected_scope == "backend":
            config_override['node_types'] = ['Endpoint', 'Function', 'Model', 'Class']
        # else: no filter (hybrid or unknown)

        # Perform search
        logger.info(f"Searching: query='{request.query}', scope={detected_scope}, strategy={strategy}")

        search_result = retriever.search(
            query=request.query,
            strategy=strategy,
            config_override=config_override
        )

        # Convert to API models - SearchResult has primary_nodes list
        results = [
            _convert_to_search_result(node)
            for node in search_result.primary_nodes[:request.limit]
        ]

        elapsed_ms = (time.time() - start_time) * 1000

        return SearchResponse(
            query=request.query,
            scope=detected_scope,
            strategy=request.strategy or "semantic",
            results=results,
            total_found=len(results),
            took_ms=elapsed_ms
        )

    except Exception as e:
        logger.error(f"Search failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}"
        )


@router.post(
    "/ask",
    response_model=AskResponse,
    summary="Agent-powered Q&A",
    description="Use intelligent agent to iteratively explore codebase and answer complex questions"
)
async def ask(
    request: AskRequest,
    orchestrator: QueryOrchestrator = Depends(require_agents),
    agent_cache: Optional[AgentCache] = Depends(get_agent_cache),
    current_key: APIKey = Depends(require_auth),
):
    """
    Agent-powered deep exploration endpoint.

    Uses LLM-powered agents to:
    1. Classify the question type
    2. Select appropriate agent (code explorer, document retriever, etc.)
    3. Iteratively explore codebase using multiple tools
    4. Synthesize comprehensive answer

    **Requires OPENROUTER_API_KEY to be set.**

    **Example:**
    ```json
    {
        "question": "Explain the complete checkout flow from UI to database",
        "max_iterations": 10,
        "timeout": 120
    }
    ```
    """
    start_time = time.time()

    # Check if agents are available
    if not orchestrator:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AI agents are disabled. Set OPENROUTER_API_KEY environment variable to enable."
        )

    try:
        # Check cache first (exact match)
        cached_result = None
        similar_question = None
        
        if agent_cache and agent_cache.config.enabled:
            # 1. Try exact cache first
            cached_result = agent_cache.get_query_result(request.question)

            if cached_result:
                logger.info(f"✅ Exact cache HIT for question: '{request.question[:50]}...'")
                elapsed_ms = (time.time() - start_time) * 1000

                # Return cached response
                return AskResponse(
                    question=request.question,
                    question_type=cached_result.get('question_type', 'CODE'),
                    agent_used=cached_result.get('agent_used', 'code_explorer'),
                    answer=cached_result['answer'],
                    sources=[
                        _convert_to_search_result(src, src.get('score', 1.0))
                        for src in cached_result.get('sources', [])
                    ],
                    iterations_used=cached_result.get('iterations_used', 0),
                    tools_used=cached_result.get('tools_used', []),
                    complete=cached_result.get('complete', True),
                    took_ms=elapsed_ms,
                    cached=True
                )
            
            # 2. Try semantic cache (similar questions)
            similar_match = agent_cache.get_similar_query(request.question)
            if similar_match:
                similar_question, cached_result = similar_match
                logger.info(f"✅ Semantic cache HIT: '{similar_question[:50]}...' similar to '{request.question[:50]}...'")
                elapsed_ms = (time.time() - start_time) * 1000

                # Return cached response with note about similar question
                answer = cached_result['answer']
                if similar_question != request.question:
                    answer = f"*Similar to: \"{similar_question[:100]}...\"*\n\n{answer}"

                return AskResponse(
                    question=request.question,
                    question_type=cached_result.get('question_type', 'CODE'),
                    agent_used=cached_result.get('agent_used', 'code_explorer'),
                    answer=answer,
                    sources=[
                        _convert_to_search_result(src, src.get('score', 1.0))
                        for src in cached_result.get('sources', [])
                    ],
                    iterations_used=cached_result.get('iterations_used', 0),
                    tools_used=cached_result.get('tools_used', []),
                    complete=cached_result.get('complete', True),
                    took_ms=elapsed_ms,
                    cached=True
                )

        logger.info(f"❌ Cache MISS - Running agent for: '{request.question[:50]}...'")

        # Run agent
        context = request.context or {}
        context['max_iterations'] = request.max_iterations or 10
        context['timeout'] = request.timeout or 120
        context['detail_level'] = request.detail_level or "detailed"
        context['verbose'] = request.verbose  # Enable debug trace if requested

        # Collect result from async generator
        result = None
        async for item in orchestrator.answer_question(
            question=request.question,
            context=context,
            stream=False  # TODO: Implement streaming via WebSocket
        ):
            result = item  # Get the final result

        # Extract data from result if it has nested structure
        if result and result.get('type') == 'result' and 'data' in result:
            result = result['data']

        # Check if we got a valid result
        if not result or 'answer' not in result:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Agent failed to produce a valid answer. Result: {result}"
            )

        elapsed_ms = (time.time() - start_time) * 1000

        # Convert sources to SearchResult models
        sources = [
            _convert_to_search_result(src, src.get('score', 1.0))
            for src in result.get('sources', [])
        ]

        # Cache the result
        if agent_cache and agent_cache.config.enabled:
            cache_data = {
                'question_type': result.get('question_type', 'CODE'),
                'agent_used': result.get('agent_used', 'code_explorer'),
                'answer': result.get('answer', ''),
                'sources': result.get('sources', []),
                'iterations_used': result.get('iterations_used', 0),
                'tools_used': result.get('tools_used', []),
                'complete': result.get('complete', True),
            }
            # Save to exact query cache
            agent_cache.set_query_result(request.question, cache_data)
            # Also add to semantic cache for similar question matching
            agent_cache.add_to_semantic_cache(request.question)

        return AskResponse(
            question=request.question,
            question_type=result.get('question_type', 'CODE'),
            agent_used=result.get('agent_used', 'code_explorer'),
            answer=result['answer'],
            sources=sources,
            iterations_used=result.get('iterations_used', 0),
            tools_used=result.get('tools_used', []),
            complete=result.get('complete', True),
            took_ms=elapsed_ms,
            cached=False,
            debug=result.get('debug')  # Include debug trace if verbose mode was enabled
        )

    except HTTPException:
        # Re-raise HTTP exceptions
        raise

    except Exception as e:
        logger.error(f"Agent failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Agent exploration failed: {str(e)}"
        )
