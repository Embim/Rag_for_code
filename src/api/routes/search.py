"""
Search and Q&A endpoints.

Provides:
- POST /search - Quick code search
- POST /ask - Agent-powered deep exploration
"""

import os
import time
import asyncio
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
from ..langfuse_tracing import get_langfuse, create_rag_trace, log_retrieval, log_generation, log_quality_score
from ...code_rag.retrieval import CodeRetriever, SearchStrategy, SearchConfig
from ...code_rag.retrieval.scope_detector import ScopeDetector, QueryScope
from ...agents import QueryOrchestrator, AgentCache
from ...logger import get_logger


logger = get_logger(__name__)


# ============================================================================
# LangGraph RAG (lazy init)
# ============================================================================
_langgraph_rag = None


def get_langgraph_rag(retriever: CodeRetriever):
    """Get or create LangGraph RAG instance."""
    global _langgraph_rag

    if _langgraph_rag is not None:
        return _langgraph_rag

    try:
        from ...langgraph_server.rag_graph import create_graph, RAGState
        _langgraph_rag = create_graph()
        logger.info("LangGraph RAG initialized")
        return _langgraph_rag
    except Exception as e:
        logger.error(f"Failed to init LangGraph RAG: {e}")
        return None

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


@router.post(
    "/ask/langgraph",
    response_model=AskResponse,
    summary="LangGraph Agent Q&A",
    description="Use LangGraph-based agent with quality feedback loop and Langfuse tracing"
)
async def ask_langgraph(
    request: AskRequest,
    retriever: CodeRetriever = Depends(get_retriever),
    current_key: APIKey = Depends(require_auth),
):
    """
    LangGraph-powered Q&A with Langfuse tracing.

    Flow:
    1. Collect context from vector DB
    2. Check context quality with LLM
    3. If quality < 0.6: rewrite query and retry
    4. Generate answer from context

    All steps are traced in Langfuse for observability.

    **Example:**
    ```json
    {
        "question": "How does authentication work?",
        "max_iterations": 3
    }
    ```
    """
    start_time = time.time()

    # Create Langfuse trace in a separate thread to avoid blocking event loop
    trace = await asyncio.to_thread(create_rag_trace, request.question)

    try:
        max_iterations = request.max_iterations or 3
        logger.info(
            f"Running LangGraph RAG for: {request.question[:50]}... "
            f"(max_iterations={max_iterations})"
        )

        # Try to use LangGraph dev server first (for Studio UI visibility)
        langgraph_url = os.getenv("LANGGRAPH_SERVER_URL", "http://127.0.0.1:2024")
        use_dev_server = os.getenv("LANGGRAPH_USE_DEV_SERVER", "true").lower() == "true"
        final_state = None
        
        if use_dev_server:
            try:
                from langgraph_sdk import get_client
                try:
                    import httpx
                except ImportError:
                    httpx = None
                
                # Check if dev server is available before attempting to connect
                async def check_dev_server_available(url: str, timeout: float = 2.0) -> bool:
                    """Check if LangGraph dev server is available."""
                    if not httpx:
                        return False
                    try:
                        async with httpx.AsyncClient(timeout=timeout) as client:
                            # Try /docs endpoint (always available in dev server)
                            response = await client.get(f"{url}/docs")
                            return response.status_code == 200
                    except Exception as e:
                        logger.debug(f"Dev server check failed: {e}")
                        return False
                
                logger.info(f"Checking LangGraph dev server availability at {langgraph_url}...")
                server_available = await check_dev_server_available(langgraph_url)
                if not server_available:
                    logger.warning(f"LangGraph dev server not available at {langgraph_url}, falling back to direct execution")
                    use_dev_server = False
                else:
                    logger.info(f"LangGraph dev server is available at {langgraph_url}")
                    # Try to connect to LangGraph dev server
                    client = get_client(url=langgraph_url)
                
                # Prepare input state for the graph
                graph_input = {
                    "query": request.question,
                    "current_query": request.question,
                    "context": [],
                    "quality_score": 0.0,
                    "quality_feedback": "",
                    "answer": "",
                    "sources": [],
                    "iterations": 0,
                    "max_iterations": max_iterations,
                }
                
                # Calculate recursion limit
                recursion_limit = max(50, max_iterations * 5 + 10)
                
                # Helper functions for handling run responses
                def get_status(run_obj):
                    if isinstance(run_obj, dict):
                        return run_obj.get("status")
                    return getattr(run_obj, "status", None)
                
                def get_run_id(run_obj):
                    if isinstance(run_obj, dict):
                        return run_obj.get("run_id") or run_obj.get("id")
                    return getattr(run_obj, "run_id", None) or getattr(run_obj, "id", None)
                
                logger.info(f"Sending request to LangGraph dev server at {langgraph_url}")
                
                # Create or get a thread for this run with timeout
                # Threads maintain state between runs
                logger.info("Creating thread...")
                try:
                    thread = await asyncio.wait_for(
                        client.threads.create(),
                        timeout=5.0
                    )
                    thread_id = thread["thread_id"] if isinstance(thread, dict) else thread.thread_id
                    logger.info(f"Thread created: {thread_id}")
                except asyncio.TimeoutError:
                    logger.warning("Thread creation timed out after 5 seconds")
                    raise  # Will trigger fallback
                except Exception as e:
                    logger.error(f"Thread creation failed: {e}")
                    raise  # Will trigger fallback
                
                # Create run through dev server (this will appear in Studio UI) with timeout
                # thread_id is a required parameter
                logger.info("Creating run...")
                try:
                    run = await asyncio.wait_for(
                        client.runs.create(
                            thread_id=thread_id,
                            assistant_id="rag",
                            input=graph_input,
                            config={
                                "recursion_limit": recursion_limit,
                            }
                        ),
                        timeout=10.0
                    )
                    run_id = get_run_id(run)
                    logger.info(f"Run created: {run_id}")
                except asyncio.TimeoutError:
                    logger.warning("Run creation timed out after 10 seconds")
                    raise  # Will trigger fallback
                except Exception as e:
                    logger.error(f"Run creation failed: {e}")
                    raise  # Will trigger fallback
                
                # Wait for completion - poll until done
                max_wait_time = 120  # 2 minutes max
                wait_start = time.time()
                poll_interval = 1.0  # Poll every 1 second instead of 0.5
                
                run_status = get_status(run)  # Renamed to avoid conflict with FastAPI status
                
                logger.info(f"Run status after creation: {run_status}")
                
                # Check if run is already completed
                if run_status in ["success", "error", "cancelled", "completed"]:
                    logger.info(f"Run completed immediately with status: {run_status}")
                else:
                    # Poll for completion with longer intervals and better error handling
                    logger.info(f"Polling run status (current: {run_status})...")
                    poll_count = 0
                    max_polls = int(max_wait_time / poll_interval)
                    
                    while run_status not in ["success", "error", "cancelled", "completed"]:
                        elapsed_time = time.time() - wait_start
                        if elapsed_time > max_wait_time:
                            raise TimeoutError("LangGraph run timed out")
                        
                        # Fallback to direct execution if run stuck in "running" status for >60s
                        if run_status == "running" and elapsed_time > 60:
                            logger.warning(
                                f"Run stuck in 'running' status for >60s ({elapsed_time:.1f}s), "
                                "falling back to direct execution"
                            )
                            use_dev_server = False
                            final_state = None  # Will trigger direct execution
                            break
                        
                        if poll_count >= max_polls:
                            logger.warning(f"Max polls reached ({max_polls}), breaking")
                            break
                        
                        await asyncio.sleep(poll_interval)
                        poll_count += 1
                        
                        if run_id and thread_id:
                            # Get run status with retry logic
                            status_updated = False
                            for attempt in range(3):  # Try up to 3 times
                                try:
                                    # Try different API formats
                                    try:
                                        run = await client.runs.get(thread_id=thread_id, run_id=run_id)
                                    except (TypeError, AttributeError):
                                        try:
                                            run = await client.runs.get(run_id=run_id)
                                        except (TypeError, AttributeError):
                                            run = await client.threads.runs.get(thread_id=thread_id, run_id=run_id)
                                    
                                    new_status = get_status(run)
                                    if new_status:
                                        if new_status != run_status:
                                            logger.info(f"Run status updated: {run_status} -> {new_status}")
                                        run_status = new_status
                                        status_updated = True
                                        break
                                except (httpx.ReadError, httpx.ConnectError, httpx.TimeoutException) as e:
                                    if attempt < 2:  # Retry on connection errors
                                        logger.debug(f"Connection error on attempt {attempt + 1}, retrying...")
                                        await asyncio.sleep(0.5 * (attempt + 1))
                                        continue
                                    else:
                                        logger.warning(f"Connection error after {attempt + 1} attempts: {e}")
                                        # Continue with last known status
                                        break
                                except Exception as e:
                                    import traceback
                                    error_details = traceback.format_exc()
                                    logger.error(
                                        f"Error getting run status: {e}, "
                                        f"type: {type(e).__name__}, "
                                        f"traceback: {error_details}"
                                    )
                                    # Continue with last known status
                                    break
                            
                            if not status_updated and poll_count % 10 == 0:
                                logger.info(f"Still waiting for run completion (status: {run_status}, poll: {poll_count})")
                        else:
                            logger.warning("No run_id or thread_id available, breaking")
                            break
                    
                    logger.info(f"Run polling completed. Final status: {run_status}")
                
                if run_status == "error":
                    error_msg = None
                    if isinstance(run, dict):
                        error_msg = run.get("error") or run.get("error_message")
                    else:
                        error_msg = getattr(run, 'error', None) or getattr(run, 'error_message', None)
                    raise Exception(f"LangGraph run failed: {error_msg or 'Unknown error'}")
                
                # Extract result from completed run
                # CRITICAL: Extract ONLY from run.output, not from run metadata
                logger.info("Extracting result from completed run...")
                
                if isinstance(run, dict):
                    # Check that run is completed
                    if run.get("status") not in ["success", "completed"]:
                        raise Exception(f"Run not completed, status: {run.get('status')}")
                    
                    # Extract ONLY from output
                    final_state = run.get("output")
                    if not final_state:
                        # If output is empty, this is an error
                        logger.error(f"Run completed but output is empty. Run keys: {list(run.keys())}")
                        raise Exception("Run completed but output is empty")
                else:
                    # Check status
                    run_status_check = getattr(run, 'status', None)
                    if run_status_check not in ["success", "completed"]:
                        raise Exception(f"Run not completed, status: {run_status_check}")
                    
                    # Extract ONLY from output
                    final_state = getattr(run, 'output', None)
                    if not final_state:
                        logger.error(f"Run completed but output is empty. Run attributes: {dir(run)}")
                        raise Exception("Run completed but output is empty")
                
                # Ensure final_state is a dict with graph result
                if not isinstance(final_state, dict):
                    logger.error(f"final_state is not a dict: {type(final_state)}")
                    raise Exception(f"Invalid output format: {type(final_state)}")
                
                logger.info(f"Result extracted. Keys: {list(final_state.keys())}")
                
                # Ensure final_state is a dict and has all required fields
                if not isinstance(final_state, dict):
                    # If it's not a dict, try to convert or use empty dict
                    if hasattr(final_state, '__dict__'):
                        final_state = final_state.__dict__
                    else:
                        logger.warning(f"final_state is not a dict: {type(final_state)}, using empty dict")
                        final_state = {}
                
                # Ensure required fields exist with defaults
                if "sources" not in final_state:
                    final_state["sources"] = []
                if "answer" not in final_state:
                    final_state["answer"] = ""
                if "iterations" not in final_state:
                    final_state["iterations"] = 0
                if "quality_score" not in final_state:
                    final_state["quality_score"] = 0.0
                if "quality_feedback" not in final_state:
                    final_state["quality_feedback"] = ""
                if "current_query" not in final_state:
                    final_state["current_query"] = request.question
                
                logger.info(f"Request completed via LangGraph dev server. State keys: {list(final_state.keys())}")
                
            except ImportError:
                logger.warning("langgraph-sdk not installed, falling back to direct graph execution")
                use_dev_server = False
            except Exception as e:
                # Check if it's a connection error
                error_str = str(e).lower()
                if httpx and ("connection" in error_str or "timeout" in error_str or "refused" in error_str):
                    pass  # Will fall through to direct execution
                else:
                    raise
                logger.warning(f"LangGraph dev server unavailable at {langgraph_url}: {e}")
                logger.info("Falling back to direct graph execution")
                use_dev_server = False
        
        # Fallback to direct graph execution if dev server is unavailable or disabled
        if not use_dev_server or final_state is None:
            # Get LangGraph RAG for direct execution
            graph = get_langgraph_rag(retriever)

            if not graph:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="LangGraph RAG not available. Check logs."
                )

            # Prepare initial state
            from ...langgraph_server.rag_graph import RAGState

            initial_state: RAGState = {
                "query": request.question,
                "current_query": request.question,
                "context": [],
                "quality_score": 0.0,
                "quality_feedback": "",
                "answer": "",
                "sources": [],
                "iterations": 0,
                "max_iterations": max_iterations,
            }

            # Run graph with increased recursion limit
            recursion_limit = max(50, max_iterations * 5 + 10)
            
            logger.debug(f"Setting recursion_limit to {recursion_limit} for max_iterations={max_iterations}")
            
            # Configure LangSmith tracing for this run
            config = {
                "recursion_limit": recursion_limit,
                "configurable": {
                    "thread_id": f"rag-{int(time.time() * 1000)}",
                }
            }
            
            final_state = graph.invoke(initial_state, config=config)

        elapsed_ms = (time.time() - start_time) * 1000

        # Log to Langfuse
        if trace:
            log_quality_score(trace, final_state["quality_score"], final_state["quality_feedback"])
            trace.update(
                output={
                    "answer_length": len(final_state["answer"]),
                    "iterations": final_state["iterations"],
                    "quality_score": final_state["quality_score"],
                },
                metadata={"duration_ms": elapsed_ms},
            )

        # Convert sources
        sources = [
            _convert_to_search_result(src, src.get("score", 1.0))
            for src in final_state["sources"]
        ]

        logger.info(
            f"LangGraph RAG completed: {final_state['iterations']} iterations, "
            f"quality={final_state['quality_score']:.2f}, {elapsed_ms:.0f}ms"
        )

        return AskResponse(
            question=request.question,
            question_type="CODE",
            agent_used="langgraph_rag",
            answer=final_state["answer"],
            sources=sources,
            iterations_used=final_state["iterations"],
            tools_used=["context_collector", "quality_checker", "answer_generator"],
            complete=True,
            took_ms=elapsed_ms,
            cached=False,
            debug={
                "quality_score": final_state["quality_score"],
                "quality_feedback": final_state["quality_feedback"],
                "final_query": final_state["current_query"],
            } if request.verbose else None,
        )

    except HTTPException:
        raise

    except Exception as e:
        logger.error(f"LangGraph RAG failed: {e}", exc_info=True)

        if trace:
            trace.update(output={"error": str(e)}, level="ERROR")

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"LangGraph RAG failed: {str(e)}"
        )
