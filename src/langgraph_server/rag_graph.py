"""
RAG Agent Graph for LangGraph Server.

Run with: langgraph dev
"""

import os
import sys
from typing import TypedDict, List, Dict, Any
from pathlib import Path

# LangGraph uses CUDA for embedding model
# FastAPI should NOT load embedding model (to avoid GPU conflicts)

from langgraph.graph import StateGraph, END
from openai import OpenAI

# Add project root to path for imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import logger after path is set
from src.logger import get_logger, setup_logging

# Setup logging to ensure logs go to pipeline.log
setup_logging(level="INFO", enable_console=True)

logger = get_logger(__name__)

# Load environment variables from project root .env file
# This ensures LANGSMITH_API_KEY and other vars are available for LangGraph
# This MUST happen before LangGraph checks for environment variables
# We load from both root .env (for all vars) and local .env (for langgraph dev)
try:
    from dotenv import load_dotenv
    
    # First, load from root .env (has all environment variables)
    root_env_path = Path(PROJECT_ROOT) / ".env"
    if root_env_path.exists():
        load_dotenv(root_env_path, override=True)
    
    # Then, load from local .env if it exists (langgraph.json points to this)
    # This ensures langgraph dev can find LANGSMITH_API_KEY
    local_env_path = Path(__file__).parent / ".env"
    if local_env_path.exists():
        load_dotenv(local_env_path, override=False)  # Don't override, root .env takes precedence
    
    # Explicitly ensure LANGSMITH_API_KEY is in os.environ
    langsmith_key = os.getenv("LANGSMITH_API_KEY")
    if langsmith_key:
        os.environ["LANGSMITH_API_KEY"] = langsmith_key
        # Also write to local .env if it doesn't exist or doesn't have the key
        if not local_env_path.exists() or not langsmith_key:
            try:
                with open(local_env_path, 'w', encoding='utf-8') as f:
                    f.write(f"LANGSMITH_API_KEY={langsmith_key}\n")
            except Exception:
                pass  # Ignore if we can't write
except ImportError:
    # dotenv not available, skip
    pass
except Exception as e:
    # Log but don't fail if .env loading fails
    import sys
    print(f"Warning: Could not load .env file: {e}", file=sys.stderr)


# ============== STATE ==============

class RAGState(TypedDict):
    """State for the agentic RAG pipeline."""
    query: str
    current_query: str
    context: List[Dict[str, Any]]
    quality_score: float
    quality_feedback: str
    answer: str
    sources: List[Dict[str, Any]]
    iterations: int
    max_iterations: int


# ============== LLM ==============

# Model configuration for different tasks
# Using standard large models (not experimental) from OpenRouter
# See docs/MODEL_RECOMMENDATIONS_OPENROUTER.md for alternatives
MODELS = {
    # Fast small model for quality scoring (simple 0-1 output)
    # Standard model: Meta Llama 3.3 8B (fast, reliable)
    "quality": os.getenv("RAG_QUALITY_MODEL", "meta-llama/llama-3.3-8b-instruct:free"),
    
    # Small model for query rewriting (short output)
    # Standard model: Meta Llama 3.3 8B (fast, reliable)
    "rewrite": os.getenv("RAG_REWRITE_MODEL", "meta-llama/llama-3.3-8b-instruct:free"),
    
    # Large model for answer generation (quality matters)
    # Standard large models (recommended in order):
    # 1. meta-llama/llama-3.1-405b-instruct:free (FREE, MAXIMUM QUALITY, 405B parameters)
    # 2. meta-llama/llama-3.3-70b-instruct:free (FREE, faster, good quality)
    # 3. meta-llama/llama-3.3-70b-instruct (paid, faster, no limits)
    # 4. meta-llama/llama-3.1-70b-instruct (proven, stable)
    # 5. qwen/qwen-2.5-72b-instruct (good alternative)
    # Current default: Llama 3.1 405B FREE (maximum quality, standard, not experimental)
    "answer": os.getenv("RAG_ANSWER_MODEL", "meta-llama/llama-3.1-405b-instruct:free"),
}

# Fallback models if primary fails (standard models only)
FALLBACK_MODELS = {
    # Fallback to 70B model if 405B is unavailable (smaller but still powerful)
    "answer": "meta-llama/llama-3.3-70b-instruct:free",
    "quality": "meta-llama/llama-3.3-8b-instruct:free",
    "rewrite": "meta-llama/llama-3.3-8b-instruct:free",
}


def get_llm():
    """Get OpenAI client for OpenRouter."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        logger.warning("[WARNING] OPENROUTER_API_KEY not found! LLM calls will fail!")
        logger.debug(f"[DEBUG] Current env vars with 'OPENROUTER': {[k for k in os.environ.keys() if 'OPENROUTER' in k]}")
        print(f"[WARNING] OPENROUTER_API_KEY not found! LLM calls will fail!", flush=True)
        print(f"[DEBUG] Current env vars with 'OPENROUTER': {[k for k in os.environ.keys() if 'OPENROUTER' in k]}", flush=True)
    else:
        logger.info(f"[INFO] OPENROUTER_API_KEY found (length: {len(api_key)})")
        print(f"[INFO] OPENROUTER_API_KEY found (length: {len(api_key)})", flush=True)
    return OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
    )


def call_llm(prompt: str, task: str = "answer", max_tokens: int = 2048) -> str:
    """
    Call LLM with task-specific model.
    
    Tries primary model first, falls back to fallback model if primary fails.

    Args:
        prompt: The prompt to send
        task: Task type - "quality", "rewrite", or "answer"
        max_tokens: Max tokens for response
    """
    model = MODELS.get(task, MODELS["answer"])
    fallback_model = FALLBACK_MODELS.get(task, FALLBACK_MODELS["answer"])
    client = get_llm()

    logger.info(f"[LLM] task={task}, model={model.split('/')[-1]}")
    logger.info(f"[LLM] Calling OpenRouter API...")
    print(f"[LLM] task={task}, model={model.split('/')[-1]}", flush=True)
    print(f"[LLM] Calling OpenRouter API...", flush=True)

    # Try primary model first
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=max_tokens,
        )
        result = response.choices[0].message.content.strip()
        logger.info(f"[LLM] Success! Response length: {len(result)}")
        print(f"[LLM] Success! Response length: {len(result)}", flush=True)
        return result
    except Exception as e:
        error_msg = str(e)
        logger.error(f"[LLM] Error with primary model: {error_msg}")
        print(f"[LLM] Error with primary model: {error_msg}", flush=True)
        # Check if it's a 404 (model not found) or other error
        if "404" in error_msg or "not found" in error_msg.lower() or "No endpoints" in error_msg:
            logger.warning(f"[LLM] Primary model {model} not available, trying fallback {fallback_model}")
            print(f"[LLM] Primary model {model} not available, trying fallback {fallback_model}", flush=True)
            try:
                # Try fallback model
                response = client.chat.completions.create(
                    model=fallback_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=max_tokens,
                )
                result = response.choices[0].message.content.strip()
                logger.info(f"[LLM] Fallback success! Response length: {len(result)}")
                print(f"[LLM] Fallback success! Response length: {len(result)}", flush=True)
                return result
            except Exception as e2:
                logger.error(f"[LLM] Fallback also failed: {str(e2)}")
                print(f"[LLM] Fallback also failed: {str(e2)}", flush=True)
                raise Exception(
                    f"Both primary model ({model}) and fallback model ({fallback_model}) failed. "
                    f"Primary error: {error_msg}. Fallback error: {str(e2)}"
                )
        else:
            # Re-raise if it's not a model availability issue
            raise


# ============== PROMPTS ==============

QUALITY_CHECK_PROMPT = """You are an expert at evaluating context quality for code questions.

User question: {query}

Retrieved context:
{context}

Evaluate the context quality:
1. Relevance: how related is the context to the question
2. Completeness: is there enough information to answer
3. Specificity: are there concrete code examples, functions, classes

Respond STRICTLY in format:
SCORE: <number from 0.0 to 1.0>
FEEDBACK: <brief explanation and what to search for if score is low>"""


QUERY_REWRITE_PROMPT = """You are an expert at code search.

Original query: {original_query}
Current query: {current_query}
Context quality feedback: {feedback}

Rewrite the query to find more relevant code.
Use technical terms, function names, class names.

Respond with ONLY the new query, no explanations."""


ANSWER_GENERATION_PROMPT = """You are a code expert. Answer the question based on context.

## Question
{query}

## Context from codebase

### Primary Results (Direct matches)
{primary_context}

### Related Code (Graph-expanded, connected entities)
{graph_context}

## Instructions
1. Answer based ONLY on the provided context
2. Include file paths and line numbers where possible
3. Format code in markdown blocks
4. If information is insufficient, say so honestly
5. Explain connections between primary and related code when relevant

## Response format

### Brief Answer
<2-3 sentences - the essence>

### Implementation Details
<details with code examples>

### Code Connections
<how the code elements are connected - calls, imports, inheritance>

### Related Files
<list of files>"""


# ============== STRATEGY DETECTION ==============
# Local enum to avoid importing from code_rag.retrieval (which triggers git import chain)

class SearchStrategy:
    """Search strategies (local copy to avoid blocking imports)."""
    UI_TO_DATABASE = "ui_to_database"
    DATABASE_TO_UI = "database_to_ui"
    IMPACT_ANALYSIS = "impact_analysis"
    PATTERN_SEARCH = "pattern_search"
    SEMANTIC_ONLY = "semantic_only"


def _detect_search_strategy(query: str) -> str:
    """
    Detect the best search strategy based on query content.

    Returns strategy string value.
    """
    query_lower = query.lower()

    # UI to Database: user asks about frontend/UI elements
    ui_keywords = [
        'ui', 'view', 'frontend', 'button', 'form', 'component',
        'template', 'html', 'css', 'react', 'vue', 'angular',
        'page', 'screen', 'widget', 'render', 'display', 'show'
    ]
    if any(kw in query_lower for kw in ui_keywords):
        # Check if asking about database connection from UI
        db_connection_keywords = ['database', 'model', 'data', 'api', 'backend', 'server']
        if any(kw in query_lower for kw in db_connection_keywords):
            return SearchStrategy.UI_TO_DATABASE

    # Database to UI: user asks about data models/database
    db_keywords = [
        'database', 'model', 'table', 'schema', 'migration', 'orm',
        'django model', 'sqlalchemy', 'entity', 'repository', 'dao'
    ]
    if any(kw in query_lower for kw in db_keywords):
        # Check if asking about UI usage
        ui_usage_keywords = ['used', 'displayed', 'shown', 'view', 'frontend', 'ui']
        if any(kw in query_lower for kw in ui_usage_keywords):
            return SearchStrategy.DATABASE_TO_UI

    # Impact Analysis: user asks about changes, dependencies
    impact_keywords = [
        'impact', 'affect', 'change', 'depend', 'break', 'modify',
        'refactor', 'update', 'remove', 'delete', 'what happens',
        'who calls', 'who uses', 'where used', 'references'
    ]
    if any(kw in query_lower for kw in impact_keywords):
        return SearchStrategy.IMPACT_ANALYSIS

    # Pattern Search: user looking for similar code
    pattern_keywords = [
        'pattern', 'similar', 'like', 'example', 'same as',
        'how to', 'implement', 'usage', 'best practice'
    ]
    if any(kw in query_lower for kw in pattern_keywords):
        return SearchStrategy.PATTERN_SEARCH

    # Default: PATTERN_SEARCH
    return SearchStrategy.PATTERN_SEARCH


# ============== SINGLETON CLIENTS ==============
# Cache clients to avoid reinitializing on every call
_neo4j_client = None
_async_weaviate_client = None


async def _get_neo4j():
    """Get or create Neo4j client (singleton pattern)."""
    global _neo4j_client

    if _neo4j_client is not None:
        return _neo4j_client

    # Direct import to avoid loading entire graph module (which imports git → os.getcwd blocking)
    from src.code_rag.graph.neo4j_client import Neo4jClient

    logger.info("[INIT] Creating Neo4j client...")
    print("[INIT] Creating Neo4j client...", flush=True)
    _neo4j_client = Neo4jClient(
        uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        user=os.getenv("NEO4J_USER", "neo4j"),
        password=os.getenv("NEO4J_PASSWORD", ""),
    )
    logger.info("[INIT] Neo4j client created")
    print("[INIT] Neo4j client created", flush=True)

    return _neo4j_client


async def _get_async_weaviate():
    """Get or create async Weaviate client (singleton pattern)."""
    global _async_weaviate_client

    if _async_weaviate_client is not None:
        return _async_weaviate_client

    try:
        # Direct import to avoid loading entire graph module (which imports git → os.getcwd blocking)
        logger.info("[INIT] Importing AsyncWeaviateIndexer...")
        print("[INIT] Importing AsyncWeaviateIndexer...", flush=True)
        from src.code_rag.graph.weaviate_indexer import AsyncWeaviateIndexer
        logger.info("[INIT] AsyncWeaviateIndexer imported successfully")
        print("[INIT] AsyncWeaviateIndexer imported successfully", flush=True)

        logger.info("[INIT] Creating async Weaviate client...")
        print("[INIT] Creating async Weaviate client...", flush=True)
        
        weaviate_url = os.getenv("WEAVIATE_URL", "http://localhost:8080")
        logger.info(f"[INIT] Weaviate URL: {weaviate_url}")
        print(f"[INIT] Weaviate URL: {weaviate_url}", flush=True)

        _async_weaviate_client = AsyncWeaviateIndexer(
            weaviate_url=weaviate_url,
        )
        logger.info("[INIT] AsyncWeaviateIndexer instance created, connecting...")
        print("[INIT] AsyncWeaviateIndexer instance created, connecting...", flush=True)
        
        await _async_weaviate_client.connect()
        logger.info("[INIT] Async Weaviate client connected")
        print("[INIT] Async Weaviate client connected", flush=True)

        logger.info("[INIT] Async Weaviate client created successfully")
        print("[INIT] Async Weaviate client created successfully", flush=True)

        return _async_weaviate_client
    except Exception as e:
        import sys
        import traceback
        error_msg = f"[INIT] Failed to create async Weaviate client: {e}"
        logger.error(error_msg)
        print(error_msg, file=sys.stderr, flush=True)
        traceback.print_exc(file=sys.stderr)
        raise


# ============== NODES ==============

async def context_collector(state: RAGState) -> Dict[str, Any]:
    """
    Collect context from vector database + graph expansion.

    Uses:
    1. AsyncWeaviateIndexer for semantic search (async, no event loop conflicts)
    2. Neo4j for graph expansion based on search strategy
    """
    query = state["current_query"]

    logger.info(f"[CONTEXT] Starting context collection for: {query[:50]}...")
    print(f"[CONTEXT] Starting context collection for: {query[:50]}...", flush=True)

    context = []
    primary_results = []

    try:
        # Get async Weaviate client
        logger.info("[CONTEXT] Getting async Weaviate client...")
        print("[CONTEXT] Getting async Weaviate client...", flush=True)
        weaviate_client = await _get_async_weaviate()
        logger.info("[CONTEXT] Async Weaviate client obtained")
        print("[CONTEXT] Async Weaviate client obtained", flush=True)

        # Determine search strategy based on query
        strategy = _detect_search_strategy(query)
        logger.info(f"[CONTEXT] Using strategy: {strategy}")
        print(f"[CONTEXT] Using strategy: {strategy}", flush=True)

        # Async vector search in Weaviate
        logger.info("[CONTEXT] Searching in Weaviate...")
        print("[CONTEXT] Searching in Weaviate...", flush=True)
        try:
            primary_results = await weaviate_client.search(
                query=query,
                limit=15,
                alpha=0.5  # Hybrid search balance
            )
            logger.info(f"[CONTEXT] Found {len(primary_results)} primary results")
            print(f"[CONTEXT] Found {len(primary_results)} primary results", flush=True)
        except Exception as search_error:
            error_msg = str(search_error)
            logger.error(f"[CONTEXT] Search failed: {search_error}")
            print(f"[CONTEXT] Search failed: {search_error}", flush=True)
            if "closed" in error_msg.lower() or "not connected" in error_msg.lower():
                logger.warning("[CONTEXT] Client closed, attempting reconnect and retry...")
                print("[CONTEXT] Client closed, attempting reconnect and retry...", flush=True)
                try:
                    await weaviate_client.connect()
                    primary_results = await weaviate_client.search(query=query, limit=15, alpha=0.5)
                    logger.info(f"[CONTEXT] Retry successful, found {len(primary_results)} primary results")
                    print(f"[CONTEXT] Retry successful, found {len(primary_results)} primary results", flush=True)
                except Exception as retry_error:
                    logger.error(f"[CONTEXT] Retry also failed: {retry_error}")
                    print(f"[CONTEXT] Retry also failed: {retry_error}", flush=True)
                    primary_results = []
            else:
                primary_results = []

        # Add primary nodes (direct search results)
        for node in primary_results[:10]:
            context.append({
                "id": node.get("node_id", ""),
                "name": node.get("name", "Unknown"),
                "type": node.get("node_type", "Unknown"),
                "file": node.get("file_path", ""),
                "code": node.get("content", "")[:1000],
                "score": node.get("score", 0.0),
                "source": "primary",
            })

        # ============== NEO4J GRAPH EXPANSION ==============
        # Expand context using graph relationships based on strategy
        if primary_results:
            try:
                logger.info("[CONTEXT] Starting Neo4j graph expansion...")
                print("[CONTEXT] Starting Neo4j graph expansion...", flush=True)
                neo4j_client = await _get_neo4j()

                # Get node IDs from primary results
                node_ids = [r.get("node_id") for r in primary_results[:10] if r.get("node_id")]

                graph_results = []

                if strategy == SearchStrategy.UI_TO_DATABASE:
                    # Trace path from UI to database
                    logger.info("[CONTEXT] Tracing UI → Database path...")
                    print("[CONTEXT] Tracing UI → Database path...", flush=True)
                    for node_id in node_ids[:5]:  # Limit to top 5 to avoid explosion
                        path_nodes = neo4j_client.trace_ui_to_database(node_id, limit=10)
                        graph_results.extend(path_nodes)

                elif strategy == SearchStrategy.DATABASE_TO_UI:
                    # Trace path from database to UI
                    logger.info("[CONTEXT] Tracing Database → UI path...")
                    print("[CONTEXT] Tracing Database → UI path...", flush=True)
                    for node_id in node_ids[:5]:
                        path_nodes = neo4j_client.trace_database_to_ui(node_id, limit=10)
                        graph_results.extend(path_nodes)

                elif strategy == SearchStrategy.IMPACT_ANALYSIS:
                    # Get impact analysis
                    logger.info("[CONTEXT] Running impact analysis...")
                    print("[CONTEXT] Running impact analysis...", flush=True)
                    for node_id in node_ids[:3]:  # Limit more for impact analysis
                        impact = neo4j_client.get_impact_analysis(node_id, depth=2)
                        graph_results.extend(impact.get("callers", []))
                        graph_results.extend(impact.get("importers", []))
                        graph_results.extend(impact.get("inheritors", []))
                        graph_results.extend(impact.get("model_users", []))

                else:
                    # PATTERN_SEARCH / SEMANTIC_ONLY - simple 1-hop expansion
                    logger.info("[CONTEXT] Running 1-hop graph expansion...")
                    print("[CONTEXT] Running 1-hop graph expansion...", flush=True)
                    graph_results = neo4j_client.get_related_nodes(
                        node_ids=node_ids,
                        depth=1,
                        direction="both",
                        limit=20
                    )

                # Add graph-expanded nodes to context (deduplicate by ID)
                seen_ids = {c["id"] for c in context}
                for node in graph_results:
                    node_id = node.get("id", "")
                    if node_id and node_id not in seen_ids:
                        seen_ids.add(node_id)
                        # Get relationship info
                        rel_type = (
                            node.get("_relationship") or
                            node.get("_relationship_types", ["RELATED"])[0] if node.get("_relationship_types") else
                            node.get("_impact_type", "RELATED")
                        )
                        context.append({
                            "id": node_id,
                            "name": node.get("name", "Unknown"),
                            "type": node.get("type", "Unknown"),
                            "file": node.get("file_path", ""),
                            "code": node.get("docstring", "") or node.get("signature", "") or "",
                            "score": 0.0,  # Graph nodes don't have semantic scores
                            "source": "graph",
                            "relationship": rel_type,
                        })

                logger.info(f"[CONTEXT] Added {len(context) - len(primary_results[:10])} graph nodes")
                print(f"[CONTEXT] Added {len(context) - len(primary_results[:10])} graph nodes", flush=True)

            except Exception as neo4j_error:
                logger.warning(f"[CONTEXT] Neo4j expansion failed (continuing with Weaviate only): {neo4j_error}")
                print(f"[CONTEXT] Neo4j expansion failed: {neo4j_error}", flush=True)

    except Exception as e:
        import sys
        import traceback
        error_msg = f"Context collection error: {e}"
        logger.error(error_msg)
        print(error_msg, file=sys.stderr, flush=True)
        traceback.print_exc(file=sys.stderr)
        context = []

    logger.info(f"[CONTEXT] Total context: {len(context)} nodes")
    print(f"[CONTEXT] Total context: {len(context)} nodes", flush=True)

    return {
        "context": context,
        "iterations": state["iterations"] + 1,
    }


def quality_checker(state: RAGState) -> Dict[str, Any]:
    """Check quality of retrieved context."""
    if not state["context"]:
        return {"quality_score": 0.0, "quality_feedback": "No context found"}

    # Count sources for logging
    primary_count = sum(1 for c in state["context"] if c.get("source") == "primary")
    graph_count = sum(1 for c in state["context"] if c.get("source") == "graph")
    logger.info(f"[QUALITY] Checking {primary_count} primary + {graph_count} graph nodes")
    print(f"[QUALITY] Checking {primary_count} primary + {graph_count} graph nodes", flush=True)

    # Format context with source info
    context_str = "\n\n".join([
        f"[{c['type']}] {c['name']} ({c['file']}) - {c.get('source', 'unknown')} match\n```\n{c['code'][:500]}\n```"
        for c in state["context"][:8]  # Check more nodes (primary + graph)
    ])

    prompt = QUALITY_CHECK_PROMPT.format(query=state["query"], context=context_str)

    try:
        response = call_llm(prompt, task="quality", max_tokens=512)

        score = 0.5
        feedback = "Could not parse quality"

        for line in response.split("\n"):
            line = line.strip()
            if line.startswith("SCORE:"):
                try:
                    score = float(line.replace("SCORE:", "").strip())
                    score = max(0.0, min(1.0, score))
                except ValueError:
                    pass
            elif line.startswith("FEEDBACK:"):
                feedback = line.replace("FEEDBACK:", "").strip()

    except Exception as e:
        score = 0.5
        feedback = str(e)

    return {"quality_score": score, "quality_feedback": feedback}


def query_rewriter(state: RAGState) -> Dict[str, Any]:
    """Rewrite query to improve retrieval."""
    prompt = QUERY_REWRITE_PROMPT.format(
        original_query=state["query"],
        current_query=state["current_query"],
        feedback=state["quality_feedback"],
    )

    try:
        new_query = call_llm(prompt, task="rewrite", max_tokens=256)
        new_query = new_query.strip().strip('"\'')
    except Exception as e:
        new_query = state["current_query"]

    return {"current_query": new_query}


def answer_generator(state: RAGState) -> Dict[str, Any]:
    """Generate final answer from context."""
    # Separate primary and graph-expanded context
    primary_nodes = [c for c in state["context"] if c.get("source") == "primary"]
    graph_nodes = [c for c in state["context"] if c.get("source") == "graph"]

    # Format primary context
    primary_context = "\n\n".join([
        f"### {c['name']} ({c['type']})\n**File:** `{c['file']}`\n```\n{c['code']}\n```"
        for c in primary_nodes
    ]) or "No direct matches found."

    # Format graph-expanded context with relationship info
    graph_context = "\n\n".join([
        f"### {c['name']} ({c['type']}) - {c.get('relationship', 'RELATED')}\n**File:** `{c['file']}`\n```\n{c['code']}\n```"
        for c in graph_nodes
    ]) or "No related code found via graph traversal."

    prompt = ANSWER_GENERATION_PROMPT.format(
        query=state["query"],
        primary_context=primary_context,
        graph_context=graph_context
    )

    try:
        answer = call_llm(prompt, task="answer", max_tokens=4096)
    except Exception as e:
        answer = f"Error generating answer: {e}"

    sources = [{"name": c["name"], "file": c["file"], "type": c["type"]} for c in state["context"]]

    return {"answer": answer, "sources": sources}


def should_rewrite(state: RAGState) -> str:
    """Decide: rewrite query or generate answer."""
    threshold = float(os.getenv("QUALITY_THRESHOLD", "0.6"))
    quality_score = state.get("quality_score", 0.0)
    iterations = state.get("iterations", 0)
    max_iterations = state.get("max_iterations", 3)
    
    logger.info(f"[ROUTING] should_rewrite: quality_score={quality_score}, threshold={threshold}, iterations={iterations}/{max_iterations}")
    print(f"[ROUTING] should_rewrite: quality_score={quality_score}, threshold={threshold}, iterations={iterations}/{max_iterations}", flush=True)
    
    if quality_score >= threshold:
        logger.info("[ROUTING] Decision: generate (quality good enough)")
        print("[ROUTING] Decision: generate (quality good enough)", flush=True)
        return "generate"
    if iterations >= max_iterations:
        logger.info(f"[ROUTING] Decision: generate (max iterations reached: {iterations})")
        print(f"[ROUTING] Decision: generate (max iterations reached: {iterations})", flush=True)
        return "generate"
    logger.info("[ROUTING] Decision: rewrite (quality too low)")
    print("[ROUTING] Decision: rewrite (quality too low)", flush=True)
    return "rewrite"


# ============== BUILD GRAPH ==============

def create_graph():
    """Build the RAG agent graph."""
    logger.info("[GRAPH] Creating RAG graph...")
    print("[GRAPH] Creating RAG graph...", flush=True)
    builder = StateGraph(RAGState)

    logger.info("[GRAPH] Adding nodes...")
    print("[GRAPH] Adding nodes...", flush=True)
    builder.add_node("context_collector", context_collector)
    builder.add_node("quality_checker", quality_checker)
    builder.add_node("query_rewriter", query_rewriter)
    builder.add_node("answer_generator", answer_generator)

    builder.set_entry_point("context_collector")
    builder.add_edge("context_collector", "quality_checker")

    builder.add_conditional_edges(
        "quality_checker",
        should_rewrite,
        {"generate": "answer_generator", "rewrite": "query_rewriter"}
    )

    builder.add_edge("query_rewriter", "context_collector")
    builder.add_edge("answer_generator", END)

    logger.info("[GRAPH] Compiling graph...")
    print("[GRAPH] Compiling graph...", flush=True)
    compiled = builder.compile()
    logger.info("[GRAPH] Graph compiled successfully!")
    print("[GRAPH] Graph compiled successfully!", flush=True)
    return compiled


# Export for LangGraph Server
logger.info("[GRAPH] Initializing graph for LangGraph Server...")
print("[GRAPH] Initializing graph for LangGraph Server...", flush=True)
graph = create_graph()
logger.info("[GRAPH] Graph initialized and ready for LangGraph Server!")
print("[GRAPH] Graph initialized and ready for LangGraph Server!", flush=True)


# ============== CLI ==============

def run_rag(query: str, max_iterations: int = 3) -> dict:
    """Run RAG pipeline."""
    initial_state: RAGState = {
        "query": query,
        "current_query": query,
        "context": [],
        "quality_score": 0.0,
        "quality_feedback": "",
        "answer": "",
        "sources": [],
        "iterations": 0,
        "max_iterations": max_iterations,
    }

    final_state = graph.invoke(initial_state)

    return {
        "answer": final_state["answer"],
        "sources": final_state["sources"],
        "iterations": final_state["iterations"],
        "quality_score": final_state["quality_score"],
    }


if __name__ == "__main__":
    import sys
    query = sys.argv[1] if len(sys.argv) > 1 else "How does authentication work?"
    result = run_rag(query)
    print(f"\n{'='*60}")
    print(f"Query: {query}")
    print(f"Iterations: {result['iterations']}, Quality: {result['quality_score']:.2f}")
    print(f"{'='*60}\n")
    print(result["answer"])
