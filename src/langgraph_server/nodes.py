"""
RAG Pipeline Nodes.

Each node is a function that takes state and returns updated state.
Integrated with Langfuse for tracing.
"""

import os
import sys
import time
from typing import Dict, Any
from pathlib import Path

# LangGraph uses CUDA for embedding model

from langfuse import Langfuse
from langfuse.callback import CallbackHandler
from openai import OpenAI

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.logger import get_logger, setup_logging

# Setup logging to ensure logs go to pipeline.log
setup_logging(level="INFO", enable_console=True)

from .state import RAGState

logger = get_logger(__name__)


def log_both(level: str, message: str):
    """Log to both logger and stdout for dev server visibility."""
    if level == "info":
        logger.info(message)
        print(f"[INFO] {message}", flush=True)
    elif level == "warning":
        logger.warning(message)
        print(f"[WARNING] {message}", flush=True)
    elif level == "error":
        logger.error(message)
        print(f"[ERROR] {message}", flush=True)
    elif level == "debug":
        logger.debug(message)
        print(f"[DEBUG] {message}", flush=True)


# ============== LANGFUSE SETUP ==============

langfuse = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
)


def get_langfuse_handler(session_id: str = None, user_id: str = None) -> CallbackHandler:
    """Create Langfuse callback handler for tracing."""
    return CallbackHandler(
        public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
        secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
        host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
        session_id=session_id,
        user_id=user_id,
    )


# ============== LLM CLIENT ==============

def get_llm_client() -> OpenAI:
    """Get OpenAI client configured for OpenRouter."""
    return OpenAI(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
    )


def call_llm(prompt: str, trace_name: str, max_tokens: int = 2048) -> str:
    """Call LLM with Langfuse tracing."""
    # Standard large model (not experimental)
    # Using 405B for maximum quality (FREE version available!)
    model = os.getenv("CODE_EXPLORER_MODEL", "meta-llama/llama-3.1-405b-instruct:free")
    client = get_llm_client()

    # Create Langfuse trace
    trace = langfuse.trace(name=trace_name)
    generation = trace.generation(
        name="llm_call",
        model=model,
        input=prompt,
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=max_tokens,
        )
        result = response.choices[0].message.content.strip()

        generation.end(output=result)
        return result

    except Exception as e:
        generation.end(output=str(e), level="ERROR")
        raise


# ============== PROMPTS ==============

QUALITY_CHECK_PROMPT = """Ты - эксперт по оценке качества контекста для ответа на вопросы о коде.

Вопрос пользователя: {query}

Найденный контекст:
{context}

Оцени качество найденного контекста:
1. Релевантность: насколько контекст относится к вопросу
2. Полнота: достаточно ли информации для полного ответа
3. Специфичность: есть ли конкретный код, функции, классы

Ответь СТРОГО в формате:
SCORE: <число от 0.0 до 1.0>
FEEDBACK: <что нужно найти дополнительно, или почему контекст хороший>

Примеры:
SCORE: 0.9
FEEDBACK: Контекст содержит все нужные функции и классы для ответа

SCORE: 0.3
FEEDBACK: Найдены только общие файлы, нужно искать конкретные функции аутентификации"""


QUERY_REWRITE_PROMPT = """Ты - эксперт по поиску в кодовой базе.

Оригинальный запрос: {original_query}
Текущий запрос: {current_query}
Проблема с контекстом: {feedback}

Перепиши запрос чтобы найти более релевантный код.
Используй:
- Технические термины (названия функций, классов)
- Ключевые слова из домена
- Более узкий фокус на конкретную часть

Ответь ТОЛЬКО новым запросом, без пояснений."""


ANSWER_GENERATION_PROMPT = """Ты - эксперт по кодовой базе. Ответь на вопрос на основе контекста.

## Вопрос
{query}

## Контекст из кодовой базы

### Основные результаты (прямые совпадения)
{primary_context}

### Связанный код (найден через граф зависимостей)
{graph_context}

## Инструкции
1. Отвечай ТОЛЬКО на основе предоставленного контекста
2. Указывай файлы и номера строк: `file.py:42`
3. Код оформляй в markdown блоках с подсветкой
4. Если информации недостаточно - честно скажи
5. Объясняй связи между основным и связанным кодом

## Формат ответа

### Краткий ответ
<2-3 предложения - суть>

### Реализация
<детали с примерами кода>

### Связи кода
<как элементы кода связаны - вызовы, импорты, наследование>

### Файлы
<список файлов>"""


# ============== STRATEGY DETECTION ==============

def _detect_search_strategy(query: str):
    """
    Detect the best search strategy based on query content.

    Strategies:
    - UI_TO_DATABASE: Query about UI → find related DB models
    - DATABASE_TO_UI: Query about DB → find related UI components
    - IMPACT_ANALYSIS: Query about changes, dependencies, effects
    - PATTERN_SEARCH: Query looking for similar code patterns
    - SEMANTIC_ONLY: Default semantic search with graph expansion
    """
    from ..code_rag.retrieval import SearchStrategy

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

    # Default: PATTERN_SEARCH instead of SEMANTIC_ONLY
    # This enables graph expansion for all queries
    return SearchStrategy.PATTERN_SEARCH


# ============== SINGLETON CLIENTS ==============
# Cache clients to avoid reinitializing on every call
_neo4j_client = None
_weaviate_client = None
_code_retriever = None


def _get_retriever():
    """Get or create CodeRetriever (singleton pattern)."""
    global _neo4j_client, _weaviate_client, _code_retriever

    if _code_retriever is not None:
        return _code_retriever

    from ..code_rag.retrieval import CodeRetriever
    from ..code_rag.graph import Neo4jClient, WeaviateIndexer

    log_both("info", "[INIT] Creating Neo4j client...")
    _neo4j_client = Neo4jClient(
        uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        user=os.getenv("NEO4J_USER", "neo4j"),
        password=os.getenv("NEO4J_PASSWORD", ""),
    )
    log_both("info", "[INIT] Neo4j client created")

    log_both("info", "[INIT] Creating Weaviate client...")

    _weaviate_client = WeaviateIndexer(
        weaviate_url=os.getenv("WEAVIATE_URL", "http://localhost:8080"),
    )
    log_both("info", "[INIT] Weaviate client created")

    log_both("info", "[INIT] Creating CodeRetriever...")
    _code_retriever = CodeRetriever(_neo4j_client, _weaviate_client)
    log_both("info", "[INIT] CodeRetriever created")

    return _code_retriever


# ============== NODE FUNCTIONS ==============

def context_collector(state: RAGState) -> Dict[str, Any]:
    """
    Collect context from vector database + graph expansion.

    Uses:
    1. Weaviate for semantic search (find relevant code)
    2. Neo4j for graph expansion (find related entities)
    """
    start_time = time.time()
    from ..code_rag.retrieval import SearchStrategy

    query = state["current_query"]
    log_both("info", f"[NODE] context_collector: Starting for query: {query[:50]}...")

    # Create trace
    trace = langfuse.trace(name="context_collector")
    span = trace.span(name="retrieval", input={"query": query})

    try:
        # Use cached retriever (avoids reloading embedding model on each call)
        retriever = _get_retriever()

        # Auto-detect search strategy based on query
        strategy = _detect_search_strategy(query)
        log_both("info", f"[NODE] context_collector: Using strategy: {strategy.value}")

        search_start = time.time()
        result = retriever.search(
            query=query,
            strategy=strategy,  # Auto-detected strategy
            config_override={
                "top_k_vector": 20,
                "top_k_final": 10,
                "expand_results": True,  # Enable graph expansion
                "max_hops": 2,           # Follow 2 hops in graph
            }
        )
        search_time = time.time() - search_start
        log_both("info", f"[NODE] context_collector: Search completed in {search_time:.2f}s, found {len(result.primary_nodes)} primary + {len(result.expanded_nodes)} expanded nodes")

        log_both("info", "[NODE] context_collector: Formatting context...")
        context = []

        # Add primary nodes
        for node in result.primary_nodes[:10]:
            context.append({
                "id": node.get("node_id", node.get("id", "")),
                "name": node.get("name", "Unknown"),
                "type": node.get("node_type", node.get("type", "Unknown")),
                "file": node.get("file_path", node.get("file", "")),
                "code": node.get("content", node.get("code", ""))[:1000],
                "score": node.get("score", 0.0),
                "source": "primary",
            })

        # Add expanded nodes from graph
        for node in result.expanded_nodes[:10]:
            context.append({
                "id": node.get("node_id", node.get("id", "")),
                "name": node.get("name", "Unknown"),
                "type": node.get("node_type", node.get("type", "Unknown")),
                "file": node.get("file_path", node.get("file", "")),
                "code": node.get("content", node.get("code", ""))[:1000],
                "score": node.get("score", 0.0),
                "source": "graph",
                "relationship": node.get("relationship", "RELATED"),
            })

        span.end(output={"chunks_found": len(context)})
        total_time = time.time() - start_time
        log_both("info", f"[NODE] context_collector: Found {len(context)} chunks, total time: {total_time:.2f}s")

    except Exception as e:
        span.end(output={"error": str(e)}, level="ERROR")
        total_time = time.time() - start_time
        log_both("error", f"[NODE] context_collector: ERROR after {total_time:.2f}s - {e}")
        import traceback
        log_both("error", f"[NODE] context_collector: Traceback: {traceback.format_exc()}")
        context = []

    return {
        "context": context,
        "iterations": state["iterations"] + 1,
    }


def quality_checker(state: RAGState) -> Dict[str, Any]:
    """
    Check quality of retrieved context.

    Returns quality score and feedback.
    """
    start_time = time.time()
    
    # Check OPENROUTER_API_KEY availability
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        log_both("error", "[NODE] quality_checker: OPENROUTER_API_KEY not found! LLM call will fail!")
    else:
        log_both("info", f"[NODE] quality_checker: OPENROUTER_API_KEY found (length: {len(api_key)})")
    
    log_both("info", f"[NODE] quality_checker: Starting. State keys: {list(state.keys())}")
    log_both("info", f"[NODE] quality_checker: Iterations: {state.get('iterations', 0)}, Max: {state.get('max_iterations', 3)}")
    
    if not state["context"]:
        log_both("warning", "[NODE] quality_checker: No context found")
        return {
            "quality_score": 0.0,
            "quality_feedback": "Контекст не найден. Нужно переформулировать запрос."
        }

    log_both("info", f"[NODE] quality_checker: Starting with {len(state['context'])} context chunks")
    # Format context for prompt
    context_str = "\n\n".join([
        f"[{c['type']}] {c['name']} ({c['file']})\n```\n{c['code'][:500]}\n```"
        for c in state["context"][:5]
    ])

    prompt = QUALITY_CHECK_PROMPT.format(
        query=state["query"],
        context=context_str
    )

    log_both("info", f"[NODE] quality_checker: Calling LLM...")
    llm_start = time.time()
    response = call_llm(prompt, "quality_check", max_tokens=512)
    llm_time = time.time() - llm_start
    log_both("info", f"[NODE] quality_checker: LLM response received in {llm_time:.2f}s")

    # Parse response
    score = 0.5
    feedback = "Не удалось оценить качество"

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

    total_time = time.time() - start_time
    log_both("info", f"[NODE] quality_checker: Score={score}, Feedback={feedback[:50]}..., total time: {total_time:.2f}s")
    return {
        "quality_score": score,
        "quality_feedback": feedback,
    }


def query_rewriter(state: RAGState) -> Dict[str, Any]:
    """
    Rewrite query to improve retrieval.
    """
    start_time = time.time()
    
    # Check OPENROUTER_API_KEY availability
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        log_both("error", "[NODE] query_rewriter: OPENROUTER_API_KEY not found! LLM call will fail!")
    else:
        log_both("info", f"[NODE] query_rewriter: OPENROUTER_API_KEY found (length: {len(api_key)})")
    
    log_both("info", f"[NODE] query_rewriter: Starting rewrite based on feedback: {state['quality_feedback'][:50]}...")
    prompt = QUERY_REWRITE_PROMPT.format(
        original_query=state["query"],
        current_query=state["current_query"],
        feedback=state["quality_feedback"],
    )

    log_both("info", f"[NODE] query_rewriter: Calling LLM...")
    llm_start = time.time()
    new_query = call_llm(prompt, "query_rewrite", max_tokens=256)
    llm_time = time.time() - llm_start
    log_both("info", f"[NODE] query_rewriter: LLM response received in {llm_time:.2f}s")
    new_query = new_query.strip().strip('"\'')
    total_time = time.time() - start_time
    log_both("info", f"[NODE] query_rewriter: New query: {new_query[:50]}..., total time: {total_time:.2f}s")

    return {"current_query": new_query}


def answer_generator(state: RAGState) -> Dict[str, Any]:
    """
    Generate final answer from context.

    Separates primary (direct search) and graph-expanded context.
    """
    start_time = time.time()

    # Check OPENROUTER_API_KEY availability
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        log_both("error", "[NODE] answer_generator: OPENROUTER_API_KEY not found! LLM call will fail!")
    else:
        log_both("info", f"[NODE] answer_generator: OPENROUTER_API_KEY found (length: {len(api_key)})")

    # Separate primary and graph-expanded context
    primary_nodes = [c for c in state["context"] if c.get("source") == "primary"]
    graph_nodes = [c for c in state["context"] if c.get("source") == "graph"]

    log_both("info", f"[NODE] answer_generator: Starting with {len(primary_nodes)} primary + {len(graph_nodes)} graph context chunks")

    # Format primary context
    primary_context = "\n\n".join([
        f"### {c['name']} ({c['type']})\n**Файл:** `{c['file']}`\n```\n{c['code']}\n```"
        for c in primary_nodes
    ]) or "Прямых совпадений не найдено."

    # Format graph-expanded context with relationship info
    graph_context = "\n\n".join([
        f"### {c['name']} ({c['type']}) - {c.get('relationship', 'RELATED')}\n**Файл:** `{c['file']}`\n```\n{c['code']}\n```"
        for c in graph_nodes
    ]) or "Связанный код через граф не найден."

    prompt = ANSWER_GENERATION_PROMPT.format(
        query=state["query"],
        primary_context=primary_context,
        graph_context=graph_context,
    )

    log_both("info", f"[NODE] answer_generator: Calling LLM...")
    llm_start = time.time()
    answer = call_llm(prompt, "answer_generation", max_tokens=4096)
    llm_time = time.time() - llm_start
    total_time = time.time() - start_time
    log_both("info", f"[NODE] answer_generator: Answer generated, length: {len(answer)}, LLM time: {llm_time:.2f}s, total time: {total_time:.2f}s")

    # Extract sources
    sources = [
        {"name": c["name"], "file": c["file"], "type": c["type"]}
        for c in state["context"]
    ]

    return {
        "answer": answer,
        "sources": sources,
    }


# ============== ROUTING ==============

def should_rewrite(state: RAGState) -> str:
    """
    Decide: rewrite query or generate answer.

    Returns:
        "generate" - quality is good enough
        "rewrite" - need to improve query
    """
    quality_threshold = float(os.getenv("QUALITY_THRESHOLD", "0.6"))

    if state["quality_score"] >= quality_threshold:
        return "generate"

    if state["iterations"] >= state["max_iterations"]:
        return "generate"  # Give up, generate with what we have

    return "rewrite"
