"""
RAG Pipeline Nodes.

Each node is a function that takes state and returns updated state.
Integrated with Langfuse v3 for tracing via start_as_current_generation().
"""

import os
import sys
import time
from typing import Dict, Any
from pathlib import Path

from openai import OpenAI
from langfuse import get_client as _get_langfuse

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.logger import get_logger, setup_logging

setup_logging(level="INFO", enable_console=True)

from src.langgraph_server.state import RAGState

logger = get_logger(__name__)


def log_both(level: str, message: str):
    """Log to both logger and stdout for visibility."""
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


# ============== LLM CLIENT ==============

def get_llm_client() -> OpenAI:
    """Get OpenAI client configured for OpenRouter."""
    from src.config.agent import AgentConfig
    config = AgentConfig()
    return OpenAI(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url=config.api_base,
    )


def call_llm(prompt: str, max_tokens: int = 2048, model: str = None, name: str = "llm_call") -> str:
    """Call LLM with Langfuse v3 generation tracing."""
    if model is None:
        from src.config.agent import AgentConfig
        model = AgentConfig().rag_answer_model
    client = get_llm_client()
    langfuse = _get_langfuse()

    with langfuse.start_as_current_observation(name=name, as_type="generation"):
        langfuse.update_current_generation(model=model, input=prompt)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=max_tokens,
        )
        msg = response.choices[0].message
        content = msg.content
        if content is None:
            # Reasoning models (o1, gpt-oss, etc.) may put output in reasoning fields
            content = (getattr(msg, 'reasoning', None) or
                       getattr(msg, 'reasoning_content', None) or "")
            if content:
                log_both("warning", f"[LLM] {model}: content=None, using reasoning field ({len(content)} chars)")
            else:
                log_both("warning", f"[LLM] {model}: content=None and no reasoning. finish_reason={response.choices[0].finish_reason}")
        result = content.strip()
        langfuse.update_current_generation(output=result)

    return result


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
    """Detect the best search strategy based on query content."""
    from src.code_rag.retrieval import SearchStrategy

    query_lower = query.lower()

    ui_keywords = [
        'ui', 'view', 'frontend', 'button', 'form', 'component',
        'template', 'html', 'css', 'react', 'vue', 'angular',
        'page', 'screen', 'widget', 'render', 'display', 'show'
    ]
    if any(kw in query_lower for kw in ui_keywords):
        db_connection_keywords = ['database', 'model', 'data', 'api', 'backend', 'server']
        if any(kw in query_lower for kw in db_connection_keywords):
            return SearchStrategy.UI_TO_DATABASE

    db_keywords = [
        'database', 'model', 'table', 'schema', 'migration', 'orm',
        'django model', 'sqlalchemy', 'entity', 'repository', 'dao'
    ]
    if any(kw in query_lower for kw in db_keywords):
        ui_usage_keywords = ['used', 'displayed', 'shown', 'view', 'frontend', 'ui']
        if any(kw in query_lower for kw in ui_usage_keywords):
            return SearchStrategy.DATABASE_TO_UI

    impact_keywords = [
        'impact', 'affect', 'change', 'depend', 'break', 'modify',
        'refactor', 'update', 'remove', 'delete', 'what happens',
        'who calls', 'who uses', 'where used', 'references'
    ]
    if any(kw in query_lower for kw in impact_keywords):
        return SearchStrategy.IMPACT_ANALYSIS

    pattern_keywords = [
        'pattern', 'similar', 'like', 'example', 'same as',
        'how to', 'implement', 'usage', 'best practice'
    ]
    if any(kw in query_lower for kw in pattern_keywords):
        return SearchStrategy.PATTERN_SEARCH

    return SearchStrategy.PATTERN_SEARCH


# ============== SINGLETON CLIENTS ==============

_neo4j_client = None
_weaviate_client = None
_code_retriever = None


def _get_retriever():
    """Get or create CodeRetriever (singleton pattern)."""
    global _neo4j_client, _weaviate_client, _code_retriever

    if _code_retriever is not None:
        return _code_retriever

    from src.code_rag.retrieval import CodeRetriever
    from src.code_rag.graph import Neo4jClient, WeaviateIndexer

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
    """Collect context from vector database + graph expansion."""
    start_time = time.time()
    query = state["current_query"]
    log_both("info", f"[NODE] context_collector: Starting for query: {query[:50]}...")

    try:
        retriever = _get_retriever()
        strategy = _detect_search_strategy(query)
        log_both("info", f"[NODE] context_collector: Using strategy: {strategy.value}")

        search_start = time.time()
        result = retriever.search(
            query=query,
            strategy=strategy,
            config_override={
                "top_k_vector": 80,
                "top_k_final": 40,
                "expand_results": True,
                "max_hops": 3,
                "hybrid_alpha": 0.7,  # More semantic (was 0.3 BM25-heavy)
            }
        )
        search_time = time.time() - search_start
        log_both("info", f"[NODE] context_collector: Search done in {search_time:.2f}s, "
                         f"found {len(result.primary_nodes)} primary + {len(result.expanded_nodes)} expanded nodes")

        context = []
        seen_ids: set = set()

        def _dedup_key(node: dict) -> str:
            node_id = node.get("node_id", node.get("id", ""))
            if node_id:
                return node_id
            return f"{node.get('name', '')}::{node.get('file_path', node.get('file', ''))}"

        for node in result.primary_nodes[:40]:
            key = _dedup_key(node)
            if key in seen_ids:
                continue
            seen_ids.add(key)
            context.append({
                "id": node.get("node_id", node.get("id", "")),
                "name": node.get("name", "Unknown"),
                "type": node.get("node_type", node.get("type", "Unknown")),
                "file": node.get("file_path", node.get("file", "")),
                "code": (node.get("code") or node.get("content", ""))[:2000],
                "score": node.get("score", 0.0),
                "source": "primary",
            })

        for node in result.expanded_nodes[:30]:
            key = _dedup_key(node)
            if key in seen_ids:
                continue
            seen_ids.add(key)
            context.append({
                "id": node.get("node_id", node.get("id", "")),
                "name": node.get("name", "Unknown"),
                "type": node.get("node_type", node.get("type", "Unknown")),
                "file": node.get("file_path", node.get("file", "")),
                "code": (node.get("code") or node.get("content", ""))[:2000],
                "score": node.get("score", 0.0),
                "source": "graph",
                "relationship": node.get("relationship", "RELATED"),
            })

        total_time = time.time() - start_time
        log_both("info", f"[NODE] context_collector: Found {len(context)} chunks in {total_time:.2f}s")

    except Exception as e:
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
    """Check quality of retrieved context."""
    start_time = time.time()

    if not state["context"]:
        log_both("warning", "[NODE] quality_checker: No context found")
        return {
            "quality_score": 0.0,
            "quality_feedback": "Контекст не найден. Нужно переформулировать запрос."
        }

    log_both("info", f"[NODE] quality_checker: Checking {len(state['context'])} context chunks")

    context_str = "\n\n".join([
        f"[{c.get('type') or 'Unknown'}] {c.get('name') or 'Unknown'} ({c.get('file') or ''})\n```\n{str(c.get('code') or '')[:2500]}\n```"
        for c in state["context"][:20]
    ])

    prompt = QUALITY_CHECK_PROMPT.format(
        query=state["query"],
        context=context_str
    )

    from src.config.agent import AgentConfig
    llm_start = time.time()
    response = call_llm(prompt, max_tokens=512, model=AgentConfig().rag_quality_model, name="quality_check")
    log_both("info", f"[NODE] quality_checker: LLM response in {time.time() - llm_start:.2f}s")

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
    log_both("info", f"[NODE] quality_checker: Score={score}, Feedback={feedback[:50]}..., time={total_time:.2f}s")

    return {
        "quality_score": score,
        "quality_feedback": feedback,
    }


def query_rewriter(state: RAGState) -> Dict[str, Any]:
    """Rewrite query to improve retrieval."""
    start_time = time.time()

    log_both("info", f"[NODE] query_rewriter: Rewriting based on feedback: {state['quality_feedback'][:50]}...")

    prompt = QUERY_REWRITE_PROMPT.format(
        original_query=state["query"],
        current_query=state["current_query"],
        feedback=state["quality_feedback"],
    )

    from src.config.agent import AgentConfig
    new_query = call_llm(prompt, max_tokens=256, model=AgentConfig().rag_rewrite_model, name="query_rewrite")
    new_query = new_query.strip().strip('"\'') or state["current_query"]
    total_time = time.time() - start_time
    log_both("info", f"[NODE] query_rewriter: New query: {new_query[:50]}..., time={total_time:.2f}s")

    return {"current_query": new_query}


def answer_generator(state: RAGState) -> Dict[str, Any]:
    """Generate final answer from context."""
    start_time = time.time()

    primary_nodes = [c for c in state["context"] if c.get("source") == "primary"]
    graph_nodes = [c for c in state["context"] if c.get("source") == "graph"]

    log_both("info", f"[NODE] answer_generator: {len(primary_nodes)} primary + {len(graph_nodes)} graph chunks")

    primary_context = "\n\n".join([
        f"### {c.get('name') or 'Unknown'} ({c.get('type') or 'Unknown'})\n**Файл:** `{c.get('file') or ''}`\n```\n{str(c.get('code') or '')}\n```"
        for c in primary_nodes
    ]) or "Прямых совпадений не найдено."

    graph_context = "\n\n".join([
        f"### {c.get('name') or 'Unknown'} ({c.get('type') or 'Unknown'}) - {c.get('relationship', 'RELATED')}\n**Файл:** `{c.get('file') or ''}`\n```\n{str(c.get('code') or '')}\n```"
        for c in graph_nodes
    ]) or "Связанный код через граф не найден."

    prompt = ANSWER_GENERATION_PROMPT.format(
        query=state["query"],
        primary_context=primary_context,
        graph_context=graph_context,
    )

    from src.config.agent import AgentConfig
    answer = call_llm(prompt, max_tokens=4096, model=AgentConfig().rag_answer_model, name="answer_generator")
    total_time = time.time() - start_time
    log_both("info", f"[NODE] answer_generator: Answer length={len(answer)}, total={total_time:.2f}s")

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
    """Decide: rewrite query or generate answer."""
    quality_threshold = float(os.getenv("QUALITY_THRESHOLD", "0.6"))

    if state["quality_score"] >= quality_threshold:
        return "generate"

    if state["iterations"] >= state["max_iterations"]:
        return "generate"  # Give up, generate with what we have

    return "rewrite"
