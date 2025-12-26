"""
LangGraph Agentic RAG Pipeline.

Implements a graph-based RAG with agents for:
- Context collection from vector DB (code or documents)
- Quality checking of retrieved context
- Query rewriting for better retrieval
- Answer generation based on instructions
"""

import os
from typing import TypedDict, List, Dict, Any, Optional, Annotated
from operator import add
from enum import Enum

from langgraph.graph import StateGraph, END
from openai import OpenAI

from ..code_rag.retrieval import CodeRetriever, SearchStrategy, DocumentRetriever
from ..logger import get_logger


logger = get_logger(__name__)


# ============== SEARCH MODE ==============

class SearchMode(str, Enum):
    """Search mode for RAG pipeline."""
    CODE_ONLY = "code_only"      # Search only in code
    DOCS_ONLY = "docs_only"      # Search only in documentation (SOP, policies)
    HYBRID = "hybrid"             # Search in both code and docs


# ============== STATE ==============

class RAGState(TypedDict):
    """State for the RAG pipeline."""
    query: str                              # Original user query
    current_query: str                      # Current query (may be rewritten)
    search_mode: str                        # Search mode (code_only/docs_only/hybrid)
    context: List[Dict[str, Any]]           # Retrieved context chunks
    quality_score: float                    # Context quality score (0-1)
    quality_feedback: str                   # Feedback on context quality
    answer: str                             # Generated answer
    iterations: int                         # Number of retrieval iterations
    max_iterations: int                     # Maximum allowed iterations
    sources: List[Dict[str, Any]]           # Sources used in answer


# ============== PROMPTS ==============

QUALITY_CHECK_PROMPT = """Ты - эксперт по оценке качества контекста для ответа на вопросы о коде.

Вопрос пользователя: {query}

Найденный контекст:
{context}

Оцени качество найденного контекста по следующим критериям:
1. Релевантность: насколько контекст относится к вопросу
2. Полнота: достаточно ли информации для ответа
3. Специфичность: есть ли конкретные примеры кода, функции, классы

Ответь строго в формате:
SCORE: <число от 0.0 до 1.0>
FEEDBACK: <краткое объяснение оценки и что улучшить>

Примеры:
- SCORE: 0.9 - контекст полностью релевантен и достаточен
- SCORE: 0.5 - контекст частично релевантен, нужно больше деталей
- SCORE: 0.2 - контекст слабо связан с вопросом"""


QUERY_REWRITE_PROMPT = """Ты - эксперт по поиску в кодовой базе. Перепиши запрос для лучшего поиска.

Оригинальный запрос: {original_query}
Текущий запрос: {current_query}
Обратная связь о качестве контекста: {feedback}

Учитывая обратную связь, создай новый поисковый запрос который:
1. Более точно описывает искомое
2. Использует технические термины (функции, классы, API)
3. Фокусируется на конкретных аспектах которых не хватало

Ответь ТОЛЬКО новым запросом, без пояснений."""


ANSWER_GENERATION_PROMPT = """Ты - эксперт по кодовой базе. Ответь на вопрос используя найденный контекст.

Вопрос: {query}

Контекст из кодовой базы:
{context}

Инструкции:
1. Отвечай на основе ТОЛЬКО предоставленного контекста
2. Указывай файлы и номера строк где возможно
3. Приводи примеры кода в markdown блоках
4. Если контекст недостаточен - честно скажи об этом
5. Структурируй ответ с заголовками и списками

Формат ответа:
## Краткий ответ
<2-3 предложения с сутью>

## Детали реализации
<подробности с примерами кода>

## Связанные файлы
<список релевантных файлов>"""


# ============== AGENT NODES ==============

class LangGraphRAG:
    """LangGraph-based RAG pipeline with quality feedback loop."""

    def __init__(
        self,
        retriever: CodeRetriever,
        document_retriever: Optional[DocumentRetriever] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        api_base: str = "https://openrouter.ai/api/v1",
        max_iterations: int = 3,
        quality_threshold: float = 0.6,
    ):
        """
        Initialize LangGraph RAG.

        Args:
            retriever: CodeRetriever instance for code search
            document_retriever: DocumentRetriever for document search (optional)
            api_key: OpenRouter API key
            model: LLM model name (default from env)
            api_base: API base URL
            max_iterations: Max retrieval attempts
            quality_threshold: Min quality score to accept context
        """
        self.retriever = retriever
        self.document_retriever = document_retriever
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        # Standard large model (not experimental)
        # Using 405B for maximum quality (FREE version available!)
        self.model = model or os.getenv("CODE_EXPLORER_MODEL", "meta-llama/llama-3.1-405b-instruct:free")
        self.api_base = api_base
        self.max_iterations = max_iterations
        self.quality_threshold = quality_threshold

        self.llm = OpenAI(api_key=self.api_key, base_url=self.api_base)
        self.graph = self._build_graph()

    def _call_llm(self, prompt: str, max_tokens: int = 2048) -> str:
        """Call LLM with prompt."""
        response = self.llm.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip()

    # --- Node: Context Collector ---
    def context_collector(self, state: RAGState) -> RAGState:
        """Collect context based on search mode (code or documents)."""
        query = state["current_query"]
        search_mode = state.get("search_mode", SearchMode.CODE_ONLY.value)

        logger.info(f"Collecting context for: {query} (mode: {search_mode})")

        # Route to appropriate collector
        if search_mode == SearchMode.DOCS_ONLY.value:
            return self._collect_documents(state)
        elif search_mode == SearchMode.HYBRID.value:
            # Collect from both sources
            code_state = self._collect_code(state)
            docs_state = self._collect_documents(state)
            # Combine contexts
            combined_context = code_state["context"] + docs_state["context"]
            return {
                **state,
                "context": combined_context,
                "iterations": state["iterations"] + 1,
            }
        else:  # CODE_ONLY (default)
            return self._collect_code(state)

    def _collect_code(self, state: RAGState) -> RAGState:
        """Collect code context from vector DB with call graph enrichment."""
        query = state["current_query"]

        try:
            result = self.retriever.search(
                query=query,
                strategy=SearchStrategy.SEMANTIC_ONLY,
                config_override={
                    "top_k_vector": 20,
                    "top_k_final": 10,
                    # Enable call graph enrichment for better context
                    "enable_call_graph_enrichment": True,
                    "call_graph_depth": 1,
                    "max_callees_per_function": 5,
                    "include_parent_class": True,
                    "include_file_context": True
                }
            )

            context = []

            # Primary nodes (most relevant results)
            for node in result.primary_nodes[:10]:
                code_content = node.get("code", "") or node.get("content", "")
                context.append({
                    "id": node.get("node_id", node.get("id", "")),
                    "name": node.get("name", "Unknown"),
                    "type": node.get("node_type", node.get("type", "Unknown")),
                    "file": node.get("file_path", node.get("file", "")),
                    "code": code_content,
                    "score": node.get("score", 0.0),
                    "is_primary": True,
                    "source_type": "code"  # Mark as code source
                })

            # Expanded nodes (called functions, parent classes, file context)
            for node in result.expanded_nodes:
                code_content = node.get("code", "") or node.get("content", "")
                enrichment_source = node.get("_enrichment_source", "expanded")

                context.append({
                    "id": node.get("node_id", node.get("id", "")),
                    "name": node.get("name", "Unknown"),
                    "type": node.get("node_type", node.get("type", "Unknown")),
                    "file": node.get("file_path", node.get("file", "")),
                    "code": code_content,
                    "score": 0.0,
                    "is_primary": False,
                    "enrichment_source": enrichment_source,
                    "source_type": "code"
                })

            logger.info(
                f"Code: Found {len(context)} chunks: "
                f"{len(result.primary_nodes)} primary + {len(result.expanded_nodes)} enriched"
            )

        except Exception as e:
            logger.error(f"Code collection failed: {e}")
            context = []

        return {
            **state,
            "context": context,
            "iterations": state["iterations"] + 1,
        }

    def _collect_documents(self, state: RAGState) -> RAGState:
        """Collect document context from DocumentRetriever."""
        query = state["current_query"]

        if not self.document_retriever:
            logger.warning("Document retriever not available, skipping document search")
            return {
                **state,
                "context": [],
                "iterations": state["iterations"] + 1,
            }

        try:
            result = self.document_retriever.search(
                query=query,
                top_k=10
            )

            context = []

            # Add documents to context
            for doc in result.documents:
                context.append({
                    "id": doc.get("node_id", ""),
                    "name": doc.get("name", "Unknown"),
                    "type": doc.get("document_type", "Document"),
                    "file": doc.get("file_path", ""),
                    "code": doc.get("content", ""),  # Use 'code' field for consistency
                    "score": doc.get("score", 0.0),
                    "is_primary": True,
                    "source_type": "document",  # Mark as document source
                    "author": doc.get("author", ""),
                    "sections_count": doc.get("sections_count", 0),
                    "images_count": doc.get("images_count", 0)
                })

            logger.info(f"Documents: Found {len(context)} documents")

        except Exception as e:
            logger.error(f"Document collection failed: {e}")
            context = []

        return {
            **state,
            "context": context,
            "iterations": state["iterations"] + 1,
        }

    # --- Node: Quality Checker ---
    def quality_checker(self, state: RAGState) -> RAGState:
        """Check quality of retrieved context."""
        if not state["context"]:
            return {**state, "quality_score": 0.0, "quality_feedback": "No context found"}

        # Format context for prompt
        # Limit to first 5 chunks and max 1000 chars per chunk for quality check
        context_str = "\n\n".join([
            f"[{c['type']}] {c['name']} ({c['file']})\n```\n{c['code'][:1000]}\n```"
            for c in state["context"][:5]
        ])

        prompt = QUALITY_CHECK_PROMPT.format(
            query=state["query"],
            context=context_str
        )

        try:
            response = self._call_llm(prompt, max_tokens=512)

            # Parse response
            score = 0.5
            feedback = "Could not parse quality assessment"

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

            logger.info(f"Quality score: {score:.2f} - {feedback[:100]}")

        except Exception as e:
            logger.error(f"Quality check failed: {e}")
            score = 0.5
            feedback = str(e)

        return {**state, "quality_score": score, "quality_feedback": feedback}

    # --- Node: Query Rewriter ---
    def query_rewriter(self, state: RAGState) -> RAGState:
        """Rewrite query for better retrieval."""
        prompt = QUERY_REWRITE_PROMPT.format(
            original_query=state["query"],
            current_query=state["current_query"],
            feedback=state["quality_feedback"]
        )

        try:
            new_query = self._call_llm(prompt, max_tokens=256)
            new_query = new_query.strip().strip('"\'')
            logger.info(f"Rewritten query: {new_query}")

        except Exception as e:
            logger.error(f"Query rewrite failed: {e}")
            new_query = state["current_query"]

        return {**state, "current_query": new_query}

    # --- Node: Answer Generator ---
    def answer_generator(self, state: RAGState) -> RAGState:
        """Generate final answer from context."""
        # Separate code and document contexts
        code_context = []
        doc_context = []
        enriched_context = []

        for c in state["context"]:
            source_type = c.get("source_type", "code")

            # Build context string with markers
            enrichment_note = ""
            if not c.get("is_primary", True) and source_type == "code":
                source = c.get("enrichment_source", "expanded")
                if source == "callee":
                    enrichment_note = " [CALLED BY PRIMARY FUNCTION]"
                elif source == "parent_class":
                    enrichment_note = " [PARENT CLASS]"
                elif source == "sibling_method":
                    enrichment_note = " [SIBLING METHOD]"
                elif source == "file_context":
                    enrichment_note = " [FILE CONTEXT]"

            # Format based on source type
            if source_type == "document":
                # Document formatting
                doc_info = f"Author: {c.get('author', 'Unknown')}" if c.get('author') else ""
                formatted = f"### {c['name']} ({c['type']})\n**File:** `{c['file']}`\n{doc_info}\n\n{c['code']}"
                doc_context.append(formatted)
            else:
                # Code formatting
                formatted = f"### {c['name']} ({c['type']}){enrichment_note}\n**File:** `{c['file']}`\n```\n{c['code']}\n```"

                if c.get("is_primary", True):
                    code_context.append(formatted)
                else:
                    enriched_context.append(formatted)

        # Combine contexts
        context_parts = []

        if doc_context:
            context_parts.append("## DOCUMENTATION (СОП, Policies, Instructions)")
            context_parts.extend(doc_context)

        if code_context:
            context_parts.append("\n## CODE (Most Relevant)")
            context_parts.extend(code_context)

        if enriched_context:
            context_parts.append("\n## RELATED CODE (Called Functions, Parent Classes, File Context)")
            context_parts.extend(enriched_context)

        context_str = "\n\n".join(context_parts)

        prompt = ANSWER_GENERATION_PROMPT.format(
            query=state["query"],
            context=context_str
        )

        try:
            answer = self._call_llm(prompt, max_tokens=4096)
            logger.info(f"Generated answer: {len(answer)} chars")

        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            answer = f"Error generating answer: {e}"

        # Extract sources
        sources = [
            {"name": c["name"], "file": c["file"], "type": c["type"]}
            for c in state["context"]
        ]

        return {**state, "answer": answer, "sources": sources}

    # --- Conditional Edge ---
    def should_rewrite(self, state: RAGState) -> str:
        """Decide whether to rewrite query or generate answer."""
        if state["quality_score"] >= self.quality_threshold:
            logger.info("Quality sufficient, generating answer")
            return "generate"

        if state["iterations"] >= state["max_iterations"]:
            logger.info("Max iterations reached, generating answer anyway")
            return "generate"

        logger.info("Quality insufficient, rewriting query")
        return "rewrite"

    # --- Build Graph ---
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph pipeline."""
        graph = StateGraph(RAGState)

        # Add nodes
        graph.add_node("context_collector", self.context_collector)
        graph.add_node("quality_checker", self.quality_checker)
        graph.add_node("query_rewriter", self.query_rewriter)
        graph.add_node("answer_generator", self.answer_generator)

        # Add edges
        graph.set_entry_point("context_collector")
        graph.add_edge("context_collector", "quality_checker")

        # Conditional edge: check quality
        graph.add_conditional_edges(
            "quality_checker",
            self.should_rewrite,
            {
                "generate": "answer_generator",
                "rewrite": "query_rewriter",
            }
        )

        # Rewrite leads back to context collection
        graph.add_edge("query_rewriter", "context_collector")

        # Answer generator ends the graph
        graph.add_edge("answer_generator", END)

        return graph.compile()

    def run(
        self,
        query: str,
        search_mode: str = SearchMode.CODE_ONLY.value
    ) -> Dict[str, Any]:
        """
        Run the RAG pipeline.

        Args:
            query: User question
            search_mode: Search mode (code_only/docs_only/hybrid)

        Returns:
            Dict with answer, sources, and metadata
        """
        initial_state: RAGState = {
            "query": query,
            "current_query": query,
            "search_mode": search_mode,  # Add search mode
            "context": [],
            "quality_score": 0.0,
            "quality_feedback": "",
            "answer": "",
            "iterations": 0,
            "max_iterations": self.max_iterations,
            "sources": [],
        }

        logger.info(f"Starting LangGraph RAG for: {query} (mode: {search_mode})")

        final_state = self.graph.invoke(initial_state)

        return {
            "answer": final_state["answer"],
            "sources": final_state["sources"],
            "search_mode": search_mode,  # Include in response
            "iterations": final_state["iterations"],
            "quality_score": final_state["quality_score"],
            "success": len(final_state["answer"]) > 0,
        }

    async def arun(self, query: str) -> Dict[str, Any]:
        """Async version of run."""
        # LangGraph supports async invoke
        initial_state: RAGState = {
            "query": query,
            "current_query": query,
            "context": [],
            "quality_score": 0.0,
            "quality_feedback": "",
            "answer": "",
            "iterations": 0,
            "max_iterations": self.max_iterations,
            "sources": [],
        }

        logger.info(f"Starting async LangGraph RAG for: {query}")

        final_state = await self.graph.ainvoke(initial_state)

        return {
            "answer": final_state["answer"],
            "sources": final_state["sources"],
            "iterations": final_state["iterations"],
            "quality_score": final_state["quality_score"],
            "success": len(final_state["answer"]) > 0,
        }


# ============== FACTORY ==============

def create_langgraph_rag(
    neo4j_client,
    weaviate_indexer,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
) -> LangGraphRAG:
    """
    Create LangGraph RAG instance.

    Args:
        neo4j_client: Neo4j client
        weaviate_indexer: Weaviate indexer
        api_key: Optional API key
        model: Optional model name

    Returns:
        Configured LangGraphRAG instance
    """
    from ..code_rag.retrieval import CodeRetriever

    retriever = CodeRetriever(neo4j_client, weaviate_indexer)

    return LangGraphRAG(
        retriever=retriever,
        api_key=api_key,
        model=model,
    )
