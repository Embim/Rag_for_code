"""
RAG Pipeline — Python while-loop with Langfuse v3 tracing.

No LangGraph. Orchestration is a simple while-loop.
Nodes live in nodes.py. This module only wires the loop and handles .env loading.
"""

import os
import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.logger import get_logger, setup_logging

setup_logging(level="INFO", enable_console=True)

logger = get_logger(__name__)

# Load environment variables from project root .env file
try:
    from dotenv import load_dotenv

    root_env_path = Path(PROJECT_ROOT) / ".env"
    if root_env_path.exists():
        load_dotenv(root_env_path, override=True)
        logger.info(f"[ENV] Loaded .env from {root_env_path}")
except ImportError:
    pass
except Exception as e:
    print(f"Warning: Could not load .env file: {e}", file=sys.stderr)


# ============== STATE (re-exported for backward compat) ==============

from src.langgraph_server.state import RAGState  # noqa: F401


# ============== NODES ==============

from src.langgraph_server.nodes import (
    context_collector,
    quality_checker,
    query_rewriter,
    answer_generator,
    should_rewrite,
)


# ============== LANGFUSE ==============

from langfuse import get_client as _get_langfuse


# ============== PIPELINE ==============

def run_rag(query: str, max_iterations: int = 3) -> dict:
    """Run RAG pipeline with quality feedback loop and Langfuse v3 tracing."""
    langfuse = _get_langfuse()

    with langfuse.start_as_current_span(name="rag_pipeline"):
        langfuse.update_current_trace(input={"query": query}, tags=["rag"])

        state: RAGState = {
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

        logger.info(f"[RAG] Starting pipeline for query: {query[:60]}... (max_iterations={max_iterations})")

        while True:
            with langfuse.start_as_current_span(name="context_collector"):
                langfuse.update_current_span(
                    input={"query": state["current_query"], "iteration": state["iterations"]}
                )
                update = context_collector(state)
                state.update(update)
                langfuse.update_current_span(
                    output={
                        "chunks": len(state["context"]),
                        "primary": len([c for c in state["context"] if c.get("source") == "primary"]),
                        "graph": len([c for c in state["context"] if c.get("source") == "graph"]),
                        "node_names": [c["name"] for c in state["context"][:10]],
                    }
                )

            with langfuse.start_as_current_span(name="quality_checker"):
                langfuse.update_current_span(
                    input={"query": state["query"], "context_chunks": len(state["context"])}
                )
                update = quality_checker(state)
                state.update(update)
                langfuse.update_current_span(
                    output={"score": state["quality_score"], "feedback": state["quality_feedback"]}
                )

            decision = should_rewrite(state)
            logger.info(
                f"[RAG] Iteration {state['iterations']}: "
                f"score={state['quality_score']:.2f}, decision={decision}"
            )

            if decision == "rewrite":
                with langfuse.start_as_current_span(name="query_rewriter"):
                    langfuse.update_current_span(
                        input={"query": state["current_query"], "feedback": state["quality_feedback"]}
                    )
                    update = query_rewriter(state)
                    state.update(update)
                    langfuse.update_current_span(output={"new_query": state["current_query"]})
            else:
                break

        with langfuse.start_as_current_span(name="answer_generator"):
            langfuse.update_current_span(
                input={
                    "query": state["query"],
                    "primary_chunks": len([c for c in state["context"] if c.get("source") == "primary"]),
                    "graph_chunks": len([c for c in state["context"] if c.get("source") == "graph"]),
                }
            )
            update = answer_generator(state)
            state.update(update)
            langfuse.update_current_span(
                output={"answer_length": len(state["answer"]), "sources": len(state["sources"])}
            )

        result = {
            "answer": state["answer"],
            "sources": state["sources"],
            "iterations": state["iterations"],
            "quality_score": state["quality_score"],
            "quality_feedback": state["quality_feedback"],
            "final_query": state["current_query"],
        }

        langfuse.update_current_trace(output={"answer": result["answer"]})

        logger.info(
            f"[RAG] Pipeline done: {result['iterations']} iterations, "
            f"quality={result['quality_score']:.2f}, answer_length={len(result['answer'])}"
        )

    return result


# ============== CLI ==============

if __name__ == "__main__":
    query = sys.argv[1] if len(sys.argv) > 1 else "How does authentication work?"
    result = run_rag(query)
    print(f"\n{'='*60}")
    print(f"Query: {query}")
    print(f"Iterations: {result['iterations']}, Quality: {result['quality_score']:.2f}")
    print(f"{'='*60}\n")
    print(result["answer"])
