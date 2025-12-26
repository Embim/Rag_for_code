"""
LangGraph Server for Agentic RAG.

This module provides a LangGraph-based RAG pipeline with:
- Context collection from Weaviate vector DB
- Quality checking with LLM feedback
- Query rewriting for improved retrieval
- Answer generation with source citations

Usage:
    # Start server
    cd src/langgraph_server
    langgraph dev

    # Or use programmatically
    from src.langgraph_server import run_rag
    result = run_rag("How does authentication work?")
"""

from .state import RAGState
from .rag_graph import graph, run_rag, arun_rag

__all__ = [
    "RAGState",
    "graph",
    "run_rag",
    "arun_rag",
]
