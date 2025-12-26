"""
RAG Pipeline State.

Defines the state that flows through the LangGraph pipeline.
"""

from typing import TypedDict, List, Dict, Any, Annotated
from operator import add


class RAGState(TypedDict):
    """State for the agentic RAG pipeline."""

    # Input
    query: str                              # Original user query

    # Working state
    current_query: str                      # Current query (may be rewritten)
    context: List[Dict[str, Any]]           # Retrieved context chunks
    quality_score: float                    # Context quality score (0-1)
    quality_feedback: str                   # Feedback on why quality is low/high

    # Output
    answer: str                             # Final generated answer
    sources: List[Dict[str, Any]]           # Sources used in answer

    # Control flow
    iterations: int                         # Current iteration count
    max_iterations: int                     # Maximum allowed iterations


class ContextChunk(TypedDict):
    """Single context chunk from retrieval."""
    id: str
    name: str
    type: str
    file: str
    code: str
    score: float
