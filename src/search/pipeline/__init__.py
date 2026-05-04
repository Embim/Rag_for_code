"""
Search pipeline (executor / task / service).

Точки входа:
- pipeline.rag_search.run(query, ...) — iterative RAG.
- SearchExecutor + Task'и — для своих сценариев (например, single-shot
  без feedback loop'а или с другой стратегией контроля).
"""

from .base import SearchExecutor, SearchTask
from .pipelines import rag_search

__all__ = ["SearchExecutor", "SearchTask", "rag_search"]
