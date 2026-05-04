"""
Indexing pipeline (executor / task / service).

Точки входа:
- pipeline.full_index.run(...) — полный сценарий индексации.
- IndexingExecutor + Task'и — для своих сценариев.
"""

from .base import IndexingExecutor, IndexingTask
from .pipelines import full_index

__all__ = ["IndexingExecutor", "IndexingTask", "full_index"]
