"""
Только сборка графа: load → parse → build_graph.

Без API-линковки и без Weaviate. Полезно для быстрого пересоздания
графа без индексации embedding'ов.
"""

from typing import Any, Dict, Optional

from ..tasks import BuildGraphTask
from ._factory import build_stats, make_executor


def run(
    *,
    source: str,
    name: Optional[str] = None,
    branch: str = "main",
    clear_existing: bool = False,
    neo4j_uri: str = "bolt://localhost:7687",
    neo4j_user: str = "neo4j",
    neo4j_password: str = "password",
) -> Dict[str, Any]:
    filters = {
        "source": source,
        "name": name,
        "branch": branch,
        "clear_existing": clear_existing,
    }

    with make_executor(
        filters=filters,
        neo4j_uri=neo4j_uri,
        neo4j_user=neo4j_user,
        neo4j_password=neo4j_password,
    ) as executor:
        # BuildGraphTask сам резолвит deps:
        # LoadRepositoryTask → ParseFilesTask → BuildGraphTask.
        executor.run(BuildGraphTask)
        return build_stats(executor)
