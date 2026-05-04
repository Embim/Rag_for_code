"""
Полный индекс репозитория: load → parse → build_graph → (link_apis) → (weaviate).

Декларативная композиция шагов. Логика каждого шага — в соответствующем
Task; бизнес-логика — в Service.
"""

from typing import Any, Dict, Optional

from ..tasks import (
    BuildGraphTask,
    IndexWeaviateTask,
    LinkApisTask,
    LoadRepositoryTask,
    ParseFilesTask,
)
from ._factory import build_stats, make_executor


def run(
    *,
    source: str,
    name: Optional[str] = None,
    branch: str = "main",
    clear_existing: bool = False,
    link_apis: bool = True,
    index_weaviate: bool = True,
    neo4j_uri: str = "bolt://localhost:7687",
    neo4j_user: str = "neo4j",
    neo4j_password: str = "password",
    weaviate_url: str = "http://localhost:8080",
    embedding_model: str = "BAAI/bge-m3",
) -> Dict[str, Any]:
    """Запуск полной индексации. Возвращает сводную статистику."""

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
        weaviate_url=weaviate_url,
        embedding_model=embedding_model,
    ) as executor:
        # Загружаем репозиторий с диска / из git.
        executor.run(LoadRepositoryTask)

        # Парсим все файлы соответствующими парсерами,
        # собираем плоский список (file_path, ParseResult).
        executor.run(ParseFilesTask)

        # Строим knowledge-graph в Neo4j: ноды, связи, резолв URL-паттернов
        # для Django (включая раскрытие include() префиксов).
        executor.run(BuildGraphTask)

        if link_apis:
            # Создаём CALLS_ENDPOINT связи между ApiCall (вызов из UI) и Endpoint (бэк).
            executor.run(LinkApisTask)

        if index_weaviate:
            # Семантическое индексирование нод графа в Weaviate.
            executor.run(IndexWeaviateTask)

        return build_stats(executor)
