"""
Backward-compat shim для старого API.

Реальный пайплайн теперь живёт в ``src.indexing.pipeline``
(executor / task / service). Этот модуль сохраняет старые имена
(``GraphPipeline``, ``build_and_index``, CLI) — они дёргают новый
сценарий ``pipelines.full_index.run``.

Если пишете новый код — импортируйте напрямую из
``src.indexing.pipeline``.
"""

import argparse
from typing import Any, Dict, Optional

from src.indexing.pipeline.pipelines import full_index
from src.infra.logger import get_logger

logger = get_logger(__name__)


class GraphPipeline:
    """
    Старый фасад. Хранит конфиг подключения, при ``.run()`` запускает
    новый ``full_index.run`` со всеми параметрами.

    .. deprecated::
        Используйте ``from src.indexing.pipeline import full_index`` напрямую.
    """

    def __init__(
        self,
        neo4j_uri: str = "bolt://localhost:7687",
        neo4j_user: str = "neo4j",
        neo4j_password: str = "password",
        weaviate_url: str = "http://localhost:8080",
        embedding_model: str = "BAAI/bge-m3",
    ):
        self._cfg = {
            "neo4j_uri": neo4j_uri,
            "neo4j_user": neo4j_user,
            "neo4j_password": neo4j_password,
            "weaviate_url": weaviate_url,
            "embedding_model": embedding_model,
        }

    def run(
        self,
        source: str,
        name: Optional[str] = None,
        branch: str = "main",
        clear_existing: bool = False,
        link_apis: bool = True,
        index_weaviate: bool = True,
    ) -> Dict[str, Any]:
        return full_index.run(
            source=source,
            name=name,
            branch=branch,
            clear_existing=clear_existing,
            link_apis=link_apis,
            index_weaviate=index_weaviate,
            **self._cfg,
        )

    def close(self) -> None:
        # full_index.run сам закрывает соединения по finally.
        # Метод оставлен ради совместимости.
        return None


def build_and_index(
    repos_dir: str,
    neo4j_uri: str = "bolt://localhost:7687",
    neo4j_user: str = "neo4j",
    neo4j_password: str = "password",
    weaviate_url: str = "http://localhost:8080",
    embedding_model: str = "BAAI/bge-m3",
    clear_existing: bool = False,
    link_apis: bool = True,
    index_weaviate: bool = True,
) -> Dict[str, Any]:
    """Backward-compat обёртка."""
    return full_index.run(
        source=repos_dir,
        clear_existing=clear_existing,
        link_apis=link_apis,
        index_weaviate=index_weaviate,
        neo4j_uri=neo4j_uri,
        neo4j_user=neo4j_user,
        neo4j_password=neo4j_password,
        weaviate_url=weaviate_url,
        embedding_model=embedding_model,
    )


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Build and index knowledge graph from code repository"
    )
    parser.add_argument("source", help="Repository URL or local path")
    parser.add_argument("--name", help="Repository name")
    parser.add_argument("--branch", default="main", help="Git branch (default: main)")
    parser.add_argument("--neo4j-uri", default="bolt://localhost:7687")
    parser.add_argument("--neo4j-user", default="neo4j")
    parser.add_argument("--neo4j-password", default="password")
    parser.add_argument("--weaviate-url", default="http://localhost:8080")
    parser.add_argument("--embedding-model", default="BAAI/bge-m3")
    parser.add_argument("--clear", action="store_true", help="Clear existing graph data")
    parser.add_argument("--no-api-linking", action="store_true")
    parser.add_argument("--no-weaviate", action="store_true")

    args = parser.parse_args()

    try:
        full_index.run(
            source=args.source,
            name=args.name,
            branch=args.branch,
            clear_existing=args.clear,
            link_apis=not args.no_api_linking,
            index_weaviate=not args.no_weaviate,
            neo4j_uri=args.neo4j_uri,
            neo4j_user=args.neo4j_user,
            neo4j_password=args.neo4j_password,
            weaviate_url=args.weaviate_url,
            embedding_model=args.embedding_model,
        )
        return 0
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
