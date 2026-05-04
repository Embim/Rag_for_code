"""
RepoPathResolver — резолвинг абсолютных путей индексированных репозиториев.

Проблема: при индексации мы знаем где лежат репо (см. ``RepositoryNode.local_path``),
а при поиске (``read_code`` / ``GrepEnrichTask``) нужно эти пути восстановить.
Хардкодить один ``RAG_REPOS_DIR`` для всех репо плохо: проектов может быть 40,
все в разных местах.

Решение:
- На каждой ``RepositoryNode`` хранится ``local_path`` (атрибут).
- ``RepoPathResolver`` лениво подтягивает все ``(name, local_path)`` из Neo4j
  и кеширует в памяти. Перезагрузка по требованию через ``invalidate()``.
- Override через env ``RAG_REPOS_DIR``: если задан, для всех репо корнем
  считается ``RAG_REPOS_DIR / <name>`` — полезно для Docker-сценариев,
  когда хост-пути не совпадают с контейнерными.

Используется:
- ``GrepEnrichTask`` — получает ``list_roots()`` и передаёт всё в один ``rg``.
- ``code_loader.code_for_node`` — резолвит репо по ``node_id`` через ``resolve(name)``.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.infra.logger import get_logger

logger = get_logger(__name__)


_LIST_REPOS_QUERY = """
MATCH (r:Repository)
RETURN r.name AS name, r.local_path AS local_path
"""


class RepoPathResolver:
    """
    Lazy-cached резолвинг ``repo_name → absolute Path``.

    Безопасен для конкурентного использования (lock на загрузку кеша).

    Пример:
        resolver = RepoPathResolver(neo4j_client)
        path = resolver.resolve("api.bo")   # → Path('/home/user/repos/api.bo')
        roots = resolver.list_roots()        # → [Path(...), Path(...), ...]
    """

    def __init__(self, neo4j_client: Any) -> None:
        self._neo4j = neo4j_client
        self._cache: Optional[Dict[str, Path]] = None
        self._lock = threading.Lock()

        # Override-корень из env. Если задан, ВСЕ репо считаются как
        # ``$RAG_REPOS_DIR/<name>`` независимо от того что в Neo4j.
        self._env_override: Optional[Path] = None
        env_dir = os.environ.get("RAG_REPOS_DIR")
        if env_dir:
            self._env_override = Path(env_dir)
            logger.info(
                f"[RepoPathResolver] env override active: "
                f"all repos resolved under {self._env_override}"
            )

    def _load(self) -> Dict[str, Path]:
        """Подтянуть карту репо из Neo4j (LRU-style: сначала проверяем кеш)."""
        if self._cache is not None:
            return self._cache
        with self._lock:
            if self._cache is not None:
                return self._cache

            cache: Dict[str, Path] = {}
            try:
                rows = self._neo4j.execute_cypher(_LIST_REPOS_QUERY)
            except Exception as e:
                logger.warning(f"[RepoPathResolver] cypher failed: {e}")
                rows = []

            for row in rows:
                name = (row.get("name") or "").strip()
                lp = (row.get("local_path") or "").strip()
                if not name:
                    continue
                # При env override игнорируем local_path из Neo4j —
                # пути перезаписываются единым корнем.
                if self._env_override is not None:
                    cache[name] = self._env_override / name
                    continue
                if lp:
                    cache[name] = Path(lp)

            self._cache = cache
            logger.info(
                f"[RepoPathResolver] loaded {len(cache)} repo path(s): "
                f"{sorted(cache.keys())}"
            )
            return cache

    def resolve(self, repo_name: str) -> Optional[Path]:
        """
        Получить абсолютный путь к репо по имени.

        Возвращает ``None`` если репо не индексирован или у него не
        задан ``local_path`` и нет env override'а.
        """
        if not repo_name:
            return None
        return self._load().get(repo_name)

    def list_roots(self) -> List[Path]:
        """
        Все известные корни репо (для случаев типа GrepEnrichTask, где
        ``rg`` принимает несколько путей в одном вызове).

        Фильтрует несуществующие пути — иначе ``rg`` вернёт ошибку.
        """
        out: List[Path] = []
        for p in self._load().values():
            try:
                if p.exists():
                    out.append(p)
                else:
                    logger.debug(
                        f"[RepoPathResolver] skip non-existent root: {p}"
                    )
            except OSError:
                continue
        return out

    def invalidate(self) -> None:
        """
        Сбросить кеш. Дёргать после переиндексации (или из API endpoint'а
        ``/api/repositories/refresh-paths`` — если будет добавлен).
        """
        with self._lock:
            self._cache = None
        logger.info("[RepoPathResolver] cache invalidated")
