"""
Чтение исходного кода нод графа с диска.

Используется когда узел графа имеет ``file_path`` + ``start_line``/``end_line``,
а тело кода нужно показать (LLM-контекст, UI, debug). LRU‑кешируется,
чтобы не дёргать I/O для одних и тех же файлов на каждом запросе.
"""

from functools import lru_cache
from pathlib import Path
from typing import Optional

from src.infra.logger import get_logger

logger = get_logger(__name__)


# Модуль‑level: один кеш на процесс. 256 файлов × ~50KB ≈ 12 MB max.
@lru_cache(maxsize=256)
def _read_file_lines(absolute_path: str) -> Optional[tuple]:
    try:
        with open(absolute_path, 'r', encoding='utf-8', errors='replace') as f:
            return tuple(f.readlines())
    except FileNotFoundError:
        return None
    except Exception as e:
        logger.warning(f"[code_loader] failed to read {absolute_path}: {e}")
        return None


def _is_within(path: Path, root: Path) -> bool:
    """
    True, если ``path`` (после resolve) находится внутри ``root`` (после resolve).

    Используется как path-traversal guard: даже если в графе оказался кривой
    ``file_path`` вида ``../../etc/passwd`` — мы не вылезем за корень репо.

    Реализация через ``Path.is_relative_to`` (Python 3.9+) с fallback'ом на
    префиксное сравнение для совсем старых ОС/окружений.
    """
    try:
        path_r = path.resolve()
        root_r = root.resolve()
    except Exception:
        return False
    try:
        # Python 3.9+
        return path_r.is_relative_to(root_r)  # type: ignore[attr-defined]
    except AttributeError:
        try:
            path_r.relative_to(root_r)
            return True
        except ValueError:
            return False


def read_code(
    file_path: str,
    start_line: int,
    end_line: int,
    *,
    repo_root: Optional[Path] = None,
    max_chars: Optional[int] = None,
) -> str:
    """
    Прочитать фрагмент кода [start_line, end_line] (1-indexed inclusive).

    Args:
        file_path: путь относительно ``repo_root``, либо абсолютный.
        start_line / end_line: 1-based, inclusive. ``end_line=0`` → до конца файла.
        repo_root: если задан, итоговый путь обязан находиться **внутри**
            ``repo_root`` (после ``resolve()``). Это закрывает path-traversal:
            файлы вида ``../../etc/passwd`` или абсолютные пути за пределами
            репо отбрасываются с warning'ом.
        max_chars: опциональный hard-лимит длины (truncate).

    Returns:
        Строка с кодом, или пустая строка если файл не найден / не разрешён.
    """
    if not file_path:
        return ''

    p = Path(file_path)
    if not p.is_absolute() and repo_root is not None:
        p = repo_root / file_path

    # Path-traversal guard: если задан repo_root, итоговый путь обязан
    # находиться внутри него (включая абсолютные file_path — на них тоже
    # проверяем containment). Без guard'а отравленный индекс мог бы
    # достать ``/etc/passwd`` / ``C:\\Users\\...\\.ssh\\id_rsa``.
    if repo_root is not None and not _is_within(p, repo_root):
        logger.warning(
            f"[code_loader] path-traversal blocked: {file_path!r} "
            f"resolves outside repo_root={str(repo_root)!r}"
        )
        return ''

    lines = _read_file_lines(str(p))
    if lines is None:
        return ''

    if start_line < 1:
        start_line = 1
    if end_line <= 0 or end_line > len(lines):
        end_line = len(lines)

    fragment = ''.join(lines[start_line - 1:end_line])
    if max_chars and len(fragment) > max_chars:
        fragment = fragment[:max_chars]
    return fragment


def clear_cache() -> None:
    """Сброс LRU-кеша (для тестов или после переиндексации)."""
    _read_file_lines.cache_clear()


def code_for_node(
    node: dict,
    *,
    repos_dir: Optional[Path] = None,
    max_chars: Optional[int] = None,
) -> str:
    """
    Унифицированное получение кода для node-словаря.

    Порядок:
    1. Если в node-словаре уже есть непустое поле ``code`` (это Weaviate-результат
       со старой записью или enriched-выдача) — возвращаем его.
    2. Иначе читаем с диска через :func:`read_code` по
       ``file_path`` + ``start_line``/``end_line``.

    ``repos_dir``: корень всех репозиториев. Если не задан — берётся из
    переменной окружения ``RAG_REPOS_DIR``. ID узла должен начинаться с
    ``"repo:<name>:..."``, тогда абсолютный путь = ``repos_dir / <name> / file_path``.
    """
    existing = node.get('code') or ''
    if existing:
        return existing

    file_path = node.get('file_path') or node.get('file') or ''
    if not file_path:
        return ''

    start = int(node.get('start_line') or node.get('line') or 0)
    end = int(node.get('end_line') or 0)

    if repos_dir is None:
        import os as _os
        env = _os.environ.get('RAG_REPOS_DIR')
        if env:
            repos_dir = Path(env)

    repo_root: Optional[Path] = None
    node_id = node.get('id') or node.get('node_id') or ''
    if repos_dir is not None and node_id.startswith('repo:'):
        try:
            repo_name = node_id.split(':', 2)[1]
            repo_root = repos_dir / repo_name
        except Exception:
            repo_root = None

    return read_code(file_path, start, end, repo_root=repo_root, max_chars=max_chars)
