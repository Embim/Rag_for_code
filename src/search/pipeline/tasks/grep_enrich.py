"""
GrepEnrichTask — третий источник контекста для LLM: точный текстовый поиск
по локальным репозиториям через ``ripgrep`` (с fallback на чистый Python).

Идея: vector search ищет по семантической близости, cypher — по структуре
графа, а grep гарантированно находит **все** места, где встречается конкретный
идентификатор (например, ``net_pos``, ``risk_full_by_book``). Это закрывает
кейс "как считается X / где определяется Y", где имя — точное.

Алгоритм:
1. Из текущего ``query`` извлекаются "code-like" токены: snake_case
   (``net_pos``, ``risk_full_by_book``), camelCase (``getPosition``),
   CONST_CASE (``BASE_URL``). Минимум 4 chars, чтобы не зацепить ``id``/``pk``.
2. Если токенов нет — задача тихо завершается (нет идентификаторов = нечего
   грепать; vector search и так справится).
3. Для каждого токена запускается ``rg --fixed-strings -t py -t ts ...`` по
   ``repos_dir`` с лимитами (``max_count_per_file``, ``MAX_HITS_PER_TOKEN``).
4. Вокруг каждого хита читается ±10 строк через ``code_loader.read_code``
   (LRU-кеш на 256 файлов).
5. Чанки складываются в ``context['context']`` со ``source='grep'``.

Repeatable: вписывается в RAG-loop рядом с ``CypherEnrichTask``. Дешёвый —
ripgrep по ~50K LOC отрабатывает за 50-200 мс, без LLM-вызовов.
"""

import json as _json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from src.core.graph.code_loader import read_code
from src.infra.logger import get_logger
from ..base.task import SearchTask

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Identifier extraction
# ---------------------------------------------------------------------------

# snake_case: ≥2 слова через underscore (отсекает `id`, `pk`, etc).
# Пример матчей: ``net_pos``, ``risk_full_by_book``, ``buy_qty``.
_SNAKE_RE = re.compile(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b")

# camelCase / PascalCase: ≥2 слова, минимум один переход регистра.
# Пример: ``getPosition``, ``CheckoutHandler``, ``RiskByBook``.
_CAMEL_RE = re.compile(
    r"\b(?:[a-z]+(?:[A-Z][a-z0-9]+)+|[A-Z][a-z0-9]+(?:[A-Z][a-z0-9]+)+)\b"
)

# CONST_CASE: ALL_CAPS_WITH_UNDERSCORES, ≥2 слова.
# Пример: ``BASE_URL``, ``MAX_RETRIES``, ``DEFAULT_QUALITY_THRESHOLD``.
_CONST_RE = re.compile(r"\b[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+\b")

# Стопворды (false-positive в виде "self.something" из NLP-вопросов).
_STOPWORDS = frozenset({
    'self', 'cls', 'true', 'false', 'none', 'null', 'undefined',
})


def _extract_identifiers(query: str, *, max_tokens: int = 5) -> List[str]:
    """
    Достать из NL-вопроса до ``max_tokens`` идентификаторов кода.

    Возвращает отсортированный список (детерминированно). Если ничего не
    нашлось — пустой список.
    """
    found: Set[str] = set()
    for rx in (_SNAKE_RE, _CAMEL_RE, _CONST_RE):
        for m in rx.finditer(query):
            tok = m.group()
            if len(tok) >= 4 and tok.lower() not in _STOPWORDS:
                found.add(tok)
                if len(found) >= max_tokens:
                    return sorted(found)
    return sorted(found)


# ---------------------------------------------------------------------------
# Ripgrep wrapper (with python fallback)
# ---------------------------------------------------------------------------

_RG_PATH: Optional[str] = None  # None = не разрешали; '' = не найден; иначе — путь
_RG_RESOLVED: bool = False


def _bundled_rg_candidates() -> List[Path]:
    """
    Кросс-платформенный список типичных мест, где может лежать ``rg``,
    помимо стандартного ``PATH``. ``shutil.which`` не находит bundled-rg
    из VSCode/Cursor (они держат его в своих node_modules), но нам не
    хочется требовать установки системного ripgrep — поэтому ищем сами.
    """
    home = Path.home()
    sys_paths: List[Path] = []

    if sys.platform.startswith('win'):
        rg = 'rg.exe'
        sys_paths += [
            Path(r"C:\Program Files\cursor\resources\app\node_modules\@vscode\ripgrep\bin") / rg,
            home / r"AppData\Local\Programs\cursor\resources\app\node_modules\@vscode\ripgrep\bin" / rg,
            home / r"AppData\Local\Programs\Microsoft VS Code\resources\app\node_modules\@vscode\ripgrep\bin" / rg,
            Path(r"C:\Program Files\Microsoft VS Code\resources\app\node_modules\@vscode\ripgrep\bin") / rg,
            home / r".cargo\bin" / rg,
            home / r"scoop\apps\ripgrep\current" / rg,
            Path(r"C:\Program Files\ripgrep") / rg,
        ]
    elif sys.platform == 'darwin':  # macOS
        rg = 'rg'
        sys_paths += [
            Path('/Applications/Cursor.app/Contents/Resources/app/node_modules/@vscode/ripgrep/bin') / rg,
            Path('/Applications/Visual Studio Code.app/Contents/Resources/app/node_modules/@vscode/ripgrep/bin') / rg,
            home / 'Applications/Cursor.app/Contents/Resources/app/node_modules/@vscode/ripgrep/bin' / rg,
            home / '.cargo/bin' / rg,
            Path('/opt/homebrew/bin') / rg,
            Path('/usr/local/bin') / rg,
        ]
    else:  # linux / *bsd
        rg = 'rg'
        sys_paths += [
            # Системные установки
            Path('/usr/bin') / rg,
            Path('/usr/local/bin') / rg,
            Path('/snap/bin') / rg,
            home / '.cargo/bin' / rg,
            home / '.local/bin' / rg,
            # Cursor — deb/native установка
            Path('/opt/cursor/resources/app/node_modules/@vscode/ripgrep/bin') / rg,
            Path('/usr/share/cursor/resources/app/node_modules/@vscode/ripgrep/bin') / rg,
            # VSCode — deb/snap/официальные пакеты
            Path('/usr/share/code/resources/app/node_modules/@vscode/ripgrep/bin') / rg,
            Path('/snap/code/current/usr/share/code/resources/app/node_modules/@vscode/ripgrep/bin') / rg,
            Path('/opt/visual-studio-code/resources/app/node_modules/@vscode/ripgrep/bin') / rg,
            # Remote SSH (VSCode/Cursor server)
            home / '.vscode-server/bin' / rg,
            home / '.cursor-server/bin' / rg,
        ]

    return sys_paths


def _resolve_rg_path() -> Optional[str]:
    """
    Найти исполняемый файл ``rg``. Кешируется на процесс.

    Порядок поиска (кросс-платформенный — Windows / macOS / Linux):
    1. env ``RAG_RG_PATH`` — явный override.
    2. ``shutil.which('rg')`` — стандартный PATH (на Linux обычно достаточно
       ``apt install ripgrep`` / ``brew install ripgrep`` / ``cargo install``).
    3. Типичные bundled-локации (VSCode/Cursor встраивают ripgrep).
    """
    global _RG_PATH, _RG_RESOLVED
    if _RG_RESOLVED:
        return _RG_PATH or None

    candidate: Optional[str] = None

    env_path = os.environ.get('RAG_RG_PATH')
    if env_path and os.path.isfile(env_path):
        candidate = env_path
    else:
        which = shutil.which('rg')
        if which:
            candidate = which
        else:
            for p in _bundled_rg_candidates():
                if p.exists():
                    candidate = str(p)
                    break

    _RG_PATH = candidate or ''
    _RG_RESOLVED = True
    if candidate:
        logger.info(f"[grep] using ripgrep at {candidate}")
    else:
        logger.info("[grep] ripgrep not found — using Python fallback (slower)")
    return candidate


def _has_ripgrep() -> bool:
    return bool(_resolve_rg_path())


def _ripgrep(
    token: str,
    roots: List[Path],
    *,
    max_count_per_file: int = 3,
    max_total: int = 30,
    timeout_seconds: int = 15,
) -> List[Dict[str, Any]]:
    """
    Запускает ``rg --fixed-strings`` на нескольких ``roots`` за один subprocess.

    Поддержка списка корней: ``ripgrep`` принимает любое число path-аргументов,
    поэтому для 40 индексированных репо это всё равно один вызов на токен,
    не 40.

    Возвращает список ``{path, line, content}``, **уже диверсифицированный
    по файлам**: сначала первый хит из каждого файла, потом второй, и т.д.

    Тихо отдаёт пустой список при таймауте/ошибке rg, чтобы не ронять pipeline.
    """
    valid_roots = [r for r in roots if r.exists() and r.is_dir()]
    if not valid_roots:
        return []

    rg_path = _resolve_rg_path()
    if not rg_path:
        return []

    # --glob (а не --type) — bundled rg от VSCode/Cursor может не знать новые
    # типы вроде 'tsx'. Globs универсальнее и кросс-платформенно стабильнее.
    # --json — формат вывода, идентичный на Windows/macOS/Linux. Решает
    # проблему с двоеточием в Windows-путях ("C:\foo\bar.py:42:...") при
    # построчном парсинге.
    cmd = [
        rg_path,
        '--json',
        '--max-count', str(max_count_per_file),
        '--fixed-strings',
        # Cap размера читаемых файлов: минификаты, lock'и, сгенерированные
        # bundle'ы могут весить десятки MB и забивать поиск шумом + DoS-риск.
        # 10MB перекрывает любой нормальный исходник на порядок.
        '--max-filesize', '10M',
        '--glob', '*.py',
        '--glob', '*.js',
        '--glob', '*.jsx',
        '--glob', '*.ts',
        '--glob', '*.tsx',
        # Игнорируем виртуальные окружения и build-артефакты.
        '--glob', '!.venv/**',
        '--glob', '!venv/**',
        '--glob', '!node_modules/**',
        '--glob', '!__pycache__/**',
        '--glob', '!dist/**',
        '--glob', '!build/**',
        token,
    ]
    cmd.extend(str(r) for r in valid_roots)
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            encoding='utf-8',
            errors='replace',
        )
    except subprocess.TimeoutExpired:
        logger.warning(f"[grep] ripgrep timeout for token={token!r}")
        return []
    except Exception as e:
        logger.warning(f"[grep] ripgrep failed for token={token!r}: {e}")
        return []

    # rg exit code 1 = no matches (нормально), 2 = error.
    if proc.returncode not in (0, 1):
        logger.warning(
            f"[grep] rg exit={proc.returncode} stderr={proc.stderr[:200]}"
        )
        return []

    # JSON-stream: одна JSON-строка на событие. Берём только type=='match'.
    # Структура: {"type":"match","data":{"path":{"text":"..."},
    #             "line_number":42,"lines":{"text":"...\n"}, ...}}
    # Это надёжно работает с любыми путями (Windows drive letters, пробелы,
    # юникод) и не зависит от платформы.
    by_file: Dict[str, List[Dict[str, Any]]] = {}
    for line in proc.stdout.splitlines():
        if not line.strip():
            continue
        try:
            evt = _json.loads(line)
        except _json.JSONDecodeError:
            continue
        if evt.get('type') != 'match':
            continue
        data = evt.get('data') or {}
        path_obj = data.get('path') or {}
        path_text = path_obj.get('text')
        if not path_text:
            # rg может выдать base64 (--bytes) если путь не UTF-8. Скипаем
            # такие файлы — почти не встречаются в нормальных репах.
            continue
        ln_int = data.get('line_number')
        if not isinstance(ln_int, int):
            continue
        lines_obj = data.get('lines') or {}
        # rstrip убирает и LF, и CRLF — корректно для Windows-репов.
        content = (lines_obj.get('text') or '').rstrip('\r\n')
        by_file.setdefault(path_text, []).append({
            'path': path_text, 'line': ln_int, 'content': content,
        })

    return _diversify_hits(by_file, max_total=max_total)


def _diversify_hits(
    by_file: Dict[str, List[Dict[str, Any]]],
    *,
    max_total: int,
) -> List[Dict[str, Any]]:
    """
    Взвешенное распределение хитов по файлам.

    **Не** строгий round-robin (он давал по 1 хиту на файл и терял плотные
    источники, где токен встречается десятки раз — например ``position_by_
    book_service.py`` с 30+ упоминаниями ``net_pos`` отдавал 1 хит на инит-
    словарь, а формулы оставались за бортом).

    Стратегия: каждому файлу выделяется квота
        ``min(len(file_hits), max(1, ceil(max_total / sqrt(num_files))))``
    — то есть «болтливые» файлы могут отдать заметно больше 1, но всё равно
    не съедают весь бюджет. После этого собираем round-robin по «слотам»:
    первый хит из каждого файла, второй из каждого, и т.д. — пока не упрёмся
    в индивидуальную квоту или общий ``max_total``.
    """
    import math

    if not by_file:
        return []

    n = len(by_file)
    base_quota = max(1, math.ceil(max_total / math.sqrt(n)))

    quotas: Dict[str, int] = {
        p: min(len(hits), base_quota) for p, hits in by_file.items()
    }

    diversified: List[Dict[str, Any]] = []
    max_q = max(quotas.values()) if quotas else 0
    for i in range(max_q):
        for path, hits in by_file.items():
            if i >= quotas[path]:
                continue
            diversified.append(hits[i])
            if len(diversified) >= max_total:
                return diversified
    return diversified


_FALLBACK_EXTENSIONS = ('.py', '.js', '.jsx', '.ts', '.tsx')
_FALLBACK_SKIP_DIRS = frozenset({
    '.venv', 'venv', 'node_modules', '__pycache__', '.git', 'dist',
    'build', '.next', '.tox', '.pytest_cache', '.mypy_cache',
})
# Файлы крупнее этого размера пропускаем — это бандлы / минификаты /
# lock-файлы, которые забивают grep шумом и тянут DoS-риск.
_FALLBACK_MAX_FILESIZE = 10 * 1024 * 1024  # 10 MB


def _python_fallback_grep(
    token: str,
    roots: List[Path],
    *,
    max_count_per_file: int = 5,
    max_total: int = 30,
) -> List[Dict[str, Any]]:
    """
    Чистый Python обход на случай отсутствия ripgrep. Медленнее (~10x),
    но работает везде. Пропускает .venv/node_modules/etc.

    Принимает несколько корней — поведение ровно как у новой ``_ripgrep``.

    ``max_count_per_file`` — ограничение на хиты в одном файле, чтобы один
    "болтливый" файл не съел весь лимит ``max_total`` и мы получили
    разнообразие источников.
    """
    valid_roots = [r for r in roots if r.exists() and r.is_dir()]
    if not valid_roots:
        return []

    # Сначала собираем по файлам (как в _ripgrep), потом round-robin
    # диверсификация — иначе алфавитный обход съест max_total на первых
    # файлах и пропустит важные источники из глубины проекта.
    by_file: Dict[str, List[Dict[str, Any]]] = {}
    for root in valid_roots:
        for fp in root.rglob('*'):
            if not fp.is_file() or fp.suffix not in _FALLBACK_EXTENSIONS:
                continue
            if _FALLBACK_SKIP_DIRS & set(fp.parts):
                continue
            try:
                # Cap на размер файла — не лезем в bundle.min.js / lock-файлы.
                if fp.stat().st_size > _FALLBACK_MAX_FILESIZE:
                    continue
            except OSError:
                continue
            try:
                with open(fp, 'r', encoding='utf-8', errors='replace') as f:
                    file_hits: List[Dict[str, Any]] = []
                    for i, line in enumerate(f, 1):
                        if token in line:
                            file_hits.append({
                                'path': str(fp),
                                'line': i,
                                'content': line.rstrip('\r\n'),
                            })
                            if len(file_hits) >= max_count_per_file:
                                break
                    if file_hits:
                        by_file[str(fp)] = file_hits
            except Exception:
                continue

    return _diversify_hits(by_file, max_total=max_total)


# ---------------------------------------------------------------------------
# Repos dir resolution
# ---------------------------------------------------------------------------

def _resolve_repo_roots(executor: Any) -> List[Path]:
    """
    Достать список абсолютных корней всех индексированных репо.

    Источники в порядке приоритета:
    1) ``executor.repo_paths`` — ``RepoPathResolver`` (читает Neo4j +
       env override). Это правильный путь — путь хранится на ``Repository``
       ноде и поддерживает 40+ независимых репо.
    2) Legacy: ``executor.retriever.weaviate.repos_dir`` (один корень).
    3) Legacy: env ``RAG_REPOS_DIR`` (один корень).

    Возвращает список (≥0 путей). Пустой → таска тихо завершится.
    """
    # 1) RepoPathResolver — правильный путь.
    resolver = getattr(executor, "repo_paths", None)
    if resolver is not None:
        try:
            roots = resolver.list_roots()
            if roots:
                return roots
        except Exception as e:
            logger.warning(f"[grep] resolver.list_roots() failed: {e}")

    # 2) Fallback: один общий корень из weaviate-индексера.
    try:
        rd = executor.retriever.weaviate.repos_dir
        if rd:
            return [Path(rd)]
    except AttributeError:
        pass

    # 3) Fallback: env.
    env = os.environ.get('RAG_REPOS_DIR')
    if env:
        return [Path(env)]
    return []


# ---------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------

class GrepEnrichTask(SearchTask):
    """
    Грепает идентификаторы из ``current_query`` по локальным репозиториям и
    добавляет окружения ±10 строк как chunks ``source='grep'``.

    Если в запросе нет code-like токенов — таск тихо завершается без эффекта.
    Если нет ``repos_dir`` — таск также завершается тихо с warning.

    context['context']     ← extend with grep-chunks
    context['grep_tokens'] ← список найденных идентификаторов
    context['grep_hits']   ← число хитов всего
    """

    dependencies = []

    # Конфигурация (можно вынести в AgentConfig позже).
    CONTEXT_LINES = 10         # ±N строк вокруг каждого хита (через read_code)
    MAX_TOKENS = 5             # сколько идентификаторов вытаскиваем из query
    # Лимиты подобраны так, чтобы "плотные" файлы (где токен встречается
    # десятки раз) могли отдать достаточно разных хитов. Старые значения
    # (3 / 20) теряли формулы из таких файлов: один init-словарь съедал
    # слот, остальные функции не попадали в контекст. См. _diversify_hits.
    MAX_HITS_PER_TOKEN = 60    # ёмкость на один токен ПОСЛЕ диверсификации
    MAX_COUNT_PER_FILE = 10    # rg/fallback: не более N хитов на файл

    def run(self, filters: Dict[str, Any]) -> None:
        # ВАЖНО: грепаем по ОРИГИНАЛЬНОМУ запросу пользователя, а не по
        # ``current_query`` (rewrite). Идентификаторы кода стабильны:
        # пользователь спросил про ``net_pos`` → его и грепаем. Если LLM в
        # RewriteQueryTask добавит "past_cash, pnl, mv" — это уведёт grep
        # в шум, Sources раздуется до сотен файлов. Стабильность важнее.
        original_query = filters["query"]

        tokens = _extract_identifiers(original_query, max_tokens=self.MAX_TOKENS)
        self.context["grep_tokens"] = tokens
        if not tokens:
            self.context["grep_hits"] = self.context.get("grep_hits", 0)
            return

        repo_roots = _resolve_repo_roots(self.executor)
        if not repo_roots:
            logger.warning(
                "[grep] no repository roots resolved. "
                "Either index repos (RepositoryNode.local_path will populate "
                "RepoPathResolver) or set RAG_REPOS_DIR."
            )
            self.context["grep_hits"] = 0
            return

        # Дедуп: grep работает по ОРИГИНАЛЬНОМУ query (см. выше), а тот по
        # определению не меняется между итерациями RAG-loop'а. Поэтому
        # после первого прохода все последующие итерации мгновенно
        # завершаются — token-set уже искали, контекст в ``self.context``
        # уже накоплен.
        if self.context.get("_grep_last_query") == original_query:
            return
        self.context["_grep_last_query"] = original_query

        # Грепаем по каждому токену по ВСЕМ корням сразу (один subprocess
        # на токен — ripgrep принимает несколько path-аргументов).
        all_hits: List[Dict[str, Any]] = []
        for tok in tokens:
            if _has_ripgrep():
                hits = _ripgrep(
                    tok, repo_roots,
                    max_count_per_file=self.MAX_COUNT_PER_FILE,
                    max_total=self.MAX_HITS_PER_TOKEN,
                )
            else:
                hits = _python_fallback_grep(
                    tok, repo_roots,
                    max_count_per_file=self.MAX_COUNT_PER_FILE,
                    max_total=self.MAX_HITS_PER_TOKEN,
                )
            for h in hits:
                h['token'] = tok
            all_hits.extend(hits)

        if not all_hits:
            self.context["grep_hits"] = 0
            logger.info(f"[grep] no hits for tokens {tokens}")
            return

        # Резолвим relpath: ищем у какого из repo_roots путь является потомком.
        # Префиксы кешируем — для 40 репо это всё равно дёшево.
        resolved_roots = []
        for r in repo_roots:
            try:
                resolved_roots.append(r.resolve())
            except OSError:
                resolved_roots.append(r)

        def _to_rel(abs_path: str) -> str:
            try:
                p = Path(abs_path).resolve()
            except OSError:
                return abs_path
            for r in resolved_roots:
                try:
                    return f"{r.name}/" + str(p.relative_to(r)).replace('\\', '/')
                except ValueError:
                    continue
            return str(p).replace('\\', '/')

        def _matching_root(abs_path: str) -> Optional[Path]:
            try:
                p = Path(abs_path).resolve()
            except OSError:
                return None
            for r in resolved_roots:
                try:
                    p.relative_to(r)
                    return r
                except ValueError:
                    continue
            return None

        # Дедуп по (path, line) — разные токены могут найти одну строку.
        seen_lines = set()
        chunks: List[Dict[str, Any]] = []
        for h in all_hits:
            key = (h['path'], h['line'])
            if key in seen_lines:
                continue
            seen_lines.add(key)

            # ±N строк контекста через LRU-кешированный read_code.
            # repo_root = тот корень, под которым реально лежит этот файл —
            # активирует path-traversal guard внутри read_code и для любого
            # из нескольких репо корректно.
            matched_root = _matching_root(h['path'])
            ctx = read_code(
                h['path'],
                start_line=max(1, h['line'] - self.CONTEXT_LINES),
                end_line=h['line'] + self.CONTEXT_LINES,
                repo_root=matched_root,
            ) or h['content']

            rel = _to_rel(h['path'])

            chunks.append({
                'id': '',
                'name': f"{rel}:{h['line']} ({h['token']})",
                'type': 'GrepHit',
                'file': rel,
                'line': h['line'],
                'code': ctx,
                'score': 0.95,  # высокий: точное литеральное вхождение
                'source': 'grep',
                'token': h['token'],
            })

        existing: List[Dict[str, Any]] = self.context.get("context") or []
        existing.extend(chunks)
        self.context["context"] = existing
        self.context["grep_hits"] = len(chunks)

        logger.info(
            f"[grep] tokens={tokens} → +{len(chunks)} chunks "
            f"(total context now {len(existing)})"
        )

    @staticmethod
    def trace_input(context, filters):
        return {
            "query": context.get("current_query") or filters.get("query"),
        }

    @staticmethod
    def trace_output(context):
        return {
            "tokens": context.get("grep_tokens", []),
            "hits": context.get("grep_hits", 0),
        }
