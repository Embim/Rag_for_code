"""
GraphExpandTask — четвёртый источник контекста: целенаправленный обход графа
от уже найденных "якорей".

Идея: после того как Weaviate вернул primary chunks с конкретными ``node_id``,
мы знаем точки входа. От них за один cypher на нодуполучаем структурный контекст
который embedding/grep дать не могут:

- ``Class``    → CONTAINS Method (все методы класса — даёт полную картину
                 без обрезки большого class-чанка)
- ``Method``   → CALLS callee (downstream — кого использует)
- ``Function`` → CALLS callee + ⟵CALLS caller (1-hop в обе стороны)
- ``Endpoint`` → HANDLES_REQUEST→Function, USES_MODEL→Model
- ``Model``    → FOREIGN_KEY/MANY_TO_MANY/ONE_TO_ONE → Model (схема данных)

Каждая найденная нода превращается в chunk ``source='graph'`` с пометкой
``relationship`` (CONTAINS/CALLS/HANDLES_REQUEST/etc) и ``anchor`` (имя
исходного чанка). Код подтягивается через ``code_loader.read_code``.

Repeatable + дедуп по оригинальному query — как у ``CypherEnrichTask`` и
``GrepEnrichTask``: структура графа не меняется между итерациями RAG-loop'а,
поэтому повторный обход бесполезен.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from src.core.graph.code_loader import read_code
from src.infra.logger import get_logger
from ..base.task import SearchTask

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Cypher patterns per node type
# ---------------------------------------------------------------------------

# Class → его методы (CONTAINS). Покрывает кейс "достали класс, нужны его методы".
_CYPHER_CLASS_METHODS = """
MATCH (cls {id: $node_id})-[:CONTAINS]->(m)
WHERE (m:Method OR m:Function)
RETURN m AS node, 'CONTAINS' AS rel
ORDER BY m.start_line LIMIT $limit
"""

# Method → ВЛАДЕЮЩИЙ КЛАСС + ВСЕ ЕГО МЕТОДЫ (reverse CONTAINS + sibling expansion).
# Покрывает критический кейс: retrieval вернул один метод класса (например
# ``semi_result_cash_transfers_by_book``), но не сам класс. Без этого
# обхода мы видим один метод изолированно — а в реальности там 18 методов
# для разных типов инструментов (bonds/structured/repo/autocall/...) и
# вопрос "как считается X" требует увидеть их ВСЕ.
#
# Возвращает: владелец-класс + все его методы (с дедупом по якорному id).
_CYPHER_METHOD_OWNER_AND_SIBLINGS = """
MATCH (cls:Class)-[:CONTAINS]->(self {id: $node_id})
MATCH (cls)-[:CONTAINS]->(sibling)
WHERE (sibling:Method OR sibling:Function)
RETURN sibling AS node, 'SIBLING_OF' AS rel
ORDER BY sibling.start_line LIMIT $limit
"""

# Method/Function → кого вызывает (downstream).
_CYPHER_CALLEES = """
MATCH (src {id: $node_id})-[:CALLS]->(callee)
WHERE callee:Method OR callee:Function
RETURN callee AS node, 'CALLS' AS rel
LIMIT $limit
"""

# Multi-hop CALLS: цепочки до глубины 3. Используется для Function/Method
# якорей — даёт LLM не только прямых callees, а полное "дерево вызовов"
# с реальными helper'ами на которые опирается логика.
# Возвращает список путей; обработчик строит из них adjacency + ASCII-tree
# и выбирает top-N нод для дотаскивания кода.
_CYPHER_CALLS_TREE = """
MATCH path = (start {id: $node_id})-[:CALLS*1..3]->(target)
WHERE start <> target
WITH nodes(path) AS chain, length(path) AS depth
ORDER BY depth ASC
LIMIT $limit
RETURN [n IN chain | {
    id: n.id,
    name: n.name,
    type: head([l IN labels(n) WHERE l <> 'GraphNode']),
    file_path: n.file_path,
    start_line: n.start_line,
    end_line: n.end_line
}] AS chain, depth
"""

# Function → кто вызывает (upstream). Для Method обычно меньше полезно
# (вызовы through self.x() парсер часто не резолвит), но для top-level
# функций даёт callers.
_CYPHER_CALLERS = """
MATCH (caller)-[:CALLS]->(target {id: $node_id})
WHERE caller:Method OR caller:Function
RETURN caller AS node, 'CALLED_BY' AS rel
LIMIT $limit
"""

# Endpoint → handler-функция и используемые модели.
_CYPHER_ENDPOINT = """
MATCH (e:Endpoint {id: $node_id})
OPTIONAL MATCH (e)-[:HANDLES_REQUEST]->(h)
OPTIONAL MATCH (e)-[:USES_MODEL]->(m:Model)
WITH collect(DISTINCT {node: h, rel: 'HANDLES_REQUEST'}) AS handlers,
     collect(DISTINCT {node: m, rel: 'USES_MODEL'}) AS models
UNWIND (handlers + models) AS pair
WITH pair WHERE pair.node IS NOT NULL
RETURN pair.node AS node, pair.rel AS rel
LIMIT $limit
"""

# Model → соседи через FK/M2M/O2O (схема данных).
_CYPHER_MODEL_NEIGHBORS = """
MATCH (src:Model {id: $node_id})-[r:FOREIGN_KEY|MANY_TO_MANY|ONE_TO_ONE]->(other:Model)
RETURN other AS node, type(r) AS rel
LIMIT $limit
"""

# Component → API calls и роуты (для UI-якорей).
_CYPHER_COMPONENT = """
MATCH (c:Component {id: $node_id})
OPTIONAL MATCH (c)-[:MAKES_CALL]->(api:ApiCall)-[:CALLS_ENDPOINT]->(ep:Endpoint)
OPTIONAL MATCH (c)-[:RENDERS_AT]->(rt:Route)
WITH collect(DISTINCT {node: ep, rel: 'CALLS_ENDPOINT'}) AS endpoints,
     collect(DISTINCT {node: rt, rel: 'RENDERS_AT'}) AS routes
UNWIND (endpoints + routes) AS pair
WITH pair WHERE pair.node IS NOT NULL
RETURN pair.node AS node, pair.rel AS rel
LIMIT $limit
"""


# Node-type → (cypher, лимит). Цифровые лимиты подобраны эмпирически,
# можно вынести в AgentConfig позже.
#
# Function/Method НЕ в этой таблице — для них multi-hop CALLS-tree
# обрабатывается отдельным путём (см. ``_expand_calls_tree``), который
# возвращает не плоский список callees а полное дерево вызовов.
_EXPANSIONS: List[Tuple[str, str, int]] = [
    # (anchor_node_type, cypher, limit)
    ('Class', _CYPHER_CLASS_METHODS, 25),
    ('Endpoint', _CYPHER_ENDPOINT, 6),
    ('Model', _CYPHER_MODEL_NEIGHBORS, 8),
    ('Component', _CYPHER_COMPONENT, 6),
]


def _expand_calls_tree(
    anchor: Dict[str, Any],
    neo4j: Any,
    *,
    repos_dir: Optional[Path] = None,
    repo_paths: Optional[Any] = None,
    existing_ids: Set[str],
) -> List[Dict[str, Any]]:
    """
    Multi-hop расширение для якоря-Function/Method.

    Возвращает список chunks:
    - 1 ``CallTree`` chunk с ASCII-деревом структуры вызовов (компактно,
      покрывает до 40 узлов).
    - До ``_CALLS_TREE_TOP_N`` обычных graph chunks с ПОЛНЫМ кодом узловых
      callees (приоритет: depth=1, затем helper'ы с высоким fan_in).

    Двухуровневая структура решает trade-off "глубина vs размер контекста":
    LLM видит и архитектуру (tree), и реализацию (top-N кодом), без
    раздувания на 50+ chunks с кодом.
    """
    anchor_id = anchor.get('id') or ''
    if not anchor_id:
        return []

    try:
        rows = neo4j.execute_cypher(
            _CYPHER_CALLS_TREE,
            parameters={'node_id': anchor_id, 'limit': _CALLS_TREE_PATHS},
        )
    except Exception as e:
        logger.warning(
            f"[graph_expand] calls_tree cypher failed for {anchor_id}: {e}"
        )
        return []

    if not rows:
        logger.info(
            f"[graph_expand] anchor={anchor.get('type')} "
            f"id={anchor_id[:80]} -> 0 paths (no callees)"
        )
        return []

    adj, nodes_index, fan_in = _parse_chains(rows)
    total_nodes = len(nodes_index)
    direct_callees = len(adj.get(anchor_id, set()))

    # 1. ASCII-tree как один служебный chunk.
    tree_str = _render_call_tree_ascii(
        anchor_id=anchor_id,
        anchor_name=anchor.get('name', ''),
        adj=adj,
        nodes_index=nodes_index,
    )
    tree_chunk = {
        'id': f"calltree:{anchor_id}",  # уникальный ID, не пересечётся с node id
        'name': f"Call tree from {anchor.get('name', '')}",
        'type': 'CallTree',
        'file': anchor.get('file', ''),
        'line': anchor.get('line'),
        'code': tree_str,
        'score': 0.9,
        'source': 'graph',
        'relationship': 'CALLS_TREE',
        'anchor': anchor.get('name', ''),
    }
    chunks: List[Dict[str, Any]] = [tree_chunk]

    # 2. Top-N узловых callees — с полным кодом.
    top_nodes = _pick_top_nodes(
        anchor_id=anchor_id,
        adj=adj,
        nodes_index=nodes_index,
        fan_in=fan_in,
        max_n=_CALLS_TREE_TOP_N,
    )
    skipped_existing = 0
    for node_data in top_nodes:
        nid = node_data.get('id') or ''
        if not nid or nid in existing_ids:
            skipped_existing += 1
            continue
        chunk = _node_to_graph_chunk(
            node_data, 'CALLS', anchor,
            repos_dir=repos_dir, repo_paths=repo_paths,
        )
        if chunk is None:
            continue
        existing_ids.add(nid)
        chunks.append(chunk)

    logger.info(
        f"[graph_expand] anchor={anchor.get('type')} id={anchor_id[:80]} "
        f"-> tree({total_nodes} nodes, {direct_callees} direct), "
        f"+{len(chunks) - 1} code chunks "
        f"(skipped {skipped_existing} already-known)"
    )
    return chunks


# Лимит sibling-методов на якорь — крупный класс типа PositionByBookService
# имеет 18 методов. Берём все, но не более 25 (защита от случайных мега-классов).
_SIBLINGS_LIMIT = 25


def _expand_owner_siblings(
    anchor: Dict[str, Any],
    neo4j: Any,
    *,
    repos_dir: Optional[Path] = None,
    repo_paths: Optional[Any] = None,
    existing_ids: Set[str],
) -> List[Dict[str, Any]]:
    """
    Reverse-CONTAINS expansion для якоря-Method.

    Цель: если retrieval вытащил один метод класса (например
    ``semi_result_cash_transfers_by_book``), но не сам класс, мы всё
    равно достаём ВСЕ методы владеющего класса. Для запросов типа
    "как считается X" это критично: один метод изолированно не отвечает,
    нужны все sibling'и где X вычисляется по-разному (для разных типов
    инструментов).

    Возвращает список chunks с полным кодом каждого sibling-метода.
    Дедуп по ``existing_ids`` (не таскаем то что уже есть).
    """
    anchor_id = anchor.get('id') or ''
    if not anchor_id:
        return []

    try:
        rows = neo4j.execute_cypher(
            _CYPHER_METHOD_OWNER_AND_SIBLINGS,
            parameters={'node_id': anchor_id, 'limit': _SIBLINGS_LIMIT},
        )
    except Exception as e:
        logger.warning(
            f"[graph_expand] siblings cypher failed for {anchor_id}: {e}"
        )
        return []

    if not rows:
        return []

    chunks: List[Dict[str, Any]] = []
    skipped_existing = 0
    for row in rows:
        raw = row.get('node')
        if not raw:
            continue
        node_data = dict(raw)
        nid = node_data.get('id') or ''
        if not nid or nid == anchor_id or nid in existing_ids:
            skipped_existing += 1
            continue
        # SIBLING_OF — отдельный rel-тип, но truncate берём как для CONTAINS
        # (это и есть методы класса, могут быть крупные).
        chunk = _node_to_graph_chunk(
            node_data, 'CONTAINS', anchor,
            repos_dir=repos_dir, repo_paths=repo_paths,
        )
        if chunk is None:
            continue
        # Помечаем как SIBLING_OF — LLM поймёт что это однопорядковые методы.
        chunk['relationship'] = 'SIBLING_OF'
        existing_ids.add(nid)
        chunks.append(chunk)

    logger.info(
        f"[graph_expand] siblings of {anchor_id[:80]} -> +{len(chunks)} "
        f"(skipped {skipped_existing} already-known)"
    )
    return chunks


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Лимиты на размер кода graph-chunk'а. Раньше 2000 (≈ 60 LOC), но это
# слишком мало для крупных методов класса (например
# ``_get_position_bond_by_book`` — 561 LOC, формула ``net_pos += quantity``
# в середине, начало = init-словарь). LLM получал обрезанный фрагмент без
# формулы расчёта.
#
# CONTAINS-методы (методы класса от якоря-Class) — могут быть очень крупные,
# им даём больший лимит. Callees (CALLS) обычно мельче — 60-100 LOC.
_GRAPH_CODE_TRUNCATE_DEFAULT = 2000   # для CALLS callees
_GRAPH_CODE_TRUNCATE_CONTAINS = 12000  # для методов класса (CONTAINS)


def _truncate_for(rel: str) -> int:
    return _GRAPH_CODE_TRUNCATE_CONTAINS if rel == 'CONTAINS' else _GRAPH_CODE_TRUNCATE_DEFAULT

# Multi-hop CALLS: лимиты на размер дерева.
_CALLS_TREE_PATHS = 30        # max путей на якорь (cypher LIMIT)
_CALLS_TREE_TOP_N = 8         # max нод с полным кодом
_CALLS_TREE_ASCII_LINES = 40  # max строк ASCII-tree


def _parse_chains(rows: List[Dict[str, Any]]) -> Tuple[
    Dict[str, Set[str]], Dict[str, Dict[str, Any]], Dict[str, int]
]:
    """
    Разобрать список cypher-rows с полем ``chain`` (list of node-dicts)
    в три структуры:

    - ``adj``         {parent_id: {child_id, ...}} — DAG смежности.
    - ``nodes_index`` {id: node_data}              — все встретившиеся ноды.
    - ``fan_in``      {id: count}                  — сколько раз нода
                                                    встречалась как target.
                                                    Высокий fan_in → "узловой
                                                    helper", приоритет в
                                                    дотаскивании кода.
    """
    adj: Dict[str, Set[str]] = {}
    nodes_index: Dict[str, Dict[str, Any]] = {}
    fan_in: Dict[str, int] = {}
    for row in rows:
        chain = row.get('chain') or []
        prev_id: Optional[str] = None
        for n in chain:
            if not isinstance(n, dict):
                continue
            nid = n.get('id') or ''
            if not nid:
                prev_id = None
                continue
            nodes_index[nid] = n
            if prev_id is not None:
                adj.setdefault(prev_id, set()).add(nid)
                fan_in[nid] = fan_in.get(nid, 0) + 1
            prev_id = nid
    return adj, nodes_index, fan_in


def _render_call_tree_ascii(
    *,
    anchor_id: str,
    anchor_name: str,
    adj: Dict[str, Set[str]],
    nodes_index: Dict[str, Dict[str, Any]],
    max_lines: int = _CALLS_TREE_ASCII_LINES,
) -> str:
    """
    DFS обход дерева смежности, рендер в ASCII (как `tree` команда).

    Если дерево больше ``max_lines`` — обрезается; добавляется ``...``
    отметка чтобы LLM знал что показано не всё.
    """
    lines: List[str] = []
    visited: Set[str] = set()

    def _short(nid: str, fallback_name: str = '') -> str:
        n = nodes_index.get(nid) or {}
        nm = n.get('name') or fallback_name or nid.split(':')[-1]
        fp = n.get('file_path') or ''
        # Только имя файла, не полный путь — компактнее.
        fp_short = fp.split('/')[-1] if fp else ''
        line = n.get('start_line') or ''
        if fp_short:
            return f"{nm} ({fp_short}:{line})"
        return nm

    def _visit(nid: str, prefix: str, is_last: bool) -> None:
        if len(lines) >= max_lines:
            return
        if nid in visited:
            # Циклы не рисуем повторно — отмечаем "→ already shown".
            connector = '└─ ' if is_last else '├─ '
            lines.append(prefix + connector + _short(nid) + '  ↺')
            return
        visited.add(nid)

        connector = '└─ ' if is_last else '├─ '
        lines.append(prefix + connector + _short(nid))
        children = sorted(adj.get(nid, set()))
        next_prefix = prefix + ('   ' if is_last else '│  ')
        for i, ch in enumerate(children):
            _visit(ch, next_prefix, i == len(children) - 1)

    # Корень — якорь.
    visited.add(anchor_id)
    lines.append(_short(anchor_id, fallback_name=anchor_name))
    children = sorted(adj.get(anchor_id, set()))
    for i, ch in enumerate(children):
        _visit(ch, '', i == len(children) - 1)

    truncated = len(lines) >= max_lines
    if truncated:
        lines.append('  …(tree truncated)')
    return '\n'.join(lines)


def _pick_top_nodes(
    *,
    anchor_id: str,
    adj: Dict[str, Set[str]],
    nodes_index: Dict[str, Dict[str, Any]],
    fan_in: Dict[str, int],
    max_n: int,
) -> List[Dict[str, Any]]:
    """
    Выбрать top-N нод дерева для дотаскивания кода.

    Приоритет:
    1) прямые callees (depth=1 от якоря) — это "что делает функция";
    2) узловые helper'ы (высокий fan_in) — общие зависимости в дереве;
    3) остальные по алфавиту.

    Якорь сам не включаем — его код уже в primary chunk.
    """
    direct = list(adj.get(anchor_id, set()))

    # Все остальные ноды (исключая якорь и прямых callees) — по fan_in.
    direct_set = set(direct)
    others = [
        nid for nid in nodes_index
        if nid != anchor_id and nid not in direct_set
    ]
    others.sort(key=lambda nid: (-fan_in.get(nid, 0), nodes_index[nid].get('name', '')))

    selected_ids: List[str] = list(direct)
    for nid in others:
        if len(selected_ids) >= max_n:
            break
        selected_ids.append(nid)

    # Возвращаем node_data объекты (а не id) — caller сразу строит chunks.
    return [nodes_index[nid] for nid in selected_ids[:max_n] if nid in nodes_index]


def _node_to_graph_chunk(
    node_data: Dict[str, Any],
    rel: str,
    anchor: Dict[str, Any],
    *,
    repos_dir: Optional[Path] = None,
    repo_paths: Optional[Any] = None,
) -> Optional[Dict[str, Any]]:
    """
    Превратить узел графа в chunk ``source='graph'`` для LLM-контекста.

    Код читается с диска через ``code_loader.read_code``. Если файла нет —
    chunk всё равно создаётся, но ``code=''`` (LLM хотя бы увидит факт связи).
    """
    node_id = node_data.get('id') or ''
    file_path = node_data.get('file_path') or ''
    if not node_id and not file_path:
        return None

    # Резолв корня репо: либо через resolver (per-repo), либо общий repos_dir.
    repo_root: Optional[Path] = None
    if repo_paths is not None and node_id.startswith('repo:'):
        try:
            repo_name = node_id.split(':', 2)[1]
            repo_root = repo_paths.resolve(repo_name)
        except Exception:
            repo_root = None
    if repo_root is None and repos_dir is not None and node_id.startswith('repo:'):
        try:
            repo_name = node_id.split(':', 2)[1]
            repo_root = repos_dir / repo_name
        except Exception:
            repo_root = None

    code = ''
    if file_path:
        start = int(node_data.get('start_line') or 0)
        end = int(node_data.get('end_line') or 0)
        try:
            code = read_code(
                file_path,
                start_line=start,
                end_line=end,
                repo_root=repo_root,
                # CONTAINS-методы (тело метода класса) могут быть крупные
                # (500+ LOC, ``_get_position_bond_by_book``). Берём для них
                # больший cap, чтобы формула расчёта попала в контекст.
                max_chars=_truncate_for(rel),
            )
        except Exception:
            code = ''

    return {
        'id': node_id,
        'name': node_data.get('name', 'Unknown'),
        'type': node_data.get('node_type') or node_data.get('type', 'Unknown'),
        'file': file_path,
        'line': node_data.get('start_line'),
        'code': code,
        'score': 0.85,  # высокий: точное соответствие через граф
        'source': 'graph',
        'relationship': rel,
        'anchor': anchor.get('name', ''),
    }


# Приоритет якорей для расширения. Class даёт CONTAINS (все методы класса
# одним cypher) — это самый "богатый" expand. Endpoint тащит за собой handler
# и модели. Method/Function — только callees. Сортировка переопределяет порядок
# top-K Weaviate, но это оправдано: для запросов "как считается X" expand класса
# даёт качественно больше контекста чем 5 одноимённых функций.
_ANCHOR_TYPE_PRIORITY: Dict[str, int] = {
    'Class': 0,
    'Endpoint': 1,
    'Component': 2,
    'Model': 3,
    'Method': 4,
    'Function': 5,
}


def _select_anchors(
    chunks: List[Dict[str, Any]],
    *,
    max_anchors: int,
    skip_ids: Optional[Set[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Выбрать top-K якорей для расширения.

    Сортировка по приоритетам:
    1. **grep_match boost** — если в этом файле есть grep-хит на токен из
       вопроса, якорь "горячий". Это сильный сигнал прямой релевантности
       (в файле физически встречается слово из запроса), сильнее чем
       semantic similarity. Без буста ``PositionByBookService`` (где
       30+ упоминаний ``net_pos``) проигрывал в top-K Class'ам которые
       просто похожи по embedding'у на тему "позиции".
    2. **тип** — Class > Endpoint > Component > Model > Method > Function.
    3. **score** — desc.

    ``skip_ids`` — якоря, уже расширенные на предыдущей итерации; их пропускаем.
    """
    skip = skip_ids or set()

    # Файлы где есть grep-хиты — это файлы где физически найден токен из
    # вопроса. Якоря в этих файлах получают приоритет.
    grep_files: Set[str] = set()
    for c in chunks:
        if c.get('source') == 'grep':
            f = c.get('file') or ''
            # grep_enrich возвращает rel-path "<repo_name>/path/to/file.py"
            # — ствола имени для ноды (file_path = "path/to/file.py"). Берём
            # базовое имя файла как ключ (компактно и совпадает в обоих).
            if f:
                base = f.split('/')[-1]
                if base:
                    grep_files.add(base)

    candidates: List[Dict[str, Any]] = []
    seen_ids: set = set()
    for c in chunks:
        if c.get('source') != 'primary':
            continue
        node_id = c.get('id') or ''
        if not node_id or node_id in seen_ids or node_id in skip:
            continue
        node_type = (c.get('type') or '').strip()
        if node_type not in _ANCHOR_TYPE_PRIORITY:
            continue
        seen_ids.add(node_id)
        candidates.append(c)

    def _grep_match(c: Dict[str, Any]) -> int:
        """0 = есть grep-хиты в файле якоря (горячо), 1 = нет."""
        f = c.get('file') or ''
        base = f.split('/')[-1]
        return 0 if base and base in grep_files else 1

    # Sort: grep_match (0 < 1), priority asc, score desc.
    candidates.sort(key=lambda c: (
        _grep_match(c),
        _ANCHOR_TYPE_PRIORITY.get((c.get('type') or '').strip(), 99),
        -float(c.get('score') or 0.0),
    ))
    return candidates[:max_anchors]


# ---------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------

class GraphExpandTask(SearchTask):
    """
    Расширение от primary-якорей через Cypher по чёткому набору паттернов.

    Repeatable; дедупится по оригинальному query (структура графа
    стабильна между итерациями RAG-loop'а, повторный обход бесполезен).

    context['context']        ← extend with graph-chunks (source='graph')
    context['graph_anchors']  ← список node_id якорей, по которым шли
    context['graph_chunks']   ← число добавленных chunks
    """

    dependencies = []

    # Лимиты (можно вынести в AgentConfig).
    # 10, не 5: на крупных проектах в primary часто 6+ Class'ов с близким
    # score (TheoreticalDataCashService, StaticInfoService, RiskReport,
    # PositionByBookService, ...). Лимит 5 терял самый релевантный
    # класс из-за порядка сортировки. 10 пускает все Class из top-K.
    MAX_ANCHORS = 10
    # При 10 якорях × 25 sibling-методов = 250 потенциальных нод.
    # Поднимаем total cap чтобы CONTAINS не задавил весь budget на первом
    # классе и оставил место остальным.
    MAX_TOTAL_CHUNKS = 80

    def run(self, filters: Dict[str, Any]) -> None:
        # Дедуп через set УЖЕ обработанных якорей (не через query):
        # primary chunks меняются между итерациями (rewrite query → новый top-K),
        # и могут добавиться новые ценные якоря (например Class
        # PositionByBookService появился только на iter 2). Старые якоря
        # пропускаем; новые расширяем.
        processed: Set[str] = set(self.context.get("_graph_processed_anchors") or [])

        chunks: List[Dict[str, Any]] = self.context.get("context") or []

        # Диагностика: что именно мы видим в context на момент запуска.
        type_counts: Dict[str, int] = {}
        primary_count = 0
        primary_with_id = 0
        for c in chunks:
            if c.get('source') == 'primary':
                primary_count += 1
                if c.get('id'):
                    primary_with_id += 1
                t = (c.get('type') or '').strip()
                type_counts[t] = type_counts.get(t, 0) + 1

        anchors = _select_anchors(
            chunks, max_anchors=self.MAX_ANCHORS, skip_ids=processed
        )
        logger.info(
            f"[graph_expand] context={len(chunks)} primary={primary_count} "
            f"primary_with_id={primary_with_id} types={type_counts} "
            f"anchors_selected={len(anchors)}"
        )
        self.context["_graph_diag"] = {
            "context_total": len(chunks),
            "primary_count": primary_count,
            "primary_with_id": primary_with_id,
            "primary_types": type_counts,
        }

        if not anchors:
            self.context["graph_anchors"] = []
            self.context.setdefault("graph_chunks", 0)
            return
        # Накапливаем имена обработанных, чтобы следующая итерация
        # не повторяла те же expand'ы.
        for a in anchors:
            processed.add(a.get('id') or '')
        self.context["_graph_processed_anchors"] = list(processed)
        self.context["graph_anchors"] = [a.get('id') for a in anchors]

        neo4j = self.executor.retriever.neo4j
        repo_paths = getattr(self.executor, 'repo_paths', None)
        # Fallback на общий repos_dir у weaviate-индексера (legacy).
        repos_dir: Optional[Path] = None
        try:
            rd = self.executor.retriever.weaviate.repos_dir
            if rd:
                repos_dir = Path(rd)
        except AttributeError:
            pass

        # Существующие id, чтобы не дублировать в выдаче.
        existing_ids: Set[str] = {c.get('id') for c in chunks if c.get('id')}

        new_chunks: List[Dict[str, Any]] = []
        for anchor in anchors:
            anchor_type = (anchor.get('type') or '').strip()
            anchor_id = anchor.get('id')
            if not anchor_id:
                continue

            # Function/Method: специальный multi-hop путь через CALLS-tree.
            if anchor_type in ('Function', 'Method'):
                tree_chunks = _expand_calls_tree(
                    anchor, neo4j,
                    repos_dir=repos_dir, repo_paths=repo_paths,
                    existing_ids=existing_ids,
                )
                # Учитываем общий лимит на размер graph-блока.
                room = self.MAX_TOTAL_CHUNKS - len(new_chunks)
                if room <= 0:
                    break
                new_chunks.extend(tree_chunks[:room])

                # Reverse-CONTAINS: для Method тащим ВСЕ методы владеющего
                # класса. Критично для запросов типа "как считается X" —
                # один метод изолированно не отвечает, нужны siblings.
                if anchor_type == 'Method' and len(new_chunks) < self.MAX_TOTAL_CHUNKS:
                    sibling_chunks = _expand_owner_siblings(
                        anchor, neo4j,
                        repos_dir=repos_dir, repo_paths=repo_paths,
                        existing_ids=existing_ids,
                    )
                    room = self.MAX_TOTAL_CHUNKS - len(new_chunks)
                    new_chunks.extend(sibling_chunks[:room])

                if len(new_chunks) >= self.MAX_TOTAL_CHUNKS:
                    break
                continue

            # Class / Endpoint / Model / Component — плоский 1-hop expand.
            for node_type, cypher, limit in _EXPANSIONS:
                if node_type != anchor_type:
                    continue
                try:
                    rows = neo4j.execute_cypher(
                        cypher, parameters={'node_id': anchor_id, 'limit': limit}
                    )
                except Exception as e:
                    logger.warning(
                        f"[graph_expand] cypher failed for {anchor_id} "
                        f"({anchor_type}): {e}"
                    )
                    continue
                logger.info(
                    f"[graph_expand] anchor={anchor_type} id={anchor_id[:80]} "
                    f"-> {len(rows)} rows"
                )

                for row in rows:
                    raw_node = row.get('node')
                    rel = row.get('rel') or 'RELATED'
                    if not raw_node:
                        continue
                    # Neo4j-driver возвращает Node-объект, превращаем в dict.
                    node_data = dict(raw_node)
                    nid = node_data.get('id')
                    if not nid or nid in existing_ids:
                        continue

                    chunk = _node_to_graph_chunk(
                        node_data, rel, anchor,
                        repos_dir=repos_dir, repo_paths=repo_paths,
                    )
                    if chunk is None:
                        continue
                    existing_ids.add(nid)
                    new_chunks.append(chunk)

                    if len(new_chunks) >= self.MAX_TOTAL_CHUNKS:
                        break
                if len(new_chunks) >= self.MAX_TOTAL_CHUNKS:
                    break
            if len(new_chunks) >= self.MAX_TOTAL_CHUNKS:
                break

        if not new_chunks:
            self.context.setdefault("graph_chunks", 0)
            logger.info(
                f"[graph_expand] no new chunks "
                f"(anchors={len(anchors)}, types={[a.get('type') for a in anchors]})"
            )
            return

        chunks.extend(new_chunks)
        self.context["context"] = chunks
        # Накапливаем счётчик — мы можем расширяться на iter 1 и iter 2.
        prev = int(self.context.get("graph_chunks") or 0)
        self.context["graph_chunks"] = prev + len(new_chunks)
        logger.info(
            f"[graph_expand] anchors={len(anchors)} -> +{len(new_chunks)} graph chunks "
            f"(total graph={prev + len(new_chunks)}, total context={len(chunks)})"
        )

    @staticmethod
    def trace_input(context, filters):
        return {"query": filters.get("query")}

    @staticmethod
    def trace_output(context):
        out = {
            "anchors": context.get("graph_anchors", []),
            "chunks": context.get("graph_chunks", 0),
        }
        diag = context.get("_graph_diag")
        if diag:
            out["diag"] = diag
        return out
