"""
CypherGenerationService — LLM генерирует Cypher по NL-вопросу пользователя.

Подаём LLM:
- Hardcoded схема графа (NodeType / RelationshipType + ключевые поля).
- Несколько few-shot примеров `вопрос → cypher`.
- Запрос на STRICT-формат: либо валидный cypher, либо токен ``SKIP``.

Ответ парсится, прогоняется через ``_validate_read_only`` (whitelist
ключевых слов), исполняется через ``Neo4jClient.execute_cypher``.

Идея взята из HF cookbook'а "RAG with Knowledge Graphs Neo4j"
(https://huggingface.co/learn/cookbook/rag_with_knowledge_graphs_neo4j).
"""

import re
from typing import Any, Dict, List, Optional, Tuple

from src.infra.logger import get_logger
from ._llm import LLMRole

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Hardcoded schema prompt — обновлять при изменении NodeType/RelationshipType
# ---------------------------------------------------------------------------

GRAPH_SCHEMA = """\
NODES:
- Repository {id, name, branch, languages, frameworks}
- File {id, name, file_path, language, line_count}
- Function {id, name, signature, file_path, start_line, end_line, decorators, is_async}
- Method {id, name, signature, file_path, start_line, end_line, is_method=true}
- Class {id, name, base_classes, file_path, start_line, end_line}
- Component {id, name, props_type, hooks_used, file_path}
- Endpoint {id, name, http_method, path, normalized_path, view_ref, framework}
- ApiCall {id, name, http_method, url, normalized_url, call_type, file_path}
- Model {id, name, model_type, field_count}
- Route {id, name, path}

RELATIONSHIPS:
- (Repository|File|Class)-[:CONTAINS]->(File|Function|Class|Method|Component|...)
- (File)-[:IMPORTS]->(File)
- (Function|Method)-[:CALLS]->(Function|Method)
- (Class)-[:INHERITS]->(Class)
- (Component)-[:MAKES_CALL]->(ApiCall)
- (ApiCall)-[:CALLS_ENDPOINT]->(Endpoint)
- (Endpoint)-[:HANDLES_REQUEST]->(Function|Method)
- (Endpoint|Function)-[:USES_MODEL]->(Model)
- (Component)-[:RENDERS_AT]->(Route)
- (Model)-[:FOREIGN_KEY|MANY_TO_MANY|ONE_TO_ONE]->(Model)
"""


PROMPT = """\
ВЫВЕДИ РОВНО ОДНУ строку: либо валидный Cypher-запрос, либо литерал SKIP.
ЗАПРЕЩЕНО: рассуждения, "We need to", "Output:", markdown, ```cypher```,
комментарии, преамбула, объяснения, перевод строки внутри ответа.

Схема графа:

{schema}

Разрешённые операции: MATCH, OPTIONAL MATCH, WITH, WHERE, RETURN, UNWIND,
ORDER BY, LIMIT, SKIP, DISTINCT, count(), collect().
Запрещённые: CREATE, DELETE, MERGE, SET, REMOVE, DROP, CALL, LOAD CSV.
Всегда добавляй LIMIT (≤ 50), если в вопросе нет "сколько" / "count".
Если вопрос — общий/философский/не про структуру графа → ответь: SKIP

Примеры:
Q: Сколько endpoint'ов в репозитории "api"?
A: MATCH (e:Endpoint) WHERE e.id STARTS WITH "repo:api:" RETURN count(e) AS endpoints

Q: Какие модели использует UserView?
A: MATCH (c:Class {{name: "UserView"}})-[:USES_MODEL]->(m:Model) RETURN m.name AS model LIMIT 50

Q: Какие компоненты дёргают /api/users/?
A: MATCH (c:Component)-[:MAKES_CALL]->(a:ApiCall)-[:CALLS_ENDPOINT]->(e:Endpoint) WHERE e.normalized_path = "/api/users" RETURN DISTINCT c.name AS component, c.file_path AS file LIMIT 50

Q: Сколько классов наследуется от APIView?
A: MATCH (c:Class)-[:INHERITS]->(p:Class {{name: "APIView"}}) RETURN count(c) AS subclasses

Q: Что такое RAG?
A: SKIP

Q: Найди код для POST /api/checkout flow
A: MATCH (c:Component)-[:MAKES_CALL]->(a:ApiCall)-[:CALLS_ENDPOINT]->(e:Endpoint) WHERE e.normalized_path = "/api/checkout" AND a.http_method = "POST" OPTIONAL MATCH (e)-[:HANDLES_REQUEST]->(h) RETURN c.name AS component, e.path AS endpoint, h.name AS handler LIMIT 50

Q: {query}
A:"""


# ---------------------------------------------------------------------------
# Safety: whitelist read-only keywords
# ---------------------------------------------------------------------------

_FORBIDDEN_PATTERNS = (
    r'\bCREATE\b', r'\bDELETE\b', r'\bMERGE\b', r'\bSET\b',
    r'\bREMOVE\b', r'\bDROP\b', r'\bDETACH\s+DELETE\b',
    r'\bLOAD\s+CSV\b',
    # CALL без YIELD может выполнить mutating procedures
    r'\bCALL\s+(?!.*\bYIELD\b)',
)


def _validate_read_only(cypher: str) -> Tuple[bool, str]:
    """Проверяет, что запрос содержит только разрешённые операции."""
    upper = cypher.upper()
    if not re.search(r'\bMATCH\b', upper):
        return False, 'no MATCH clause'
    # Любой валидный read-only cypher должен заканчиваться RETURN (или
    # содержать его). Без RETURN нельзя получить результат — да и обычно
    # это значит что мы поймали "огрызок" reasoning'а LLM, а не запрос.
    if not re.search(r'\bRETURN\b', upper):
        return False, 'no RETURN clause'
    for pat in _FORBIDDEN_PATTERNS:
        if re.search(pat, upper):
            return False, f'forbidden keyword: {pat}'
    return True, ''


_CYPHER_START_RE = re.compile(
    # Cypher всегда начинается с ``MATCH`` (или ``OPTIONAL MATCH``).
    # ``WITH`` и ``UNWIND`` — subordinate clauses ВНУТРИ запроса, нельзя
    # ими начинать. Раньше это приводило к false-positive: LLM писал
    # "with properties name, base_classes..." (английское слово 'with'!),
    # regex принимал это за начало cypher и тащил весь reasoning в запрос.
    r'\b(MATCH|OPTIONAL\s+MATCH)\b',
    re.IGNORECASE,
)
_CYPHER_END_MARKERS_RE = re.compile(
    # Граница, где "ответ" заканчивается:
    # пустая строка, "Q:", "A:" нового примера, начало нового prose-блока.
    r'(\n\s*\n|\n\s*Q:|\n\s*A:|\n\s*Output\b|\n\s*###|\n\s*```)',
    re.IGNORECASE,
)


def _extract_cypher(raw: str) -> str:
    """
    Из произвольного LLM-ответа достать чистый Cypher или 'SKIP'.

    Покрывает кейсы:
    - reasoning-модели с ``<think>...</think>`` (deepseek-r1, o1-style)
    - prose-обёртка ``Output\\n\\nWe need to...``
    - markdown `````cypher ...`````
    - ответы в формате ``A: <cypher>`` из few-shot примеров
    - смешанный ответ "пояснение → MATCH (...) ... → ещё пояснение"

    Возвращает:
    - ``""`` если не удалось вытащить запрос (caller трактует как rejected)
    - ``"SKIP"`` если LLM сказал пропустить
    - чистый однострочный (или многострочный без trailing prose) Cypher
    """
    if not raw:
        return ""

    text = raw.strip()

    # 1) Снять reasoning-блоки моделей вроде deepseek-r1 / o1
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r'<reasoning>.*?</reasoning>', '', text, flags=re.IGNORECASE | re.DOTALL)

    # 2) Снять markdown-обёртки (могут встречаться внутри текста)
    text = re.sub(r'```(?:cypher|sql)?\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'```', '', text)

    # 3) Если в тексте есть префикс "A:" (наш формат) — берём от последнего "A:"
    a_match = list(re.finditer(r'^\s*A:\s*', text, flags=re.MULTILINE))
    if a_match:
        text = text[a_match[-1].end():]

    text = text.strip()
    if not text:
        return ""

    # 4) Явный SKIP в любой части ответа (часто LLM пишет "I should SKIP this")
    if re.search(r'(?:^|\W)SKIP(?:\W|$)', text) and not _CYPHER_START_RE.search(text):
        return "SKIP"

    # 5) Найти начало cypher-выражения (MATCH / OPTIONAL MATCH / WITH / UNWIND)
    start = _CYPHER_START_RE.search(text)
    if not start:
        return ""

    cypher = text[start.start():]

    # 6) Обрезать prose, если он начался после cypher
    end = _CYPHER_END_MARKERS_RE.search(cypher)
    if end:
        cypher = cypher[:end.start()]

    # 7) Нормализация whitespace: схлопнуть множественные пробелы/переводы строк
    cypher = re.sub(r'\s+', ' ', cypher).strip()
    return cypher


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Smart skip: heuristic gate перед LLM-вызовом
# ---------------------------------------------------------------------------

# Паттерны структурных запросов. Если хоть один матчится — query похож на
# вопрос про структуру графа, имеет смысл звать LLM. Если ни один —
# сразу SKIP без LLM (экономим ~5 секунд на запрос типа "как считается X").
_STRUCTURAL_PATTERNS: Tuple[re.Pattern, ...] = (
    # Counting / aggregation
    re.compile(r'\bсколько\b|\bколичество\b', re.IGNORECASE),
    re.compile(r'\b(?:how\s+many|count|number\s+of|total)\b', re.IGNORECASE),
    # Inheritance / hierarchy
    re.compile(r'\b(?:наследу|подклас|базовый\s+класс|родитель)', re.IGNORECASE),
    re.compile(r'\b(?:inherits?|extends?|subclass|parent\s+class|derived\s+from)\b', re.IGNORECASE),
    # Graph traversal verbs
    re.compile(r'\b(?:вызывает|использует|связан|зависит|импортирует|рендерит)', re.IGNORECASE),
    re.compile(r'\b(?:calls?|uses?|invokes?|imports?|renders?|depends?\s+on)\b', re.IGNORECASE),
    # URL-паттерны (HTTP-route в вопросе → структурный вопрос про endpoint'ы)
    re.compile(r'/(?:api|backend|v\d+)/[\w\-/{}<>:?]+', re.IGNORECASE),
    re.compile(r'\b(?:GET|POST|PUT|PATCH|DELETE|HEAD|OPTIONS)\s+/', re.IGNORECASE),
    # Названия типов нод схемы — указывают на структурный запрос
    re.compile(r'\b(?:Endpoint|Component|Model|Route|ApiCall|Function|Method|Class)s?\b'),
    # CamelCase имена (UserView, OrderViewSet, etc) — обычно классы из кодовой базы
    re.compile(r'\b[A-Z][a-z]+[A-Z][A-Za-z0-9]*\b'),
)


def _is_structural_query(query: str) -> bool:
    """
    Дешёвая heuristic: похож ли вопрос на структурный (про граф) или нет.

    Если хоть один паттерн матчится — возвращаем True, и LLM будет генерить
    cypher. Если ни один — `False`, и CypherEnrichTask пропускается без
    LLM-вызова (экономит ~5s на типичных code-запросах вроде «как считается
    X / где определяется Y», для которых cypher всегда возвращает SKIP).

    На false-negative'ах (структурный запрос ошибочно классифицирован как
    "не структурный") ничего страшного не происходит — pipeline просто
    не получит cypher-фактов, остальные источники (primary/grep/graph)
    дадут ответ.

    На false-positive'ах (не-структурный classified как структурный) — LLM
    обычно сам ответит SKIP, ничего не сломается.
    """
    if not query or not query.strip():
        return False
    return any(p.search(query) for p in _STRUCTURAL_PATTERNS)


class CypherGenerationService:
    """
    NL → Cypher через LLM.

    Использование:
        service = CypherGenerationService(llm_client, model="gpt-4o")
        cypher = service.generate("Какие компоненты вызывают /api/users/?")
        if cypher:
            results = service.execute(neo4j_client, cypher)
    """

    def __init__(self, llm: LLMRole):
        self.llm = llm

    def generate(self, query: str) -> Optional[str]:
        """Генерация cypher. Возвращает строку cypher или None если SKIP/невалидно."""
        # Smart skip ПЕРЕД LLM-вызовом: если query явно не структурный
        # (например «как считается net_pos» — про код, не про граф),
        # пропускаем LLM. Экономит 5s на типичных запросах "как X" / "где Y".
        if not _is_structural_query(query):
            logger.info(f"[cypher] heuristic skip (non-structural): {query[:80]}")
            return None

        prompt = PROMPT.format(schema=GRAPH_SCHEMA, query=query)
        raw = self.llm.call(
            prompt,
            name="cypher_generation",
            max_tokens=512,
            temperature=0.0,
        )
        cleaned = _extract_cypher(raw)

        if not cleaned:
            # Не нашли ни cypher, ни SKIP — это plain-prose мусор от LLM.
            preview = (raw or '').replace('\n', ' ')[:160]
            logger.warning(f"[cypher] could not extract cypher from response: {preview}")
            return None

        if cleaned.upper() == 'SKIP':
            logger.info(f"[cypher] skip: query is non-structural")
            return None

        ok, reason = _validate_read_only(cleaned)
        if not ok:
            logger.warning(f"[cypher] generated query rejected ({reason}): {cleaned[:120]}")
            return None

        return cleaned

    @staticmethod
    def execute(
        neo4j_client: Any,
        cypher: str,
        *,
        max_results: int = 50,
    ) -> List[Dict[str, Any]]:
        """Безопасное выполнение. Кейс ошибки → пустой список + лог."""
        try:
            rows = neo4j_client.execute_cypher(cypher)
        except Exception as e:
            logger.warning(f"[cypher] execution failed: {e}; query={cypher[:200]}")
            return []
        return list(rows)[:max_results]
