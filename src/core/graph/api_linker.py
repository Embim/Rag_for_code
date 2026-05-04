"""
API Linker — связывает ``ApiCallNode`` (вызовы из UI) с ``EndpointNode``
(серверные роуты) через cypher.

После refactor #2: api_calls — это first-class ``ApiCallNode``, а не JSON
на компоненте. Линковка выполняется одним cypher-запросом с exact-матчем по
``normalized_url`` (одна и та же функция нормализации записывается на оба
вида нод при их создании).

API:
- ``APILinker.link(neo4j_client) -> int`` — создаёт CALLS_ENDPOINT, возвращает
  число новых связей.
- ``APILinker.find_orphaned_api_calls(neo4j_client)`` — список ApiCallNode без
  совпавшего endpoint'а. Полезно для дебага.

Старый Python-цикл с SequenceMatcher (similarity matching) удалён.
"""

from typing import Any, Dict, List

from src.infra.logger import get_logger

logger = get_logger(__name__)


# Связь: ApiCall → Endpoint, exact-матч по normalized URL и http_method
# (с учётом wildcard 'ANY' на стороне эндпоинта — для DRF ViewSet'ов и т.п.).
LINK_QUERY = """
MATCH (a:ApiCall)
MATCH (e:Endpoint)
WHERE a.normalized_url <> ''
  AND a.normalized_url <> '/'
  AND (
    // (1) Точный матч путей.
    a.normalized_url = e.normalized_path
    // (2) UI URL — это list-вариант detail-эндпоинта.
    //     Пример: UI=`/backend/autocall-trade-versioned`,
    //             endpoint=`/backend/autocall-trade-versioned/{param}`.
    //     Это происходит когда UI делает axios.put(api_url + id + '/'):
    //     парсер видит только литеральную базу URL, без id.
    OR e.normalized_path = a.normalized_url + '/{param}'
    // (3) Backend имеет detail-вариант, UI обращается с доп. слешем
    //     (`/backend/foo/` vs `/backend/foo/{param}`).
    OR e.normalized_path = a.normalized_url + '{param}'
  )
  AND (a.http_method = e.http_method OR e.http_method = 'ANY')
MERGE (a)-[r:CALLS_ENDPOINT]->(e)
ON CREATE SET
    r.created_at = timestamp(),
    r.confidence = CASE
        WHEN a.normalized_url = e.normalized_path THEN 1.0
        WHEN e.http_method = 'ANY' THEN 0.7
        ELSE 0.85
    END
RETURN count(r) AS rel_count
"""


# ApiCall'ы, не получившие связи с эндпоинтами (для отчёта/дебага).
ORPHAN_QUERY = """
MATCH (a:ApiCall)
WHERE NOT (a)-[:CALLS_ENDPOINT]->(:Endpoint)
OPTIONAL MATCH (c:Component)-[:MAKES_CALL]->(a)
RETURN
    a.id            AS id,
    a.http_method   AS method,
    a.url           AS url,
    a.normalized_url AS normalized_url,
    a.file_path     AS file_path,
    a.start_line    AS start_line,
    c.name          AS component
ORDER BY a.file_path, a.start_line
LIMIT $limit
"""


class APILinker:
    """
    Тонкий фасад над cypher-запросами линковки.

    Параметры конструктора оставлены ради обратной совместимости с
    индексационным сценарием — фактически они не используются.
    """

    def __init__(self, use_openapi: bool = True):
        self.use_openapi = use_openapi

    def link(self, neo4j_client: Any) -> int:
        """
        Создать ``CALLS_ENDPOINT`` для всех совпавших ApiCall ↔ Endpoint.

        Идемпотентно (``MERGE``): повторный запуск не создаёт дублей.
        """
        try:
            result = neo4j_client.execute_cypher(LINK_QUERY)
            created = int(result[0]['rel_count']) if result else 0
            logger.info(f"[api_linker] CALLS_ENDPOINT links: {created}")
            return created
        except Exception as e:
            logger.error(f"[api_linker] link failed: {e}")
            return 0

    def find_orphaned_api_calls(
        self, neo4j_client: Any, *, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        ApiCall'ы, для которых не нашлось матчащего EndpointNode.
        """
        try:
            return neo4j_client.execute_cypher(ORPHAN_QUERY, limit=limit)
        except Exception as e:
            logger.error(f"[api_linker] orphan query failed: {e}")
            return []
