"""
Только перепривязка UI ↔ Backend поверх уже существующего графа.

Не парсит файлы и не пересобирает Neo4j. Дёргает RelinkApisTask,
который через ApiLinkingService достаёт ComponentNode/EndpointNode
из Neo4j и создаёт CALLS_ENDPOINT связи.

Когда полезно:
- крутили правила api_linker и хотим увидеть результат, не пересобирая граф;
- добавили новую модель/маппер — нужно перелинковать без перепарса.
"""

from typing import Any, Dict

from ..tasks import RelinkApisTask
from ._factory import make_executor


def run(
    *,
    neo4j_uri: str = "bolt://localhost:7687",
    neo4j_user: str = "neo4j",
    neo4j_password: str = "password",
) -> Dict[str, Any]:
    with make_executor(
        neo4j_uri=neo4j_uri,
        neo4j_user=neo4j_user,
        neo4j_password=neo4j_password,
    ) as executor:
        executor.run(RelinkApisTask)
        return {"api_links_created": executor.context.get("api_links_created", 0)}
