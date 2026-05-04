from typing import Any, Dict, List

from src.core.graph import APILinker, Neo4jClient
from src.infra.logger import get_logger

logger = get_logger(__name__)


class ApiLinkingService:
    """
    Создание ``CALLS_ENDPOINT`` связей между ``ApiCallNode`` (UI) и
    ``EndpointNode`` (backend). Делегирует одной cypher-команде в APILinker.
    """

    def __init__(self, neo4j_client: Neo4jClient, api_linker: APILinker):
        self.neo4j_client = neo4j_client
        self.api_linker = api_linker

    def link(self) -> int:
        created = self.api_linker.link(self.neo4j_client)

        if created == 0:
            orphans = self.api_linker.find_orphaned_api_calls(
                self.neo4j_client, limit=10
            )
            if orphans:
                logger.warning(
                    f"[link] 0 links created; {len(orphans)} orphan ApiCalls (showing up to 5):"
                )
                for o in orphans[:5]:
                    logger.warning(
                        f"  - {o.get('component') or '?'}: "
                        f"{o.get('method')} {o.get('url')} "
                        f"(normalized={o.get('normalized_url')})"
                    )
        return created
