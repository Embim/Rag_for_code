from typing import Any, Optional

from src.core.repo_loader import RepositoryInfo, RepositoryLoader
from src.infra.logger import get_logger

logger = get_logger(__name__)


class RepositoryService:
    """Загрузка репозитория с диска или из git."""

    def __init__(self, repo_loader: RepositoryLoader):
        self.repo_loader = repo_loader

    def load(
        self,
        source: str,
        name: Optional[str] = None,
        branch: str = "main",
    ) -> RepositoryInfo:
        repo_info = self.repo_loader.load(source=source, name=name, branch=branch)

        logger.info(f"Repository loaded: {repo_info.name}")
        logger.info(f"  Type: {repo_info.project_type}")
        logger.info(f"  Languages: {', '.join(repo_info.languages or [])}")
        logger.info(f"  Frameworks: {', '.join(repo_info.frameworks or [])}")

        return repo_info
