from ..base.task import IndexingTask
from ..services import RepositoryService


class LoadRepositoryTask(IndexingTask):
    """
    Загружает репозиторий с диска / клонирует из git.

    filters: source, name (optional), branch (default 'main').
    context["repo_info"] ← RepositoryInfo.
    """

    dependencies = []

    def run(self, filters):
        service = RepositoryService(self.executor.repo_loader)
        self.context["repo_info"] = service.load(
            source=filters["source"],
            name=filters.get("name"),
            branch=filters.get("branch", "main"),
        )
