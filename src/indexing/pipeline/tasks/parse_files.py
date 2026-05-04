from src.infra.logger import get_logger
from ..base.task import IndexingTask
from ..services import ParsingService
from .load_repository import LoadRepositoryTask

logger = get_logger(__name__)


class ParseFilesTask(IndexingTask):
    """
    Парсит все файлы репозитория соответствующими парсерами.

    context["parse_results"] ← List[(Path, ParseResult)].
    Также пишет в context метрики files_parsed / entities_found.
    """

    dependencies = [LoadRepositoryTask]

    def run(self, filters):
        repo_info = self.context["repo_info"]
        service = ParsingService()
        parse_results = service.parse_repository(repo_info)

        files_parsed = len(parse_results)
        entities_found = sum(len(r.entities) for _, r in parse_results)

        self.context["parse_results"] = parse_results
        self.context["files_parsed"] = files_parsed
        self.context["entities_found"] = entities_found

        logger.info(
            f"Parsed {files_parsed} files, found {entities_found} entities"
        )
