from pathlib import Path
from typing import List, Tuple

from tqdm import tqdm

from src.core.parsers import ParseResult, get_parser
from src.core.repo_loader import RepositoryInfo
from src.infra.logger import get_logger

logger = get_logger(__name__)


# Расширения, которые мы парсим. Расширять список — здесь.
_CODE_EXTENSIONS = ("*.py", "*.js", "*.jsx", "*.ts", "*.tsx")
# Папки/паттерны, которые пропускаем при обходе репозитория.
_SKIP_PATH_FRAGMENTS = (
    "node_modules",
    "venv",
    ".venv",
    "dist",
    "build",
    "__pycache__",
)


class ParsingService:
    """
    Обходит файлы репозитория и парсит каждый соответствующим парсером.

    Не знает про graph и context — на вход RepositoryInfo, на выходе
    плоский список (file_path, ParseResult).
    """

    def collect_code_files(self, repo_info: RepositoryInfo) -> List[Path]:
        files: List[Path] = []
        for pattern in _CODE_EXTENSIONS:
            files.extend(repo_info.path.rglob(pattern))

        cleaned: List[Path] = []
        for file_path in files:
            if not file_path.is_file():
                continue
            name_low = file_path.name.lower()
            path_str = str(file_path)
            if "test" in name_low or "migration" in path_str:
                continue
            if any(frag in path_str for frag in _SKIP_PATH_FRAGMENTS):
                continue
            cleaned.append(file_path)
        return cleaned

    def parse_repository(
        self, repo_info: RepositoryInfo
    ) -> List[Tuple[Path, ParseResult]]:
        code_files = self.collect_code_files(repo_info)
        results: List[Tuple[Path, ParseResult]] = []

        for file_path in tqdm(code_files, desc="Parsing files", unit="file"):
            parser = get_parser(file_path)
            if not parser:
                continue
            try:
                result = parser.parse_file(file_path)
                results.append((file_path, result))
                logger.debug(f"Parsed {file_path.name}: {len(result.entities)} entities")
            except Exception as e:
                logger.error(f"Failed to parse {file_path}: {e}")

        return results
