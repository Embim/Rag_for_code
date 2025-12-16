"""
Git repository loader.

This module handles:
- Cloning repositories from Git URLs
- Working with local repository paths
- Filtering files (via .ragignore)
- Tracking changes for incremental updates
"""

import os
import shutil
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass
from datetime import datetime

import git
from git import Repo, GitCommandError
import pathspec

from src.logger import get_logger


logger = get_logger(__name__)


@dataclass
class RepositoryInfo:
    """Information about a loaded repository."""

    name: str
    path: Path
    url: Optional[str] = None
    branch: str = "main"
    commit_hash: Optional[str] = None
    last_updated: Optional[datetime] = None

    # Detected project info
    project_type: Optional[str] = None  # "frontend", "backend", "fullstack"
    languages: List[str] = None
    frameworks: List[str] = None

    def __post_init__(self):
        if self.languages is None:
            self.languages = []
        if self.frameworks is None:
            self.frameworks = []


class RepositoryLoader:
    """
    Loader for Git repositories.

    Handles cloning, updating, and file filtering for code repositories.
    """

    # Default ignore patterns (like .gitignore)
    DEFAULT_IGNORE_PATTERNS = [
        # Dependencies
        "node_modules/",
        "vendor/",
        "venv/",
        "env/",
        ".venv/",
        "__pycache__/",
        "*.pyc",
        ".Python",
        "pip-log.txt",
        "pip-delete-this-directory.txt",

        # Build artifacts
        "dist/",
        "build/",
        ".next/",
        "out/",
        "target/",
        "*.egg-info/",

        # IDE
        ".vscode/",
        ".idea/",
        "*.swp",
        "*.swo",
        "*~",
        ".DS_Store",

        # Version control
        ".git/",
        ".svn/",
        ".hg/",

        # Compiled files
        "*.so",
        "*.dylib",
        "*.dll",
        "*.exe",

        # Minified/bundled
        "*.min.js",
        "*.min.css",
        "*.bundle.js",
        "*.chunk.js",
        "*.map",

        # Lock files (too large, not useful)
        "package-lock.json",
        "yarn.lock",
        "poetry.lock",
        "Pipfile.lock",
        "Cargo.lock",

        # Logs
        "*.log",
        "logs/",

        # Test coverage
        "coverage/",
        ".coverage",
        ".nyc_output/",

        # Environment
        ".env",
        ".env.local",
        ".env.*.local",

        # Large media files
        "*.jpg",
        "*.jpeg",
        "*.png",
        "*.gif",
        "*.svg",
        "*.ico",
        "*.pdf",
        "*.zip",
        "*.tar.gz",
    ]

    def __init__(self, repos_dir: Path = None):
        """
        Initialize repository loader.

        Args:
            repos_dir: Directory where repositories are stored (default: data/repos)
        """
        self.repos_dir = repos_dir or Path("data/repos")
        self.repos_dir.mkdir(parents=True, exist_ok=True)

    def load(
        self,
        source: str,
        name: Optional[str] = None,
        branch: str = "main",
        force_clone: bool = False,
        custom_ignore: Optional[List[str]] = None
    ) -> RepositoryInfo:
        """
        Load a repository from URL or local path.

        Args:
            source: Git URL or local path to repository
            name: Name for this repository (auto-generated if not provided)
            branch: Branch to checkout (default: main)
            force_clone: Force re-clone even if exists
            custom_ignore: Additional ignore patterns

        Returns:
            RepositoryInfo with loaded repository details

        Raises:
            ValueError: If source is invalid
            GitCommandError: If git operations fail
        """
        is_url = source.startswith(('http://', 'https://', 'git@'))

        if is_url:
            return self._load_from_url(source, name, branch, force_clone, custom_ignore)
        else:
            return self._load_from_path(Path(source), name, custom_ignore)

    def _load_from_url(
        self,
        url: str,
        name: Optional[str],
        branch: str,
        force_clone: bool,
        custom_ignore: Optional[List[str]]
    ) -> RepositoryInfo:
        """Load repository from Git URL."""
        # Generate name from URL if not provided
        if not name:
            name = self._extract_repo_name_from_url(url)

        repo_path = self.repos_dir / name

        # Check if already exists
        if repo_path.exists():
            if force_clone:
                logger.info(f"Removing existing repository at {repo_path}")
                shutil.rmtree(repo_path)
            else:
                logger.info(f"Repository already exists at {repo_path}, pulling latest changes")
                return self._update_existing_repo(repo_path, name, url, branch, custom_ignore)

        # Clone repository
        logger.info(f"Cloning repository from {url} to {repo_path}")
        try:
            repo = Repo.clone_from(url, repo_path, branch=branch, depth=1)
            logger.info(f"✓ Successfully cloned {url}")
        except GitCommandError as e:
            logger.error(f"Failed to clone repository: {e}")
            raise

        # Get commit hash
        commit_hash = repo.head.commit.hexsha

        # Create repository info
        repo_info = RepositoryInfo(
            name=name,
            path=repo_path,
            url=url,
            branch=branch,
            commit_hash=commit_hash,
            last_updated=datetime.now()
        )

        # Save metadata
        self._save_metadata(repo_info, custom_ignore)

        return repo_info

    def _load_from_path(
        self,
        path: Path,
        name: Optional[str],
        custom_ignore: Optional[List[str]]
    ) -> RepositoryInfo:
        """Load repository from local path."""
        if not path.exists():
            raise ValueError(f"Path does not exist: {path}")

        if not path.is_dir():
            raise ValueError(f"Path is not a directory: {path}")

        # Use directory name if name not provided
        if not name:
            name = path.name

        # Check if it's a git repository
        try:
            repo = Repo(path)
            commit_hash = repo.head.commit.hexsha
            branch = repo.active_branch.name
        except Exception:
            logger.warning(f"Path {path} is not a git repository, using as-is")
            commit_hash = None
            branch = None

        # Create repository info
        repo_info = RepositoryInfo(
            name=name,
            path=path,
            url=None,
            branch=branch,
            commit_hash=commit_hash,
            last_updated=datetime.now()
        )

        return repo_info

    def _update_existing_repo(
        self,
        repo_path: Path,
        name: str,
        url: str,
        branch: str,
        custom_ignore: Optional[List[str]]
    ) -> RepositoryInfo:
        """Update existing repository."""
        try:
            repo = Repo(repo_path)

            # Pull latest changes
            origin = repo.remotes.origin
            origin.pull(branch)

            logger.info(f"✓ Updated repository {name}")

            commit_hash = repo.head.commit.hexsha

            repo_info = RepositoryInfo(
                name=name,
                path=repo_path,
                url=url,
                branch=branch,
                commit_hash=commit_hash,
                last_updated=datetime.now()
            )

            self._save_metadata(repo_info, custom_ignore)

            return repo_info

        except GitCommandError as e:
            logger.error(f"Failed to update repository: {e}")
            raise

    def get_files(
        self,
        repo_info: RepositoryInfo,
        custom_ignore: Optional[List[str]] = None
    ) -> List[Path]:
        """
        Get list of files in repository, respecting ignore patterns.

        Args:
            repo_info: Repository information
            custom_ignore: Additional ignore patterns

        Returns:
            List of file paths to process
        """
        # Build ignore spec
        ignore_patterns = self.DEFAULT_IGNORE_PATTERNS.copy()

        # Load .ragignore if exists
        ragignore_path = repo_info.path / ".ragignore"
        if ragignore_path.exists():
            with open(ragignore_path) as f:
                ignore_patterns.extend(
                    line.strip() for line in f
                    if line.strip() and not line.startswith('#')
                )

        # Add custom patterns
        if custom_ignore:
            ignore_patterns.extend(custom_ignore)

        spec = pathspec.PathSpec.from_lines('gitwildmatch', ignore_patterns)

        # Walk directory and collect files
        files = []
        for root, dirs, filenames in os.walk(repo_info.path):
            root_path = Path(root)
            rel_root = root_path.relative_to(repo_info.path)

            # Filter directories (modify in-place to affect os.walk)
            dirs[:] = [
                d for d in dirs
                if not spec.match_file(str(rel_root / d) + '/')
            ]

            # Filter files
            for filename in filenames:
                rel_path = rel_root / filename
                if not spec.match_file(str(rel_path)):
                    files.append(root_path / filename)

        logger.info(f"Found {len(files)} files in {repo_info.name} (after filtering)")

        return files

    def get_changed_files(
        self,
        repo_info: RepositoryInfo,
        since_commit: str
    ) -> List[Path]:
        """
        Get list of files changed since a specific commit.

        Args:
            repo_info: Repository information
            since_commit: Commit hash to compare against

        Returns:
            List of changed file paths
        """
        try:
            repo = Repo(repo_info.path)

            # Get diff between commits
            current_commit = repo.head.commit
            old_commit = repo.commit(since_commit)

            diff = old_commit.diff(current_commit)

            changed_files = []
            for change in diff:
                # Handle different change types
                if change.a_path:
                    changed_files.append(repo_info.path / change.a_path)
                if change.b_path and change.b_path != change.a_path:
                    changed_files.append(repo_info.path / change.b_path)

            logger.info(f"Found {len(changed_files)} changed files since {since_commit[:8]}")

            return changed_files

        except Exception as e:
            logger.error(f"Failed to get changed files: {e}")
            # Fallback: return all files
            return self.get_files(repo_info)

    def _extract_repo_name_from_url(self, url: str) -> str:
        """Extract repository name from Git URL."""
        # Handle different URL formats
        # https://github.com/user/repo.git -> repo
        # git@github.com:user/repo.git -> repo
        name = url.rstrip('/').split('/')[-1]
        if name.endswith('.git'):
            name = name[:-4]
        return name

    def _save_metadata(
        self,
        repo_info: RepositoryInfo,
        custom_ignore: Optional[List[str]] = None
    ) -> None:
        """Save repository metadata to .rag-meta.json."""
        import json

        meta_path = repo_info.path / ".rag-meta.json"

        metadata = {
            'name': repo_info.name,
            'url': repo_info.url,
            'branch': repo_info.branch,
            'commit_hash': repo_info.commit_hash,
            'last_updated': repo_info.last_updated.isoformat() if repo_info.last_updated else None,
            'project_type': repo_info.project_type,
            'languages': repo_info.languages,
            'frameworks': repo_info.frameworks,
            'custom_ignore': custom_ignore or []
        }

        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)

    def load_metadata(self, repo_path: Path) -> Optional[RepositoryInfo]:
        """Load repository metadata from .rag-meta.json."""
        import json

        meta_path = repo_path / ".rag-meta.json"

        if not meta_path.exists():
            return None

        try:
            with open(meta_path) as f:
                data = json.load(f)

            return RepositoryInfo(
                name=data['name'],
                path=repo_path,
                url=data.get('url'),
                branch=data.get('branch', 'main'),
                commit_hash=data.get('commit_hash'),
                last_updated=datetime.fromisoformat(data['last_updated']) if data.get('last_updated') else None,
                project_type=data.get('project_type'),
                languages=data.get('languages', []),
                frameworks=data.get('frameworks', [])
            )
        except Exception as e:
            logger.error(f"Failed to load metadata: {e}")
            return None
