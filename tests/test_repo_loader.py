"""
Tests for Git repository loader.
"""

import pytest
import tempfile
import shutil
from pathlib import Path

from src.code_rag.repo_loader import RepositoryLoader, RepositoryInfo
from src.code_rag.project_detector import ProjectDetector


class TestRepositoryLoader:
    """Test suite for RepositoryLoader."""

    @pytest.fixture
    def temp_repos_dir(self):
        """Create temporary directory for test repositories."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def loader(self, temp_repos_dir):
        """Create repository loader with temporary directory."""
        return RepositoryLoader(repos_dir=temp_repos_dir)

    def test_load_from_local_path(self, loader, tmp_path):
        """Test loading repository from local path."""
        # Create a fake repository structure
        repo_path = tmp_path / "test_repo"
        repo_path.mkdir()

        # Create some files
        (repo_path / "main.py").write_text("print('hello')")
        (repo_path / "README.md").write_text("# Test Repo")

        # Load repository
        repo_info = loader.load(
            source=str(repo_path),
            name="test_repo"
        )

        assert repo_info.name == "test_repo"
        assert repo_info.path == repo_path
        assert repo_info.url is None

    def test_get_files_with_filtering(self, loader, tmp_path):
        """Test file filtering with ignore patterns."""
        # Create repository structure
        repo_path = tmp_path / "test_repo"
        repo_path.mkdir()

        # Create files that should be included
        (repo_path / "main.py").write_text("print('hello')")
        (repo_path / "utils.py").write_text("def helper(): pass")

        # Create files that should be ignored
        node_modules = repo_path / "node_modules"
        node_modules.mkdir()
        (node_modules / "package.js").write_text("// package")

        pycache = repo_path / "__pycache__"
        pycache.mkdir()
        (pycache / "main.pyc").write_text("compiled")

        # Create repo info
        repo_info = RepositoryInfo(
            name="test_repo",
            path=repo_path
        )

        # Get files
        files = loader.get_files(repo_info)

        # Should only include .py files, not node_modules or __pycache__
        file_names = [f.name for f in files]
        assert "main.py" in file_names
        assert "utils.py" in file_names
        assert "package.js" not in file_names
        assert "main.pyc" not in file_names

    def test_ragignore_support(self, loader, tmp_path):
        """Test custom .ragignore file support."""
        repo_path = tmp_path / "test_repo"
        repo_path.mkdir()

        # Create files
        (repo_path / "main.py").write_text("print('hello')")
        (repo_path / "test.py").write_text("def test(): pass")
        (repo_path / "secret.py").write_text("API_KEY = 'secret'")

        # Create .ragignore
        (repo_path / ".ragignore").write_text("secret.py\ntest.py")

        repo_info = RepositoryInfo(
            name="test_repo",
            path=repo_path
        )

        files = loader.get_files(repo_info)
        file_names = [f.name for f in files]

        assert "main.py" in file_names
        assert "test.py" not in file_names
        assert "secret.py" not in file_names


class TestProjectDetector:
    """Test suite for ProjectDetector."""

    @pytest.fixture
    def detector(self):
        """Create project detector."""
        return ProjectDetector()

    def test_detect_django_project(self, detector, tmp_path):
        """Test detection of Django project."""
        # Create Django project structure
        (tmp_path / "manage.py").write_text("#!/usr/bin/env python")
        (tmp_path / "settings.py").write_text("from django.conf import settings")

        detection = detector.detect(tmp_path)

        assert "python" in detection['languages']
        assert "django" in detection['frameworks']
        assert detection['project_type'] == 'backend'

    def test_detect_react_project(self, detector, tmp_path):
        """Test detection of React project."""
        # Create React project structure
        import json

        package_json = {
            "name": "my-app",
            "dependencies": {
                "react": "^18.0.0",
                "react-dom": "^18.0.0"
            }
        }

        (tmp_path / "package.json").write_text(json.dumps(package_json))

        src_dir = tmp_path / "src"
        src_dir.mkdir()
        (src_dir / "App.tsx").write_text("import React from 'react';")

        detection = detector.detect(tmp_path)

        assert "typescript" in detection['languages']
        assert "react" in detection['frameworks']
        assert detection['project_type'] == 'frontend'

    def test_detect_fullstack_project(self, detector, tmp_path):
        """Test detection of fullstack project."""
        import json

        # Backend (FastAPI)
        (tmp_path / "main.py").write_text("from fastapi import FastAPI")
        (tmp_path / "requirements.txt").write_text("fastapi\nuvicorn")

        # Frontend (React)
        package_json = {
            "dependencies": {"react": "^18.0.0"}
        }
        (tmp_path / "package.json").write_text(json.dumps(package_json))

        detection = detector.detect(tmp_path)

        assert "python" in detection['languages']
        assert "fastapi" in detection['frameworks']
        assert "react" in detection['frameworks']
        assert detection['project_type'] == 'fullstack'


# Integration test
class TestGitSourceIntegration:
    """Integration tests for GitSource."""

    def test_git_source_load_local_repo(self, tmp_path):
        """Test GitSource with local repository."""
        from src.sources.git_source import GitSource

        # Create test repository
        repo_path = tmp_path / "test_repo"
        repo_path.mkdir()

        # Create Python files
        (repo_path / "main.py").write_text("""
def hello():
    '''Say hello'''
    print('Hello, World!')

if __name__ == '__main__':
    hello()
""")

        (repo_path / "utils.py").write_text("""
def add(a, b):
    '''Add two numbers'''
    return a + b
""")

        # Create package.json (to test multi-language detection)
        import json
        (repo_path / "package.json").write_text(json.dumps({
            "name": "test",
            "dependencies": {}
        }))

        # Create GitSource
        source = GitSource(
            source=str(repo_path),
            name="test_repo"
        )

        # Load files
        items = list(source.load())

        # Should have 2 Python files
        assert len(items) == 2

        # Check metadata
        py_items = [item for item in items if item.metadata.language == 'python']
        assert len(py_items) == 2

        # Check content
        main_item = next(item for item in items if 'main.py' in item.id)
        assert 'Hello, World!' in main_item.content
        assert main_item.metadata.source_type == 'code'
        assert main_item.metadata.repository == 'test_repo'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
