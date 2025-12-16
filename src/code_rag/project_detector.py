"""
Project type detector.

Automatically detects project type, languages, and frameworks
by analyzing repository structure and files.
"""

from pathlib import Path
from typing import List, Dict, Set, Tuple
import json

from src.logger import get_logger


logger = get_logger(__name__)


class ProjectDetector:
    """
    Detector for project type, languages, and frameworks.

    Analyzes repository structure to determine:
    - Project type (frontend, backend, fullstack)
    - Programming languages used
    - Frameworks (Django, FastAPI, React, Vue, etc.)
    """

    # Framework detection patterns
    FRAMEWORK_PATTERNS = {
        # Python Backend
        'django': {
            'files': ['manage.py'],
            'imports': ['django', 'from django'],
            'dirs': ['migrations'],
            'package_deps': {'django'}
        },
        'fastapi': {
            'imports': ['from fastapi import', 'fastapi.'],
            'package_deps': {'fastapi'}
        },
        'flask': {
            'imports': ['from flask import', 'flask.'],
            'package_deps': {'flask'}
        },

        # JavaScript/TypeScript Frontend
        'react': {
            'files': [],
            'imports': ['from react', 'from "react"', "from 'react'"],
            'package_deps': {'react'},
            'file_patterns': ['.jsx', '.tsx']
        },
        'vue': {
            'imports': ['from vue', 'from "vue"', "from 'vue'"],
            'package_deps': {'vue'},
            'file_patterns': ['.vue']
        },
        'angular': {
            'files': ['angular.json'],
            'package_deps': {'@angular/core'}
        },
        'next': {
            'files': ['next.config.js', 'next.config.ts'],
            'package_deps': {'next'}
        },

        # Python Data Science
        'jupyter': {
            'file_patterns': ['.ipynb']
        }
    }

    # Language detection patterns
    LANGUAGE_EXTENSIONS = {
        'python': ['.py', '.pyx', '.pyi'],
        'javascript': ['.js', '.mjs', '.cjs'],
        'typescript': ['.ts', '.tsx'],
        'java': ['.java'],
        'go': ['.go'],
        'rust': ['.rs'],
        'c': ['.c', '.h'],
        'cpp': ['.cpp', '.hpp', '.cc', '.cxx'],
        'ruby': ['.rb'],
        'php': ['.php'],
        'swift': ['.swift'],
        'kotlin': ['.kt', '.kts'],
    }

    def detect(self, repo_path: Path) -> Dict[str, any]:
        """
        Detect project type, languages, and frameworks.

        Args:
            repo_path: Path to repository

        Returns:
            Dictionary with:
            - project_type: 'frontend', 'backend', 'fullstack', 'library', 'unknown'
            - languages: List of detected languages
            - frameworks: List of detected frameworks
            - confidence: Confidence score (0-1)
        """
        logger.info(f"Detecting project type for {repo_path.name}")

        # Detect languages
        languages = self._detect_languages(repo_path)
        logger.info(f"Detected languages: {', '.join(languages)}")

        # Detect frameworks
        frameworks = self._detect_frameworks(repo_path)
        logger.info(f"Detected frameworks: {', '.join(frameworks)}")

        # Determine project type
        project_type = self._determine_project_type(languages, frameworks)
        logger.info(f"Project type: {project_type}")

        return {
            'project_type': project_type,
            'languages': languages,
            'frameworks': frameworks,
            'confidence': self._calculate_confidence(languages, frameworks)
        }

    def _detect_languages(self, repo_path: Path) -> List[str]:
        """Detect programming languages by file extensions."""
        extension_counts = {}

        # Count file extensions
        for ext_list in self.LANGUAGE_EXTENSIONS.values():
            for ext in ext_list:
                extension_counts[ext] = 0

        # Walk repository
        for file_path in repo_path.rglob('*'):
            if file_path.is_file():
                ext = file_path.suffix.lower()
                if ext in extension_counts:
                    extension_counts[ext] += 1

        # Map extensions to languages
        languages = set()
        for language, extensions in self.LANGUAGE_EXTENSIONS.items():
            count = sum(extension_counts.get(ext, 0) for ext in extensions)
            if count > 0:
                languages.add(language)

        return sorted(list(languages))

    def _detect_frameworks(self, repo_path: Path) -> List[str]:
        """Detect frameworks by analyzing project files."""
        detected = set()

        # Check package.json for JavaScript frameworks
        package_json = repo_path / 'package.json'
        if package_json.exists():
            js_frameworks = self._check_package_json(package_json)
            detected.update(js_frameworks)

        # Check requirements.txt / pyproject.toml for Python frameworks
        requirements_txt = repo_path / 'requirements.txt'
        if requirements_txt.exists():
            py_frameworks = self._check_requirements_txt(requirements_txt)
            detected.update(py_frameworks)

        pyproject_toml = repo_path / 'pyproject.toml'
        if pyproject_toml.exists():
            py_frameworks = self._check_pyproject_toml(pyproject_toml)
            detected.update(py_frameworks)

        # Check for specific files
        for framework, patterns in self.FRAMEWORK_PATTERNS.items():
            if 'files' in patterns:
                for filename in patterns['files']:
                    if (repo_path / filename).exists():
                        detected.add(framework)
                        break

        # Check for file patterns (extensions)
        for framework, patterns in self.FRAMEWORK_PATTERNS.items():
            if 'file_patterns' in patterns:
                for pattern in patterns['file_patterns']:
                    files = list(repo_path.rglob(f'*{pattern}'))
                    if files:
                        detected.add(framework)
                        break

        # Check source files for imports
        detected.update(self._check_imports(repo_path))

        return sorted(list(detected))

    def _check_package_json(self, package_json_path: Path) -> Set[str]:
        """Check package.json for JavaScript/TypeScript frameworks."""
        try:
            with open(package_json_path) as f:
                data = json.load(f)

            dependencies = set()
            dependencies.update(data.get('dependencies', {}).keys())
            dependencies.update(data.get('devDependencies', {}).keys())

            detected = set()
            for framework, patterns in self.FRAMEWORK_PATTERNS.items():
                if 'package_deps' in patterns:
                    if patterns['package_deps'] & dependencies:
                        detected.add(framework)

            return detected

        except Exception as e:
            logger.warning(f"Failed to parse package.json: {e}")
            return set()

    def _check_requirements_txt(self, requirements_path: Path) -> Set[str]:
        """Check requirements.txt for Python frameworks."""
        try:
            with open(requirements_path) as f:
                requirements = set()
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        # Extract package name (before ==, >=, etc.)
                        pkg = line.split('==')[0].split('>=')[0].split('<=')[0].strip()
                        requirements.add(pkg.lower())

            detected = set()
            for framework, patterns in self.FRAMEWORK_PATTERNS.items():
                if 'package_deps' in patterns:
                    if patterns['package_deps'] & requirements:
                        detected.add(framework)

            return detected

        except Exception as e:
            logger.warning(f"Failed to parse requirements.txt: {e}")
            return set()

    def _check_pyproject_toml(self, pyproject_path: Path) -> Set[str]:
        """Check pyproject.toml for Python frameworks."""
        try:
            # Simple text parsing (avoid toml dependency for now)
            with open(pyproject_path) as f:
                content = f.read().lower()

            detected = set()
            for framework, patterns in self.FRAMEWORK_PATTERNS.items():
                if 'package_deps' in patterns:
                    for dep in patterns['package_deps']:
                        if dep in content:
                            detected.add(framework)

            return detected

        except Exception as e:
            logger.warning(f"Failed to parse pyproject.toml: {e}")
            return set()

    def _check_imports(self, repo_path: Path) -> Set[str]:
        """Check source files for framework imports."""
        detected = set()

        # Check Python files
        py_files = list(repo_path.rglob('*.py'))[:50]  # Sample first 50
        for py_file in py_files:
            try:
                with open(py_file, encoding='utf-8', errors='ignore') as f:
                    content = f.read(5000)  # First 5KB

                for framework, patterns in self.FRAMEWORK_PATTERNS.items():
                    if 'imports' in patterns:
                        for import_pattern in patterns['imports']:
                            if import_pattern in content:
                                detected.add(framework)
                                break

            except Exception:
                continue

        # Check JS/TS files
        js_files = list(repo_path.rglob('*.{js,jsx,ts,tsx}'))[:50]
        for js_file in js_files:
            try:
                with open(js_file, encoding='utf-8', errors='ignore') as f:
                    content = f.read(5000)

                for framework, patterns in self.FRAMEWORK_PATTERNS.items():
                    if 'imports' in patterns:
                        for import_pattern in patterns['imports']:
                            if import_pattern in content:
                                detected.add(framework)
                                break

            except Exception:
                continue

        return detected

    def _determine_project_type(
        self,
        languages: List[str],
        frameworks: List[str]
    ) -> str:
        """
        Determine project type based on languages and frameworks.

        Returns:
            'frontend', 'backend', 'fullstack', 'library', or 'unknown'
        """
        frontend_frameworks = {'react', 'vue', 'angular', 'next'}
        backend_frameworks = {'django', 'fastapi', 'flask'}

        has_frontend = bool(frontend_frameworks & set(frameworks))
        has_backend = bool(backend_frameworks & set(frameworks))

        # Check languages too
        frontend_langs = {'javascript', 'typescript'}
        backend_langs = {'python', 'java', 'go', 'ruby', 'php'}

        has_frontend_lang = bool(frontend_langs & set(languages))
        has_backend_lang = bool(backend_langs & set(languages))

        # Determine type
        if has_frontend and has_backend:
            return 'fullstack'
        elif has_frontend or (has_frontend_lang and not has_backend_lang):
            return 'frontend'
        elif has_backend or (has_backend_lang and not has_frontend_lang):
            return 'backend'
        elif languages:
            return 'library'
        else:
            return 'unknown'

    def _calculate_confidence(
        self,
        languages: List[str],
        frameworks: List[str]
    ) -> float:
        """Calculate confidence score for detection."""
        # High confidence if we found both languages and frameworks
        if languages and frameworks:
            return 0.9
        elif languages:
            return 0.7
        elif frameworks:
            return 0.8
        else:
            return 0.3
