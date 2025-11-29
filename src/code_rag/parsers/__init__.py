"""
Code parsers for different languages and frameworks.

Parsers extract structured information from code files:
- Functions, classes, methods
- Imports and dependencies
- Docstrings and comments
- Type information

Available parsers:
- PythonParser: Basic Python parsing with AST
- DjangoParser: Django-specific entities (models, views, etc.)
- FastAPIParser: FastAPI endpoints and Pydantic models
- ReactParser: React components and hooks
"""

from pathlib import Path
from typing import Optional

from .base import (
    BaseParser,
    CodeEntity,
    EntityType,
    ParseResult,
    ParsingError,
    register_parser,
)
from .python_parser import PythonParser
from .django_parser import DjangoParser
from .fastapi_parser import FastAPIParser
from .react_parser import ReactParser


def get_parser(file_path: Path, framework: Optional[str] = None):
    """
    Get appropriate parser for a file based on extension and framework.

    Args:
        file_path: Path to the file
        framework: Optional framework hint (django, fastapi, react, etc.)

    Returns:
        Parser instance
    """
    extension = file_path.suffix.lower()

    # Python files
    if extension == '.py':
        if framework == 'django':
            return DjangoParser()
        elif framework == 'fastapi':
            return FastAPIParser()
        else:
            # Default to Python parser
            return PythonParser()

    # JavaScript/TypeScript files
    elif extension in {'.js', '.jsx', '.ts', '.tsx'}:
        if framework in {'react', 'next', 'nextjs'}:
            return ReactParser()
        else:
            # For now, use React parser for all JS/TS
            return ReactParser()

    else:
        # Unsupported file type, return None or base parser
        return None


__all__ = [
    'BaseParser',
    'CodeEntity', 
    'EntityType',
    'ParseResult',
    'ParsingError',
    'register_parser',
    'PythonParser', 
    'DjangoParser', 
    'FastAPIParser', 
    'ReactParser', 
    'get_parser',
]
