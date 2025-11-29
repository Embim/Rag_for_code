"""
Tests for Python AST parser.
"""

import pytest
from pathlib import Path
import tempfile

from src.code_rag.parsers.python_parser import PythonParser
from src.core.parsers import EntityType


class TestPythonParser:
    """Test suite for PythonParser."""

    @pytest.fixture
    def parser(self):
        """Create Python parser instance."""
        return PythonParser()

    def test_parser_supports_python_extensions(self, parser):
        """Test that parser recognizes Python file extensions."""
        assert '.py' in parser.get_supported_extensions()
        assert '.pyi' in parser.get_supported_extensions()
        assert '.pyx' in parser.get_supported_extensions()

    def test_parse_simple_function(self, parser, tmp_path):
        """Test parsing a simple function."""
        # Create test file
        code = '''
def hello(name: str) -> str:
    """Say hello to someone."""
    return f"Hello, {name}!"
'''
        file_path = tmp_path / "test.py"
        file_path.write_text(code)

        # Parse
        result = parser.parse_file(file_path)

        # Assertions
        assert result.success
        assert len(result.entities) == 1

        func = result.entities[0]
        assert func.type == EntityType.FUNCTION
        assert func.name == "hello"
        assert func.docstring == "Say hello to someone."
        assert func.return_type == "str"
        assert len(func.parameters) == 1
        assert func.parameters[0]['name'] == 'name'
        assert func.parameters[0]['type'] == 'str'

    def test_parse_class_with_methods(self, parser, tmp_path):
        """Test parsing a class with methods."""
        code = '''
class Calculator:
    """A simple calculator."""

    def add(self, a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    @staticmethod
    def multiply(a: int, b: int) -> int:
        """Multiply two numbers."""
        return a * b
'''
        file_path = tmp_path / "test.py"
        file_path.write_text(code)

        result = parser.parse_file(file_path)

        assert result.success
        assert len(result.entities) == 3  # Class + 2 methods

        # Find entities by type
        classes = [e for e in result.entities if e.type == EntityType.CLASS]
        methods = [e for e in result.entities if e.type == EntityType.METHOD]

        assert len(classes) == 1
        assert len(methods) == 2

        # Check class
        calc_class = classes[0]
        assert calc_class.name == "Calculator"
        assert calc_class.docstring == "A simple calculator."

        # Check methods
        add_method = next(m for m in methods if m.name == "add")
        assert add_method.parent == "Calculator"
        assert add_method.docstring == "Add two numbers."
        assert not add_method.metadata['is_staticmethod']

        multiply_method = next(m for m in methods if m.name == "multiply")
        assert multiply_method.parent == "Calculator"
        assert multiply_method.metadata['is_staticmethod']

    def test_parse_with_decorators(self, parser, tmp_path):
        """Test parsing functions with decorators."""
        code = '''
@app.route('/api/users')
@login_required
def get_users():
    """Get all users."""
    return []
'''
        file_path = tmp_path / "test.py"
        file_path.write_text(code)

        result = parser.parse_file(file_path)

        assert result.success
        func = result.entities[0]
        assert len(func.decorators) == 2
        # Note: decorators are returned in reverse order (bottom-up)
        assert '@app.route' in func.decorators[0] or '@login_required' in func.decorators[0]

    def test_parse_async_function(self, parser, tmp_path):
        """Test parsing async function."""
        code = '''
async def fetch_data(url: str) -> dict:
    """Fetch data from URL."""
    return {"status": "ok"}
'''
        file_path = tmp_path / "test.py"
        file_path.write_text(code)

        result = parser.parse_file(file_path)

        assert result.success
        func = result.entities[0]
        assert func.metadata['is_async']
        assert 'async' in func.signature

    def test_parse_with_imports(self, parser, tmp_path):
        """Test extracting imports."""
        code = '''
import os
import sys
from pathlib import Path
from typing import List, Dict

def process():
    pass
'''
        file_path = tmp_path / "test.py"
        file_path.write_text(code)

        result = parser.parse_file(file_path)

        assert result.success
        assert 'os' in result.imports
        assert 'sys' in result.imports
        assert 'pathlib.Path' in result.imports
        assert 'typing.List' in result.imports or 'typing.Dict' in result.imports

    def test_parse_dataclass(self, parser, tmp_path):
        """Test parsing dataclass."""
        code = '''
from dataclasses import dataclass

@dataclass
class User:
    """User model."""
    name: str
    email: str
    age: int = 0
'''
        file_path = tmp_path / "test.py"
        file_path.write_text(code)

        result = parser.parse_file(file_path)

        assert result.success
        user_class = next(e for e in result.entities if e.type == EntityType.CLASS)
        assert user_class.metadata['is_dataclass']

    def test_parse_property(self, parser, tmp_path):
        """Test parsing property decorator."""
        code = '''
class Circle:
    def __init__(self, radius):
        self._radius = radius

    @property
    def diameter(self):
        """Get diameter."""
        return self._radius * 2
'''
        file_path = tmp_path / "test.py"
        file_path.write_text(code)

        result = parser.parse_file(file_path)

        assert result.success
        diameter_method = next(
            e for e in result.entities
            if e.name == "diameter" and e.type == EntityType.METHOD
        )
        assert diameter_method.metadata['is_property']

    def test_parse_with_syntax_error(self, parser, tmp_path):
        """Test handling syntax errors gracefully."""
        code = '''
def broken(
    # Missing closing parenthesis
    return 42
'''
        file_path = tmp_path / "test.py"
        file_path.write_text(code)

        result = parser.parse_file(file_path)

        # Should not crash, but return empty entities with error
        assert not result.success or len(result.errors) > 0
        assert len(result.entities) == 0

    def test_parse_complex_signature(self, parser, tmp_path):
        """Test parsing complex function signature."""
        code = '''
def complex_func(
    pos_arg: str,
    *args: int,
    kwonly: bool = True,
    **kwargs: str
) -> list[str]:
    """Complex signature."""
    return []
'''
        file_path = tmp_path / "test.py"
        file_path.write_text(code)

        result = parser.parse_file(file_path)

        assert result.success
        func = result.entities[0]
        assert len(func.parameters) == 4

        # Check parameter kinds
        params_by_name = {p['name']: p for p in func.parameters}
        assert params_by_name['pos_arg']['kind'] == 'positional'
        assert params_by_name['args']['kind'] == 'var_positional'
        assert params_by_name['kwonly']['kind'] == 'keyword_only'
        assert params_by_name['kwargs']['kind'] == 'var_keyword'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
