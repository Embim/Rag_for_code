"""
Python AST parser.

Uses Python's built-in ast module to extract structured information
from Python source code: functions, classes, imports, etc.
"""

import ast
from pathlib import Path
from typing import List, Optional, Dict, Any

from .base import (
    BaseParser,
    CodeEntity,
    EntityType,
    ParseResult,
    ParsingError,
    register_parser
)
from src.logger import get_logger


logger = get_logger(__name__)


class PythonParser(BaseParser):
    """
    Parser for Python source code.

    Uses Python's ast module to extract:
    - Functions and their signatures
    - Classes and methods
    - Imports
    - Docstrings
    - Type hints
    - Decorators
    """

    def get_supported_extensions(self) -> List[str]:
        """Python file extensions."""
        return ['.py', '.pyx', '.pyi']

    def parse_file(self, file_path: Path) -> ParseResult:
        """
        Parse a Python file and extract entities.

        Args:
            file_path: Path to Python file

        Returns:
            ParseResult with extracted entities
        """
        try:
            # Read file
            with open(file_path, 'r', encoding='utf-8') as f:
                source = f.read()

            # Parse AST
            try:
                tree = ast.parse(source)
            except SyntaxError as e:
                return ParseResult(
                    file_path=file_path,
                    language='python',
                    entities=[],
                    errors=[f"Syntax error: {e}"]
                )

            # Extract entities
            entities = []
            imports = []

            # Extract module-level imports
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ''
                    for alias in node.names:
                        imports.append(f"{module}.{alias.name}" if module else alias.name)

            # Process module body
            for node in tree.body:
                if isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
                    entity = self._extract_function(node, file_path, source)
                    if entity:
                        entities.append(entity)

                elif isinstance(node, ast.ClassDef):
                    class_entity = self._extract_class(node, file_path, source)
                    if class_entity:
                        entities.append(class_entity)

                    # Extract methods
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef) or isinstance(item, ast.AsyncFunctionDef):
                            method_entity = self._extract_function(
                                item,
                                file_path,
                                source,
                                parent_class=node.name
                            )
                            if method_entity:
                                entities.append(method_entity)

            return ParseResult(
                file_path=file_path,
                language='python',
                entities=entities,
                imports=imports,
                errors=[]
            )

        except Exception as e:
            logger.error(f"Failed to parse {file_path}: {e}")
            return ParseResult(
                file_path=file_path,
                language='python',
                entities=[],
                errors=[str(e)]
            )

    def _extract_function(
        self,
        node: ast.FunctionDef,
        file_path: Path,
        source: str,
        parent_class: Optional[str] = None
    ) -> Optional[CodeEntity]:
        """Extract function/method information."""
        try:
            # Get function name
            name = node.name

            # Get line numbers
            start_line = node.lineno
            end_line = node.end_lineno or start_line

            # Extract code
            lines = source.split('\n')
            code = '\n'.join(lines[start_line - 1:end_line])

            # Get docstring
            docstring = ast.get_docstring(node)

            # Get decorators
            decorators = [self._get_decorator_name(dec) for dec in node.decorator_list]

            # Build signature
            signature = self._build_signature(node)

            # Extract parameters
            parameters = self._extract_parameters(node.args)

            # Get return type
            return_type = None
            if node.returns:
                return_type = ast.unparse(node.returns)

            # Extract function calls
            calls = self._extract_function_calls(node)

            # Determine entity type
            entity_type = EntityType.METHOD if parent_class else EntityType.FUNCTION

            return CodeEntity(
                type=entity_type,
                name=name,
                file_path=file_path,
                start_line=start_line,
                end_line=end_line,
                code=code,
                signature=signature,
                docstring=docstring,
                parent=parent_class,
                calls=calls,  # Add extracted calls
                decorators=decorators,
                parameters=parameters,
                return_type=return_type,
                metadata={
                    'decorators': decorators,
                    'is_async': isinstance(node, ast.AsyncFunctionDef),
                    'is_property': '@property' in decorators,
                    'is_staticmethod': '@staticmethod' in decorators,
                    'is_classmethod': '@classmethod' in decorators,
                }
            )

        except Exception as e:
            logger.warning(f"Failed to extract function {node.name}: {e}")
            return None

    def _extract_class(
        self,
        node: ast.ClassDef,
        file_path: Path,
        source: str
    ) -> Optional[CodeEntity]:
        """Extract class information."""
        try:
            name = node.name
            start_line = node.lineno
            end_line = node.end_lineno or start_line

            # Extract code (class definition)
            lines = source.split('\n')
            code = '\n'.join(lines[start_line - 1:end_line])

            # Get docstring
            docstring = ast.get_docstring(node)

            # Get base classes
            bases = []
            for base in node.bases:
                if isinstance(base, ast.Name):
                    bases.append(base.id)
                elif isinstance(base, ast.Attribute):
                    bases.append(ast.unparse(base))

            # Get decorators
            decorators = [self._get_decorator_name(dec) for dec in node.decorator_list]

            return CodeEntity(
                type=EntityType.CLASS,
                name=name,
                file_path=file_path,
                start_line=start_line,
                end_line=end_line,
                code=code,
                signature=f"class {name}({', '.join(bases)})" if bases else f"class {name}",
                docstring=docstring,
                decorators=decorators,
                metadata={
                    'base_classes': bases,
                    'is_dataclass': '@dataclass' in decorators,
                }
            )

        except Exception as e:
            logger.warning(f"Failed to extract class {node.name}: {e}")
            return None

    def _build_signature(self, node: ast.FunctionDef) -> str:
        """Build function signature string."""
        try:
            # Get function name
            name = node.name

            # Get arguments
            args_str = self._format_arguments(node.args)

            # Get return type
            returns = ''
            if node.returns:
                returns = f" -> {ast.unparse(node.returns)}"

            # Add async prefix if needed
            prefix = 'async ' if isinstance(node, ast.AsyncFunctionDef) else ''

            return f"{prefix}def {name}({args_str}){returns}"

        except Exception:
            return f"def {node.name}(...)"

    def _format_arguments(self, args: ast.arguments) -> str:
        """Format function arguments."""
        try:
            parts = []

            # Regular arguments
            for i, arg in enumerate(args.args):
                arg_str = arg.arg
                if arg.annotation:
                    arg_str += f": {ast.unparse(arg.annotation)}"

                # Add default value if present
                defaults_offset = len(args.args) - len(args.defaults)
                default_idx = i - defaults_offset
                if default_idx >= 0:
                    default = ast.unparse(args.defaults[default_idx])
                    arg_str += f" = {default}"

                parts.append(arg_str)

            # *args
            if args.vararg:
                vararg_str = f"*{args.vararg.arg}"
                if args.vararg.annotation:
                    vararg_str += f": {ast.unparse(args.vararg.annotation)}"
                parts.append(vararg_str)

            # Keyword-only arguments
            for i, arg in enumerate(args.kwonlyargs):
                arg_str = arg.arg
                if arg.annotation:
                    arg_str += f": {ast.unparse(arg.annotation)}"
                if i < len(args.kw_defaults) and args.kw_defaults[i]:
                    arg_str += f" = {ast.unparse(args.kw_defaults[i])}"
                parts.append(arg_str)

            # **kwargs
            if args.kwarg:
                kwarg_str = f"**{args.kwarg.arg}"
                if args.kwarg.annotation:
                    kwarg_str += f": {ast.unparse(args.kwarg.annotation)}"
                parts.append(kwarg_str)

            return ', '.join(parts)

        except Exception:
            return '...'

    def _extract_parameters(self, args: ast.arguments) -> List[Dict[str, Any]]:
        """Extract parameter information."""
        parameters = []

        # Regular arguments
        for arg in args.args:
            param = {
                'name': arg.arg,
                'type': ast.unparse(arg.annotation) if arg.annotation else None,
                'kind': 'positional'
            }
            parameters.append(param)

        # *args
        if args.vararg:
            param = {
                'name': args.vararg.arg,
                'type': ast.unparse(args.vararg.annotation) if args.vararg.annotation else None,
                'kind': 'var_positional'
            }
            parameters.append(param)

        # Keyword-only
        for arg in args.kwonlyargs:
            param = {
                'name': arg.arg,
                'type': ast.unparse(arg.annotation) if arg.annotation else None,
                'kind': 'keyword_only'
            }
            parameters.append(param)

        # **kwargs
        if args.kwarg:
            param = {
                'name': args.kwarg.arg,
                'type': ast.unparse(args.kwarg.annotation) if args.kwarg.annotation else None,
                'kind': 'var_keyword'
            }
            parameters.append(param)

        return parameters

    def _get_decorator_name(self, decorator: ast.expr) -> str:
        """Get decorator name as string."""
        try:
            if isinstance(decorator, ast.Name):
                return f"@{decorator.id}"
            elif isinstance(decorator, ast.Call):
                if isinstance(decorator.func, ast.Name):
                    return f"@{decorator.func.id}"
                else:
                    return f"@{ast.unparse(decorator.func)}"
            else:
                return f"@{ast.unparse(decorator)}"
        except Exception:
            return "@unknown"

    def _extract_function_calls(self, node: ast.FunctionDef) -> List[str]:
        """
        Extract all function calls made within a function.

        Returns a list of function names that this function calls.
        Handles:
        - Simple calls: foo()
        - Method calls: obj.method()
        - Chained calls: obj.method().another()
        """
        calls = []
        seen = set()  # Avoid duplicates

        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                call_name = None

                if isinstance(child.func, ast.Name):
                    # Simple function call: foo()
                    call_name = child.func.id

                elif isinstance(child.func, ast.Attribute):
                    # Method call: obj.method() or module.function()
                    # Extract just the method/function name
                    call_name = child.func.attr

                    # Optionally include the full path for clarity
                    try:
                        full_name = ast.unparse(child.func)
                        # Only include if it's a module.function pattern (not self.method)
                        if not full_name.startswith('self.'):
                            call_name = full_name
                    except Exception:
                        pass

                if call_name and call_name not in seen:
                    seen.add(call_name)
                    calls.append(call_name)

        return calls
