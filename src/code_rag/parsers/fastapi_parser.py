"""
FastAPI parser.

Extracts FastAPI-specific entities:
- API endpoints (@app.get, @app.post, etc.)
- Pydantic models (BaseModel)
- Dependencies (Depends)
- APIRouter instances
"""

import ast
import re
from pathlib import Path
from typing import List, Optional, Dict, Any, Set

from .base import (
    BaseParser,
    CodeEntity,
    EntityType,
    ParseResult,
    ParsingError,
    register_parser
)
from .python_parser import PythonParser
from src.logger import get_logger


logger = get_logger(__name__)


@register_parser
class FastAPIParser(BaseParser):
    """
    Parser for FastAPI code.

    Detects and extracts:
    - API endpoints with decorators
    - Pydantic models for request/response
    - Dependencies and middleware
    - APIRouter configuration
    """

    # HTTP methods supported by FastAPI
    HTTP_METHODS = ['get', 'post', 'put', 'delete', 'patch', 'options', 'head', 'trace']

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.python_parser = PythonParser(config)

    def get_supported_extensions(self) -> List[str]:
        """FastAPI uses Python files."""
        return ['.py']

    def parse_file(self, file_path: Path) -> ParseResult:
        """
        Parse FastAPI file.

        Args:
            file_path: Path to Python file

        Returns:
            ParseResult with FastAPI-specific entities
        """
        # First, use Python parser
        result = self.python_parser.parse_file(file_path)

        if not result.success:
            return result

        # Read source for analysis
        with open(file_path, 'r', encoding='utf-8') as f:
            source = f.read()

        # Check if this is FastAPI code
        if not self._is_fastapi_file(result.imports, source):
            return result

        # Enhance entities
        self._enhance_endpoints(result, source)
        self._enhance_pydantic_models(result, source)

        return result

    def _is_fastapi_file(self, imports: List[str], source: str) -> bool:
        """Check if file contains FastAPI code."""
        # Check imports
        fastapi_imports = [
            'fastapi', 'FastAPI', 'APIRouter',
            'Depends', 'Query', 'Path', 'Body'
        ]

        for imp in imports:
            if any(fi in imp for fi in fastapi_imports):
                return True

        # Check source for FastAPI patterns
        if 'from fastapi import' in source or 'import fastapi' in source:
            return True

        return False

    def _enhance_endpoints(self, result: ParseResult, source: str) -> None:
        """Enhance functions that are FastAPI endpoints."""
        for entity in result.entities:
            if entity.type != EntityType.FUNCTION:
                continue

            # Check decorators for endpoint markers
            endpoint_info = self._extract_endpoint_info(entity)

            if endpoint_info:
                # Change type to ENDPOINT
                entity.type = EntityType.ENDPOINT

                # Add endpoint metadata
                entity.metadata.update(endpoint_info)

                # Extract request/response models
                models = self._extract_endpoint_models(entity, source)
                entity.metadata['request_model'] = models.get('request')
                entity.metadata['response_model'] = models.get('response')

                # Extract dependencies
                deps = self._extract_dependencies(entity, source)
                entity.metadata['dependencies'] = deps

                # Extract path/query parameters
                params = self._extract_endpoint_parameters(entity)
                entity.metadata['path_params'] = params.get('path', [])
                entity.metadata['query_params'] = params.get('query', [])
                entity.metadata['body_params'] = params.get('body', [])

    def _extract_endpoint_info(
        self,
        entity: CodeEntity
    ) -> Optional[Dict[str, Any]]:
        """Extract endpoint information from decorators."""
        for decorator in entity.decorators:
            # Match patterns like @app.get("/users") or @router.post("/create")
            for method in self.HTTP_METHODS:
                # Pattern: @<app_name>.<method>("<path>", ...)
                pattern = rf"@\w+\.{method}\s*\(['\"]([^'\"]+)['\"]"
                match = re.search(pattern, decorator)

                if match:
                    path = match.group(1)
                    return {
                        'http_method': method.upper(),
                        'path': path,
                        'full_path': path,  # Will be updated with router prefix later
                        'endpoint_decorator': decorator
                    }

        return None

    def _extract_endpoint_models(
        self,
        entity: CodeEntity,
        source: str
    ) -> Dict[str, Optional[str]]:
        """Extract request and response models from endpoint."""
        models = {'request': None, 'response': None}

        # Check function signature for response model
        if entity.return_type:
            models['response'] = entity.return_type

        # Check for request body in parameters
        for param in entity.parameters:
            param_type = param.get('type')
            if param_type and not param_type in ['str', 'int', 'bool', 'float']:
                # Likely a Pydantic model
                models['request'] = param_type

        # Check decorator for response_model
        for decorator in entity.decorators:
            if 'response_model' in decorator:
                # Extract response_model=... from decorator
                match = re.search(r'response_model\s*=\s*(\w+)', decorator)
                if match:
                    models['response'] = match.group(1)

        return models

    def _extract_dependencies(
        self,
        entity: CodeEntity,
        source: str
    ) -> List[Dict[str, str]]:
        """Extract dependencies (Depends) from function parameters."""
        dependencies = []

        # Look for Depends() in function code
        try:
            # Parse just the function to get parameter defaults
            tree = ast.parse(entity.code)
            func_node = tree.body[0]

            if isinstance(func_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Check each parameter for Depends
                for i, arg in enumerate(func_node.args.args):
                    # Check if there's a default value
                    defaults_offset = len(func_node.args.args) - len(func_node.args.defaults)
                    default_idx = i - defaults_offset

                    if default_idx >= 0 and default_idx < len(func_node.args.defaults):
                        default = func_node.args.defaults[default_idx]

                        # Check if it's a Depends() call
                        if isinstance(default, ast.Call):
                            if isinstance(default.func, ast.Name) and default.func.id == 'Depends':
                                # Get the dependency function
                                if default.args:
                                    dep_func = ast.unparse(default.args[0])
                                    dependencies.append({
                                        'param': arg.arg,
                                        'dependency': dep_func
                                    })

        except Exception as e:
            logger.debug(f"Failed to extract dependencies: {e}")

        return dependencies

    def _extract_endpoint_parameters(
        self,
        entity: CodeEntity
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Extract path, query, and body parameters."""
        params = {
            'path': [],
            'query': [],
            'body': []
        }

        try:
            # Parse function to check parameter annotations
            tree = ast.parse(entity.code)
            func_node = tree.body[0]

            if isinstance(func_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for arg in func_node.args.args:
                    param_name = arg.arg

                    # Skip 'self' and 'cls'
                    if param_name in ['self', 'cls']:
                        continue

                    # Check annotation for Path(), Query(), Body()
                    param_type = 'unknown'

                    # Check defaults for Path/Query/Body
                    arg_idx = func_node.args.args.index(arg)
                    defaults_offset = len(func_node.args.args) - len(func_node.args.defaults)
                    default_idx = arg_idx - defaults_offset

                    if default_idx >= 0 and default_idx < len(func_node.args.defaults):
                        default = func_node.args.defaults[default_idx]

                        if isinstance(default, ast.Call):
                            if isinstance(default.func, ast.Name):
                                func_name = default.func.id

                                if func_name == 'Path':
                                    param_type = 'path'
                                elif func_name == 'Query':
                                    param_type = 'query'
                                elif func_name == 'Body':
                                    param_type = 'body'

                    # Add to appropriate list
                    if param_type != 'unknown':
                        param_info = {
                            'name': param_name,
                            'type': ast.unparse(arg.annotation) if arg.annotation else None
                        }
                        params[param_type].append(param_info)

        except Exception as e:
            logger.debug(f"Failed to extract parameters: {e}")

        return params

    def _enhance_pydantic_models(self, result: ParseResult, source: str) -> None:
        """Enhance Pydantic models."""
        for entity in result.entities:
            if entity.type != EntityType.CLASS:
                continue

            # Check if it's a Pydantic model
            if self._is_pydantic_model(entity):
                entity.type = EntityType.MODEL
                entity.metadata['is_pydantic'] = True

                # Extract model fields
                fields = self._extract_pydantic_fields(entity, source)
                entity.metadata['fields'] = fields

    def _is_pydantic_model(self, entity: CodeEntity) -> bool:
        """Check if class is a Pydantic model."""
        base_classes = entity.metadata.get('base_classes', [])

        # Check for BaseModel
        if 'BaseModel' in base_classes:
            return True

        # Check for pydantic.BaseModel
        if any('BaseModel' in base for base in base_classes):
            return True

        return False

    def _extract_pydantic_fields(
        self,
        entity: CodeEntity,
        source: str
    ) -> List[Dict[str, Any]]:
        """Extract Pydantic model fields."""
        fields = []

        try:
            tree = ast.parse(entity.code)
            class_node = tree.body[0]

            for node in class_node.body:
                # Pydantic fields are class-level annotated assignments
                if isinstance(node, ast.AnnAssign):
                    if isinstance(node.target, ast.Name):
                        field_name = node.target.name

                        # Skip private fields
                        if field_name.startswith('_'):
                            continue

                        field_type = ast.unparse(node.annotation) if node.annotation else None

                        # Get default value
                        default_value = None
                        if node.value:
                            try:
                                default_value = ast.literal_eval(node.value)
                            except Exception:
                                try:
                                    default_value = ast.unparse(node.value)
                                except Exception:
                                    pass

                        field_info = {
                            'name': field_name,
                            'type': field_type,
                            'default': default_value
                        }

                        fields.append(field_info)

        except Exception as e:
            logger.warning(f"Failed to extract Pydantic fields: {e}")

        return fields
