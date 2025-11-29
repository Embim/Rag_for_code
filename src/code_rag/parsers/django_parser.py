"""
Django-specific parser.

Extends Python parser to extract Django-specific entities:
- Models (django.db.models.Model)
- Views (function-based and class-based)
- URL patterns (urls.py)
- DRF ViewSets and Serializers
"""

import ast
import re
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
from .python_parser import PythonParser
from src.logger import get_logger


logger = get_logger(__name__)


@register_parser
class DjangoParser(BaseParser):
    """
    Parser for Django-specific code.

    Detects and extracts:
    - Django models with fields and relationships
    - Function-based and class-based views
    - URL patterns
    - DRF ViewSets and Serializers
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.python_parser = PythonParser(config)

    def get_supported_extensions(self) -> List[str]:
        """Django uses Python files."""
        return ['.py']

    def parse_file(self, file_path: Path) -> ParseResult:
        """
        Parse Django file.

        Args:
            file_path: Path to Python file

        Returns:
            ParseResult with Django-specific entities
        """
        # First, use Python parser to get base entities
        result = self.python_parser.parse_file(file_path)

        if not result.success:
            return result

        # Read source for analysis
        with open(file_path, 'r', encoding='utf-8') as f:
            source = f.read()

        # Determine file type
        file_name = file_path.name

        # Parse based on file type
        if file_name == 'models.py':
            self._enhance_models(result, source)
        elif file_name == 'views.py':
            self._enhance_views(result, source)
        elif file_name == 'urls.py':
            self._parse_urls(result, file_path, source)
        elif file_name == 'serializers.py':
            self._enhance_serializers(result, source)

        return result

    def _enhance_models(self, result: ParseResult, source: str) -> None:
        """Enhance model classes with Django-specific information."""
        for entity in result.entities:
            if entity.type != EntityType.CLASS:
                continue

            # Check if this is a Django model
            if not self._is_django_model(entity, source):
                continue

            # Change type to MODEL
            entity.type = EntityType.MODEL

            # Extract fields
            fields = self._extract_model_fields(entity, source)
            entity.metadata['fields'] = fields

            # Extract relationships
            relationships = self._extract_relationships(entity, source)
            entity.metadata['relationships'] = relationships

            # Extract Meta class info
            meta = self._extract_meta_class(entity, source)
            if meta:
                entity.metadata['meta'] = meta

    def _is_django_model(self, entity: CodeEntity, source: str) -> bool:
        """Check if class is a Django model."""
        # Check base classes in metadata
        base_classes = entity.metadata.get('base_classes', [])

        # Direct inheritance
        if 'Model' in base_classes:
            return True

        # Check for models.Model in source
        if 'models.Model' in entity.code:
            return True

        return False

    def _extract_model_fields(
        self,
        entity: CodeEntity,
        source: str
    ) -> List[Dict[str, Any]]:
        """Extract Django model fields."""
        fields = []

        # Parse the class code
        try:
            tree = ast.parse(entity.code)
            class_node = tree.body[0]

            for node in class_node.body:
                if isinstance(node, ast.Assign):
                    # Field assignment: name = models.CharField(...)
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            field_name = target.name

                            # Skip private fields and Meta
                            if field_name.startswith('_') or field_name == 'Meta':
                                continue

                            # Check if right side is a field type
                            if isinstance(node.value, ast.Call):
                                field_type = self._get_field_type(node.value)

                                if field_type:
                                    field_info = {
                                        'name': field_name,
                                        'type': field_type,
                                        'options': self._extract_field_options(node.value)
                                    }
                                    fields.append(field_info)

        except Exception as e:
            logger.warning(f"Failed to extract model fields: {e}")

        return fields

    def _get_field_type(self, call_node: ast.Call) -> Optional[str]:
        """Get Django field type from call node."""
        try:
            if isinstance(call_node.func, ast.Attribute):
                # models.CharField -> CharField
                return call_node.func.attr
            elif isinstance(call_node.func, ast.Name):
                # CharField (imported directly)
                return call_node.func.id
            return None
        except Exception:
            return None

    def _extract_field_options(self, call_node: ast.Call) -> Dict[str, Any]:
        """Extract field options (max_length, null, blank, etc.)."""
        options = {}

        try:
            # Extract keyword arguments
            for keyword in call_node.keywords:
                key = keyword.arg
                try:
                    # Try to evaluate simple values
                    value = ast.literal_eval(keyword.value)
                    options[key] = value
                except Exception:
                    # For complex values, just store as string
                    try:
                        options[key] = ast.unparse(keyword.value)
                    except Exception:
                        options[key] = str(keyword.value)

        except Exception as e:
            logger.debug(f"Failed to extract field options: {e}")

        return options

    def _extract_relationships(
        self,
        entity: CodeEntity,
        source: str
    ) -> List[Dict[str, Any]]:
        """Extract model relationships (ForeignKey, ManyToMany, OneToOne)."""
        relationships = []

        relationship_types = ['ForeignKey', 'ManyToManyField', 'OneToOneField']

        try:
            tree = ast.parse(entity.code)
            class_node = tree.body[0]

            for node in class_node.body:
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            field_name = target.name

                            if isinstance(node.value, ast.Call):
                                field_type = self._get_field_type(node.value)

                                if field_type in relationship_types:
                                    # Get related model (first positional argument)
                                    related_model = None
                                    if node.value.args:
                                        try:
                                            arg = node.value.args[0]
                                            if isinstance(arg, ast.Constant):
                                                related_model = arg.value
                                            elif isinstance(arg, ast.Name):
                                                related_model = arg.id
                                            elif isinstance(arg, ast.Attribute):
                                                related_model = ast.unparse(arg)
                                        except Exception:
                                            pass

                                    rel_info = {
                                        'field_name': field_name,
                                        'type': field_type,
                                        'related_model': related_model,
                                        'options': self._extract_field_options(node.value)
                                    }
                                    relationships.append(rel_info)

        except Exception as e:
            logger.warning(f"Failed to extract relationships: {e}")

        return relationships

    def _extract_meta_class(
        self,
        entity: CodeEntity,
        source: str
    ) -> Optional[Dict[str, Any]]:
        """Extract Meta class information."""
        try:
            tree = ast.parse(entity.code)
            class_node = tree.body[0]

            for node in class_node.body:
                if isinstance(node, ast.ClassDef) and node.name == 'Meta':
                    meta_info = {}

                    for item in node.body:
                        if isinstance(item, ast.Assign):
                            for target in item.targets:
                                if isinstance(target, ast.Name):
                                    key = target.name
                                    try:
                                        value = ast.literal_eval(item.value)
                                        meta_info[key] = value
                                    except Exception:
                                        try:
                                            meta_info[key] = ast.unparse(item.value)
                                        except Exception:
                                            pass

                    return meta_info

        except Exception as e:
            logger.debug(f"Failed to extract Meta class: {e}")

        return None

    def _enhance_views(self, result: ParseResult, source: str) -> None:
        """Enhance views with Django-specific information."""
        # Check for Django view imports
        has_django_views = any(
            'django.views' in imp or 'django.http' in imp
            for imp in result.imports
        )

        if not has_django_views:
            return

        for entity in result.entities:
            # Function-based views
            if entity.type == EntityType.FUNCTION:
                # Check if it looks like a view (has request parameter)
                if entity.parameters and entity.parameters[0]['name'] == 'request':
                    entity.metadata['is_view'] = True
                    entity.metadata['view_type'] = 'function'

            # Class-based views
            elif entity.type == EntityType.CLASS:
                base_classes = entity.metadata.get('base_classes', [])

                # Check if it inherits from View or generic views
                view_base_classes = [
                    'View', 'TemplateView', 'ListView', 'DetailView',
                    'CreateView', 'UpdateView', 'DeleteView', 'FormView'
                ]

                if any(base in base_classes for base in view_base_classes):
                    entity.metadata['is_view'] = True
                    entity.metadata['view_type'] = 'class'
                    entity.metadata['view_base'] = next(
                        base for base in base_classes
                        if base in view_base_classes
                    )

    def _parse_urls(
        self,
        result: ParseResult,
        file_path: Path,
        source: str
    ) -> None:
        """Parse URL patterns from urls.py."""
        try:
            tree = ast.parse(source)

            # Find urlpatterns variable
            for node in tree.body:
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name) and target.name == 'urlpatterns':
                            # Parse URL patterns
                            patterns = self._extract_url_patterns(node.value, source)

                            for pattern in patterns:
                                # Create route entity
                                entity = CodeEntity(
                                    type=EntityType.ROUTE,
                                    name=pattern['name'] or pattern['pattern'],
                                    file_path=file_path,
                                    start_line=pattern.get('line', 1),
                                    end_line=pattern.get('line', 1),
                                    code=pattern['code'],
                                    metadata={
                                        'pattern': pattern['pattern'],
                                        'view': pattern['view'],
                                        'name': pattern['name']
                                    }
                                )
                                result.entities.append(entity)

        except Exception as e:
            logger.error(f"Failed to parse URLs: {e}")

    def _extract_url_patterns(
        self,
        list_node: ast.expr,
        source: str
    ) -> List[Dict[str, Any]]:
        """Extract URL patterns from urlpatterns list."""
        patterns = []

        if not isinstance(list_node, ast.List):
            return patterns

        for element in list_node.elts:
            if isinstance(element, ast.Call):
                # path() or re_path() call
                pattern = self._parse_url_pattern_call(element, source)
                if pattern:
                    patterns.append(pattern)

        return patterns

    def _parse_url_pattern_call(
        self,
        call_node: ast.Call,
        source: str
    ) -> Optional[Dict[str, Any]]:
        """Parse a single path() or re_path() call."""
        try:
            # Get function name
            func_name = None
            if isinstance(call_node.func, ast.Name):
                func_name = call_node.func.id

            if func_name not in ['path', 're_path', 'url']:
                return None

            # First argument is the pattern
            pattern = None
            if call_node.args:
                arg = call_node.args[0]
                if isinstance(arg, ast.Constant):
                    pattern = arg.value

            # Second argument is the view
            view = None
            if len(call_node.args) > 1:
                arg = call_node.args[1]
                try:
                    view = ast.unparse(arg)
                except Exception:
                    view = "unknown"

            # Look for 'name' keyword argument
            name = None
            for keyword in call_node.keywords:
                if keyword.arg == 'name':
                    if isinstance(keyword.value, ast.Constant):
                        name = keyword.value.value

            return {
                'pattern': pattern,
                'view': view,
                'name': name,
                'code': ast.unparse(call_node),
                'line': call_node.lineno
            }

        except Exception as e:
            logger.debug(f"Failed to parse URL pattern: {e}")
            return None

    def _enhance_serializers(self, result: ParseResult, source: str) -> None:
        """Enhance DRF serializers."""
        # Check for DRF imports
        has_drf = any(
            'rest_framework' in imp or 'serializers' in imp
            for imp in result.imports
        )

        if not has_drf:
            return

        for entity in result.entities:
            if entity.type != EntityType.CLASS:
                continue

            base_classes = entity.metadata.get('base_classes', [])

            # Check if it's a serializer
            if any('Serializer' in base for base in base_classes):
                entity.metadata['is_serializer'] = True
                entity.metadata['serializer_base'] = next(
                    base for base in base_classes
                    if 'Serializer' in base
                )

                # Extract serializer fields
                fields = self._extract_serializer_fields(entity, source)
                entity.metadata['serializer_fields'] = fields

    def _extract_serializer_fields(
        self,
        entity: CodeEntity,
        source: str
    ) -> List[Dict[str, Any]]:
        """Extract serializer fields."""
        # Similar to model fields extraction
        # For simplicity, reuse the model field extraction logic
        return self._extract_model_fields(entity, source)
