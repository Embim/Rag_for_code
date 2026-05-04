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
from src.infra.logger import get_logger


logger = get_logger(__name__)


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
        super().__init__()
        self.python_parser = PythonParser()

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
                            field_name = target.id

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
                            field_name = target.id

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
                                    key = target.id
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
        # Check for Django REST Framework imports
        has_drf = any(
            'rest_framework' in imp or 'django.views' in imp or 'django.http' in imp
            for imp in result.imports
        )

        if not has_drf:
            return

        for entity in result.entities:
            # Function-based views (especially DRF @api_view)
            if entity.type == EntityType.FUNCTION:
                # Check for DRF @api_view decorator
                decorators = entity.metadata.get('decorators', [])

                if any('api_view' in dec for dec in decorators):
                    # This is a DRF API endpoint!
                    entity.type = EntityType.ENDPOINT

                    # Extract HTTP methods from @api_view(['GET', 'POST'])
                    http_methods = self._extract_api_view_methods(entity, source)
                    entity.metadata['http_method'] = http_methods[0] if http_methods else 'GET'
                    entity.metadata['http_methods'] = http_methods
                    entity.metadata['path'] = f"/{entity.name}"  # Default path
                    entity.metadata['framework'] = 'django-rest-framework'

                # Regular function-based view
                elif entity.parameters and entity.parameters[0]['name'] == 'request':
                    entity.metadata['is_view'] = True
                    entity.metadata['view_type'] = 'function'

            # Class-based views (DRF ViewSets and APIView)
            elif entity.type == EntityType.CLASS:
                base_classes = entity.metadata.get('base_classes', [])

                # Check for DRF ViewSets and APIView
                drf_base_classes = [
                    'APIView', 'ViewSet', 'ModelViewSet', 'ReadOnlyModelViewSet',
                    'GenericViewSet', 'GenericAPIView',
                    'ListAPIView', 'CreateAPIView', 'RetrieveAPIView',
                    'UpdateAPIView', 'DestroyAPIView',
                    'ListCreateAPIView', 'RetrieveUpdateAPIView',
                    'RetrieveDestroyAPIView', 'RetrieveUpdateDestroyAPIView',
                ]

                if any(base in base_classes for base in drf_base_classes):
                    # Mark as view, but don't change to ENDPOINT (class-based)
                    entity.metadata['is_view'] = True
                    entity.metadata['view_type'] = 'class-api'
                    entity.metadata['view_base'] = next(
                        base for base in base_classes
                        if base in drf_base_classes
                    )
                    entity.metadata['framework'] = 'django-rest-framework'
                    entity.metadata['http_methods'] = self._detect_drf_class_methods(
                        entity, base_classes
                    )
                else:
                    # Regular Django views
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
                        entity.metadata['http_methods'] = self._detect_drf_class_methods(
                            entity, base_classes
                        )

    def _detect_drf_class_methods(
        self,
        entity: CodeEntity,
        base_classes: List[str]
    ) -> List[str]:
        """
        Determine HTTP methods supported by a class-based view.

        Combines two sources:
        1. Defined methods in the class body (def get/post/...).
        2. Standard methods provided by the base class
           (e.g. ListCreateAPIView → GET, POST; ModelViewSet → all CRUD).
        """
        methods: set = set()

        # 1. Standard methods provided by DRF generic/mixin base classes.
        base_method_map = {
            'ListAPIView': ['GET'],
            'CreateAPIView': ['POST'],
            'RetrieveAPIView': ['GET'],
            'UpdateAPIView': ['PUT', 'PATCH'],
            'DestroyAPIView': ['DELETE'],
            'ListCreateAPIView': ['GET', 'POST'],
            'RetrieveUpdateAPIView': ['GET', 'PUT', 'PATCH'],
            'RetrieveDestroyAPIView': ['GET', 'DELETE'],
            'RetrieveUpdateDestroyAPIView': ['GET', 'PUT', 'PATCH', 'DELETE'],
            'ModelViewSet': ['GET', 'POST', 'PUT', 'PATCH', 'DELETE'],
            'ReadOnlyModelViewSet': ['GET'],
            'GenericViewSet': ['GET', 'POST', 'PUT', 'PATCH', 'DELETE'],
            'ViewSet': ['GET', 'POST', 'PUT', 'PATCH', 'DELETE'],
            # Classic Django CBV
            'ListView': ['GET'],
            'DetailView': ['GET'],
            'TemplateView': ['GET'],
            'CreateView': ['GET', 'POST'],
            'UpdateView': ['GET', 'POST'],
            'DeleteView': ['GET', 'POST'],
            'FormView': ['GET', 'POST'],
        }

        for base in base_classes:
            if base in base_method_map:
                methods.update(base_method_map[base])

        # 2. Methods explicitly defined in class body.
        try:
            tree = ast.parse(entity.code)
            class_node = tree.body[0]
            http_method_names = {'get', 'post', 'put', 'patch', 'delete',
                                 'head', 'options'}
            for node in class_node.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if node.name in http_method_names:
                        methods.add(node.name.upper())
        except Exception as e:
            logger.debug(f"Failed to parse class methods for {entity.name}: {e}")

        if not methods:
            # APIView без явных методов — ставим ANY,
            # пусть api_linker сматчит как wildcard.
            return ['ANY']

        return sorted(methods)

    def _extract_api_view_methods(self, entity: CodeEntity, source: str) -> List[str]:
        """
        Извлечь HTTP-методы из декоратора ``@api_view([...])``.

        Decorators в metadata сохраняются только по имени (например, ``'@api_view'``)
        без аргументов, поэтому ищем литерал ``@api_view([...])`` в исходнике
        вокруг строки определения функции через AST.
        """
        # Default — на случай, если @api_view есть, но без аргументов
        # (тогда DRF разрешает все методы — использует HTTP_METHOD_NAMES,
        # но это редкий и небезопасный кейс — ставим ANY).
        methods: List[str] = []

        try:
            tree = ast.parse(source)
            target_name = entity.name
            target_line = entity.start_line or 0

            for node in ast.walk(tree):
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                if node.name != target_name:
                    continue
                # Если у файла несколько функций с одинаковым именем —
                # выбираем ближайшую по start_line.
                if target_line and abs(node.lineno - target_line) > 5:
                    continue

                for dec in node.decorator_list:
                    # Ищем call вида api_view([...]) или drf.decorators.api_view([...])
                    if not isinstance(dec, ast.Call):
                        continue
                    func = dec.func
                    func_name = None
                    if isinstance(func, ast.Name):
                        func_name = func.id
                    elif isinstance(func, ast.Attribute):
                        func_name = func.attr
                    if func_name != 'api_view':
                        continue
                    # @api_view(['GET', 'POST']) — первый позиционный аргумент
                    if dec.args:
                        arg = dec.args[0]
                        if isinstance(arg, (ast.List, ast.Tuple, ast.Set)):
                            for elt in arg.elts:
                                if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                                    methods.append(elt.value.upper())
                    # @api_view(http_method_names=['GET', 'POST'])
                    for kw in dec.keywords:
                        if kw.arg in ('http_method_names', 'methods'):
                            if isinstance(kw.value, (ast.List, ast.Tuple, ast.Set)):
                                for elt in kw.value.elts:
                                    if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                                        methods.append(elt.value.upper())
                    break  # one @api_view достаточно

                if methods:
                    break

        except Exception as e:
            logger.debug(f"Failed to extract HTTP methods: {e}")

        # Дедуп с сохранением порядка
        seen = set()
        uniq = []
        for m in methods:
            if m not in seen:
                seen.add(m)
                uniq.append(m)
        return uniq if uniq else ['GET']

    def _parse_urls(
        self,
        result: ParseResult,
        file_path: Path,
        source: str
    ) -> None:
        """
        Parse URL patterns from urls.py.

        Каждый ``path('foo/', view, ...)`` превращается в ENDPOINT-сущность
        с относительным path и view_ref. Каждый ``path('foo/', include('app.urls'))``
        — в ROUTE-сущность с метаданными ``is_include=True`` для последующего
        раскрытия префиксов в graph_builder.
        """
        try:
            tree = ast.parse(source)

            # 1) Соберём все DRF DefaultRouter и его register() вызовы.
            #    router_paths: {router_var_name: [{name, view, line}]}
            router_paths = self._collect_router_registrations(tree)

            # 2) Пройдёмся по top-level statements и обработаем все варианты
            #    объявления urlpatterns:
            #      urlpatterns  = [...]      / urlpatterns2 = [...]   (Assign)
            #      urlpatterns += [...]      / urlpatterns2 += [...]  (AugAssign)
            for node in tree.body:
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if (isinstance(target, ast.Name)
                                and target.id.startswith('urlpatterns')):
                            self._process_urlpatterns_value(
                                node.value, source, file_path, result, router_paths
                            )
                elif isinstance(node, ast.AugAssign):
                    if (isinstance(node.target, ast.Name)
                            and node.target.id.startswith('urlpatterns')):
                        self._process_urlpatterns_value(
                            node.value, source, file_path, result, router_paths
                        )

        except Exception as e:
            logger.error(f"Failed to parse URLs: {e}")

    # ------------------------------------------------------------------
    # urlpatterns helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_inner_list(value_node: ast.expr) -> Optional[ast.List]:
        """
        Извлечь ast.List из RHS присваивания. Поддерживает:
        - ``[...]``                       — простой список.
        - ``[...] + static(...)``         — конкатенация с другим списком.
        - ``format_suffix_patterns(...)`` — игнорируем (вернёт None).

        Рекурсивно для BinOp(Add) — ищем первый ast.List в дереве сложений.
        """
        if isinstance(value_node, ast.List):
            return value_node
        if isinstance(value_node, ast.BinOp) and isinstance(value_node.op, ast.Add):
            for side in (value_node.left, value_node.right):
                lst = DjangoParser._extract_inner_list(side)
                if lst is not None:
                    return lst
        return None

    def _process_urlpatterns_value(
        self,
        value_node: ast.expr,
        source: str,
        file_path: Path,
        result: ParseResult,
        router_paths: Dict[str, List[Dict[str, Any]]],
    ) -> None:
        """Развернуть значение urlpatterns/urlpatterns2/urlpatterns += ..."""
        list_node = self._extract_inner_list(value_node)
        if list_node is None:
            return
        for element in list_node.elts:
            if not isinstance(element, ast.Call):
                continue
            pattern = self._parse_url_pattern_call(element, source)
            if not pattern:
                continue

            # Спец-кейс: include(<router>.urls) — раскрываем все register()
            # этого router'а в endpoint'ы под текущим префиксом.
            if pattern.get('is_router_include'):
                router_var = pattern.get('router_var')
                prefix = (pattern.get('pattern') or '').strip('/')
                for reg in router_paths.get(router_var, []):
                    full = (prefix + '/' + reg['name'].strip('/')).strip('/')
                    full_path = '/' + full + '/'
                    result.entities.append(CodeEntity(
                        type=EntityType.ENDPOINT,
                        name=reg['name'],
                        file_path=file_path,
                        start_line=reg['line'],
                        end_line=reg['line'],
                        code='',
                        metadata={
                            'path': full_path,
                            # ViewSet поддерживает CRUD → ANY (api_linker
                            # сматчит с любым методом UI).
                            'http_method': 'ANY',
                            'view_ref': reg['view'],
                            'route_name': reg['name'],
                            'framework': 'django-rest-framework',
                        }
                    ))
                continue

            entity = self._url_pattern_to_entity(pattern, file_path)
            if entity:
                result.entities.append(entity)

    # ------------------------------------------------------------------
    # DRF Router collection
    # ------------------------------------------------------------------

    def _collect_router_registrations(
        self, tree: ast.Module
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Найти все ``<router> = routers.DefaultRouter() / SimpleRouter()`` и собрать
        все ``<router>.register(r'<name>', <ViewSet>)`` в map.

        Возвращает: {router_var_name: [{name, view, line}, ...]}
        """
        result: Dict[str, List[Dict[str, Any]]] = {}

        # 1. router-переменные (строго по типу `*Router()` в RHS)
        router_vars = set()
        for node in tree.body:
            if not isinstance(node, ast.Assign):
                continue
            if not isinstance(node.value, ast.Call):
                continue
            func = node.value.func
            name = None
            if isinstance(func, ast.Attribute):
                name = func.attr
            elif isinstance(func, ast.Name):
                name = func.id
            if not name or 'Router' not in name:
                continue
            for target in node.targets:
                if isinstance(target, ast.Name):
                    router_vars.add(target.id)
                    result.setdefault(target.id, [])

        # 2. <router>.register(r'<name>', <ViewSet>, ...)
        for node in tree.body:
            if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
                continue
            call = node.value
            if not (isinstance(call.func, ast.Attribute)
                    and call.func.attr == 'register'):
                continue
            obj = call.func.value
            if not isinstance(obj, ast.Name) or obj.id not in router_vars:
                continue
            if len(call.args) < 2:
                continue
            name_arg, view_arg = call.args[0], call.args[1]
            if not (isinstance(name_arg, ast.Constant) and isinstance(name_arg.value, str)):
                continue
            try:
                view_ref = ast.unparse(view_arg)
            except Exception:
                view_ref = ''
            result[obj.id].append({
                'name': name_arg.value,
                'view': view_ref,
                'line': call.lineno,
            })

        return result

    def _url_pattern_to_entity(
        self,
        pattern: Dict[str, Any],
        file_path: Path
    ) -> Optional[CodeEntity]:
        """Build a CodeEntity from a single URL pattern dict."""
        if pattern.get('is_include'):
            # include('app.urls') — маркер для post-pass в graph_builder
            return CodeEntity(
                type=EntityType.ROUTE,
                name=f"include:{pattern.get('include_module') or 'inline'}",
                file_path=file_path,
                start_line=pattern.get('line', 1),
                end_line=pattern.get('line', 1),
                code=pattern.get('code', ''),
                metadata={
                    'is_include': True,
                    'prefix': pattern.get('pattern') or '',
                    'include_module': pattern.get('include_module'),
                    'inline_patterns': pattern.get('inline_patterns', []),
                }
            )

        # Обычный URL → ENDPOINT
        url_path = pattern.get('pattern') or ''
        view_ref = pattern.get('view') or ''

        return CodeEntity(
            type=EntityType.ENDPOINT,
            name=pattern.get('name') or url_path or view_ref or 'endpoint',
            file_path=file_path,
            start_line=pattern.get('line', 1),
            end_line=pattern.get('line', 1),
            code=pattern.get('code', ''),
            metadata={
                'path': '/' + url_path.lstrip('/') if url_path else '/',
                'http_method': 'ANY',  # Будет уточнено по view в graph_builder
                'view_ref': view_ref,
                'route_name': pattern.get('name'),
                'framework': 'django',
            }
        )

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
        """
        Parse a single path() / re_path() / url() call.

        Распознаёт второй аргумент как include('module') / include([...])
        и помечает результат флагом is_include.
        """
        try:
            # Get function name
            func_name = None
            if isinstance(call_node.func, ast.Name):
                func_name = call_node.func.id

            if func_name not in ['path', 're_path', 'url']:
                return None

            # First argument: URL pattern string
            pattern = None
            if call_node.args:
                arg = call_node.args[0]
                if isinstance(arg, ast.Constant):
                    pattern = arg.value

            # Second argument: view OR include(...)
            view = None
            is_include = False
            include_module: Optional[str] = None
            inline_patterns: List[Dict[str, Any]] = []
            is_router_include = False
            router_var: Optional[str] = None

            if len(call_node.args) > 1:
                second_arg = call_node.args[1]

                # Detect include(...) call
                if (isinstance(second_arg, ast.Call)
                        and isinstance(second_arg.func, ast.Name)
                        and second_arg.func.id == 'include'):
                    is_include = True
                    if second_arg.args:
                        first_inc_arg = second_arg.args[0]
                        if isinstance(first_inc_arg, ast.Constant) and isinstance(first_inc_arg.value, str):
                            # include('app.urls') либо include('app.urls', namespace='...')
                            include_module = first_inc_arg.value
                        elif isinstance(first_inc_arg, ast.Attribute) and first_inc_arg.attr == 'urls' \
                                and isinstance(first_inc_arg.value, ast.Name):
                            # include(router.urls) — DRF router include
                            is_router_include = True
                            router_var = first_inc_arg.value.id
                            is_include = False  # это не модульный include
                        elif isinstance(first_inc_arg, ast.Tuple) and first_inc_arg.elts:
                            # include((patterns_list, app_namespace))
                            inner = first_inc_arg.elts[0]
                            if isinstance(inner, ast.List):
                                inline_patterns = self._extract_url_patterns(inner, source)
                        elif isinstance(first_inc_arg, ast.List):
                            # include([path(...), path(...)])
                            inline_patterns = self._extract_url_patterns(first_inc_arg, source)
                else:
                    try:
                        view = ast.unparse(second_arg)
                    except Exception:
                        view = "unknown"

            # 'name' kwarg
            name = None
            for keyword in call_node.keywords:
                if keyword.arg == 'name' and isinstance(keyword.value, ast.Constant):
                    name = keyword.value.value

            return {
                'pattern': pattern,
                'view': view,
                'name': name,
                'is_include': is_include,
                'include_module': include_module,
                'inline_patterns': inline_patterns,
                'is_router_include': is_router_include,
                'router_var': router_var,
                'code': ast.unparse(call_node),
                'line': call_node.lineno,
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
