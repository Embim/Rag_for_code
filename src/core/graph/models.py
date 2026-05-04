"""
Graph node and relationship models.

Архитектура (после refactor #1+#2+#3):
- Код функций/классов/компонентов в Neo4j НЕ хранится. На ноде только
  ``file_path``, ``start_line``, ``end_line`` — потребитель читает с диска
  через ``code_loader.read_code(...)`` или из Weaviate (где хранится content).
- API-вызовы UI — отдельные ``ApiCallNode``, не JSON-строка на компоненте.
  Цепочка: ``Component -[MAKES_CALL]-> ApiCall -[CALLS_ENDPOINT]-> Endpoint``.
- Старый одно-rebро ``Component → Endpoint`` (legacy) удалено: его роль
  закрывает 2-hop путь выше.
"""

import re
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from urllib.parse import urlparse


class NodeType(str, Enum):
    """Types of nodes in the knowledge graph."""
    REPOSITORY = "Repository"
    FILE = "File"
    FUNCTION = "Function"
    CLASS = "Class"
    METHOD = "Method"
    COMPONENT = "Component"  # React/Vue components
    ENDPOINT = "Endpoint"     # API endpoints (server-side)
    API_CALL = "ApiCall"      # API call site in UI (client-side)
    MODEL = "Model"           # Data models (Django, Pydantic)
    ROUTE = "Route"           # Frontend routes
    DOCUMENT = "Document"     # Documentation files (Word, PDF, etc)


class RelationshipType(str, Enum):
    """Types of relationships in the knowledge graph."""
    # Containment
    CONTAINS = "CONTAINS"  # Repository CONTAINS File, File CONTAINS Function, etc.

    # Code dependencies
    IMPORTS = "IMPORTS"    # File IMPORTS File, Function IMPORTS Module
    CALLS = "CALLS"        # Function CALLS Function
    INHERITS = "INHERITS"  # Class INHERITS Class

    # API relationships (структурное моделирование вызовов)
    MAKES_CALL = "MAKES_CALL"           # Component MAKES_CALL ApiCall
    CALLS_ENDPOINT = "CALLS_ENDPOINT"   # ApiCall CALLS_ENDPOINT Endpoint
    HANDLES_REQUEST = "HANDLES_REQUEST" # Endpoint HANDLES_REQUEST Function

    # Data relationships
    USES_MODEL = "USES_MODEL"  # Endpoint USES_MODEL Model, Function USES_MODEL Model

    # UI relationships
    RENDERS_AT = "RENDERS_AT"  # Component RENDERS_AT Route

    # Django model relationships
    FOREIGN_KEY = "FOREIGN_KEY"        # Model FOREIGN_KEY Model
    MANY_TO_MANY = "MANY_TO_MANY"      # Model MANY_TO_MANY Model
    ONE_TO_ONE = "ONE_TO_ONE"          # Model ONE_TO_ONE Model


# ---------------------------------------------------------------------------
# URL normalization (used by both EndpointNode and ApiCallNode at write-time
# to enable exact cypher matching at link-time)
# ---------------------------------------------------------------------------

_PARAM_PATTERNS = (
    r'\$\{[^}]+\}',                                # ${id}, ${productId}
    r':[a-zA-Z_]\w*',                              # :id, :productId
    r'<[^>]+>',                                    # <int:pk>, <str:slug>
    r'\{[^}]+\}',                                  # {id}, {product_id}
)

# Сегменты, выглядящие как конкретное значение runtime-параметра
# (число, UUID): UI часто подставляет их в URL прямо в коде.
# Нормализуем чтобы матчиться с {param} backend-эндпоинтами.
_VALUE_LIKE_SEGMENT = re.compile(
    r'/(?:'
    r'\d+'                                         # 123
    r'|[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}'  # UUID
    r')(?=/|$)'
)


def normalize_url(url: str) -> str:
    """
    Normalize a URL/path for exact equality matching.

    - Strips query string and trailing slash.
    - Replaces all parameter placeholders with literal ``{param}``.
    - Replaces value-like segments (numeric ids, UUID) with ``{param}``.
    - Collapses repeated slashes.
    - Strips host (keeps only path).

    Idempotent. Same function applied at write‑time on both sides
    (``ApiCallNode.normalized_url``, ``EndpointNode.normalized_path``).
    """
    if not url:
        return ''
    url = url.split('?', 1)[0]
    url = url.rstrip('/')
    if url.startswith('http'):
        url = urlparse(url).path
    for pat in _PARAM_PATTERNS:
        url = re.sub(pat, '{param}', url)
    # value-like segments — после плейсхолдеров, чтобы случайно не схлопнуть
    # уже отнормализованный {param}.
    url = _VALUE_LIKE_SEGMENT.sub('/{param}', url)
    url = re.sub(r'/+', '/', url)
    return url


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

@dataclass
class GraphNode:
    """Base class for all graph nodes."""

    id: str
    name: str
    type: Optional[NodeType] = None

    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    properties: Dict[str, Any] = field(default_factory=dict)

    def extract_metadata_from_id(self):
        if not self.id or not self.id.startswith('repo:'):
            return
        parts = self.id.split(':', 3)
        if len(parts) >= 3:
            repository = parts[1]
            file_path = parts[2]
            self.properties.setdefault('repository', repository)
            self.properties.setdefault('file_path', file_path)

    def to_dict(self) -> Dict[str, Any]:
        self.extract_metadata_from_id()
        return {
            'id': self.id,
            'type': self.type.value if self.type else 'Unknown',
            'name': self.name,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            **self.properties,
        }


# ---------------------------------------------------------------------------
# Concrete nodes
# ---------------------------------------------------------------------------

@dataclass
class RepositoryNode(GraphNode):
    url: Optional[str] = None
    branch: str = "main"
    commit_hash: Optional[str] = None
    project_type: Optional[str] = None
    languages: List[str] = field(default_factory=list)
    frameworks: List[str] = field(default_factory=list)
    # Абсолютный путь к репо на момент индексации. Используется
    # ``RepoPathResolver`` чтобы найти исходники для ``read_code`` /
    # ``GrepEnrichTask`` без хардкода ``RAG_REPOS_DIR``. Если репо
    # перенесли после индексации, env ``RAG_REPOS_DIR`` остаётся
    # override'ом верхнего приоритета.
    local_path: Optional[str] = None

    def __post_init__(self):
        self.type = NodeType.REPOSITORY
        self.properties.update({
            'url': self.url,
            'branch': self.branch,
            'commit_hash': self.commit_hash,
            'project_type': self.project_type,
            'languages': ','.join(self.languages) if self.languages else '',
            'frameworks': ','.join(self.frameworks) if self.frameworks else '',
            'local_path': self.local_path or '',
        })


@dataclass
class FileNode(GraphNode):
    file_path: str = ""
    language: str = ""
    size_bytes: int = 0
    line_count: int = 0
    content_hash: str = ""

    def __post_init__(self):
        self.type = NodeType.FILE
        self.properties.update({
            'file_path': self.file_path,
            'language': self.language,
            'size_bytes': self.size_bytes,
            'line_count': self.line_count,
            'content_hash': self.content_hash,
        })


@dataclass
class FunctionNode(GraphNode):
    """
    Function node. Source code НЕ хранится — только координаты файла.
    Получить код: ``code_loader.read_code(file_path, start_line, end_line)``.
    """

    signature: str = ""
    docstring: Optional[str] = None
    file_path: str = ""
    start_line: int = 0
    end_line: int = 0
    is_async: bool = False
    is_method: bool = False
    decorators: List[str] = field(default_factory=list)
    parameters: List[Dict[str, Any]] = field(default_factory=list)
    return_type: Optional[str] = None

    def __post_init__(self):
        self.type = NodeType.METHOD if self.is_method else NodeType.FUNCTION
        self.properties.update({
            'signature': self.signature,
            'docstring': self.docstring or '',
            'file_path': self.file_path,
            'start_line': self.start_line,
            'end_line': self.end_line,
            'is_async': self.is_async,
            'is_method': self.is_method,
            'decorators': ','.join(self.decorators) if self.decorators else '',
            'return_type': self.return_type or '',
        })


@dataclass
class ClassNode(GraphNode):
    """Class node. Source code НЕ хранится — только координаты файла."""

    docstring: Optional[str] = None
    file_path: str = ""
    base_classes: List[str] = field(default_factory=list)
    start_line: int = 0
    end_line: int = 0
    decorators: List[str] = field(default_factory=list)

    def __post_init__(self):
        self.type = NodeType.CLASS
        self.properties.update({
            'docstring': self.docstring or '',
            'file_path': self.file_path,
            'base_classes': ','.join(self.base_classes) if self.base_classes else '',
            'start_line': self.start_line,
            'end_line': self.end_line,
            'decorators': ','.join(self.decorators) if self.decorators else '',
        })


@dataclass
class ComponentNode(GraphNode):
    """
    React/Vue component node.

    API-вызовы НЕ хранятся в свойствах: они моделируются отдельными
    ``ApiCallNode`` со связью ``Component -[MAKES_CALL]-> ApiCall``.
    """

    props_type: Optional[str] = None
    hooks_used: List[str] = field(default_factory=list)
    file_path: str = ""
    start_line: int = 0
    end_line: int = 0
    is_exported: bool = False
    is_default_export: bool = False

    def __post_init__(self):
        self.type = NodeType.COMPONENT
        self.properties.update({
            'props_type': self.props_type or '',
            'hooks_used': ','.join(self.hooks_used) if self.hooks_used else '',
            'file_path': self.file_path,
            'start_line': self.start_line,
            'end_line': self.end_line,
            'is_exported': self.is_exported,
            'is_default_export': self.is_default_export,
        })


@dataclass
class EndpointNode(GraphNode):
    """
    API endpoint (server‑side).

    ``normalized_path`` записывается на этапе создания (через :func:`normalize_url`),
    чтобы линковка с ApiCall'ами могла идти exact-матчем по индексу cypher'ом.
    """

    http_method: str = ""  # GET, POST, PUT, DELETE, ANY
    path: str = ""         # raw, e.g. "/api/users/<int:pk>/"
    request_model: Optional[str] = None
    response_model: Optional[str] = None
    requires_auth: bool = False
    view_ref: str = ""
    framework: str = ""

    def __post_init__(self):
        self.type = NodeType.ENDPOINT
        if not self.name:
            self.name = f"{self.http_method} {self.path}"
        self.properties.update({
            'http_method': self.http_method,
            'path': self.path,
            'normalized_path': normalize_url(self.path),
            'request_model': self.request_model or '',
            'response_model': self.response_model or '',
            'requires_auth': self.requires_auth,
            'view_ref': self.view_ref,
            'framework': self.framework,
        })


@dataclass
class ApiCallNode(GraphNode):
    """
    Single API call site detected in UI code (axios.get / fetch / ...).

    Это first‑class сущность графа. Связь ``Component -[MAKES_CALL]-> ApiCall``
    говорит "компонент содержит этот вызов", связь ``ApiCall -[CALLS_ENDPOINT]-> Endpoint``
    создаётся api_linker'ом по ``(normalized_url, http_method)``.
    """

    http_method: str = "GET"
    url: str = ""               # raw, как в коде
    call_type: str = ""         # "axios" / "fetch" / "react-query" / ...
    file_path: str = ""
    start_line: int = 0

    def __post_init__(self):
        self.type = NodeType.API_CALL
        if not self.name:
            self.name = f"{self.http_method} {self.url}"
        self.properties.update({
            'http_method': self.http_method,
            'url': self.url,
            'normalized_url': normalize_url(self.url),
            'call_type': self.call_type,
            'file_path': self.file_path,
            'start_line': self.start_line,
        })


@dataclass
class ModelNode(GraphNode):
    model_type: str = ""
    fields: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        self.type = NodeType.MODEL
        self.properties.update({
            'model_type': self.model_type,
            'field_count': len(self.fields),
        })


@dataclass
class RouteNode(GraphNode):
    path: str = ""

    def __post_init__(self):
        self.type = NodeType.ROUTE
        self.properties.update({
            'path': self.path,
        })


@dataclass
class GraphRelationship:
    type: RelationshipType
    source_id: str
    target_id: str
    properties: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': self.type.value,
            'source_id': self.source_id,
            'target_id': self.target_id,
            'confidence': self.confidence,
            **self.properties,
        }


@dataclass
class DocumentNode(GraphNode):
    file_path: str = ""
    document_type: str = "SOP"
    content: str = ""
    author: str = ""
    created_date: str = ""
    modified_date: str = ""
    sections_count: int = 0
    images_count: int = 0

    def __post_init__(self):
        self.type = NodeType.DOCUMENT
        self.properties.update({
            'file_path': self.file_path,
            'document_type': self.document_type,
            'content': self.content,
            'author': self.author,
            'created_date': self.created_date,
            'modified_date': self.modified_date,
            'sections_count': self.sections_count,
            'images_count': self.images_count,
        })


def create_node_id(
    repository: str,
    file_path: Optional[str] = None,
    entity_name: Optional[str] = None,
) -> str:
    """
    Create a unique node ID.

    Examples:
        - Repository: "repo:my-project"
        - File: "repo:my-project:src/models.py"
        - Function: "repo:my-project:src/models.py:create_user"
    """
    parts = [f"repo:{repository}"]
    if file_path:
        parts.append(file_path.replace('\\', '/'))
    if entity_name:
        parts.append(entity_name)
    return ':'.join(parts) if len(parts) > 1 else parts[0]
