"""
Graph node and relationship models.

Defines the structure of nodes and relationships in the knowledge graph.
"""

from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime


class NodeType(str, Enum):
    """Types of nodes in the knowledge graph."""
    REPOSITORY = "Repository"
    FILE = "File"
    FUNCTION = "Function"
    CLASS = "Class"
    METHOD = "Method"
    COMPONENT = "Component"  # React/Vue components
    ENDPOINT = "Endpoint"     # API endpoints
    MODEL = "Model"           # Data models (Django, Pydantic)
    ROUTE = "Route"           # Frontend routes
    DOCUMENT = "Document"     # Documentation files (Word, PDF, etc)
    DOCUMENT_SECTION = "DocumentSection"  # Section within document


class RelationshipType(str, Enum):
    """Types of relationships in the knowledge graph."""
    # Containment
    CONTAINS = "CONTAINS"  # Repository CONTAINS File, File CONTAINS Function, etc.

    # Code dependencies
    IMPORTS = "IMPORTS"    # File IMPORTS File, Function IMPORTS Module
    CALLS = "CALLS"        # Function CALLS Function
    INHERITS = "INHERITS"  # Class INHERITS Class

    # API relationships
    HANDLES_REQUEST = "HANDLES_REQUEST"      # Endpoint HANDLES_REQUEST Function
    SENDS_REQUEST_TO = "SENDS_REQUEST_TO"    # Component SENDS_REQUEST_TO Endpoint

    # Data relationships
    USES_MODEL = "USES_MODEL"  # Endpoint USES_MODEL Model, Function USES_MODEL Model

    # UI relationships
    RENDERS_AT = "RENDERS_AT"  # Component RENDERS_AT Route

    # Django model relationships
    FOREIGN_KEY = "FOREIGN_KEY"        # Model FOREIGN_KEY Model
    MANY_TO_MANY = "MANY_TO_MANY"      # Model MANY_TO_MANY Model
    ONE_TO_ONE = "ONE_TO_ONE"          # Model ONE_TO_ONE Model


@dataclass
class GraphNode:
    """
    Base class for all graph nodes.

    Every node in the knowledge graph extends this base class.
    """

    # Core fields
    id: str  # Unique identifier (e.g., "repo:django/models.py:Product")
    name: str
    type: Optional[NodeType] = None

    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    # Additional properties (extensible)
    properties: Dict[str, Any] = field(default_factory=dict)

    def extract_metadata_from_id(self):
        """
        Extract repository and file_path from node ID.

        ID format: repo:repository_name:file/path.py:ClassName.method_name
        Example: repo:api:app/backend/booking.py:TradeUploader.book_trade
        """
        if not self.id or not self.id.startswith('repo:'):
            return

        # Split by colons, but preserve path
        parts = self.id.split(':', 3)  # ['repo', 'api', 'app/backend/booking.py', 'TradeUploader.book_trade']

        if len(parts) >= 3:
            repository = parts[1]

            # Extract file path (part before last colon)
            if len(parts) >= 4:
                file_path = parts[2]
            else:
                file_path = parts[2]

            # Update properties
            if 'repository' not in self.properties:
                self.properties['repository'] = repository
            if 'file_path' not in self.properties:
                self.properties['file_path'] = file_path

    def to_dict(self) -> Dict[str, Any]:
        """Convert node to dictionary for Neo4j."""
        # Extract metadata from ID before converting
        self.extract_metadata_from_id()

        return {
            'id': self.id,
            'type': self.type.value if self.type else 'Unknown',
            'name': self.name,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            **self.properties
        }


@dataclass
class RepositoryNode(GraphNode):
    """Repository node in the knowledge graph."""

    url: Optional[str] = None
    branch: str = "main"
    commit_hash: Optional[str] = None
    project_type: Optional[str] = None  # frontend, backend, fullstack
    languages: List[str] = field(default_factory=list)
    frameworks: List[str] = field(default_factory=list)

    def __post_init__(self):
        self.type = NodeType.REPOSITORY
        self.properties.update({
            'url': self.url,
            'branch': self.branch,
            'commit_hash': self.commit_hash,
            'project_type': self.project_type,
            'languages': ','.join(self.languages) if self.languages else '',
            'frameworks': ','.join(self.frameworks) if self.frameworks else '',
        })


@dataclass
class FileNode(GraphNode):
    """File node in the knowledge graph."""

    file_path: str = ""  # Relative path from repository root
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
    """Function node in the knowledge graph."""

    signature: str = ""
    docstring: Optional[str] = None
    code: str = ""  # Full source code of the function
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
            'code': self.code,  # Store full source code
            'start_line': self.start_line,
            'end_line': self.end_line,
            'is_async': self.is_async,
            'is_method': self.is_method,
            'decorators': ','.join(self.decorators) if self.decorators else '',
            'return_type': self.return_type or '',
        })


@dataclass
class ClassNode(GraphNode):
    """Class node in the knowledge graph."""

    docstring: Optional[str] = None
    code: str = ""  # Full source code of the class
    base_classes: List[str] = field(default_factory=list)
    start_line: int = 0
    end_line: int = 0
    decorators: List[str] = field(default_factory=list)

    def __post_init__(self):
        self.type = NodeType.CLASS
        self.properties.update({
            'docstring': self.docstring or '',
            'code': self.code,  # Store full source code
            'base_classes': ','.join(self.base_classes) if self.base_classes else '',
            'start_line': self.start_line,
            'end_line': self.end_line,
            'decorators': ','.join(self.decorators) if self.decorators else '',
        })


@dataclass
class ComponentNode(GraphNode):
    """React/Vue component node in the knowledge graph."""

    code: str = ""  # Full source code of the component
    props_type: Optional[str] = None
    hooks_used: List[str] = field(default_factory=list)
    start_line: int = 0
    end_line: int = 0
    is_exported: bool = False
    is_default_export: bool = False

    def __post_init__(self):
        self.type = NodeType.COMPONENT
        self.properties.update({
            'code': self.code,  # Store full source code
            'props_type': self.props_type or '',
            'hooks_used': ','.join(self.hooks_used) if self.hooks_used else '',
            'start_line': self.start_line,
            'end_line': self.end_line,
            'is_exported': self.is_exported,
            'is_default_export': self.is_default_export,
        })


@dataclass
class EndpointNode(GraphNode):
    """API endpoint node in the knowledge graph."""

    http_method: str = ""  # GET, POST, PUT, DELETE, etc.
    path: str = ""         # /api/users/{id}
    request_model: Optional[str] = None
    response_model: Optional[str] = None
    requires_auth: bool = False

    def __post_init__(self):
        self.type = NodeType.ENDPOINT
        # Create a descriptive name
        self.name = f"{self.http_method} {self.path}"
        self.properties.update({
            'http_method': self.http_method,
            'path': self.path,
            'request_model': self.request_model or '',
            'response_model': self.response_model or '',
            'requires_auth': self.requires_auth,
        })


@dataclass
class ModelNode(GraphNode):
    """Data model node (Django Model, Pydantic Model, etc.)."""

    model_type: str = ""  # "django", "pydantic", "sqlalchemy"
    fields: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        self.type = NodeType.MODEL
        self.properties.update({
            'model_type': self.model_type,
            'field_count': len(self.fields),
        })


@dataclass
class RouteNode(GraphNode):
    """Frontend route node in the knowledge graph."""

    path: str = ""  # /users/:id

    def __post_init__(self):
        self.type = NodeType.ROUTE
        self.properties.update({
            'path': self.path,
        })


@dataclass
class GraphRelationship:
    """
    Relationship between two nodes in the knowledge graph.
    """

    type: RelationshipType
    source_id: str  # ID of source node
    target_id: str  # ID of target node

    # Optional properties
    properties: Dict[str, Any] = field(default_factory=dict)

    # Confidence score for inferred relationships (e.g., API linking)
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert relationship to dictionary for Neo4j."""
        return {
            'type': self.type.value,
            'source_id': self.source_id,
            'target_id': self.target_id,
            'confidence': self.confidence,
            **self.properties
        }


@dataclass
class DocumentNode(GraphNode):
    """Documentation file node (Word, PDF, Markdown, etc.)."""

    file_path: str = ""  # Path to source file
    document_type: str = "SOP"  # SOP, Policy, Manual, Instruction, etc.
    content: str = ""  # Full text content
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
            'content': self.content,  # Full searchable content
            'author': self.author,
            'created_date': self.created_date,
            'modified_date': self.modified_date,
            'sections_count': self.sections_count,
            'images_count': self.images_count,
        })


@dataclass
class DocumentSectionNode(GraphNode):
    """Section within a document."""

    document_id: str = ""  # Parent document ID
    section_title: str = ""
    section_level: int = 0  # Heading level
    content: str = ""  # Section text content
    position: int = 0  # Position in document
    has_images: bool = False
    has_tables: bool = False

    def __post_init__(self):
        self.type = NodeType.DOCUMENT_SECTION
        self.properties.update({
            'document_id': self.document_id,
            'section_title': self.section_title,
            'section_level': self.section_level,
            'content': self.content,
            'position': self.position,
            'has_images': self.has_images,
            'has_tables': self.has_tables,
        })


def create_node_id(
    repository: str,
    file_path: Optional[str] = None,
    entity_name: Optional[str] = None
) -> str:
    """
    Create a unique node ID.

    Examples:
        - Repository: "repo:my-project"
        - File: "repo:my-project/src/models.py"
        - Function: "repo:my-project/src/models.py:create_user"
        - Class: "repo:my-project/src/models.py:User"

    Args:
        repository: Repository name
        file_path: Optional file path (relative to repo root)
        entity_name: Optional entity name (function, class, etc.)

    Returns:
        Unique node ID
    """
    parts = [f"repo:{repository}"]

    if file_path:
        parts.append(file_path.replace('\\', '/'))

    if entity_name:
        parts.append(entity_name)

    return ':'.join(parts) if len(parts) > 1 else parts[0]
