"""
Base parser classes and types.

Defines EntityType, CodeEntity, and ParseResult used by all parsers.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from pathlib import Path


class EntityType(str, Enum):
    """Types of code entities that parsers can extract."""
    FUNCTION = "function"
    METHOD = "method"
    CLASS = "class"
    COMPONENT = "component"  # React/Vue components
    ENDPOINT = "endpoint"    # API endpoints
    MODEL = "model"          # Data models (Django, Pydantic)
    ROUTE = "route"          # Frontend routes
    IMPORT = "import"
    VARIABLE = "variable"


@dataclass
class CodeEntity:
    """
    Represents a code entity extracted by a parser.
    
    This is the common format that all parsers produce.
    """
    name: str
    type: EntityType
    file_path: str = ""
    start_line: int = 0
    end_line: int = 0
    
    # Code content
    code: str = ""
    signature: str = ""
    docstring: Optional[str] = None
    
    # Relationships
    parent: Optional[str] = None  # Parent class/component name
    imports: List[str] = field(default_factory=list)
    calls: List[str] = field(default_factory=list)
    
    # Type information
    return_type: Optional[str] = None
    parameters: List[Dict[str, Any]] = field(default_factory=list)
    
    # Decorators/annotations
    decorators: List[str] = field(default_factory=list)
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def full_name(self) -> str:
        """
        Get the fully qualified name of this entity.

        For methods, this includes the parent class (e.g., 'ClassName.method_name').
        For standalone entities, this is just the name.
        """
        if self.parent:
            return f"{self.parent}.{self.name}"
        return self.name

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'type': self.type.value,
            'file_path': self.file_path,
            'start_line': self.start_line,
            'end_line': self.end_line,
            'code': self.code,
            'signature': self.signature,
            'docstring': self.docstring,
            'parent': self.parent,
            'imports': self.imports,
            'calls': self.calls,
            'return_type': self.return_type,
            'parameters': self.parameters,
            'decorators': self.decorators,
            'metadata': self.metadata,
        }


@dataclass
class ParseResult:
    """Result of parsing a file."""
    file_path: Path
    language: str
    entities: List[CodeEntity] = field(default_factory=list)
    imports: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    
    @property
    def success(self) -> bool:
        """Check if parsing was successful."""
        return len(self.errors) == 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'file_path': str(self.file_path),
            'language': self.language,
            'entities': [e.to_dict() for e in self.entities],
            'imports': self.imports,
            'errors': self.errors,
        }


class ParsingError(Exception):
    """Raised when parsing fails."""
    pass


# Registry for parsers
_parser_registry: Dict[str, type] = {}


def register_parser(language: str):
    """Decorator to register a parser for a language."""
    def decorator(cls):
        _parser_registry[language] = cls
        return cls
    return decorator


def get_parser_for_language(language: str) -> Optional[type]:
    """Get parser class for a language."""
    return _parser_registry.get(language)


class BaseParser:
    """Base class for all parsers."""
    
    language: str = "unknown"
    
    def parse_file(self, file_path: Path) -> ParseResult:
        """Parse a file and extract entities."""
        raise NotImplementedError
    
    def parse_code(self, code: str, file_path: Optional[Path] = None) -> ParseResult:
        """Parse code string and extract entities."""
        raise NotImplementedError

