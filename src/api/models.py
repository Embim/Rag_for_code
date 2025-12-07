"""
Pydantic models for API request/response validation.

These models define the schema for all API endpoints.
"""

from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field
from datetime import datetime


# ============================================================================
# Search & Q&A Models
# ============================================================================

class SearchRequest(BaseModel):
    """Request for /search endpoint (quick search)."""

    query: str = Field(..., description="Search query", min_length=1)
    scope: Optional[Literal["frontend", "backend", "hybrid"]] = Field(
        default=None,
        description="Search scope (auto-detected if not provided)"
    )
    limit: int = Field(default=10, ge=1, le=50, description="Max results")
    strategy: Optional[Literal["semantic", "bm25", "hybrid", "multi_hop"]] = Field(
        default="hybrid",
        description="Search strategy"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "query": "how does authentication work?",
                "scope": "backend",
                "limit": 10,
                "strategy": "hybrid"
            }
        }


class SearchResult(BaseModel):
    """Single search result."""

    entity_id: str = Field(..., description="Entity ID in Neo4j")
    entity_type: str = Field(..., description="Type: Function, Class, Component, etc.")
    name: str = Field(..., description="Entity name")
    file_path: str = Field(..., description="File path")
    content: str = Field(..., description="Entity content (code)")
    score: float = Field(..., description="Relevance score 0.0-1.0")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    class Config:
        json_schema_extra = {
            "example": {
                "entity_id": "repo:myproject/auth.py:authenticate",
                "entity_type": "Function",
                "name": "authenticate",
                "file_path": "backend/auth.py",
                "content": "def authenticate(username, password):\n    ...",
                "score": 0.95,
                "metadata": {
                    "signature": "authenticate(username: str, password: str) -> bool",
                    "start_line": 10,
                    "end_line": 25
                }
            }
        }


class SearchResponse(BaseModel):
    """Response from /search endpoint."""

    query: str = Field(..., description="Original query")
    scope: str = Field(..., description="Detected or specified scope")
    strategy: str = Field(..., description="Used search strategy")
    results: List[SearchResult] = Field(..., description="Search results")
    total_found: int = Field(..., description="Total results found")
    took_ms: float = Field(..., description="Query execution time in ms")

    class Config:
        json_schema_extra = {
            "example": {
                "query": "how does authentication work?",
                "scope": "backend",
                "strategy": "hybrid",
                "results": [],
                "total_found": 5,
                "took_ms": 123.45
            }
        }


class AskRequest(BaseModel):
    """Request for /ask endpoint (agent-powered deep exploration)."""

    question: str = Field(..., description="Natural language question", min_length=1)
    context: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional context (repositories, previous conversation, etc.)"
    )
    max_iterations: Optional[int] = Field(
        default=10,
        ge=1,
        le=20,
        description="Max agent iterations"
    )
    timeout: Optional[int] = Field(
        default=120,
        ge=10,
        le=300,
        description="Timeout in seconds"
    )
    detail_level: Optional[Literal["brief", "normal", "detailed"]] = Field(
        default="detailed",
        description="Answer detail level: brief (concise), normal (balanced), detailed (comprehensive)"
    )
    verbose: bool = Field(
        default=False,
        description="Enable verbose debug mode (includes detailed iteration trace)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "question": "Explain the complete checkout flow from UI to database",
                "context": {"repositories": ["my-frontend", "my-backend"]},
                "max_iterations": 10,
                "timeout": 120,
                "detail_level": "detailed"
            }
        }


class AskResponse(BaseModel):
    """Response from /ask endpoint."""

    question: str = Field(..., description="Original question")
    question_type: str = Field(..., description="Classified type: CODE, DOCUMENT, VISUAL, HYBRID")
    agent_used: str = Field(..., description="Agent that handled the question")
    answer: str = Field(..., description="Comprehensive answer")
    sources: List[SearchResult] = Field(default_factory=list, description="Referenced code entities")
    iterations_used: int = Field(..., description="Number of agent iterations")
    tools_used: List[str] = Field(default_factory=list, description="Tools used by agent")
    complete: bool = Field(..., description="Whether exploration completed successfully")
    took_ms: float = Field(..., description="Total execution time in ms")
    cached: bool = Field(default=False, description="Whether result came from cache")
    debug: Optional[Dict[str, Any]] = Field(default=None, description="Debug information (only if verbose=True)")

    class Config:
        json_schema_extra = {
            "example": {
                "question": "How does checkout work?",
                "question_type": "CODE",
                "agent_used": "code_explorer",
                "answer": "The checkout process involves...",
                "sources": [],
                "iterations_used": 5,
                "tools_used": ["semantic_search", "get_entity_details"],
                "complete": True,
                "took_ms": 15234.56,
                "cached": False
            }
        }


# ============================================================================
# Repository Management Models
# ============================================================================

class RepositoryInfo(BaseModel):
    """Information about a repository."""

    name: str = Field(..., description="Repository name")
    type: str = Field(..., description="Type: frontend, backend, shared")
    path: str = Field(..., description="Local path or URL")
    language: str = Field(..., description="Primary language")
    framework: Optional[str] = Field(default=None, description="Framework (React, Django, etc.)")
    branch: str = Field(default="main", description="Git branch")
    commit_hash: Optional[str] = Field(default=None, description="Current commit hash")
    last_indexed: Optional[datetime] = Field(default=None, description="Last indexing time")
    stats: Dict[str, int] = Field(default_factory=dict, description="Statistics (files, entities, etc.)")

    class Config:
        json_schema_extra = {
            "example": {
                "name": "my-backend",
                "type": "backend",
                "path": "https://github.com/user/backend.git",
                "language": "Python",
                "framework": "Django",
                "branch": "main",
                "commit_hash": "abc123...",
                "last_indexed": "2025-11-29T12:00:00Z",
                "stats": {
                    "files": 150,
                    "functions": 450,
                    "classes": 120,
                    "endpoints": 45
                }
            }
        }


class AddRepositoryRequest(BaseModel):
    """Request to add a new repository."""

    source: str = Field(..., description="Git URL or local path")
    name: str = Field(..., description="Repository name")
    type: Literal["frontend", "backend", "shared"] = Field(..., description="Repository type")
    branch: Optional[str] = Field(default=None, description="Git branch (only for URLs, ignored for local paths)")
    auto_detect_framework: bool = Field(default=True, description="Auto-detect framework")

    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "summary": "GitHub URL",
                    "value": {
                        "source": "https://github.com/user/backend.git",
                        "name": "my-backend",
                        "type": "backend",
                        "branch": "main"
                    }
                },
                {
                    "summary": "Local path (no git)",
                    "value": {
                        "source": "C:/Projects/my-frontend",
                        "name": "my-frontend",
                        "type": "frontend"
                    }
                }
            ]
        }


class RepositoryIndexingStatus(BaseModel):
    """Status of repository indexing operation."""

    repository: str = Field(..., description="Repository name")
    status: Literal["queued", "in_progress", "completed", "failed"] = Field(..., description="Indexing status")
    progress: float = Field(default=0.0, ge=0.0, le=1.0, description="Progress 0.0-1.0")
    message: str = Field(default="", description="Status message")
    files_processed: int = Field(default=0, description="Files processed")
    entities_found: int = Field(default=0, description="Entities extracted")
    errors: List[str] = Field(default_factory=list, description="Errors encountered")
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    class Config:
        json_schema_extra = {
            "example": {
                "repository": "my-backend",
                "status": "in_progress",
                "progress": 0.65,
                "message": "Parsing Python files...",
                "files_processed": 98,
                "entities_found": 234,
                "errors": [],
                "started_at": "2025-11-29T12:00:00Z",
                "completed_at": None
            }
        }


# ============================================================================
# Visualization Models
# ============================================================================

class VisualizeRequest(BaseModel):
    """Request for /visualize endpoint."""

    diagram_type: Literal["sequence", "component", "er", "flow"] = Field(
        ...,
        description="Diagram type"
    )
    entities: List[str] = Field(
        ...,
        description="Entity IDs to include in diagram",
        min_length=1
    )
    title: Optional[str] = Field(default=None, description="Diagram title")
    options: Dict[str, Any] = Field(
        default_factory=dict,
        description="Diagram-specific options"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "diagram_type": "sequence",
                "entities": [
                    "repo:frontend/CartButton.tsx:CartButton",
                    "repo:backend/api/cart.py:add_to_cart"
                ],
                "title": "Add to Cart Flow",
                "options": {}
            }
        }


class VisualizeResponse(BaseModel):
    """Response from /visualize endpoint."""

    diagram_type: str = Field(..., description="Diagram type")
    mermaid_code: str = Field(..., description="Mermaid diagram code")
    entities_included: int = Field(..., description="Number of entities in diagram")
    image_url: Optional[str] = Field(
        default=None,
        description="URL to rendered image (via mermaid.ink)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "diagram_type": "sequence",
                "mermaid_code": "sequenceDiagram\n    User->>CartButton: click\n    ...",
                "entities_included": 5,
                "image_url": "https://mermaid.ink/img/..."
            }
        }


# ============================================================================
# Diagnostics Models
# ============================================================================

class HealthResponse(BaseModel):
    """Response from /health endpoint."""

    status: Literal["healthy", "degraded", "unhealthy"] = Field(..., description="Overall health")
    services: Dict[str, Dict[str, Any]] = Field(..., description="Service statuses")
    timestamp: datetime = Field(default_factory=datetime.now)

    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "services": {
                    "neo4j": {"status": "up", "latency_ms": 5.2},
                    "weaviate": {"status": "up", "latency_ms": 12.3},
                    "agents": {"status": "up", "enabled": True}
                },
                "timestamp": "2025-11-29T12:00:00Z"
            }
        }


class StatsResponse(BaseModel):
    """Response from /stats endpoint."""

    repositories: Dict[str, RepositoryInfo] = Field(..., description="Repository statistics")
    knowledge_graph: Dict[str, int] = Field(..., description="Graph statistics")
    search_index: Dict[str, Any] = Field(..., description="Weaviate statistics")
    agent_cache: Optional[Dict[str, Any]] = Field(default=None, description="Agent cache statistics")

    class Config:
        json_schema_extra = {
            "example": {
                "repositories": {},
                "knowledge_graph": {
                    "total_nodes": 1523,
                    "total_relationships": 3456,
                    "nodes_by_type": {
                        "Function": 450,
                        "Class": 120,
                        "Component": 80,
                        "Endpoint": 45
                    }
                },
                "search_index": {
                    "total_objects": 1523,
                    "embedding_dimensions": 768
                },
                "agent_cache": {
                    "size": 150,
                    "hit_rate": 0.82
                }
            }
        }


# ============================================================================
# Error Models
# ============================================================================

class ErrorResponse(BaseModel):
    """Standard error response."""

    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Additional details")
    timestamp: datetime = Field(default_factory=datetime.now)

    class Config:
        json_schema_extra = {
            "example": {
                "error": "ValidationError",
                "message": "Invalid request parameters",
                "details": {"field": "query", "issue": "too short"},
                "timestamp": "2025-11-29T12:00:00Z"
            }
        }
