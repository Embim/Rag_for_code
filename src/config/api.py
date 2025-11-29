"""
API configuration for Code RAG.
"""

from dataclasses import dataclass, field
from typing import List, Optional
import os

from .base import BaseConfig


@dataclass  
class APISettings(BaseConfig):
    """
    FastAPI application settings.
    
    Attributes:
        title: API title
        description: API description
        version: API version
        
        host: Server host
        port: Server port
        reload: Enable auto-reload (dev)
        workers: Number of workers (prod)
        
        cors_origins: Allowed CORS origins
        cors_allow_credentials: Allow credentials
        cors_allow_methods: Allowed methods
        cors_allow_headers: Allowed headers
        
        rate_limit_enabled: Enable rate limiting
        rate_limit_requests: Requests per window
        rate_limit_window: Window in seconds
    """
    # App info
    title: str = "Code RAG API"
    description: str = "REST API for Code RAG - intelligent code search and analysis"
    version: str = "1.0.0"
    
    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = False
    workers: int = 4
    
    # CORS
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    cors_allow_credentials: bool = True
    cors_allow_methods: List[str] = field(default_factory=lambda: ["*"])
    cors_allow_headers: List[str] = field(default_factory=lambda: ["*"])
    
    # Rate limiting
    rate_limit_enabled: bool = True
    rate_limit_requests: int = 100
    rate_limit_window: int = 60  # seconds
    
    # Auth
    api_key_header: str = "X-API-Key"
    admin_api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("ADMIN_API_KEY")
    )
    
    @classmethod
    def from_env(cls) -> 'APISettings':
        """Load from environment variables."""
        return cls(
            host=os.getenv("API_HOST", "0.0.0.0"),
            port=int(os.getenv("API_PORT", "8000")),
            reload=os.getenv("API_RELOAD", "false").lower() == "true",
            workers=int(os.getenv("API_WORKERS", "4")),
        )

