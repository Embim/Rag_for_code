"""
Cache configuration for Code RAG.
"""

from dataclasses import dataclass, field
from typing import Optional, Literal
import os

from .base import BaseConfig


@dataclass
class CacheConfig(BaseConfig):
    """
    Configuration for caching layer.
    
    Supports:
    - Query result caching
    - Semantic similarity caching
    - Tool result caching
    - LLM response caching
    
    Attributes:
        enabled: Enable caching
        backend: Cache backend (memory/redis)
        
        query_ttl: TTL for query results (seconds)
        tool_result_ttl: TTL for tool results
        llm_response_ttl: TTL for LLM responses
        
        semantic_cache_enabled: Enable semantic similarity cache
        semantic_cache_threshold: Similarity threshold (0.0-1.0)
        semantic_cache_max_size: Max entries in semantic cache
        
        redis_url: Redis connection URL
        redis_prefix: Key prefix for Redis
        
        memory_max_size: Max entries for memory backend
    """
    enabled: bool = True
    backend: Literal["memory", "redis"] = "memory"
    
    # TTL settings (seconds)
    query_ttl: int = 86400  # 24 hours
    tool_result_ttl: int = 3600  # 1 hour
    llm_response_ttl: int = 43200  # 12 hours
    
    # Semantic cache
    semantic_cache_enabled: bool = True
    semantic_cache_threshold: float = 0.92
    semantic_cache_max_size: int = 10000
    
    # Redis settings
    redis_url: str = "redis://localhost:6379"
    redis_prefix: str = "coderag:"
    
    # Memory settings
    memory_max_size: int = 1000
    
    @classmethod
    def from_env(cls) -> 'CacheConfig':
        """Load from environment variables."""
        return cls(
            enabled=os.getenv("CACHE_ENABLED", "true").lower() == "true",
            backend=os.getenv("CACHE_BACKEND", "memory"),
            redis_url=os.getenv("REDIS_URL", "redis://localhost:6379"),
            query_ttl=int(os.getenv("CACHE_QUERY_TTL", "86400")),
        )

