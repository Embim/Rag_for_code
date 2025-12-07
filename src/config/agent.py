"""
Agent configuration for Code RAG.

Unified AgentConfig - replaces duplicates in:
- src/api/config.py
- src/agents/code_explorer.py
"""

from dataclasses import dataclass, field
from typing import Optional, List, Literal
import os

from .base import BaseConfig


@dataclass
class AgentConfig(BaseConfig):
    """
    Configuration for LLM agents.
    
    Used by:
    - CodeExplorerAgent
    - QueryOrchestrator
    - BusinessAgent
    - TracebackAnalyzer
    
    Attributes:
        enabled: Whether agents are enabled
        api_key: OpenRouter API key
        api_base: API base URL
        
        # Code Explorer
        max_iterations: Maximum agent iterations
        timeout_seconds: Timeout per request
        max_tokens_per_call: Token limit per LLM call
        temperature: LLM temperature
        code_explorer_model: Model for code exploration
        
        # Orchestrator
        orchestrator_model: Model for query classification
        
        # Caching
        cache_enabled: Enable result caching
        cache_backend: Cache backend (memory/redis)
        query_ttl: Query cache TTL in seconds
        tool_result_ttl: Tool result TTL
    """
    # General
    enabled: bool = True
    api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("OPENROUTER_API_KEY")
    )
    api_base: str = "https://openrouter.ai/api/v1"
    
    # Code Explorer Agent
    max_iterations: int = 30  # Increased to 30 (agent now uses multiple tools per iteration)
    timeout_seconds: float = 600.0  # Increased to 10 minutes for slow free models
    max_tokens_per_call: int = 16384  # Increased for detailed answers (262k context available)
    max_tokens_enrichment: int = 8192  # Tokens for answer enrichment step
    temperature: float = 0.1
    detail_level: Literal["brief", "normal", "detailed"] = "detailed"  # Answer detail level
    enable_enrichment: bool = True  # Enable answer enrichment step
    enrichment_timeout: float = 180.0  # Timeout for enrichment in seconds
    code_explorer_model: str = field(
        default_factory=lambda: os.getenv("CODE_EXPLORER_MODEL", "tngtech/tng-r1t-chimera:free")
    )

    # Query Orchestrator
    orchestrator_model: str = field(
        default_factory=lambda: os.getenv("ORCHESTRATOR_MODEL", "deepseek/deepseek-r1:free")
    )

    # Business Agent / Traceback Analyzer
    analysis_model: str = field(
        default_factory=lambda: os.getenv("ANALYSIS_MODEL", "tngtech/tng-r1t-chimera:free")
    )
    
    # Caching
    cache_enabled: bool = True
    cache_backend: Literal["memory", "redis"] = "memory"
    query_ttl: int = 86400  # 24 hours
    tool_result_ttl: int = 3600  # 1 hour
    semantic_cache_threshold: float = 0.92
    
    # Redis (if backend = redis)
    redis_url: str = "redis://localhost:6379"
    
    @classmethod
    def from_env(cls) -> 'AgentConfig':
        """Load config from environment variables."""
        return cls(
            enabled=os.getenv("AGENT_ENABLED", "true").lower() == "true",
            api_key=os.getenv("OPENROUTER_API_KEY"),
            code_explorer_model=os.getenv(
                "CODE_EXPLORER_MODEL",
                "tngtech/tng-r1t-chimera:free"
            ),
            orchestrator_model=os.getenv(
                "ORCHESTRATOR_MODEL",
                "deepseek/deepseek-r1:free"
            ),
            analysis_model=os.getenv(
                "ANALYSIS_MODEL",
                "tngtech/tng-r1t-chimera:free"
            ),
            max_iterations=int(os.getenv("AGENT_MAX_ITERATIONS", "30")),
            timeout_seconds=float(os.getenv("AGENT_TIMEOUT", "600")),
            cache_enabled=os.getenv("AGENT_CACHE_ENABLED", "true").lower() == "true",
            cache_backend=os.getenv("AGENT_CACHE_BACKEND", "memory"),
        )
    
    @property
    def is_configured(self) -> bool:
        """Check if agent is properly configured."""
        return self.enabled and bool(self.api_key)
    
    @property
    def model(self) -> str:
        """Alias for code_explorer_model (backward compatibility)."""
        return self.code_explorer_model

