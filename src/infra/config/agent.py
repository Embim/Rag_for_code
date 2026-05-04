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
    # General — primary endpoint
    enabled: bool = field(
        default_factory=lambda: os.getenv("AGENT_ENABLED", "true").lower() == "true"
    )
    api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("OPENROUTER_API_KEY")
    )
    api_base: str = "https://openrouter.ai/api/v1"

    # Fallback endpoint (опционально). Если задан — используется как 2-я попытка
    # при ошибках primary. Если задан только api_key_fallback — используется
    # тот же api_base, что у primary.
    api_key_fallback: Optional[str] = field(
        default_factory=lambda: os.getenv("OPENROUTER_API_KEY_FALLBACK") or os.getenv("OPENAI_API_KEY")
    )
    api_base_fallback: Optional[str] = field(
        default_factory=lambda: os.getenv("LLM_API_BASE_FALLBACK")
    )
    
    # Code Explorer Agent
    max_iterations: int = field(
        default_factory=lambda: int(os.getenv("AGENT_MAX_ITERATIONS", "30"))
    )
    timeout_seconds: float = field(
        default_factory=lambda: float(os.getenv("AGENT_TIMEOUT", "600"))
    )
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

    # RAG pipeline models (primary)
    rag_quality_model: str = field(
        default_factory=lambda: os.getenv("RAG_QUALITY_MODEL", "meta-llama/llama-3.3-8b-instruct:free")
    )
    rag_rewrite_model: str = field(
        default_factory=lambda: os.getenv("RAG_REWRITE_MODEL", "meta-llama/llama-3.3-8b-instruct:free")
    )
    rag_answer_model: str = field(
        default_factory=lambda: os.getenv("RAG_ANSWER_MODEL", "meta-llama/llama-3.1-405b-instruct:free")
    )
    rag_cypher_model: str = field(
        default_factory=lambda: os.getenv("RAG_CYPHER_MODEL", "")  # пусто = берёт rag_answer_model
    )

    # RAG pipeline models (fallback). Если пусто — failover отключён для роли.
    rag_quality_model_fallback: Optional[str] = field(
        default_factory=lambda: os.getenv("RAG_QUALITY_MODEL_FALLBACK")
    )
    rag_rewrite_model_fallback: Optional[str] = field(
        default_factory=lambda: os.getenv("RAG_REWRITE_MODEL_FALLBACK")
    )
    rag_answer_model_fallback: Optional[str] = field(
        default_factory=lambda: os.getenv("RAG_ANSWER_MODEL_FALLBACK")
    )
    rag_cypher_model_fallback: Optional[str] = field(
        default_factory=lambda: os.getenv("RAG_CYPHER_MODEL_FALLBACK")
    )

    # Корень всех индексированных репозиториев. ID нод имеет формат
    # ``repo:<name>:<rel_path>``; полный путь к файлу = ``<repos_dir>/<name>/<rel_path>``.
    # Используется ``code_loader.read_code`` (для Weaviate-индексации и
    # GenerationService) и ``GrepEnrichTask`` (для ripgrep). Если не задан —
    # берётся из env ``RAG_REPOS_DIR`` (legacy fallback).
    repos_dir: Optional[str] = field(
        default_factory=lambda: os.getenv("RAG_REPOS_DIR")
    )

    # Caching
    cache_enabled: bool = field(
        default_factory=lambda: os.getenv("AGENT_CACHE_ENABLED", "true").lower() == "true"
    )
    cache_backend: Literal["memory", "redis"] = field(
        default_factory=lambda: os.getenv("AGENT_CACHE_BACKEND", "memory")
    )
    query_ttl: int = 86400  # 24 hours
    tool_result_ttl: int = 3600  # 1 hour
    semantic_cache_threshold: float = 0.92
    
    # Redis (if backend = redis)
    redis_url: str = "redis://localhost:6379"
    
    @property
    def is_configured(self) -> bool:
        """Check if agent is properly configured."""
        return self.enabled and bool(self.api_key)
    
    @property
    def model(self) -> str:
        """Alias for code_explorer_model (backward compatibility)."""
        return self.code_explorer_model

