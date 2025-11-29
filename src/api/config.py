"""
Configuration for FastAPI application.

Loads settings from config/base.yaml and environment variables.
"""

import os
from pathlib import Path
from typing import Optional
import yaml
from pydantic import BaseModel, Field


class Neo4jConfig(BaseModel):
    """Neo4j configuration."""
    uri: str = "bolt://localhost:7687"
    user: str = "neo4j"
    password: str
    database: str = "code_rag"


class WeaviateConfig(BaseModel):
    """Weaviate configuration."""
    url: str = "http://localhost:8080"
    timeout: int = 30


class AgentConfig(BaseModel):
    """Agent configuration."""
    enabled: bool = True
    openrouter_api_key: Optional[str] = None
    code_explorer_max_iterations: int = 10
    code_explorer_timeout: int = 120
    code_explorer_model: str = "anthropic/claude-sonnet-4"
    orchestrator_model: str = "deepseek/deepseek-r1:free"
    cache_enabled: bool = True
    cache_backend: str = "memory"
    query_ttl: int = 86400
    tool_result_ttl: int = 3600


class APISettings(BaseModel):
    """FastAPI application settings."""

    # API metadata
    title: str = "Code RAG API"
    description: str = "REST API for Code RAG System - Search, Q&A, and Repository Management"
    version: str = "1.0.0"

    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = False  # Set to True for development

    # CORS settings
    cors_origins: list = Field(default_factory=lambda: ["*"])
    cors_allow_credentials: bool = True
    cors_allow_methods: list = Field(default_factory=lambda: ["*"])
    cors_allow_headers: list = Field(default_factory=lambda: ["*"])

    # Database configs
    neo4j: Neo4jConfig
    weaviate: WeaviateConfig

    # Agent config
    agents: AgentConfig

    # Paths
    repos_dir: Path = Path("data/repos")

    @classmethod
    def from_yaml(cls, config_path: Path) -> "APISettings":
        """Load settings from YAML config file."""
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        # Parse Neo4j config
        neo4j_password = os.getenv("NEO4J_PASSWORD")
        if not neo4j_password:
            # Try to extract from ${...} placeholder
            password_placeholder = config['neo4j'].get('password', '')
            if password_placeholder.startswith('${') and password_placeholder.endswith('}'):
                env_var = password_placeholder[2:-1]
                neo4j_password = os.getenv(env_var)

            if not neo4j_password:
                raise ValueError("NEO4J_PASSWORD environment variable is required")

        neo4j_config = Neo4jConfig(
            uri=config['neo4j']['uri'],
            user=config['neo4j']['user'],
            password=neo4j_password,
            database=config['neo4j'].get('database', 'code_rag')
        )

        # Parse Weaviate config
        weaviate_config = WeaviateConfig(
            url=config['weaviate']['url'],
            timeout=config['weaviate'].get('timeout', 30)
        )

        # Parse Agent config
        openrouter_key = os.getenv("OPENROUTER_API_KEY")
        agents_enabled = openrouter_key is not None

        agent_config = AgentConfig(
            enabled=agents_enabled,
            openrouter_api_key=openrouter_key,
            code_explorer_max_iterations=config['agents']['code_explorer'].get('max_iterations', 10),
            code_explorer_timeout=config['agents']['code_explorer'].get('timeout_seconds', 120),
            code_explorer_model=config['agents']['code_explorer'].get('model', 'anthropic/claude-sonnet-4'),
            orchestrator_model=config['agents']['orchestrator'].get('model', 'deepseek/deepseek-r1:free'),
            cache_enabled=config['agents']['cache'].get('enabled', True),
            cache_backend=config['agents']['cache'].get('backend', 'memory'),
            query_ttl=config['agents']['cache'].get('query_ttl', 86400),
            tool_result_ttl=config['agents']['cache'].get('tool_result_ttl', 3600)
        )

        # Parse repos directory
        repos_dir = Path(config.get('directories', {}).get('repos', 'data/repos'))

        # Get API-specific settings from env or defaults
        host = os.getenv("API_HOST", "0.0.0.0")
        port = int(os.getenv("API_PORT", "8000"))
        reload = os.getenv("API_RELOAD", "false").lower() == "true"

        return cls(
            host=host,
            port=port,
            reload=reload,
            neo4j=neo4j_config,
            weaviate=weaviate_config,
            agents=agent_config,
            repos_dir=repos_dir
        )


# Global settings instance
_settings: Optional[APISettings] = None


def get_settings() -> APISettings:
    """Get API settings (singleton)."""
    global _settings
    if _settings is None:
        config_path = Path("config/base.yaml")
        _settings = APISettings.from_yaml(config_path)
    return _settings
