"""
Configuration for FastAPI application.

Loads settings from config/base.yaml and environment variables.
Uses canonical config classes from src/config/ — no duplicate definitions here.
"""

import os
from pathlib import Path
from typing import Optional, List
import yaml
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ─── Re-use canonical config classes from src/config ─────────────────────────
from ..config.database import Neo4jConfig, WeaviateConfig  # noqa: F401 — public re-export
from ..config.agent import AgentConfig  # noqa: F401 — public re-export


# ─── APISettings (Pydantic, YAML-backed) ─────────────────────────────────────

class APISettings(BaseModel):
    """FastAPI application settings — loaded from config/base.yaml."""

    # API metadata
    title: str = "Code RAG API"
    description: str = "REST API for Code RAG System - Search, Q&A, and Repository Management"
    version: str = "1.0.0"

    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = False

    # CORS settings
    cors_origins: List[str] = Field(default_factory=lambda: ["*"])
    cors_allow_credentials: bool = True
    cors_allow_methods: List[str] = Field(default_factory=lambda: ["*"])
    cors_allow_headers: List[str] = Field(default_factory=lambda: ["*"])

    # Sub-configs (Pydantic accepts arbitrary types via model_config)
    neo4j: Neo4jConfig = Field(default_factory=Neo4jConfig)
    weaviate: WeaviateConfig = Field(default_factory=WeaviateConfig)
    agents: AgentConfig = Field(default_factory=AgentConfig)

    # Paths
    repos_dir: Path = Path("data/repos")

    model_config = {"arbitrary_types_allowed": True}

    @classmethod
    def from_yaml(cls, config_path: Path) -> "APISettings":
        """Load settings from YAML config file."""
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        # ── Neo4j ──────────────────────────────────────────────────────────────
        neo4j_password = os.getenv("NEO4J_PASSWORD")
        if not neo4j_password:
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
            database=config['neo4j'].get('database', 'neo4j'),
        )

        # ── Weaviate ───────────────────────────────────────────────────────────
        weaviate_config = WeaviateConfig(
            url=config['weaviate']['url'],
        )

        # ── Agents ────────────────────────────────────────────────────────────
        openrouter_key = os.getenv("OPENROUTER_API_KEY")
        agents_enabled = openrouter_key is not None

        agent_config = AgentConfig(
            enabled=agents_enabled,
            api_key=openrouter_key,                                           # was: openrouter_api_key
            code_explorer_model=config['agents']['code_explorer'].get('model', 'tngtech/tng-r1t-chimera:free'),
            orchestrator_model=config['agents']['orchestrator'].get('model', 'deepseek/deepseek-r1:free'),
            max_iterations=config['agents']['code_explorer'].get('max_iterations', 20),  # was: code_explorer_max_iterations
            timeout_seconds=config['agents']['code_explorer'].get('timeout_seconds', 600.0),  # was: code_explorer_timeout
            cache_enabled=config['agents']['cache'].get('enabled', True),
            cache_backend=config['agents']['cache'].get('backend', 'memory'),
            query_ttl=config['agents']['cache'].get('query_ttl', 86400),
            tool_result_ttl=config['agents']['cache'].get('tool_result_ttl', 3600),
        )

        repos_dir = Path(config.get('directories', {}).get('repos', 'data/repos'))

        return cls(
            host=os.getenv("API_HOST", "0.0.0.0"),
            port=int(os.getenv("API_PORT", "8000")),
            reload=os.getenv("API_RELOAD", "false").lower() == "true",
            neo4j=neo4j_config,
            weaviate=weaviate_config,
            agents=agent_config,
            repos_dir=repos_dir,
        )


# ─── Singleton ────────────────────────────────────────────────────────────────

_settings: Optional[APISettings] = None


def get_settings() -> APISettings:
    """Get API settings (singleton)."""
    global _settings
    if _settings is None:
        config_path = Path("config/base.yaml")
        _settings = APISettings.from_yaml(config_path)
    return _settings
