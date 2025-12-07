"""
FastAPI dependency injection.

Provides dependencies for database connections, agents, and other services.
Uses proper lifecycle management (startup/shutdown).
"""

from typing import Optional, AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import Depends, HTTPException, status

from ..code_rag.graph import Neo4jClient, WeaviateIndexer
from ..code_rag.retrieval import CodeRetriever, ScopeDetector
from ..agents import (
    CodeExplorerAgent,
    VisualGuideAgent,
    QueryOrchestrator,
    AgentCache,
    AgentConfig as AgentConfigModel,
    CacheConfig,
)
from ..agents.tools import (
    SemanticSearchTool,
    ExactSearchTool,
    GetEntityDetailsTool,
    GetRelatedEntitiesTool,
    ListFilesTool,
    ReadFileTool,
    GrepTool,
    GetGraphPathTool,
)
from ..telegram_bot.visualizer import MermaidDiagramGenerator
from ..telegram_bot.troubleshoot import TroubleshootingAssistant
from ..logger import get_logger
from .config import get_settings, APISettings


logger = get_logger(__name__)


# ============================================================================
# Global State (initialized at app startup)
# ============================================================================

class AppState:
    """Global application state."""

    def __init__(self):
        self.neo4j_client: Optional[Neo4jClient] = None
        self.weaviate_indexer: Optional[WeaviateIndexer] = None
        self.code_retriever: Optional[CodeRetriever] = None
        self.scope_detector: Optional[ScopeDetector] = None
        self.orchestrator: Optional[QueryOrchestrator] = None
        self.agent_cache: Optional[AgentCache] = None
        self.visualizer: Optional[MermaidDiagramGenerator] = None
        self.troubleshooter: Optional[TroubleshootingAssistant] = None


# Global state instance
app_state = AppState()


# ============================================================================
# Lifecycle Management
# ============================================================================

@asynccontextmanager
async def lifespan_context(app):
    """
    FastAPI lifespan context manager.

    Handles startup and shutdown of database connections and services.
    """
    settings = get_settings()
    logger.info("🚀 Starting API server...")

    try:
        # Initialize Neo4j
        logger.info(f"Connecting to Neo4j at {settings.neo4j.uri}...")
        app_state.neo4j_client = Neo4jClient(
            uri=settings.neo4j.uri,
            user=settings.neo4j.user,
            password=settings.neo4j.password,
            database=settings.neo4j.database
        )
        app_state.neo4j_client.__enter__()
        logger.info("✅ Neo4j connected")

        # Initialize Weaviate
        logger.info(f"Connecting to Weaviate at {settings.weaviate.url}...")
        app_state.weaviate_indexer = WeaviateIndexer(
            weaviate_url=settings.weaviate.url,
            neo4j_client=app_state.neo4j_client
        )
        logger.info("✅ Weaviate connected")

        # Initialize retriever
        app_state.code_retriever = CodeRetriever(
            neo4j_client=app_state.neo4j_client,
            weaviate_indexer=app_state.weaviate_indexer
        )
        app_state.scope_detector = ScopeDetector()
        logger.info("✅ Code retriever initialized")

        # Initialize visualizer
        app_state.visualizer = MermaidDiagramGenerator(app_state.neo4j_client)
        logger.info("✅ Visualizer initialized")

        # Initialize troubleshooter
        app_state.troubleshooter = TroubleshootingAssistant(
            neo4j_client=app_state.neo4j_client,
            weaviate_indexer=app_state.weaviate_indexer
        )
        logger.info("✅ Troubleshooter initialized")

        # Initialize agents (if enabled)
        if settings.agents.enabled and settings.agents.openrouter_api_key:
            logger.info("Initializing agent system...")

            # Initialize tools
            tools = [
                SemanticSearchTool(app_state.code_retriever),
                ExactSearchTool(app_state.neo4j_client),
                GetEntityDetailsTool(app_state.neo4j_client),
                GetRelatedEntitiesTool(app_state.neo4j_client),
                ListFilesTool(settings.repos_dir),
                ReadFileTool(settings.repos_dir),
                GrepTool(app_state.neo4j_client),
                GetGraphPathTool(app_state.neo4j_client),
            ]

            # Initialize Code Explorer Agent
            agent_config = AgentConfigModel(
                max_iterations=settings.agents.code_explorer_max_iterations,
                timeout_seconds=settings.agents.code_explorer_timeout,
                temperature=0.1,
                code_explorer_model=settings.agents.code_explorer_model,
            )

            code_explorer = CodeExplorerAgent(
                tools=tools,
                api_key=settings.agents.openrouter_api_key,
                config=agent_config,
            )

            # Initialize Visual Guide Agent
            visual_agent = VisualGuideAgent(
                code_explorer=code_explorer,
                diagram_generator=app_state.visualizer,
                api_key=settings.agents.openrouter_api_key,
                config=agent_config,
            )

            # Initialize cache
            cache_config = CacheConfig(
                enabled=settings.agents.cache_enabled,
                backend=settings.agents.cache_backend,
                query_ttl=settings.agents.query_ttl,
                tool_result_ttl=settings.agents.tool_result_ttl,
            )
            app_state.agent_cache = AgentCache(cache_config)

            # Initialize orchestrator
            app_state.orchestrator = QueryOrchestrator(
                code_explorer=code_explorer,
                api_key=settings.agents.openrouter_api_key,
                model=settings.agents.orchestrator_model,
                visual_agent=visual_agent,
            )

            logger.info("✅ Agent system initialized")
        else:
            logger.warning("⚠️ Agents disabled (no OPENROUTER_API_KEY)")

        logger.info("✅ API server ready!")

        # Initialize API keys (create admin key if none exists)
        from .auth import create_initial_admin_key
        admin_key = create_initial_admin_key()
        if admin_key:
            # Log to logger
            logger.info("=" * 80)
            logger.info("🔑 INITIAL ADMIN API KEY CREATED")
            logger.info(f"   API Key: {admin_key}")
            logger.info("   ⚠️  SAVE THIS KEY! It will not be shown again.")
            logger.info("   Use this key to create additional API keys via POST /api/keys")
            logger.info("=" * 80)

            # Also print to console for visibility
            print("\n" + "=" * 80)
            print("🔑 INITIAL ADMIN API KEY CREATED")
            print(f"   API Key: {admin_key}")
            print("   ⚠️  SAVE THIS KEY! It will not be shown again.")
            print("   Use this key to create additional API keys via POST /api/keys")
            print("=" * 80 + "\n")

        # Yield control to FastAPI
        yield

    finally:
        # Cleanup on shutdown
        logger.info("🛑 Shutting down API server...")

        if app_state.weaviate_indexer:
            app_state.weaviate_indexer.close()
            logger.info("✅ Weaviate disconnected")

        if app_state.neo4j_client:
            app_state.neo4j_client.__exit__(None, None, None)
            logger.info("✅ Neo4j disconnected")

        logger.info("👋 API server stopped")


# ============================================================================
# Dependency Functions
# ============================================================================

async def get_neo4j() -> Neo4jClient:
    """Get Neo4j client dependency."""
    if app_state.neo4j_client is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Neo4j client not initialized"
        )
    return app_state.neo4j_client


async def get_weaviate() -> WeaviateIndexer:
    """Get Weaviate indexer dependency."""
    if app_state.weaviate_indexer is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Weaviate indexer not initialized"
        )
    return app_state.weaviate_indexer


async def get_retriever() -> CodeRetriever:
    """Get code retriever dependency."""
    if app_state.code_retriever is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Code retriever not initialized"
        )
    return app_state.code_retriever


async def get_scope_detector() -> ScopeDetector:
    """Get scope detector dependency."""
    if app_state.scope_detector is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Scope detector not initialized"
        )
    return app_state.scope_detector


async def get_orchestrator() -> Optional[QueryOrchestrator]:
    """Get query orchestrator dependency (optional, None if agents disabled)."""
    return app_state.orchestrator


async def get_agent_cache() -> Optional[AgentCache]:
    """Get agent cache dependency (optional, None if agents disabled)."""
    return app_state.agent_cache


async def get_visualizer() -> MermaidDiagramGenerator:
    """Get diagram visualizer dependency."""
    if app_state.visualizer is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Visualizer not initialized"
        )
    return app_state.visualizer


async def get_troubleshooter() -> TroubleshootingAssistant:
    """Get troubleshooting assistant dependency."""
    if app_state.troubleshooter is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Troubleshooter not initialized"
        )
    return app_state.troubleshooter


def require_agents(orchestrator: Optional[QueryOrchestrator] = Depends(get_orchestrator)):
    """
    Dependency that requires agents to be enabled.

    Raises 503 if agents are not available.
    """
    if orchestrator is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Agent system not available. Ensure OPENROUTER_API_KEY is set."
        )
    return orchestrator
