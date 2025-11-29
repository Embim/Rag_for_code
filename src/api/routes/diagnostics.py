"""
Diagnostics endpoints.

Provides:
- GET /health - Health check
- GET /stats - System statistics
"""

import time
from fastapi import APIRouter, Depends, HTTPException, status
from datetime import datetime

from ..models import HealthResponse, StatsResponse, RepositoryInfo
from ..dependencies import (
    get_neo4j,
    get_weaviate,
    get_agent_cache,
)
from ..auth import require_auth, APIKey
from ...code_rag.graph import Neo4jClient, WeaviateIndexer
from ...agents import AgentCache
from ...logger import get_logger


logger = get_logger(__name__)

router = APIRouter(prefix="/api", tags=["diagnostics"])


# ============================================================================
# Endpoints
# ============================================================================

@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Check the health status of all system components"
)
async def health_check(
    neo4j: Neo4jClient = Depends(get_neo4j),
    weaviate: WeaviateIndexer = Depends(get_weaviate),
    current_key: APIKey = Depends(require_auth),
):
    """
    Health check endpoint.

    Checks connectivity and responsiveness of:
    - Neo4j (knowledge graph)
    - Weaviate (vector search)
    - Agent system (if enabled)

    Returns overall status: healthy, degraded, or unhealthy.
    """
    services = {}
    overall_status = "healthy"

    # Check Neo4j
    try:
        start = time.time()
        neo4j_stats = neo4j.get_statistics()
        latency_ms = (time.time() - start) * 1000

        services["neo4j"] = {
            "status": "up",
            "latency_ms": round(latency_ms, 2),
            "total_nodes": neo4j_stats.get('total_nodes', 0),
            "total_relationships": neo4j_stats.get('total_relationships', 0),
        }
        logger.info(f"Neo4j health: OK ({latency_ms:.2f}ms)")

    except Exception as e:
        logger.error(f"Neo4j health check failed: {e}")
        services["neo4j"] = {
            "status": "down",
            "error": str(e)
        }
        overall_status = "unhealthy"

    # Check Weaviate
    try:
        start = time.time()
        weaviate_stats = weaviate.get_statistics()
        latency_ms = (time.time() - start) * 1000

        services["weaviate"] = {
            "status": "up",
            "latency_ms": round(latency_ms, 2),
            "total_objects": weaviate_stats.get('total_nodes', 0),
        }
        logger.info(f"Weaviate health: OK ({latency_ms:.2f}ms)")

    except Exception as e:
        logger.error(f"Weaviate health check failed: {e}")
        services["weaviate"] = {
            "status": "down",
            "error": str(e)
        }
        overall_status = "unhealthy"

    # Check agents (optional)
    try:
        from ..dependencies import app_state
        if app_state.orchestrator:
            services["agents"] = {
                "status": "up",
                "enabled": True,
            }
        else:
            services["agents"] = {
                "status": "disabled",
                "enabled": False,
            }
    except Exception as e:
        logger.error(f"Agent check failed: {e}")
        services["agents"] = {
            "status": "down",
            "error": str(e),
            "enabled": False,
        }
        # Agents are optional, don't change overall status

    # Determine overall status
    if overall_status == "healthy" and any(s.get("status") == "down" for s in services.values()):
        overall_status = "degraded"

    return HealthResponse(
        status=overall_status,
        services=services,
        timestamp=datetime.now()
    )


@router.get(
    "/stats",
    response_model=StatsResponse,
    summary="System statistics",
    description="Get detailed statistics about repositories, knowledge graph, and search index"
)
async def get_stats(
    neo4j: Neo4jClient = Depends(get_neo4j),
    weaviate: WeaviateIndexer = Depends(get_weaviate),
    agent_cache: AgentCache = Depends(get_agent_cache),
    current_key: APIKey = Depends(require_auth),
):
    """
    Get system statistics.

    Returns detailed information about:
    - Indexed repositories
    - Knowledge graph (nodes and relationships by type)
    - Search index (total objects, embeddings)
    - Agent cache (if enabled)
    """
    # Get Neo4j statistics
    neo4j_stats = neo4j.get_statistics()
    knowledge_graph = {
        "total_nodes": neo4j_stats.get('total_nodes', 0),
        "total_relationships": neo4j_stats.get('total_relationships', 0),
        "nodes_by_type": neo4j_stats.get('nodes_by_type', {}),
        "relationships_by_type": neo4j_stats.get('relationships_by_type', {}),
    }

    # Get Weaviate statistics
    weaviate_stats = weaviate.get_statistics()
    search_index = {
        "total_objects": weaviate_stats.get('total_nodes', 0),
        "nodes_by_type": weaviate_stats.get('nodes_by_type', {}),
        "embedding_dimensions": 768,  # BGE-M3
    }

    # Get repository information from Neo4j
    # Query all Repository nodes
    repos_query = """
    MATCH (r:Repository)
    RETURN r
    """
    repo_results = neo4j.execute_cypher(repos_query)

    repositories = {}
    for record in repo_results:
        repo_node = record['r']
        repo_name = repo_node.get('name', 'unknown')

        # Get stats for this repo
        stats_query = f"""
        MATCH (r:Repository {{name: $repo_name}})-[:CONTAINS*]->(n)
        RETURN labels(n)[0] as type, count(n) as count
        """
        stats_results = neo4j.execute_cypher(stats_query, {"repo_name": repo_name})

        repo_stats = {}
        for stat in stats_results:
            repo_stats[stat['type']] = stat['count']

        repositories[repo_name] = RepositoryInfo(
            name=repo_name,
            type=repo_node.get('type', 'unknown'),
            path=repo_node.get('path', ''),
            language=repo_node.get('language', ''),
            framework=repo_node.get('framework'),
            branch=repo_node.get('branch', 'main'),
            commit_hash=repo_node.get('commit_hash'),
            last_indexed=repo_node.get('last_indexed'),
            stats=repo_stats
        )

    # Get agent cache statistics (if available)
    agent_cache_stats = None
    if agent_cache and agent_cache.config.enabled:
        agent_cache_stats = agent_cache.get_stats()

    return StatsResponse(
        repositories=repositories,
        knowledge_graph=knowledge_graph,
        search_index=search_index,
        agent_cache=agent_cache_stats
    )
