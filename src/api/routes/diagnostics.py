"""
Diagnostics endpoints.

Provides:
- GET /health - Health check
- GET /stats - System statistics
- GET /graph-schema - Graph database schema visualization
"""

import time
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status, Query
from fastapi.responses import Response
from datetime import datetime
import base64

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
        stats_results = neo4j.execute_cypher(stats_query, repo_name=repo_name)

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


@router.get(
    "/graph-schema",
    summary="Graph database schema visualization",
    description="Generate Mermaid diagram showing Neo4j graph structure (node types and relationships)"
)
async def get_graph_schema(
    neo4j: Neo4jClient = Depends(get_neo4j),
    format: str = Query(default="mermaid", description="Output format: 'mermaid' or 'url'"),
    current_key: APIKey = Depends(require_auth),
):
    """
    Generate graph schema visualization.

    Shows:
    - All node types in the graph
    - All relationship types between nodes
    - Node counts for each type

    **Formats:**
    - `mermaid`: Returns Mermaid code
    - `url`: Returns rendered image URL
    """
    try:
        # Get statistics to understand node types
        stats = neo4j.get_statistics()
        nodes_by_type = stats.get('nodes_by_type', {})
        relationships_by_type = stats.get('relationships_by_type', {})

        # Query to get relationship patterns
        relationship_query = """
        MATCH (a)-[r]->(b)
        RETURN DISTINCT labels(a)[0] as from_type, type(r) as rel_type, labels(b)[0] as to_type, count(r) as count
        ORDER BY count DESC
        LIMIT 50
        """
        rel_results = neo4j.execute_cypher(relationship_query)

        # Build Mermaid diagram
        mermaid_lines = ["graph TB"]
        mermaid_lines.append("    %% Node Types")

        # Add nodes with counts
        for node_type, count in sorted(nodes_by_type.items(), key=lambda x: -x[1]):
            safe_type = node_type.replace(' ', '_').replace('-', '_')
            mermaid_lines.append(f"    {safe_type}[\"**{node_type}**<br/>{count} nodes\"]")
            # Style based on type
            if node_type == 'Repository':
                mermaid_lines.append(f"    style {safe_type} fill:#e1f5ff,stroke:#01579b,stroke-width:2px")
            elif node_type in ['Function', 'Method', 'Class']:
                mermaid_lines.append(f"    style {safe_type} fill:#f3e5f5,stroke:#4a148c,stroke-width:2px")
            elif node_type in ['Component', 'File']:
                mermaid_lines.append(f"    style {safe_type} fill:#e8f5e9,stroke:#1b5e20,stroke-width:2px")
            elif node_type == 'Endpoint':
                mermaid_lines.append(f"    style {safe_type} fill:#fff3e0,stroke:#e65100,stroke-width:2px")
            elif node_type == 'Model':
                mermaid_lines.append(f"    style {safe_type} fill:#fce4ec,stroke:#880e4f,stroke-width:2px")

        # Add relationships
        mermaid_lines.append("")
        mermaid_lines.append("    %% Relationships")

        for record in rel_results:
            from_type = record['from_type'].replace(' ', '_').replace('-', '_')
            to_type = record['to_type'].replace(' ', '_').replace('-', '_')
            rel_type = record['rel_type']
            count = record['count']

            # Format relationship label
            rel_label = f"{rel_type}<br/>{count}"
            mermaid_lines.append(f"    {from_type} -->|{rel_label}| {to_type}")

        mermaid_code = '\n'.join(mermaid_lines)

        if format == "url":
            # Generate URL to rendered diagram
            encoded = base64.b64encode(mermaid_code.encode('utf-8')).decode('utf-8')
            url = f"https://mermaid.ink/img/{encoded}"

            return {
                "mermaid_code": mermaid_code,
                "diagram_url": url,
                "node_types": len(nodes_by_type),
                "relationship_types": len(relationships_by_type),
                "total_nodes": stats.get('total_nodes', 0),
                "total_relationships": stats.get('total_relationships', 0),
            }
        else:
            # Return Mermaid code
            return Response(
                content=mermaid_code,
                media_type="text/plain",
                headers={
                    "X-Node-Types": str(len(nodes_by_type)),
                    "X-Total-Nodes": str(stats.get('total_nodes', 0)),
                }
            )

    except Exception as e:
        logger.error(f"Graph schema generation failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate graph schema: {str(e)}"
        )


@router.get(
    "/search-graph",
    summary="Search results graph visualization",
    description="Visualize search results as a graph showing found nodes and their relationships"
)
async def visualize_search_graph(
    query: str = Query(..., description="Search query"),
    limit: int = Query(default=10, ge=1, le=20, description="Max nodes to show"),
    format: str = Query(default="url", description="Output format: 'mermaid' or 'url'"),
    neo4j: Neo4jClient = Depends(get_neo4j),
    current_key: APIKey = Depends(require_auth),
):
    """
    Visualize search results as a graph.

    Shows:
    - Nodes matching the search query
    - Relationships between found nodes
    - Node types and names

    **Example:** `/api/search-graph?query=equity%20trade&limit=10&format=url`
    """
    try:
        # Search for nodes matching query (simple name-based search)
        search_query = """
        MATCH (n)
        WHERE n.name CONTAINS $query OR n.id CONTAINS $query
        RETURN n
        ORDER BY n.name
        LIMIT $limit
        """
        search_results = neo4j.execute_cypher(search_query, query=query, limit=limit)

        if not search_results:
            return {
                "message": f"No nodes found for query: '{query}'",
                "mermaid_code": "",
                "diagram_url": "",
            }

        # Get node IDs
        node_ids = [record['n'].get('id') for record in search_results if record['n'].get('id')]

        # Find relationships between these nodes
        if len(node_ids) > 1:
            rel_query = """
            MATCH (a)-[r]->(b)
            WHERE a.id IN $node_ids AND b.id IN $node_ids
            RETURN a, type(r) as rel_type, b
            LIMIT 50
            """
            rel_results = neo4j.execute_cypher(rel_query, node_ids=node_ids)
        else:
            rel_results = []

        # Build Mermaid diagram
        mermaid_lines = ["graph LR"]
        mermaid_lines.append(f"    %% Search results for: {query}")
        mermaid_lines.append("")

        # Add nodes
        node_map = {}
        for i, record in enumerate(search_results):
            node = record['n']
            node_id = node.get('id', f'node_{i}')
            node_name = node.get('name', 'Unknown')
            node_type = node.get('type') or (node.labels[0] if hasattr(node, 'labels') and node.labels else 'Unknown')

            # Create safe identifier
            safe_id = f"N{i}"
            node_map[node_id] = safe_id

            # Format node label
            label = f"{node_name}<br/><small>{node_type}</small>"
            mermaid_lines.append(f"    {safe_id}[\"{label}\"]")

            # Style based on type
            if node_type in ['Function', 'Method']:
                mermaid_lines.append(f"    style {safe_id} fill:#f3e5f5,stroke:#4a148c")
            elif node_type in ['Component', 'File']:
                mermaid_lines.append(f"    style {safe_id} fill:#e8f5e9,stroke:#1b5e20")
            elif node_type == 'Endpoint':
                mermaid_lines.append(f"    style {safe_id} fill:#fff3e0,stroke:#e65100")
            elif node_type == 'Model':
                mermaid_lines.append(f"    style {safe_id} fill:#fce4ec,stroke:#880e4f")
            elif node_type == 'Class':
                mermaid_lines.append(f"    style {safe_id} fill:#e3f2fd,stroke:#0d47a1")

        # Add relationships
        if rel_results:
            mermaid_lines.append("")
            mermaid_lines.append("    %% Relationships")
            for record in rel_results:
                a_id = record['a'].get('id')
                b_id = record['b'].get('id')
                rel_type = record['rel_type']

                if a_id in node_map and b_id in node_map:
                    safe_a = node_map[a_id]
                    safe_b = node_map[b_id]
                    mermaid_lines.append(f"    {safe_a} -->|{rel_type}| {safe_b}")

        mermaid_code = '\n'.join(mermaid_lines)

        if format == "url":
            # Generate URL to rendered diagram
            encoded = base64.b64encode(mermaid_code.encode('utf-8')).decode('utf-8')
            url = f"https://mermaid.ink/img/{encoded}"

            return {
                "query": query,
                "nodes_found": len(search_results),
                "relationships_found": len(rel_results),
                "mermaid_code": mermaid_code,
                "diagram_url": url,
            }
        else:
            # Return Mermaid code
            return Response(
                content=mermaid_code,
                media_type="text/plain",
                headers={
                    "X-Query": query,
                    "X-Nodes-Found": str(len(search_results)),
                }
            )

    except Exception as e:
        logger.error(f"Search graph visualization failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to visualize search graph: {str(e)}"
        )


@router.get(
    "/repository-structure",
    summary="Repository structure visualization",
    description="Visualize repository structure showing files, classes, functions hierarchy"
)
async def visualize_repository_structure(
    repository: str = Query(..., description="Repository name "),
    max_depth: int = Query(default=2, ge=1, le=4, description="Max depth to traverse"),
    format: str = Query(default="url", description="Output format: 'mermaid' or 'url'"),
    neo4j: Neo4jClient = Depends(get_neo4j),
    current_key: APIKey = Depends(require_auth),
):
    """
    Visualize repository structure as a tree.

    Shows hierarchical structure:
    - Repository → Files → Classes/Functions → Methods

    **Example:** `/api/repository-structure?repository=api&max_depth=2`
    """
    try:
        # Query repository structure
        structure_query = """
        MATCH path = (r:Repository {name: $repo_name})-[:CONTAINS*1..3]->(n)
        RETURN path
        LIMIT 100
        """
        results = neo4j.execute_cypher(structure_query, repo_name=repository)

        if not results:
            return {
                "message": f"Repository '{repository}' not found or has no structure",
                "mermaid_code": "",
                "diagram_url": "",
            }

        # Build hierarchical structure
        mermaid_lines = ["graph TD"]
        mermaid_lines.append(f"    %% Repository: {repository}")
        mermaid_lines.append("")

        # Repository root
        repo_id = "REPO"
        mermaid_lines.append(f"    {repo_id}[\"{repository}<br/><small>Repository</small>\"]")
        mermaid_lines.append(f"    style {repo_id} fill:#e1f5ff,stroke:#01579b,stroke-width:3px")

        # Track nodes and relationships
        nodes_added = {repository: repo_id}
        node_counter = 0

        for record in results[:50]:  # Limit to prevent diagram from being too large
            path = record['path']
            nodes = path.nodes if hasattr(path, 'nodes') else []

            prev_id = repo_id
            for node in nodes[1:]:  # Skip repository node
                node_name = node.get('name', 'Unknown')
                node_type = node.get('type') or (node.labels[0] if hasattr(node, 'labels') and node.labels else 'Unknown')
                node_key = f"{node_name}_{node_type}"

                # Check if node already added
                if node_key not in nodes_added:
                    node_counter += 1
                    safe_id = f"N{node_counter}"
                    nodes_added[node_key] = safe_id

                    # Shorten long names
                    display_name = node_name if len(node_name) <= 30 else node_name[:27] + "..."

                    # Add node
                    label = f"{display_name}<br/><small>{node_type}</small>"
                    mermaid_lines.append(f"    {safe_id}[\"{label}\"]")

                    # Style based on type
                    if node_type in ['File']:
                        mermaid_lines.append(f"    style {safe_id} fill:#e8f5e9,stroke:#1b5e20")
                    elif node_type in ['Class']:
                        mermaid_lines.append(f"    style {safe_id} fill:#e3f2fd,stroke:#0d47a1")
                    elif node_type in ['Function', 'Method']:
                        mermaid_lines.append(f"    style {safe_id} fill:#f3e5f5,stroke:#4a148c")
                    elif node_type == 'Endpoint':
                        mermaid_lines.append(f"    style {safe_id} fill:#fff3e0,stroke:#e65100")

                    # Add relationship
                    mermaid_lines.append(f"    {prev_id} --> {safe_id}")

                prev_id = nodes_added[node_key]

        mermaid_code = '\n'.join(mermaid_lines)

        if format == "url":
            # Generate URL to rendered diagram
            encoded = base64.b64encode(mermaid_code.encode('utf-8')).decode('utf-8')
            url = f"https://mermaid.ink/img/{encoded}"

            return {
                "repository": repository,
                "nodes_shown": len(nodes_added),
                "mermaid_code": mermaid_code,
                "diagram_url": url,
            }
        else:
            # Return Mermaid code
            return Response(
                content=mermaid_code,
                media_type="text/plain",
                headers={
                    "X-Repository": repository,
                    "X-Nodes": str(len(nodes_added)),
                }
            )

    except Exception as e:
        logger.error(f"Repository structure visualization failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to visualize repository structure: {str(e)}"
        )
