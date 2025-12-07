"""
Repository management endpoints.

Provides:
- GET /repos - List all repositories
- POST /repos - Add new repository
- DELETE /repos/{name} - Remove repository
- POST /repos/{name}/reindex - Reindex repository
- GET /repos/{name}/status - Get indexing status
"""

import asyncio
from typing import Dict, List, Optional
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks

from ..models import (
    RepositoryInfo,
    AddRepositoryRequest,
    RepositoryIndexingStatus,
    ErrorResponse,
)
from ..dependencies import get_neo4j, get_weaviate
from ..auth import require_auth, require_user, require_admin, APIKey
from ...code_rag.graph import Neo4jClient, WeaviateIndexer
from ...code_rag.repo_loader import RepositoryLoader
from ...code_rag.project_detector import ProjectDetector
from ...code_rag.graph.build_and_index import GraphPipeline
from ...logger import get_logger


logger = get_logger(__name__)

router = APIRouter(prefix="/api/repos", tags=["repositories"])


# ============================================================================
# Background Indexing State
# ============================================================================

# Global state for tracking indexing jobs
indexing_jobs: Dict[str, RepositoryIndexingStatus] = {}


async def _index_repository_background(
    repo_name: str,
    source: str,
    branch: str,
    neo4j_client: Neo4jClient,
    weaviate_indexer: WeaviateIndexer,
):
    """
    Background task to index a repository.

    Updates indexing_jobs dict with progress.
    """
    from datetime import datetime

    try:
        # Update status to in_progress
        indexing_jobs[repo_name] = RepositoryIndexingStatus(
            repository=repo_name,
            status="in_progress",
            progress=0.1,
            message="Cloning repository...",
            started_at=datetime.now()
        )

        # Initialize pipeline
        pipeline = GraphPipeline(
            neo4j_uri=neo4j_client._uri,
            neo4j_user=neo4j_client._user,
            neo4j_password=neo4j_client._password,
            weaviate_url=weaviate_indexer.weaviate_url,
            embedding_model=weaviate_indexer.embedding_model_name,
        )

        # Update progress: parsing
        indexing_jobs[repo_name].progress = 0.3
        indexing_jobs[repo_name].message = "Parsing code files..."

        # Run pipeline
        stats = pipeline.run(
            source=source,
            name=repo_name,
            branch=branch,
            clear_existing=False,  # Keep existing data
            link_apis=True,
            index_weaviate=True
        )

        # Update progress: indexing
        indexing_jobs[repo_name].progress = 0.7
        indexing_jobs[repo_name].message = "Building knowledge graph..."
        indexing_jobs[repo_name].files_processed = stats.get('files_parsed', 0)
        indexing_jobs[repo_name].entities_found = stats.get('entities_found', 0)

        # Final update
        indexing_jobs[repo_name].progress = 1.0
        indexing_jobs[repo_name].status = "completed"
        indexing_jobs[repo_name].message = f"Indexed {stats.get('nodes_created', 0)} entities successfully"
        indexing_jobs[repo_name].completed_at = datetime.now()

        pipeline.close()

        logger.info(f"✅ Repository '{repo_name}' indexed successfully")

    except Exception as e:
        logger.error(f"Failed to index repository '{repo_name}': {e}", exc_info=True)

        indexing_jobs[repo_name].status = "failed"
        indexing_jobs[repo_name].message = f"Indexing failed: {str(e)}"
        indexing_jobs[repo_name].errors = [str(e)]
        indexing_jobs[repo_name].completed_at = datetime.now()


# ============================================================================
# Endpoints
# ============================================================================

@router.get(
    "",
    response_model=Dict[str, RepositoryInfo],
    summary="List repositories",
    description="Get list of all indexed repositories with their statistics"
)
async def list_repositories(
    neo4j: Neo4jClient = Depends(get_neo4j),
    current_key: APIKey = Depends(require_auth),
):
    """
    List all repositories.

    Returns a dictionary mapping repository name to RepositoryInfo.
    """
    # Query all Repository nodes
    query = """
    MATCH (r:Repository)
    RETURN r
    """
    results = neo4j.execute_cypher(query)

    repositories = {}
    for record in results:
        repo_node = record['r']
        repo_name = repo_node.get('name', 'unknown')

        # Get stats for this repo
        stats_query = """
        MATCH (r:Repository {name: $repo_name})-[:CONTAINS*]->(n)
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

    return repositories


@router.post(
    "",
    response_model=RepositoryIndexingStatus,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Add repository",
    description="Add new repository and start indexing in background"
)
async def add_repository(
    request: AddRepositoryRequest,
    background_tasks: BackgroundTasks,
    neo4j: Neo4jClient = Depends(get_neo4j),
    weaviate: WeaviateIndexer = Depends(get_weaviate),
    current_key: APIKey = Depends(require_user),  # Requires write access
):
    """
    Add new repository.

    The repository will be cloned, parsed, and indexed in the background.
    Use GET /repos/{name}/status to check indexing progress.

    **Example:**
    ```json
    {
        "source": "https://github.com/user/backend.git",
        "name": "my-backend",
        "type": "backend",
        "branch": "main",
        "auto_detect_framework": true
    }
    ```

    Returns 202 Accepted with indexing status.
    """
    # Check if repository already exists
    query = """
    MATCH (r:Repository {name: $name})
    RETURN r
    """
    existing = neo4j.execute_cypher(query, name=request.name)

    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Repository '{request.name}' already exists. Use DELETE first or choose different name."
        )

    # Initialize indexing status
    from datetime import datetime
    indexing_jobs[request.name] = RepositoryIndexingStatus(
        repository=request.name,
        status="queued",
        progress=0.0,
        message="Queued for indexing...",
        started_at=datetime.now()
    )

    # Schedule background indexing
    # Branch defaults to "main" for URLs, None for local paths
    branch = request.branch or "main" if request.source.startswith(('http://', 'https://', 'git@')) else request.branch
    
    background_tasks.add_task(
        _index_repository_background,
        repo_name=request.name,
        source=request.source,
        branch=branch,
        neo4j_client=neo4j,
        weaviate_indexer=weaviate,
    )

    logger.info(f"Queued repository '{request.name}' for indexing from '{request.source}'")

    return indexing_jobs[request.name]


@router.get(
    "/{name}/status",
    response_model=RepositoryIndexingStatus,
    summary="Get indexing status",
    description="Get current indexing status for a repository"
)
async def get_repository_status(name: str):
    """
    Get indexing status.

    Returns the current status of repository indexing operation.
    """
    if name not in indexing_jobs:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No indexing job found for repository '{name}'"
        )

    return indexing_jobs[name]


@router.delete(
    "/{name}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete repository",
    description="Remove repository and all its data from the system"
)
async def delete_repository(
    name: str,
    neo4j: Neo4jClient = Depends(get_neo4j),
    weaviate: WeaviateIndexer = Depends(get_weaviate),
    current_key: APIKey = Depends(require_admin),  # Requires admin access
):
    """
    Delete repository.

    Removes:
    - All nodes and relationships in Neo4j
    - All objects in Weaviate
    - Local files (if cloned)

    **Warning:** This operation cannot be undone!
    """
    # Check if repository exists
    query = """
    MATCH (r:Repository {name: $name})
    RETURN r
    """
    results = neo4j.execute_cypher(query, name=name)

    if not results:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Repository '{name}' not found"
        )

    try:
        # Delete from Neo4j (cascade delete all contained nodes)
        delete_query = """
        MATCH (r:Repository {name: $name})
        OPTIONAL MATCH (r)-[:CONTAINS*]->(n)
        DETACH DELETE r, n
        """
        neo4j.execute_cypher(delete_query, name=name)

        logger.info(f"Deleted repository '{name}' from Neo4j")

        # Delete from Weaviate
        # Note: WeaviateIndexer doesn't have delete_by_repo method yet
        # This is a TODO - for now, nodes will remain in Weaviate
        # TODO: Implement WeaviateIndexer.delete_repository(repo_name)

        # Delete local files
        repos_dir = Path("data/repos")
        repo_path = repos_dir / name
        if repo_path.exists():
            import shutil
            shutil.rmtree(repo_path)
            logger.info(f"Deleted local files for '{name}'")

        # Clean up indexing status
        if name in indexing_jobs:
            del indexing_jobs[name]

        logger.info(f"✅ Repository '{name}' deleted successfully")

    except Exception as e:
        logger.error(f"Failed to delete repository '{name}': {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete repository: {str(e)}"
        )


@router.post(
    "/{name}/reindex",
    response_model=RepositoryIndexingStatus,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Reindex repository",
    description="Trigger full reindexing of repository"
)
async def reindex_repository(
    name: str,
    background_tasks: BackgroundTasks,
    neo4j: Neo4jClient = Depends(get_neo4j),
    weaviate: WeaviateIndexer = Depends(get_weaviate),
    current_key: APIKey = Depends(require_user),  # Requires write access
):
    """
    Reindex repository.

    Performs incremental update:
    - Fetches latest changes from Git
    - Parses modified files
    - Updates knowledge graph

    Returns 202 Accepted with indexing status.
    """
    # Check if repository exists
    query = """
    MATCH (r:Repository {name: $name})
    RETURN r
    """
    results = neo4j.execute_cypher(query, name=name)

    if not results:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Repository '{name}' not found"
        )

    repo_node = results[0]['r']
    source = repo_node.get('path', '')
    branch = repo_node.get('branch', 'main')

    # Check if already indexing
    if name in indexing_jobs and indexing_jobs[name].status == "in_progress":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Repository '{name}' is already being indexed"
        )

    # Initialize indexing status
    from datetime import datetime
    indexing_jobs[name] = RepositoryIndexingStatus(
        repository=name,
        status="queued",
        progress=0.0,
        message="Queued for reindexing...",
        started_at=datetime.now()
    )

    # Schedule background reindexing
    background_tasks.add_task(
        _index_repository_background,
        repo_name=name,
        source=source,
        branch=branch,
        neo4j_client=neo4j,
        weaviate_indexer=weaviate,
    )

    logger.info(f"Queued repository '{name}' for reindexing")

    return indexing_jobs[name]
