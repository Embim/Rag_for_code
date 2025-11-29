"""
API routes for repository reindexing.

Provides:
- Manual reindex trigger
- Webhook for CI/CD integration
- Reindex status monitoring
"""

from typing import Optional
from fastapi import APIRouter, BackgroundTasks, HTTPException, Depends
from pydantic import BaseModel

from ...indexing import AutoReindexService
from ...logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/reindex", tags=["reindex"])

# Global service instance (will be set by main app)
_reindex_service: Optional[AutoReindexService] = None


def get_reindex_service() -> AutoReindexService:
    """Dependency to get reindex service."""
    if _reindex_service is None:
        raise HTTPException(503, "Reindex service not initialized")
    return _reindex_service


def set_reindex_service(service: AutoReindexService):
    """Set the reindex service instance."""
    global _reindex_service
    _reindex_service = service


class WebhookPayload(BaseModel):
    """Webhook payload for triggering reindex."""
    repository: str
    ref: Optional[str] = None  # Git ref (branch/tag)
    action: str = "push"  # push, manual, schedule


class ReindexRequest(BaseModel):
    """Request to reindex specific repository."""
    repository: str
    force: bool = False


class ReindexAllRequest(BaseModel):
    """Request to reindex all repositories."""
    force: bool = False


@router.get("/status")
async def get_reindex_status(
    service: AutoReindexService = Depends(get_reindex_service)
):
    """
    Get status of all repositories and reindex service.
    
    Returns:
        Status of scheduler and all repositories
    """
    return service.get_status()


@router.get("/repos")
async def list_repositories(
    service: AutoReindexService = Depends(get_reindex_service)
):
    """
    List all discovered repositories and their index status.
    
    Returns:
        List of repositories with last indexed time, commit hash, etc.
    """
    repos = service.discover_repos()
    return {
        "count": len(repos),
        "repositories": [
            {
                "name": r.name,
                "path": r.path,
                "last_indexed": r.last_indexed.isoformat() if r.last_indexed else None,
                "current_commit": r.current_commit_hash,
                "last_commit": r.last_commit_hash,
                "needs_reindex": r.needs_reindex,
                "indexing": r.indexing_in_progress,
                "node_count": r.node_count,
                "file_count": r.file_count,
                "error": r.last_error
            }
            for r in repos
        ]
    }


@router.get("/repos/{repo_name}")
async def get_repository_status(
    repo_name: str,
    service: AutoReindexService = Depends(get_reindex_service)
):
    """
    Get status of a specific repository.
    
    Args:
        repo_name: Repository name
        
    Returns:
        Repository status details
    """
    repos = service.discover_repos()
    
    for r in repos:
        if r.name == repo_name:
            return {
                "name": r.name,
                "path": r.path,
                "last_indexed": r.last_indexed.isoformat() if r.last_indexed else None,
                "current_commit": r.current_commit_hash,
                "last_commit": r.last_commit_hash,
                "needs_reindex": r.needs_reindex,
                "indexing": r.indexing_in_progress,
                "node_count": r.node_count,
                "file_count": r.file_count,
                "error": r.last_error
            }
    
    raise HTTPException(404, f"Repository '{repo_name}' not found")


@router.post("/trigger")
async def trigger_reindex(
    request: ReindexRequest,
    background_tasks: BackgroundTasks,
    service: AutoReindexService = Depends(get_reindex_service)
):
    """
    Trigger reindex for a specific repository.
    
    Args:
        request: Repository name and force flag
        background_tasks: FastAPI background tasks
        
    Returns:
        Status message
    """
    repos = service.discover_repos()
    repo_names = [r.name for r in repos]
    
    if request.repository not in repo_names:
        raise HTTPException(
            404,
            f"Repository '{request.repository}' not found. Available: {repo_names}"
        )
    
    # Schedule reindex in background
    background_tasks.add_task(
        service.reindex_repo,
        request.repository,
        request.force
    )
    
    return {
        "status": "scheduled",
        "repository": request.repository,
        "force": request.force,
        "message": f"Reindex scheduled for '{request.repository}'"
    }


@router.post("/trigger-all")
async def trigger_reindex_all(
    request: ReindexAllRequest,
    background_tasks: BackgroundTasks,
    service: AutoReindexService = Depends(get_reindex_service)
):
    """
    Trigger reindex for all repositories.
    
    Args:
        request: Force flag
        background_tasks: FastAPI background tasks
        
    Returns:
        List of scheduled repositories
    """
    repos = service.discover_repos()
    
    if not request.force:
        repos_to_reindex = [r.name for r in repos if r.needs_reindex]
    else:
        repos_to_reindex = [r.name for r in repos]
    
    if not repos_to_reindex:
        return {
            "status": "skipped",
            "message": "All repositories are up to date"
        }
    
    # Schedule all reindexes
    background_tasks.add_task(service.reindex_all, request.force)
    
    return {
        "status": "scheduled",
        "repositories": repos_to_reindex,
        "force": request.force,
        "message": f"Reindex scheduled for {len(repos_to_reindex)} repositories"
    }


@router.post("/pull-and-reindex/{repo_name}")
async def pull_and_reindex(
    repo_name: str,
    background_tasks: BackgroundTasks,
    service: AutoReindexService = Depends(get_reindex_service)
):
    """
    Git pull and reindex a repository.
    
    Args:
        repo_name: Repository name
        background_tasks: FastAPI background tasks
        
    Returns:
        Status message
    """
    repos = service.discover_repos()
    repo_names = [r.name for r in repos]
    
    if repo_name not in repo_names:
        raise HTTPException(404, f"Repository '{repo_name}' not found")
    
    background_tasks.add_task(service.pull_and_reindex, repo_name)
    
    return {
        "status": "scheduled",
        "repository": repo_name,
        "message": f"Pull and reindex scheduled for '{repo_name}'"
    }


# Webhook endpoint for CI/CD integration
@router.post("/webhook")
async def webhook_handler(
    payload: WebhookPayload,
    background_tasks: BackgroundTasks,
    service: AutoReindexService = Depends(get_reindex_service)
):
    """
    Webhook endpoint for CI/CD integration (GitHub Actions, GitLab CI, etc.)
    
    Example GitHub Actions workflow:
    ```yaml
    - name: Trigger reindex
      run: |
        curl -X POST "http://your-api/api/reindex/webhook" \\
          -H "Content-Type: application/json" \\
          -d '{"repository": "${{ github.repository }}", "action": "push"}'
    ```
    
    Args:
        payload: Webhook payload with repository name and action
        background_tasks: FastAPI background tasks
        
    Returns:
        Status message
    """
    logger.info(f"Webhook received: {payload.repository} - {payload.action}")
    
    # Extract repo name from full path (e.g., "owner/repo" -> "repo")
    repo_name = payload.repository.split('/')[-1]
    
    # Check if we know this repo
    repos = service.discover_repos()
    repo_names = [r.name for r in repos]
    
    if repo_name not in repo_names:
        # Try to find by partial match
        matches = [n for n in repo_names if repo_name in n or n in repo_name]
        if matches:
            repo_name = matches[0]
        else:
            raise HTTPException(
                404,
                f"Repository '{repo_name}' not found. Known repos: {repo_names}"
            )
    
    # Schedule appropriate action
    if payload.action in ("push", "merge", "release"):
        background_tasks.add_task(service.pull_and_reindex, repo_name)
        action_desc = "pull and reindex"
    else:
        background_tasks.add_task(service.reindex_repo, repo_name, force=True)
        action_desc = "reindex"
    
    return {
        "status": "scheduled",
        "repository": repo_name,
        "action": action_desc,
        "webhook_action": payload.action,
        "ref": payload.ref
    }


@router.post("/scheduler/start")
async def start_scheduler(
    interval_hours: float = 24,
    check_interval_minutes: float = 60,
    service: AutoReindexService = Depends(get_reindex_service)
):
    """
    Start the background reindex scheduler.
    
    Args:
        interval_hours: Minimum hours between reindex per repo
        check_interval_minutes: How often to check for changes
        
    Returns:
        Status message
    """
    service.start_scheduler(
        interval_hours=interval_hours,
        check_interval_minutes=check_interval_minutes
    )
    
    return {
        "status": "started",
        "interval_hours": interval_hours,
        "check_interval_minutes": check_interval_minutes
    }


@router.post("/scheduler/stop")
async def stop_scheduler(
    service: AutoReindexService = Depends(get_reindex_service)
):
    """Stop the background reindex scheduler."""
    service.stop_scheduler()
    return {"status": "stopped"}

