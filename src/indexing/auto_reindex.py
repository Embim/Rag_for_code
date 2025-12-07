"""
Auto Reindex Service - automatically reindex repositories on schedule or git changes.

Features:
- Scheduled reindexing (cron-style)
- Git-based change detection (only reindex if commits changed)
- Webhook support for CI/CD integration
- Repository health monitoring

Usage:
    # Scheduled reindexing
    service = AutoReindexService(
        neo4j_client=neo4j,
        weaviate_indexer=weaviate,
        repos_dir="data/repos"
    )
    service.start_scheduler(interval_hours=24)
    
    # Manual trigger
    await service.reindex_all()
    await service.reindex_repo("my-repo")
"""

import asyncio
import hashlib
import json
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable
from threading import Thread
import time

from ..logger import get_logger

logger = get_logger(__name__)


@dataclass
class RepoStatus:
    """Status of a repository."""
    name: str
    path: str
    last_indexed: Optional[datetime] = None
    last_commit_hash: Optional[str] = None
    current_commit_hash: Optional[str] = None
    needs_reindex: bool = False
    indexing_in_progress: bool = False
    last_error: Optional[str] = None
    node_count: int = 0
    file_count: int = 0


@dataclass
class ReindexResult:
    """Result of a reindex operation."""
    repo_name: str
    success: bool
    duration_seconds: float = 0.0
    nodes_added: int = 0
    nodes_removed: int = 0
    files_processed: int = 0
    error: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)


class AutoReindexService:
    """
    Service for automatic repository reindexing.
    
    Monitors repositories for changes and reindexes as needed.
    """
    
    def __init__(
        self,
        neo4j_client=None,
        weaviate_indexer=None,
        repos_dir: str = "data/repos",
        state_file: str = "data/reindex_state.json",
        on_reindex_complete: Optional[Callable[[ReindexResult], None]] = None
    ):
        """
        Initialize auto reindex service.
        
        Args:
            neo4j_client: Neo4j client
            weaviate_indexer: Weaviate indexer
            repos_dir: Directory containing repositories
            state_file: File to persist indexing state
            on_reindex_complete: Callback when reindex completes
        """
        self.neo4j = neo4j_client
        self.weaviate = weaviate_indexer
        self.repos_dir = Path(repos_dir)
        self.state_file = Path(state_file)
        self.on_reindex_complete = on_reindex_complete
        
        self._scheduler_running = False
        self._scheduler_thread: Optional[Thread] = None
        
        # Load persisted state
        self._state: Dict[str, RepoStatus] = {}
        self._load_state()
    
    def _load_state(self):
        """Load persisted state from file."""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    data = json.load(f)
                    
                for repo_name, repo_data in data.get('repos', {}).items():
                    self._state[repo_name] = RepoStatus(
                        name=repo_name,
                        path=repo_data.get('path', ''),
                        last_indexed=datetime.fromisoformat(repo_data['last_indexed']) 
                            if repo_data.get('last_indexed') else None,
                        last_commit_hash=repo_data.get('last_commit_hash'),
                        node_count=repo_data.get('node_count', 0),
                        file_count=repo_data.get('file_count', 0)
                    )
                logger.info(f"Loaded reindex state for {len(self._state)} repos")
            except Exception as e:
                logger.error(f"Error loading reindex state: {e}")
    
    def _save_state(self):
        """Save state to file."""
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'repos': {
                name: {
                    'path': status.path,
                    'last_indexed': status.last_indexed.isoformat() if status.last_indexed else None,
                    'last_commit_hash': status.last_commit_hash,
                    'node_count': status.node_count,
                    'file_count': status.file_count
                }
                for name, status in self._state.items()
            },
            'updated_at': datetime.now().isoformat()
        }
        
        with open(self.state_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def get_current_commit(self, repo_path: Path) -> Optional[str]:
        """Get current git commit hash for a repository."""
        try:
            result = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                cwd=str(repo_path),
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
                timeout=10
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception as e:
            logger.warning(f"Could not get git commit for {repo_path}: {e}")
        return None
    
    def discover_repos(self) -> List[RepoStatus]:
        """Discover repositories in the repos directory."""
        repos = []
        
        if not self.repos_dir.exists():
            logger.warning(f"Repos directory does not exist: {self.repos_dir}")
            return repos
        
        for item in self.repos_dir.iterdir():
            if item.is_dir() and (item / '.git').exists():
                name = item.name
                current_commit = self.get_current_commit(item)
                
                # Get or create status
                if name in self._state:
                    status = self._state[name]
                    status.path = str(item)
                    status.current_commit_hash = current_commit
                    status.needs_reindex = (
                        status.last_commit_hash is None or
                        status.last_commit_hash != current_commit
                    )
                else:
                    status = RepoStatus(
                        name=name,
                        path=str(item),
                        current_commit_hash=current_commit,
                        needs_reindex=True
                    )
                    self._state[name] = status
                
                repos.append(status)
        
        return repos
    
    async def reindex_repo(
        self,
        repo_name: str,
        force: bool = False
    ) -> ReindexResult:
        """
        Reindex a single repository.
        
        Args:
            repo_name: Name of the repository
            force: Force reindex even if not needed
            
        Returns:
            ReindexResult with operation details
        """
        start_time = time.time()
        
        if repo_name not in self._state:
            return ReindexResult(
                repo_name=repo_name,
                success=False,
                error=f"Repository '{repo_name}' not found"
            )
        
        status = self._state[repo_name]
        
        if not force and not status.needs_reindex:
            logger.info(f"Repository {repo_name} is up to date, skipping")
            return ReindexResult(
                repo_name=repo_name,
                success=True,
                duration_seconds=0
            )
        
        logger.info(f"Starting reindex for {repo_name}...")
        status.indexing_in_progress = True
        status.last_error = None
        
        try:
            # Import here to avoid circular imports
            from ..code_rag.repo_loader import RepositoryLoader
            from ..code_rag.graph import GraphBuilder
            
            repo_path = Path(status.path)
            
            # Load repository
            loader = RepositoryLoader()
            repo_data = loader.load_from_path(repo_path, name=repo_name)
            
            # Count files
            files_processed = len(repo_data.get('files', []))
            
            # Clear old data for this repo
            if self.neo4j:
                self.neo4j.execute_cypher(
                    "MATCH (n {repository: $repo}) DETACH DELETE n",
                    {'repo': repo_name}
                )
            
            # Build graph
            nodes_added = 0
            if self.neo4j:
                builder = GraphBuilder(self.neo4j)
                nodes_added = builder.build_from_repository(repo_data)
            
            # Index in Weaviate
            if self.weaviate:
                self.weaviate.index_from_neo4j(repository=repo_name)
            
            # Update status
            duration = time.time() - start_time
            status.last_indexed = datetime.now()
            status.last_commit_hash = status.current_commit_hash
            status.needs_reindex = False
            status.node_count = nodes_added
            status.file_count = files_processed
            status.indexing_in_progress = False
            
            self._save_state()
            
            result = ReindexResult(
                repo_name=repo_name,
                success=True,
                duration_seconds=duration,
                nodes_added=nodes_added,
                files_processed=files_processed
            )
            
            logger.info(
                f"✅ Reindex complete for {repo_name}: "
                f"{nodes_added} nodes, {files_processed} files in {duration:.1f}s"
            )
            
            if self.on_reindex_complete:
                self.on_reindex_complete(result)
            
            return result
            
        except Exception as e:
            duration = time.time() - start_time
            error_msg = str(e)
            
            status.last_error = error_msg
            status.indexing_in_progress = False
            
            logger.error(f"❌ Reindex failed for {repo_name}: {error_msg}")
            
            result = ReindexResult(
                repo_name=repo_name,
                success=False,
                duration_seconds=duration,
                error=error_msg
            )
            
            if self.on_reindex_complete:
                self.on_reindex_complete(result)
            
            return result
    
    async def reindex_all(self, force: bool = False) -> List[ReindexResult]:
        """
        Reindex all repositories that need updating.
        
        Args:
            force: Force reindex all repos
            
        Returns:
            List of ReindexResult for each repo
        """
        repos = self.discover_repos()
        results = []
        
        for status in repos:
            if force or status.needs_reindex:
                result = await self.reindex_repo(status.name, force=force)
                results.append(result)
        
        return results
    
    def check_needs_reindex(self) -> List[str]:
        """
        Check which repositories need reindexing.
        
        Returns:
            List of repository names that need reindexing
        """
        self.discover_repos()
        return [
            name for name, status in self._state.items()
            if status.needs_reindex
        ]
    
    def start_scheduler(
        self,
        interval_hours: float = 24,
        check_interval_minutes: float = 60
    ):
        """
        Start background scheduler for automatic reindexing.
        
        Args:
            interval_hours: Minimum hours between reindex attempts per repo
            check_interval_minutes: How often to check for changes
        """
        if self._scheduler_running:
            logger.warning("Scheduler already running")
            return
        
        self._scheduler_running = True
        
        def scheduler_loop():
            logger.info(
                f"Starting reindex scheduler (check every {check_interval_minutes} min, "
                f"reindex every {interval_hours} hours)"
            )
            
            while self._scheduler_running:
                try:
                    # Check for repos needing reindex
                    repos_to_reindex = self.check_needs_reindex()
                    
                    if repos_to_reindex:
                        logger.info(f"Found {len(repos_to_reindex)} repos needing reindex")
                        
                        # Run reindex in async context
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        
                        try:
                            for repo_name in repos_to_reindex:
                                loop.run_until_complete(
                                    self.reindex_repo(repo_name)
                                )
                        finally:
                            loop.close()
                    
                    # Also check time-based reindex
                    for name, status in self._state.items():
                        if status.last_indexed:
                            time_since = datetime.now() - status.last_indexed
                            if time_since > timedelta(hours=interval_hours):
                                status.needs_reindex = True
                    
                except Exception as e:
                    logger.error(f"Scheduler error: {e}")
                
                # Sleep until next check
                time.sleep(check_interval_minutes * 60)
        
        self._scheduler_thread = Thread(target=scheduler_loop, daemon=True)
        self._scheduler_thread.start()
        logger.info("Reindex scheduler started")
    
    def stop_scheduler(self):
        """Stop the background scheduler."""
        self._scheduler_running = False
        if self._scheduler_thread:
            self._scheduler_thread.join(timeout=5)
            self._scheduler_thread = None
        logger.info("Reindex scheduler stopped")
    
    def get_status(self) -> Dict[str, Any]:
        """Get overall service status."""
        repos = self.discover_repos()
        
        return {
            'scheduler_running': self._scheduler_running,
            'repos_count': len(repos),
            'repos_needing_reindex': sum(1 for r in repos if r.needs_reindex),
            'repos_indexing': sum(1 for r in repos if r.indexing_in_progress),
            'repos': {
                r.name: {
                    'last_indexed': r.last_indexed.isoformat() if r.last_indexed else None,
                    'needs_reindex': r.needs_reindex,
                    'indexing': r.indexing_in_progress,
                    'node_count': r.node_count,
                    'file_count': r.file_count,
                    'error': r.last_error
                }
                for r in repos
            }
        }
    
    async def pull_and_reindex(self, repo_name: str) -> ReindexResult:
        """
        Git pull and reindex a repository.
        
        Args:
            repo_name: Name of the repository
            
        Returns:
            ReindexResult
        """
        if repo_name not in self._state:
            return ReindexResult(
                repo_name=repo_name,
                success=False,
                error=f"Repository '{repo_name}' not found"
            )
        
        status = self._state[repo_name]
        repo_path = Path(status.path)
        
        # Git pull
        try:
            result = subprocess.run(
                ['git', 'pull', '--ff-only'],
                cwd=str(repo_path),
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
                timeout=60
            )
            
            if result.returncode != 0:
                logger.warning(f"Git pull failed for {repo_name}: {result.stderr}")
            else:
                logger.info(f"Git pull successful for {repo_name}")
                
        except Exception as e:
            logger.error(f"Git pull error for {repo_name}: {e}")
        
        # Update commit hash
        status.current_commit_hash = self.get_current_commit(repo_path)
        status.needs_reindex = True
        
        # Reindex
        return await self.reindex_repo(repo_name, force=True)


# FastAPI webhook endpoint for CI/CD integration
def create_webhook_router(service: AutoReindexService):
    """
    Create FastAPI router for webhook endpoints.
    
    Usage:
        from fastapi import FastAPI
        app = FastAPI()
        app.include_router(create_webhook_router(reindex_service))
    """
    from fastapi import APIRouter, BackgroundTasks, HTTPException
    from pydantic import BaseModel
    
    router = APIRouter(prefix="/webhooks", tags=["webhooks"])
    
    class WebhookPayload(BaseModel):
        repository: str
        ref: Optional[str] = None
        action: str = "push"  # push, manual, etc.
    
    @router.post("/reindex")
    async def trigger_reindex(
        payload: WebhookPayload,
        background_tasks: BackgroundTasks
    ):
        """Trigger repository reindex (e.g., from GitHub webhook)."""
        if payload.repository not in service._state:
            raise HTTPException(404, f"Repository '{payload.repository}' not found")
        
        # Schedule reindex in background
        background_tasks.add_task(
            service.pull_and_reindex,
            payload.repository
        )
        
        return {
            "status": "scheduled",
            "repository": payload.repository,
            "message": "Reindex will start shortly"
        }
    
    @router.get("/status")
    async def get_status():
        """Get reindex service status."""
        return service.get_status()
    
    @router.post("/reindex-all")
    async def reindex_all(background_tasks: BackgroundTasks, force: bool = False):
        """Trigger reindex of all repositories."""
        repos_needing = service.check_needs_reindex()
        
        if not force and not repos_needing:
            return {"status": "skipped", "message": "All repositories up to date"}
        
        background_tasks.add_task(service.reindex_all, force)
        
        return {
            "status": "scheduled",
            "repositories": repos_needing if not force else list(service._state.keys())
        }
    
    return router

