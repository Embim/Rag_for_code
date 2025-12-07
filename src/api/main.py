"""
FastAPI main application.

This is the entry point for the Code RAG REST API.

Usage:
    # Development
    uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

    # Production
    uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 4

    # Or use the create_app factory
    python -m src.api.main
"""

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
import time

from .config import get_settings
from .dependencies import lifespan_context
from .models import ErrorResponse
from .routes import search, repos, diagnostics, visualize, auth, reindex
from .auth import create_initial_admin_key
from ..logger import get_logger, setup_logging
import os


# Initialize logging
setup_logging(
    level=os.getenv("LOG_LEVEL", "INFO"),
    enable_console=True
)

logger = get_logger(__name__)


def create_app() -> FastAPI:
    """
    Create and configure FastAPI application.

    Returns:
        FastAPI application instance
    """
    settings = get_settings()

    # Create FastAPI app
    app = FastAPI(
        title=settings.title,
        description=settings.description,
        version=settings.version,
        lifespan=lifespan_context,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # ========================================================================
    # Middleware
    # ========================================================================

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=settings.cors_allow_credentials,
        allow_methods=settings.cors_allow_methods,
        allow_headers=settings.cors_allow_headers,
    )

    # GZip compression
    app.add_middleware(GZipMiddleware, minimum_size=1000)

    # Request timing middleware
    @app.middleware("http")
    async def add_process_time_header(request: Request, call_next):
        """Add X-Process-Time header to all responses."""
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = f"{process_time:.4f}"
        return response

    # ========================================================================
    # Exception Handlers
    # ========================================================================

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        """Global exception handler for unhandled errors."""
        logger.error(f"Unhandled exception: {exc}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ErrorResponse(
                error="InternalServerError",
                message="An unexpected error occurred",
                details={"exception": str(exc)}
            ).model_dump(mode='json')  # Serialize datetime to string
        )

    # ========================================================================
    # Routes
    # ========================================================================

    # Include routers
    app.include_router(auth.router)  # Authentication endpoints (no auth required)
    app.include_router(search.router)
    app.include_router(repos.router)
    app.include_router(diagnostics.router)
    app.include_router(visualize.router)
    app.include_router(reindex.router)

    # Root endpoint
    @app.get("/", tags=["root"])
    async def root():
        """Root endpoint - API information."""
        return {
            "name": settings.title,
            "version": settings.version,
            "status": "running",
            "docs": "/docs",
            "health": "/api/health",
        }

    return app


# ============================================================================
# Application Instance
# ============================================================================

app = create_app()


# ============================================================================
# CLI Entry Point
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    settings = get_settings()

    logger.info(f"🚀 Starting {settings.title} v{settings.version}")
    logger.info(f"📍 Server: http://{settings.host}:{settings.port}")
    logger.info(f"📖 Docs: http://{settings.host}:{settings.port}/docs")

    uvicorn.run(
        "src.api.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
        log_level="info",
    )
