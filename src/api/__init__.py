"""
FastAPI REST API for Code RAG System.

This module provides a REST API interface for:
- Code search and Q&A
- Repository management
- System diagnostics
- Visualization generation

The API can be used by the Telegram bot or any other client.
"""

from .main import app, create_app

__all__ = ["app", "create_app"]
