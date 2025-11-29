"""
Unified configuration module for Code RAG.

All configuration classes in one place:
- SearchConfig: Search parameters
- AgentConfig: LLM agent settings  
- CacheConfig: Caching settings
- Neo4jConfig: Database connection
- WeaviateConfig: Vector store connection
- APISettings: REST API settings

Usage:
    from src.config import SearchConfig, AgentConfig
    
    config = SearchConfig(top_k=20, enable_reranking=True)
"""

import os
from pathlib import Path

from .search import SearchConfig, SearchStrategy
from .agent import AgentConfig
from .cache import CacheConfig
from .database import Neo4jConfig, WeaviateConfig
from .api import APISettings
from .base import BaseConfig

# =============================================================================
# Path constants (previously in src/config.py)
# =============================================================================
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# Create directories
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# Weaviate defaults
# =============================================================================
WEAVIATE_URL = os.environ.get("WEAVIATE_URL", "http://localhost:8080")
WEAVIATE_CLASS_NAME = "CodeEntity"

# =============================================================================
# Embedding defaults
# =============================================================================
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-m3")
EMBEDDING_DEVICE = "cuda" if os.environ.get("FORCE_CPU", "").lower() != "true" else "cpu"
EMBEDDING_BATCH_SIZE = 128 if EMBEDDING_DEVICE == "cuda" else 16

# =============================================================================
# Search defaults
# =============================================================================
TOP_K_DENSE = 50
TOP_K_BM25 = 50
TOP_K_RERANK = 20
HYBRID_ALPHA = 0.7

__all__ = [
    # Config classes
    'SearchConfig',
    'SearchStrategy',
    'AgentConfig', 
    'CacheConfig',
    'Neo4jConfig',
    'WeaviateConfig',
    'APISettings',
    'BaseConfig',
    # Path constants
    'PROJECT_ROOT',
    'DATA_DIR', 
    'OUTPUTS_DIR', 
    'MODELS_DIR', 
    'PROCESSED_DIR',
    # Weaviate
    'WEAVIATE_URL', 
    'WEAVIATE_CLASS_NAME',
    # Embedding
    'EMBEDDING_MODEL', 
    'EMBEDDING_DEVICE', 
    'EMBEDDING_BATCH_SIZE',
    # Search
    'TOP_K_DENSE', 
    'TOP_K_BM25', 
    'TOP_K_RERANK', 
    'HYBRID_ALPHA',
]

