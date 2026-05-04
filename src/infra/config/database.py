"""
Database configuration for Code RAG.
"""

from dataclasses import dataclass, field
from typing import Optional
import os

from .base import BaseConfig


@dataclass
class Neo4jConfig(BaseConfig):
    """
    Neo4j connection configuration.
    
    Attributes:
        uri: Bolt URI (bolt://host:port)
        user: Username
        password: Password
        database: Database name (default: neo4j)
        max_connection_pool_size: Connection pool size
        connection_timeout: Timeout in seconds
    """
    uri: str = "bolt://localhost:7687"
    user: str = "neo4j"
    password: str = field(
        default_factory=lambda: os.getenv("NEO4J_PASSWORD", "password")
    )
    database: str = "neo4j"
    max_connection_pool_size: int = 50
    connection_timeout: float = 30.0
    
    @classmethod
    def from_env(cls) -> 'Neo4jConfig':
        """Load from environment variables."""
        return cls(
            uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            user=os.getenv("NEO4J_USER", "neo4j"),
            password=os.getenv("NEO4J_PASSWORD", "password"),
            database=os.getenv("NEO4J_DATABASE", "neo4j"),
        )


@dataclass
class WeaviateConfig(BaseConfig):
    """
    Weaviate connection configuration.
    
    Attributes:
        url: Weaviate URL
        api_key: Optional API key
        class_name: Collection/class name
        embedding_model: Model for embeddings
        embedding_device: Device for embedding (cpu/cuda)
        batch_size: Batch size for indexing
    """
    url: str = "http://localhost:8080"
    api_key: Optional[str] = None
    class_name: str = "CodeEntity"
    
    # Embedding settings
    embedding_model: str = "BAAI/bge-m3"
    embedding_device: str = "cuda"
    embedding_batch_size: int = 32
    embedding_dimensions: int = 1024
    
    @classmethod
    def from_env(cls) -> 'WeaviateConfig':
        """Load from environment variables."""
        return cls(
            url=os.getenv("WEAVIATE_URL", "http://localhost:8080"),
            api_key=os.getenv("WEAVIATE_API_KEY"),
            class_name=os.getenv("WEAVIATE_CLASS_NAME", "CodeEntity"),
            embedding_model=os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3"),
            embedding_device=os.getenv("EMBEDDING_DEVICE", "cuda"),
        )

