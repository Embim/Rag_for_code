"""
Knowledge graph module.

This module handles:
- Graph node models (File, Function, Class, Component, Endpoint, etc.)
- Graph relationships (CONTAINS, CALLS, IMPORTS, etc.)
- Graph building from parsed code
- Neo4j integration
- Weaviate vector indexing
"""

from .models import (
    GraphNode, GraphRelationship,
    NodeType, RelationshipType,
    RepositoryNode, FileNode, FunctionNode, ClassNode,
    ComponentNode, EndpointNode, ApiCallNode, ModelNode, RouteNode,
    DocumentNode,
    create_node_id, normalize_url,
)
from .neo4j_client import Neo4jClient
from .graph_builder import GraphBuilder
from .api_linker import APILinker
from .weaviate_indexer import WeaviateIndexer
from .code_loader import read_code, code_for_node

__all__ = [
    # Models
    'GraphNode', 'GraphRelationship',
    'NodeType', 'RelationshipType',
    'RepositoryNode', 'FileNode', 'FunctionNode', 'ClassNode',
    'ComponentNode', 'EndpointNode', 'ApiCallNode', 'ModelNode', 'RouteNode',
    'DocumentNode',
    'create_node_id', 'normalize_url',

    # Clients and builders
    'Neo4jClient',
    'GraphBuilder',
    'APILinker',
    'WeaviateIndexer',

    # Code loader (read source by file_path + line range)
    'read_code',
    'code_for_node',
]

__version__ = "0.1.0"
