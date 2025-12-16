"""
Graph builder - builds knowledge graph from parsed code entities.

This is the core component that:
1. Takes parsed entities from all parsers
2. Creates graph nodes for each entity
3. Establishes relationships between nodes
4. Resolves imports and calls
"""

import ast
from pathlib import Path
from typing import List, Dict, Set, Optional, Tuple
from collections import defaultdict

from ..parsers.base import CodeEntity, EntityType as ParserEntityType, ParseResult
from .models import (
    GraphNode, GraphRelationship,
    RepositoryNode, FileNode, FunctionNode, ClassNode,
    ComponentNode, EndpointNode, ModelNode, RouteNode,
    NodeType, RelationshipType,
    create_node_id
)
from .neo4j_client import Neo4jClient
from ..repo_loader import RepositoryInfo
from src.logger import get_logger


logger = get_logger(__name__)


class GraphBuilder:
    """
    Builds knowledge graph from parsed code entities.

    Two-pass algorithm:
    1. First pass: Create nodes for all entities
    2. Second pass: Create relationships (requires all nodes to exist)
    """

    def __init__(self, neo4j_client: Neo4jClient):
        """
        Initialize graph builder.

        Args:
            neo4j_client: Neo4j client for graph operations
        """
        self.client = neo4j_client

        # Temporary storage during graph building
        self.nodes: Dict[str, GraphNode] = {}  # node_id -> GraphNode
        self.relationships: List[GraphRelationship] = []

        # Indexes for faster lookup
        self.nodes_by_file: Dict[str, List[str]] = defaultdict(list)  # file_path -> [node_ids]
        self.imports_by_file: Dict[str, List[str]] = defaultdict(list)  # file_path -> [imports]

    def build_graph(
        self,
        repo_info: RepositoryInfo,
        parse_results: List[Tuple[Path, ParseResult]]
    ) -> Dict[str, int]:
        """
        Build knowledge graph from parse results.

        Args:
            repo_info: Repository information
            parse_results: List of (file_path, ParseResult) tuples

        Returns:
            Statistics dictionary with counts
        """
        logger.info(f"Building knowledge graph for repository: {repo_info.name}")

        # Clear temporary storage
        self.nodes.clear()
        self.relationships.clear()
        self.nodes_by_file.clear()
        self.imports_by_file.clear()

        # Pass 1: Create repository node
        repo_node = self._create_repository_node(repo_info)
        self.nodes[repo_node.id] = repo_node

        # Pass 2: Create file and entity nodes
        for file_path, parse_result in parse_results:
            self._process_file(repo_info, file_path, parse_result)

        # Pass 3: Create relationships
        self._create_relationships(repo_info, parse_results)

        # Save to Neo4j
        stats = self._save_to_neo4j()

        logger.info(f"Graph building complete: {stats}")
        return stats

    def _create_repository_node(self, repo_info: RepositoryInfo) -> RepositoryNode:
        """Create repository node."""
        node_id = create_node_id(repo_info.name)

        return RepositoryNode(
            id=node_id,
            name=repo_info.name,
            url=repo_info.url,
            branch=repo_info.branch,
            commit_hash=repo_info.commit_hash,
            project_type=repo_info.project_type,
            languages=repo_info.languages or [],
            frameworks=repo_info.frameworks or []
        )

    def _process_file(
        self,
        repo_info: RepositoryInfo,
        file_path: Path,
        parse_result: ParseResult
    ):
        """
        Process a single file and create nodes for it and its entities.
        """
        # Create file node
        rel_path = str(file_path.relative_to(repo_info.path))
        file_node_id = create_node_id(repo_info.name, rel_path)

        # Count lines with proper encoding handling
        line_count = 0
        if file_path.exists():
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    line_count = len(f.readlines())
            except Exception:
                line_count = 0

        file_node = FileNode(
            id=file_node_id,
            name=file_path.name,
            file_path=rel_path,
            language=parse_result.language,
            size_bytes=file_path.stat().st_size if file_path.exists() else 0,
            line_count=line_count
        )

        self.nodes[file_node.id] = file_node
        self.nodes_by_file[rel_path].append(file_node.id)

        # Repository CONTAINS File
        self.relationships.append(GraphRelationship(
            type=RelationshipType.CONTAINS,
            source_id=create_node_id(repo_info.name),
            target_id=file_node.id
        ))

        # Store imports for later resolution
        self.imports_by_file[rel_path] = parse_result.imports

        # Create entity nodes
        for entity in parse_result.entities:
            entity_node = self._create_entity_node(repo_info, rel_path, entity)

            if entity_node:
                self.nodes[entity_node.id] = entity_node
                self.nodes_by_file[rel_path].append(entity_node.id)

                # File CONTAINS Entity
                self.relationships.append(GraphRelationship(
                    type=RelationshipType.CONTAINS,
                    source_id=file_node.id,
                    target_id=entity_node.id
                ))

    def _create_entity_node(
        self,
        repo_info: RepositoryInfo,
        file_path: str,
        entity: CodeEntity
    ) -> Optional[GraphNode]:
        """
        Create a graph node from a code entity.
        """
        try:
            entity_id = create_node_id(repo_info.name, file_path, entity.full_name)

            # Map parser EntityType to graph NodeType
            if entity.type == ParserEntityType.FUNCTION:
                return FunctionNode(
                    id=entity_id,
                    name=entity.name,
                    signature=entity.signature or f"def {entity.name}(...)",
                    docstring=entity.docstring,
                    start_line=entity.start_line,
                    end_line=entity.end_line,
                    is_async=entity.metadata.get('is_async', False),
                    is_method=False,
                    decorators=entity.decorators,
                    parameters=entity.parameters,
                    return_type=entity.return_type
                )

            elif entity.type == ParserEntityType.METHOD:
                return FunctionNode(
                    id=entity_id,
                    name=entity.name,
                    signature=entity.signature or f"def {entity.name}(...)",
                    docstring=entity.docstring,
                    start_line=entity.start_line,
                    end_line=entity.end_line,
                    is_async=entity.metadata.get('is_async', False),
                    is_method=True,
                    decorators=entity.decorators,
                    parameters=entity.parameters,
                    return_type=entity.return_type
                )

            elif entity.type == ParserEntityType.CLASS:
                return ClassNode(
                    id=entity_id,
                    name=entity.name,
                    docstring=entity.docstring,
                    base_classes=entity.metadata.get('base_classes', []),
                    start_line=entity.start_line,
                    end_line=entity.end_line,
                    decorators=entity.decorators
                )

            elif entity.type == ParserEntityType.COMPONENT:
                hooks = entity.metadata.get('hooks_used', [])
                hook_names = [h['name'] if isinstance(h, dict) else h for h in hooks]

                return ComponentNode(
                    id=entity_id,
                    name=entity.name,
                    props_type=entity.metadata.get('props_type'),
                    hooks_used=hook_names,
                    start_line=entity.start_line,
                    end_line=entity.end_line,
                    is_exported=entity.metadata.get('is_exported', False),
                    is_default_export=entity.metadata.get('is_default_export', False)
                )

            elif entity.type == ParserEntityType.ENDPOINT:
                return EndpointNode(
                    id=entity_id,
                    name=entity.name,
                    http_method=entity.metadata.get('http_method', 'GET'),
                    path=entity.metadata.get('path', '/'),
                    request_model=entity.metadata.get('request_model'),
                    response_model=entity.metadata.get('response_model'),
                    requires_auth=False  # TODO: Detect from decorators
                )

            elif entity.type == ParserEntityType.MODEL:
                model_type = 'pydantic' if entity.metadata.get('is_pydantic') else 'django'

                return ModelNode(
                    id=entity_id,
                    name=entity.name,
                    model_type=model_type,
                    fields=entity.metadata.get('fields', [])
                )

            elif entity.type == ParserEntityType.ROUTE:
                return RouteNode(
                    id=entity_id,
                    name=entity.metadata.get('path', entity.name),
                    path=entity.metadata.get('path', entity.name)
                )

            else:
                logger.warning(f"Unknown entity type: {entity.type}")
                return None

        except Exception as e:
            logger.error(f"Failed to create entity node for {entity.name}: {e}")
            return None

    def _create_relationships(
        self,
        repo_info: RepositoryInfo,
        parse_results: List[Tuple[Path, ParseResult]]
    ):
        """
        Create relationships between nodes (second pass).
        """
        for file_path, parse_result in parse_results:
            rel_path = str(file_path.relative_to(repo_info.path))

            for entity in parse_result.entities:
                entity_id = create_node_id(repo_info.name, rel_path, entity.full_name)

                # Class-Method containment
                if entity.type == ParserEntityType.METHOD and entity.parent:
                    parent_id = create_node_id(repo_info.name, rel_path, entity.parent)

                    if parent_id in self.nodes:
                        self.relationships.append(GraphRelationship(
                            type=RelationshipType.CONTAINS,
                            source_id=parent_id,
                            target_id=entity_id
                        ))

                # Class inheritance
                if entity.type == ParserEntityType.CLASS:
                    base_classes = entity.metadata.get('base_classes', [])

                    for base_class in base_classes:
                        # Try to find base class node
                        base_id = self._find_class_node(repo_info.name, base_class)

                        if base_id:
                            self.relationships.append(GraphRelationship(
                                type=RelationshipType.INHERITS,
                                source_id=entity_id,
                                target_id=base_id
                            ))

                # Django model relationships
                if entity.type == ParserEntityType.MODEL:
                    relationships = entity.metadata.get('relationships', [])

                    for rel in relationships:
                        rel_type_str = rel.get('type', '')
                        related_model = rel.get('related_model')

                        if not related_model:
                            continue

                        # Find related model node
                        related_id = self._find_model_node(repo_info.name, related_model)

                        if related_id:
                            # Map Django relationship types to graph relationships
                            if 'ForeignKey' in rel_type_str:
                                rel_type = RelationshipType.FOREIGN_KEY
                            elif 'ManyToMany' in rel_type_str:
                                rel_type = RelationshipType.MANY_TO_MANY
                            elif 'OneToOne' in rel_type_str:
                                rel_type = RelationshipType.ONE_TO_ONE
                            else:
                                continue

                            self.relationships.append(GraphRelationship(
                                type=rel_type,
                                source_id=entity_id,
                                target_id=related_id,
                                properties={'field_name': rel.get('field_name')}
                            ))

                # Endpoint → Function (HANDLES_REQUEST)
                if entity.type == ParserEntityType.ENDPOINT:
                    # The endpoint IS the function in FastAPI
                    # For Django, we'd need to extract the view function name
                    pass

                # Component → Route (RENDERS_AT)
                # Would need to parse router configuration
                # TODO: Implement in dedicated router parser

                # Function/Method CALLS other functions
                if entity.type in (ParserEntityType.FUNCTION, ParserEntityType.METHOD):
                    calls = entity.calls or []

                    for called_func_name in calls:
                        # Try to find the called function in the graph
                        # First try exact name match in same file
                        target_id = self._find_function_in_file(
                            repo_info.name,
                            rel_path,
                            called_func_name
                        )

                        # If not found in same file, try across all files
                        if not target_id:
                            target_id = self._find_function_by_name(
                                repo_info.name,
                                called_func_name
                            )

                        if target_id:
                            self.relationships.append(GraphRelationship(
                                type=RelationshipType.CALLS,
                                source_id=entity_id,
                                target_id=target_id
                            ))

        # Create IMPORTS relationships (file-level)
        for file_path, parse_result in parse_results:
            rel_path = str(file_path.relative_to(repo_info.path))
            file_node_id = create_node_id(repo_info.name, rel_path)

            for import_name in parse_result.imports:
                # Try to find the imported file/module
                target_file_id = self._find_file_by_import(repo_info.name, import_name, rel_path)

                if target_file_id:
                    self.relationships.append(GraphRelationship(
                        type=RelationshipType.IMPORTS,
                        source_id=file_node_id,
                        target_id=target_file_id
                    ))

        logger.info(f"Created {len(self.relationships)} relationships")

    def _find_class_node(self, repository: str, class_name: str) -> Optional[str]:
        """Find class node by name (search across all files)."""
        for node_id, node in self.nodes.items():
            if isinstance(node, ClassNode) and node.name == class_name:
                return node_id
        return None

    def _find_model_node(self, repository: str, model_name: str) -> Optional[str]:
        """Find model node by name."""
        for node_id, node in self.nodes.items():
            if isinstance(node, ModelNode) and node.name == model_name:
                return node_id
        return None

    def _find_function_in_file(
        self,
        repository: str,
        file_path: str,
        func_name: str
    ) -> Optional[str]:
        """
        Find function node by name within a specific file.

        Handles:
        - Simple names: "foo"
        - Method names: "ClassName.method"
        - Full paths: "module.Class.method"
        """
        # Get all nodes in this file
        file_nodes = self.nodes_by_file.get(file_path, [])

        for node_id in file_nodes:
            node = self.nodes.get(node_id)
            if not node:
                continue

            # Check if this is a function/method node
            if isinstance(node, FunctionNode):
                # Try exact match
                if node.name == func_name:
                    return node_id

                # Try matching last part of qualified name (e.g., "method" matches "Class.method")
                if '.' in func_name:
                    parts = func_name.split('.')
                    if node.name == parts[-1]:
                        return node_id

        return None

    def _find_function_by_name(self, repository: str, func_name: str) -> Optional[str]:
        """
        Find function node by name across all files in repository.

        Returns first match. Prioritizes exact matches over partial matches.
        """
        # Extract simple name if it's a qualified name
        simple_name = func_name.split('.')[-1] if '.' in func_name else func_name

        # First pass: exact name match
        for node_id, node in self.nodes.items():
            if isinstance(node, FunctionNode):
                if node.name == simple_name or node.name == func_name:
                    return node_id

        # Second pass: try matching against full_name in node ID
        # (for methods like "ClassName.method_name")
        for node_id, node in self.nodes.items():
            if isinstance(node, FunctionNode):
                if simple_name in node_id or func_name in node_id:
                    return node_id

        return None

    def _find_file_by_import(
        self,
        repository: str,
        import_name: str,
        current_file: str
    ) -> Optional[str]:
        """
        Find file node by import statement.

        Handles:
        - Relative imports: "from .module import foo" → look in same directory
        - Absolute imports: "from app.models import User" → look for app/models.py
        - Module imports: "import json" → ignore (standard library)
        """
        # Ignore standard library imports
        stdlib_modules = {
            'os', 'sys', 'json', 'time', 'datetime', 'pathlib', 'typing',
            'collections', 'itertools', 'functools', 'logging', 'asyncio',
            'dataclasses', 'enum', 'abc', 'copy', 're', 'math', 'random'
        }

        # Extract module name from import
        # "from app.models import User" → "app.models"
        # "import app.models" → "app.models"
        module_name = import_name.split(' import ')[0].replace('from ', '').replace('import ', '').strip()

        # Check if it's a standard library module
        if module_name.split('.')[0] in stdlib_modules:
            return None

        # Convert module path to file path
        # "app.models" → "app/models.py"
        potential_paths = [
            module_name.replace('.', '/') + '.py',
            module_name.replace('.', '/') + '/__init__.py',
        ]

        # Try to find matching file node
        for node_id, node in self.nodes.items():
            if isinstance(node, FileNode):
                for potential_path in potential_paths:
                    if node.file_path.endswith(potential_path):
                        return node_id

        return None

    def _save_to_neo4j(self) -> Dict[str, int]:
        """Save all nodes and relationships to Neo4j."""
        logger.info("Saving graph to Neo4j...")

        # Create indexes first
        self.client.create_indexes()

        # Save nodes in batch
        nodes_list = list(self.nodes.values())
        nodes_created = self.client.create_nodes_batch(nodes_list)

        # Save relationships in batch
        rels_created = self.client.create_relationships_batch(self.relationships)

        return {
            'nodes_created': nodes_created,
            'relationships_created': rels_created,
            'total_nodes': len(self.nodes),
            'total_relationships': len(self.relationships)
        }
