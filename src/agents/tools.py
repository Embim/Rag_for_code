"""
Tools for Code Explorer Agent.

Each tool is a capability the agent can use to gather information about the codebase.
Tools are designed to be composable and can be used in different combinations
by the agent depending on the question.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from pathlib import Path
import json

from ..code_rag.retrieval import CodeRetriever, SearchStrategy
from ..code_rag.retrieval.scope_detector import ScopeDetector, QueryScope
from ..logger import get_logger


logger = get_logger(__name__)


class Tool(ABC):
    """
    Base class for agent tools.

    Each tool has:
    - name: Identifier for the tool
    - description: What the tool does (for LLM to understand when to use it)
    - execute: The actual logic
    """

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    @abstractmethod
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """
        Execute the tool with given parameters.

        Returns:
            Dict with:
            - success: bool
            - result: tool output
            - error: error message (if success=False)
        """
        pass

    def to_dict(self) -> Dict[str, Any]:
        """Convert tool to dict for LLM function calling."""
        return {
            'name': self.name,
            'description': self.description,
        }


class SemanticSearchTool(Tool):
    """
    Semantic search across the knowledge graph.

    Uses vector similarity to find code entities relevant to the query.
    Good for: "find functions related to user authentication"
    """

    def __init__(self, retriever: CodeRetriever):
        super().__init__(
            name="semantic_search",
            description="Search for code entities semantically related to a query. "
                       "Returns relevant functions, classes, components with their code. "
                       "Use when you need to find code by concept or behavior, not exact name."
        )
        self.retriever = retriever
        self.scope_detector = ScopeDetector()

    async def execute(self, query: str, scope: Optional[str] = None, top_k: int = 10) -> Dict[str, Any]:
        """
        Args:
            query: What to search for
            scope: Optional scope hint (frontend/backend/hybrid)
            top_k: Number of results to return
        """
        try:
            # Detect scope if not provided
            if not scope:
                scope_hint = self.scope_detector.detect_scope(query)
                scope = scope_hint.scope.value

            # Determine strategy
            query_lower = query.lower()
            if 'как работает' in query_lower or 'ui to database' in query_lower:
                strategy = SearchStrategy.UI_TO_DATABASE
            elif 'где отображается' in query_lower or 'database to ui' in query_lower:
                strategy = SearchStrategy.DATABASE_TO_UI
            else:
                strategy = SearchStrategy.SEMANTIC_ONLY

            # Perform search with correct config_override format (flat dict)
            config_override = {
                'top_k_vector': top_k,
                'top_k_final': top_k,
                'expand_results': False,  # Just semantic search
            }

            search_result = self.retriever.search(
                query=query,
                strategy=strategy,
                config_override=config_override
            )

            # Format results - SearchResult has primary_nodes and all_nodes()
            formatted_results = []
            for node in search_result.primary_nodes[:top_k]:
                formatted_results.append({
                    'id': node.get('node_id') or node.get('id', ''),
                    'name': node.get('name', 'Unknown'),
                    'type': node.get('node_type', node.get('type', 'Unknown')),
                    'file': node.get('file_path', node.get('file', 'Unknown')),
                    'code_snippet': node.get('content', node.get('code', ''))[:500],
                    'score': node.get('score', 0.0),
                })

            return {
                'success': True,
                'result': {
                    'query': query,
                    'scope': scope,
                    'entities_found': len(formatted_results),
                    'entities': formatted_results,
                },
                'error': None,
            }

        except Exception as e:
            logger.error(f"SemanticSearchTool error: {e}")
            return {
                'success': False,
                'result': None,
                'error': str(e),
            }


class ExactSearchTool(Tool):
    """
    Search for entity by exact name.

    Good for: "find the function called calculate_total"
    """

    def __init__(self, neo4j_client):
        super().__init__(
            name="exact_search",
            description="Find code entity by exact name. Use when you know the exact name "
                       "of a function, class, or component you're looking for."
        )
        self.neo4j = neo4j_client

    async def execute(self, name: str, entity_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Args:
            name: Exact name of entity
            entity_type: Optional type filter (Function, Class, Component, etc.)
        """
        try:
            # Build Cypher query
            if entity_type:
                cypher = """
                MATCH (e {name: $name})
                WHERE $entity_type IN labels(e)
                RETURN e, labels(e) as types
                LIMIT 10
                """
                params = {'name': name, 'entity_type': entity_type}
            else:
                cypher = """
                MATCH (e {name: $name})
                RETURN e, labels(e) as types
                LIMIT 10
                """
                params = {'name': name}

            results = self.neo4j.execute_cypher(cypher, parameters=params)

            entities = []
            for record in results:
                node = record['e']
                entities.append({
                    'id': node.element_id,
                    'name': node.get('name', 'Unknown'),
                    'type': record['types'][0] if record['types'] else 'Unknown',
                    'file': node.get('file', 'Unknown'),
                    'line': node.get('line', None),
                    'code': node.get('code', ''),
                })

            return {
                'success': True,
                'result': {
                    'name_searched': name,
                    'entity_type': entity_type,
                    'entities_found': len(entities),
                    'entities': entities,
                },
                'error': None,
            }

        except Exception as e:
            logger.error(f"ExactSearchTool error: {e}")
            return {
                'success': False,
                'result': None,
                'error': str(e),
            }


class GetEntityDetailsTool(Tool):
    """
    Get full details about a specific entity.

    Returns: complete code, docstring, relationships, file location.
    """

    def __init__(self, neo4j_client):
        super().__init__(
            name="get_entity_details",
            description="Get full details about a specific code entity by its ID. "
                       "Returns complete code, documentation, relationships, and file location."
        )
        self.neo4j = neo4j_client

    async def execute(self, entity_id: str) -> Dict[str, Any]:
        """
        Args:
            entity_id: Neo4j element ID
        """
        try:
            cypher = """
            MATCH (e)
            WHERE elementId(e) = $entity_id
            OPTIONAL MATCH (e)-[r]->(related)
            RETURN e, labels(e) as types,
                   collect({type: type(r), target: related.name}) as relationships
            """

            results = list(self.neo4j.execute_cypher(cypher, parameters={'entity_id': entity_id}))

            if not results:
                return {
                    'success': False,
                    'result': None,
                    'error': f'Entity not found: {entity_id}',
                }

            record = results[0]
            node = record['e']

            return {
                'success': True,
                'result': {
                    'id': entity_id,
                    'name': node.get('name', 'Unknown'),
                    'type': record['types'][0] if record['types'] else 'Unknown',
                    'file': node.get('file', 'Unknown'),
                    'line': node.get('line', None),
                    'code': node.get('code', ''),
                    'docstring': node.get('docstring', ''),
                    'relationships': record['relationships'],
                },
                'error': None,
            }

        except Exception as e:
            logger.error(f"GetEntityDetailsTool error: {e}")
            return {
                'success': False,
                'result': None,
                'error': str(e),
            }


class GetRelatedEntitiesTool(Tool):
    """
    Get entities related to a given entity.

    Example: "get all functions called by this function"
    """

    def __init__(self, neo4j_client):
        super().__init__(
            name="get_related_entities",
            description="Get entities related to a given entity via specific relationship type. "
                       "Useful for understanding dependencies and connections."
        )
        self.neo4j = neo4j_client

    async def execute(
        self,
        entity_id: str,
        relation_type: Optional[str] = None,
        direction: str = 'outgoing'
    ) -> Dict[str, Any]:
        """
        Args:
            entity_id: Source entity ID
            relation_type: Optional filter (CALLS, IMPORTS, USES_MODEL, etc.)
            direction: 'outgoing' or 'incoming'
        """
        try:
            # Build Cypher based on direction
            if direction == 'outgoing':
                if relation_type:
                    cypher = """
                    MATCH (e)-[r:%s]->(related)
                    WHERE elementId(e) = $entity_id
                    RETURN related, type(r) as rel_type, labels(related) as types
                    LIMIT 50
                    """ % relation_type
                else:
                    cypher = """
                    MATCH (e)-[r]->(related)
                    WHERE elementId(e) = $entity_id
                    RETURN related, type(r) as rel_type, labels(related) as types
                    LIMIT 50
                    """
            else:  # incoming
                if relation_type:
                    cypher = """
                    MATCH (related)-[r:%s]->(e)
                    WHERE elementId(e) = $entity_id
                    RETURN related, type(r) as rel_type, labels(related) as types
                    LIMIT 50
                    """ % relation_type
                else:
                    cypher = """
                    MATCH (related)-[r]->(e)
                    WHERE elementId(e) = $entity_id
                    RETURN related, type(r) as rel_type, labels(related) as types
                    LIMIT 50
                    """

            results = self.neo4j.execute_cypher(cypher, parameters={'entity_id': entity_id})

            entities = []
            for record in results:
                node = record['related']
                entities.append({
                    'id': node.element_id,
                    'name': node.get('name', 'Unknown'),
                    'type': record['types'][0] if record['types'] else 'Unknown',
                    'relationship': record['rel_type'],
                    'file': node.get('file', 'Unknown'),
                })

            return {
                'success': True,
                'result': {
                    'entity_id': entity_id,
                    'relation_type': relation_type,
                    'direction': direction,
                    'related_count': len(entities),
                    'related_entities': entities,
                },
                'error': None,
            }

        except Exception as e:
            logger.error(f"GetRelatedEntitiesTool error: {e}")
            return {
                'success': False,
                'result': None,
                'error': str(e),
            }


class ListFilesTool(Tool):
    """
    List files in a directory.

    Useful for understanding project structure.
    """

    def __init__(self, repos_dir: Path):
        super().__init__(
            name="list_files",
            description="List files in a directory to understand project structure."
        )
        self.repos_dir = repos_dir

    async def execute(self, directory: str = '', pattern: str = '*') -> Dict[str, Any]:
        """
        Args:
            directory: Relative path from repos directory
            pattern: Glob pattern (e.g., '*.py', '**/*.tsx')
        """
        try:
            target_dir = self.repos_dir / directory

            if not target_dir.exists():
                return {
                    'success': False,
                    'result': None,
                    'error': f'Directory not found: {directory}',
                }

            # List files matching pattern
            if '**' in pattern:
                files = list(target_dir.glob(pattern))
            else:
                files = list(target_dir.glob(pattern))

            # Format results
            file_list = []
            for file in sorted(files)[:100]:  # Limit to 100 files
                relative = file.relative_to(self.repos_dir)
                file_list.append({
                    'path': str(relative),
                    'name': file.name,
                    'is_file': file.is_file(),
                    'size': file.stat().st_size if file.is_file() else None,
                })

            return {
                'success': True,
                'result': {
                    'directory': directory,
                    'pattern': pattern,
                    'files_found': len(file_list),
                    'files': file_list,
                },
                'error': None,
            }

        except Exception as e:
            logger.error(f"ListFilesTool error: {e}")
            return {
                'success': False,
                'result': None,
                'error': str(e),
            }


class ReadFileTool(Tool):
    """
    Read a file or fragment of a file.

    Useful for getting context around an entity.
    """

    def __init__(self, repos_dir: Path):
        super().__init__(
            name="read_file",
            description="Read a file or a specific range of lines from a file. "
                       "Useful for understanding context around code entities."
        )
        self.repos_dir = repos_dir

    async def execute(
        self,
        path: str,
        start_line: Optional[int] = None,
        end_line: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Args:
            path: Relative path from repos directory
            start_line: Optional start line (1-indexed)
            end_line: Optional end line (inclusive)
        """
        try:
            file_path = self.repos_dir / path

            if not file_path.exists():
                return {
                    'success': False,
                    'result': None,
                    'error': f'File not found: {path}',
                }

            # Read file
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()

            # Extract range if specified
            if start_line is not None:
                start_idx = max(0, start_line - 1)  # Convert to 0-indexed
                end_idx = end_line if end_line else len(lines)
                lines = lines[start_idx:end_idx]

            content = ''.join(lines)

            return {
                'success': True,
                'result': {
                    'path': path,
                    'start_line': start_line or 1,
                    'end_line': end_line or len(lines),
                    'content': content,
                    'total_lines': len(lines),
                },
                'error': None,
            }

        except Exception as e:
            logger.error(f"ReadFileTool error: {e}")
            return {
                'success': False,
                'result': None,
                'error': str(e),
            }


class GrepTool(Tool):
    """
    Search for pattern in code using regex.

    Useful for finding usages of variables, patterns.
    """

    def __init__(self, neo4j_client):
        super().__init__(
            name="grep",
            description="Search for a regex pattern in code. Useful for finding usages "
                       "of specific variables, function calls, or code patterns."
        )
        self.neo4j = neo4j_client

    async def execute(self, pattern: str, scope: Optional[str] = None) -> Dict[str, Any]:
        """
        Args:
            pattern: Regex pattern to search
            scope: Optional scope filter (frontend/backend)
        """
        try:
            # Search in code field of all entities
            cypher = """
            MATCH (e)
            WHERE e.code =~ $pattern
            RETURN e, labels(e) as types
            LIMIT 50
            """

            results = self.neo4j.execute_cypher(cypher, parameters={'pattern': f'.*{pattern}.*'})

            matches = []
            for record in results:
                node = record['e']
                matches.append({
                    'id': node.element_id,
                    'name': node.get('name', 'Unknown'),
                    'type': record['types'][0] if record['types'] else 'Unknown',
                    'file': node.get('file', 'Unknown'),
                    'line': node.get('line', None),
                    'code_snippet': node.get('code', '')[:300],  # Show snippet
                })

            return {
                'success': True,
                'result': {
                    'pattern': pattern,
                    'matches_found': len(matches),
                    'matches': matches,
                },
                'error': None,
            }

        except Exception as e:
            logger.error(f"GrepTool error: {e}")
            return {
                'success': False,
                'result': None,
                'error': str(e),
            }


class GetGraphPathTool(Tool):
    """
    Find path between two entities in the knowledge graph.

    Useful for understanding how two pieces of code are connected.
    """

    def __init__(self, neo4j_client):
        super().__init__(
            name="get_graph_path",
            description="Find shortest path between two entities in the code graph. "
                       "Shows how two pieces of code are connected."
        )
        self.neo4j = neo4j_client

    async def execute(self, from_entity: str, to_entity: str, max_depth: int = 5) -> Dict[str, Any]:
        """
        Args:
            from_entity: Source entity ID or name
            to_entity: Target entity ID or name
            max_depth: Maximum path length
        """
        try:
            # Find path
            cypher = """
            MATCH path = shortestPath(
                (start)-[*1..%d]-(end)
            )
            WHERE (elementId(start) = $from_entity OR start.name = $from_entity)
              AND (elementId(end) = $to_entity OR end.name = $to_entity)
            RETURN path
            LIMIT 1
            """ % max_depth

            results = list(self.neo4j.execute_cypher(
                cypher,
                parameters={'from_entity': from_entity, 'to_entity': to_entity}
            ))

            if not results:
                return {
                    'success': False,
                    'result': None,
                    'error': f'No path found between {from_entity} and {to_entity}',
                }

            path = results[0]['path']

            # Format path
            nodes = []
            for node in path.nodes:
                nodes.append({
                    'id': node.element_id,
                    'name': node.get('name', 'Unknown'),
                    'type': list(node.labels)[0] if node.labels else 'Unknown',
                })

            relationships = []
            for rel in path.relationships:
                relationships.append({
                    'type': type(rel).__name__,
                })

            return {
                'success': True,
                'result': {
                    'from_entity': from_entity,
                    'to_entity': to_entity,
                    'path_length': len(relationships),
                    'path_nodes': nodes,
                    'path_relationships': relationships,
                },
                'error': None,
            }

        except Exception as e:
            logger.error(f"GetGraphPathTool error: {e}")
            return {
                'success': False,
                'result': None,
                'error': str(e),
            }
