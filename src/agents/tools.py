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

    async def execute(
        self,
        query: str,
        scope: Optional[str] = None,
        top_k: int = 50,  # Increased from 10 to 50 for more comprehensive results
        repositories: Optional[List[str]] = None,
        repo: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Args:
            query: What to search for
            scope: Optional scope hint (frontend/backend/hybrid)
            top_k: Number of results to return (default: 50 for agent exploration)
            repositories: Optional list of repository names to filter by
            repo: Alias for a single repository (converted to list)
        """
        try:
            # Detect scope if not provided
            if not scope:
                scope_hint = self.scope_detector.detect_scope(query)
                scope = scope_hint.scope.value

            # Handle repo parameter (single repo as alias for repositories)
            if repo and not repositories:
                repositories = [repo]

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

            # Add repository filter if provided
            if repositories:
                config_override['repositories'] = repositories

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

    async def execute(
        self,
        name: Optional[str] = None,
        entity_name: Optional[str] = None,
        query: Optional[str] = None,
        entity_type: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Args:
            name: Exact name of entity (preferred)
            entity_name: Alias for name
            query: Alias for name
            entity_type: Optional type filter (Function, Class, Component, etc.)
        """
        try:
            # Try to extract entity name from various parameter names
            entity_search_name = name or entity_name or query

            if not entity_search_name:
                return {
                    'success': False,
                    'result': None,
                    'error': 'Missing required parameter: name, entity_name, or query must be provided',
                }


            # Build Cypher query
            if entity_type:
                cypher = """
                MATCH (e {name: $name})
                WHERE $entity_type IN labels(e)
                RETURN e, labels(e) as types
                LIMIT 30
                """
                params = {'name': entity_search_name, 'entity_type': entity_type}
            else:
                cypher = """
                MATCH (e {name: $name})
                RETURN e, labels(e) as types
                LIMIT 30
                """
                params = {'name': entity_search_name}

            results = self.neo4j.execute_cypher(cypher, parameters=params)

            entities = []
            for record in results:
                node = record['e']
                entities.append({
                    'id': node.get('id', node.element_id),
                    'name': node.get('name', 'Unknown'),
                    'type': record['types'][0] if record['types'] else 'Unknown',
                    'file': node.get('file_path', node.get('file', 'Unknown')),
                    'line': node.get('start_line', node.get('line', None)),
                    'code': node.get('code', ''),
                })

            return {
                'success': True,
                'result': {
                    'name_searched': entity_search_name,
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

    async def execute(
        self,
        entity_id: Optional[str] = None,
        id: Optional[str] = None,
        name: Optional[str] = None,
        entity_name: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Args:
            entity_id: Entity ID in format "repo:name:path:entity" (from semantic_search)
            id: Alias for entity_id
            name: Entity name (fallback if ID not provided)
            entity_name: Alias for name
        """
        try:
            # Try to extract entity identifier from various parameter names
            identifier = entity_id or id or name or entity_name

            if not identifier:
                return {
                    'success': False,
                    'result': None,
                    'error': 'Missing required parameter: entity_id, id, or name must be provided',
                }

            # Try to find by custom ID field (e.id) - used by semantic_search
            cypher = """
            MATCH (e)
            WHERE e.id = $entity_id
            OPTIONAL MATCH (e)-[r]->(related)
            RETURN e, labels(e) as types,
                   collect({type: type(r), target: related.name}) as relationships
            LIMIT 1
            """
            results = list(self.neo4j.execute_cypher(cypher, parameters={'entity_id': identifier}))

            # Fallback: try searching by name if not found by ID
            if not results:
                cypher = """
                MATCH (e {name: $name})
                OPTIONAL MATCH (e)-[r]->(related)
                RETURN e, labels(e) as types,
                       collect({type: type(r), target: related.name}) as relationships
                LIMIT 1
                """
                results = list(self.neo4j.execute_cypher(cypher, parameters={'name': identifier}))

            if not results:
                return {
                    'success': False,
                    'result': None,
                    'error': f'Entity not found: {identifier}',
                }

            record = results[0]
            node = record['e']

            return {
                'success': True,
                'result': {
                    'entities_found': 1,
                    'entities': [{
                        'id': node.get('id', identifier),
                        'name': node.get('name', 'Unknown'),
                        'type': record['types'][0] if record['types'] else 'Unknown',
                        'file': node.get('file_path', node.get('file', 'Unknown')),
                        'line': node.get('start_line', node.get('line', None)),
                        'code': node.get('code', ''),
                        'docstring': node.get('docstring', ''),
                        'relationships': record['relationships'],
                    }],
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
        entity_id: Optional[str] = None,
        id: Optional[str] = None,
        name: Optional[str] = None,
        entity_name: Optional[str] = None,
        relation_type: Optional[str] = None,
        direction: str = 'outgoing',
        **kwargs
    ) -> Dict[str, Any]:
        """
        Args:
            entity_id: Source entity ID in format "repo:name:path:entity" (from semantic_search)
            id: Alias for entity_id
            name: Entity name (fallback if ID not provided)
            entity_name: Alias for name
            relation_type: Optional filter (CALLS, IMPORTS, USES_MODEL, etc.)
            direction: 'outgoing' or 'incoming'
        """
        try:
            # Try to extract entity identifier from various parameter names
            identifier = entity_id or id or name or entity_name

            if not identifier:
                return {
                    'success': False,
                    'result': None,
                    'error': 'Missing required parameter: entity_id, id, or name must be provided',
                }

            # Build Cypher based on direction - use e.id instead of elementId(e)
            if direction == 'outgoing':
                if relation_type:
                    cypher = """
                    MATCH (e)-[r:%s]->(related)
                    WHERE e.id = $entity_id
                    RETURN related, type(r) as rel_type, labels(related) as types
                    LIMIT 50
                    """ % relation_type
                else:
                    cypher = """
                    MATCH (e)-[r]->(related)
                    WHERE e.id = $entity_id
                    RETURN related, type(r) as rel_type, labels(related) as types
                    LIMIT 50
                    """
            else:  # incoming
                if relation_type:
                    cypher = """
                    MATCH (related)-[r:%s]->(e)
                    WHERE e.id = $entity_id
                    RETURN related, type(r) as rel_type, labels(related) as types
                    LIMIT 50
                    """ % relation_type
                else:
                    cypher = """
                    MATCH (related)-[r]->(e)
                    WHERE e.id = $entity_id
                    RETURN related, type(r) as rel_type, labels(related) as types
                    LIMIT 50
                    """

            results = self.neo4j.execute_cypher(cypher, parameters={'entity_id': identifier})

            entities = []
            for record in results:
                node = record['related']
                entities.append({
                    'id': node.get('id', node.element_id),
                    'name': node.get('name', 'Unknown'),
                    'type': record['types'][0] if record['types'] else 'Unknown',
                    'relationship': record['rel_type'],
                    'file': node.get('file_path', node.get('file', 'Unknown')),
                })

            return {
                'success': True,
                'result': {
                    'entity_id': identifier,
                    'relation_type': relation_type,
                    'direction': direction,
                    'entities_found': len(entities),
                    'entities': entities,
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

    async def execute(self, directory: str = '', pattern: str = '*', path: str = None, **kwargs) -> Dict[str, Any]:
        """
        Args:
            directory: Relative path from repos directory
            pattern: Glob pattern (e.g., '*.py', '**/*.tsx')
            path: Alias for directory (for backward compatibility)
        """
        try:
            # Support both 'path' and 'directory' parameters
            dir_to_use = path if path is not None else directory

            # Sanitize path: remove leading slashes, backslashes, and dangerous patterns
            dir_to_use = str(dir_to_use).lstrip('/\\').replace('..', '')

            # Reject system directories
            dangerous_patterns = ['$Recycle.Bin', 'System Volume Information', 'Windows', 'Program Files']
            if any(pattern in dir_to_use for pattern in dangerous_patterns):
                return {
                    'success': False,
                    'result': None,
                    'error': f'Access denied: cannot list system directories',
                }

            target_dir = self.repos_dir / dir_to_use

            # Ensure the target directory is within repos_dir
            try:
                target_dir = target_dir.resolve()
                target_dir.relative_to(self.repos_dir.resolve())
            except ValueError:
                return {
                    'success': False,
                    'result': None,
                    'error': f'Access denied: path must be within repository directory',
                }

            if not target_dir.exists():
                return {
                    'success': False,
                    'result': None,
                    'error': f'Directory not found: {dir_to_use}',
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
                    'directory': dir_to_use,
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
        end_line: Optional[int] = None,
        **kwargs
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
                    'error': f'File not found: {path}. Note: Physical files may not be available - use semantic_search or get_entity_details to get code instead.',
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

    async def execute(self, pattern: str, scope: Optional[str] = None, **kwargs) -> Dict[str, Any]:
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

    async def execute(self, from_entity: str, to_entity: str, max_depth: int = 5, **kwargs) -> Dict[str, Any]:
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
