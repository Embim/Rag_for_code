"""
React/TypeScript parser.

Extracts React-specific entities:
- React components (function and class)
- Hooks usage (useState, useEffect, custom hooks)
- Props and their types
- API calls (fetch, axios, React Query)
- React Router routes

Uses @babel/parser (via Node.js subprocess) for accurate AST parsing.
Falls back to regex-based parsing if Node.js is unavailable.
"""

import re
import json
import subprocess
from pathlib import Path
from typing import List, Optional, Dict, Any, Set

from .base import (
    BaseParser,
    CodeEntity,
    EntityType,
    ParseResult,
    ParsingError,
    register_parser
)
from src.logger import get_logger


logger = get_logger(__name__)


class ReactParser(BaseParser):
    """
    Parser for React/TypeScript code.

    Detects and extracts:
    - React components (FC and class components)
    - Component props with TypeScript types
    - Hooks usage
    - API calls
    - Routes
    - Forms

    Uses Babel for accurate parsing when available, regex as fallback.
    """

    # React hook patterns
    REACT_HOOKS = [
        'useState', 'useEffect', 'useContext', 'useReducer',
        'useCallback', 'useMemo', 'useRef', 'useImperativeHandle',
        'useLayoutEffect', 'useDebugValue',
        # React Router
        'useNavigate', 'useParams', 'useLocation', 'useSearchParams',
        # React Query
        'useQuery', 'useMutation', 'useQueryClient',
        # Form libraries
        'useForm', 'useFormContext', 'useController'
    ]

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.parser_mode = 'babel'  # 'babel' or 'regex'
        self.babel_parser_path = None

        # Initialize babel parser
        try:
            self._init_babel_parser()
            logger.info("✓ Using Babel parser for React/TypeScript (accurate AST parsing)")
        except Exception as e:
            logger.warning(f"Babel parser unavailable: {e}")
            logger.warning("⚠ Falling back to regex-based parsing (less accurate)")
            self.parser_mode = 'regex'

    def _init_babel_parser(self):
        """Initialize Babel parser (check if available)."""
        # Find js_parser.js
        parser_dir = Path(__file__).parent
        js_parser_path = parser_dir / "js_parser.js"

        if not js_parser_path.exists():
            raise FileNotFoundError(f"Babel parser script not found: {js_parser_path}")

        # Check if node_modules exists
        node_modules = parser_dir / "node_modules"
        if not node_modules.exists():
            raise FileNotFoundError(
                f"Babel dependencies not installed. Run: cd {parser_dir} && npm install"
            )

        # Test the parser
        try:
            result = subprocess.run(
                ['node', str(js_parser_path), '--version'],
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
                timeout=5,
                cwd=str(parser_dir)  # Run from parser directory to find node_modules
            )
            # If it doesn't error, we're good
            self.babel_parser_path = js_parser_path
        except subprocess.TimeoutExpired:
            raise RuntimeError("Babel parser test timed out")
        except FileNotFoundError:
            raise RuntimeError("Node.js not found in PATH")
        except Exception as e:
            raise RuntimeError(f"Babel parser test failed: {e}")

    def get_supported_extensions(self) -> List[str]:
        """React/TypeScript file extensions."""
        return ['.jsx', '.tsx', '.ts', '.js']

    def parse_file(self, file_path: Path) -> ParseResult:
        """
        Parse React/TypeScript file.

        Args:
            file_path: Path to React file

        Returns:
            ParseResult with React-specific entities
        """
        # Determine language early so it's available in error handler
        ext = file_path.suffix.lower()
        language = 'typescript' if ext in ['.ts', '.tsx'] else 'javascript'

        try:
            # Read file
            with open(file_path, 'r', encoding='utf-8') as f:
                source = f.read()

            # Parse based on mode
            if self.parser_mode == 'babel' and self.babel_parser_path:
                return self._parse_with_babel(file_path, source, language)
            else:
                return self._parse_with_regex(file_path, source, language)

        except Exception as e:
            logger.error(f"Failed to parse {file_path}: {e}")
            return ParseResult(
                file_path=file_path,
                language=language,
                entities=[],
                errors=[str(e)]
            )

    def _parse_with_babel(
        self,
        file_path: Path,
        source: str,
        language: str
    ) -> ParseResult:
        """Parse using Babel (accurate AST parsing)."""
        try:
            # Call Node.js parser
            # IMPORTANT: Set cwd to parser directory so Node.js can find node_modules
            parser_dir = self.babel_parser_path.parent
            result = subprocess.run(
                ['node', str(self.babel_parser_path), str(file_path)],
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',  # Replace invalid characters instead of crashing
                timeout=30,
                cwd=str(parser_dir)  # Run from parser directory to find node_modules
            )

            if result.returncode != 0:
                logger.error(f"Babel parser failed: {result.stderr}")
                # Fallback to regex
                return self._parse_with_regex(file_path, source, language)

            # Check if stdout is empty
            if not result.stdout or not result.stdout.strip():
                logger.error(f"Babel parser returned empty output for {file_path}")
                return self._parse_with_regex(file_path, source, language)

            # Parse JSON result
            data = json.loads(result.stdout)

            # Check for parse errors
            if data.get('errors'):
                logger.warning(f"Parse errors in {file_path}: {data['errors']}")
                # Continue with partial results

            # Convert to CodeEntity objects
            entities = []
            imports = data.get('imports', [])

            # Convert imports to simple list
            import_list = []
            for imp in imports:
                import_list.append(imp['source'])

            # Convert components
            for comp in data.get('components', []):
                entity = self._babel_component_to_entity(comp, file_path)
                if entity:
                    entities.append(entity)

            # Convert routes
            for route in data.get('routes', []):
                entity = self._babel_route_to_entity(route, file_path)
                if entity:
                    entities.append(entity)

            return ParseResult(
                file_path=file_path,
                language=language,
                entities=entities,
                imports=import_list,
                errors=data.get('errors', [])
            )

        except subprocess.TimeoutExpired:
            logger.error(f"Babel parser timed out for {file_path}")
            return self._parse_with_regex(file_path, source, language)
        except UnicodeDecodeError as e:
            logger.error(f"Encoding error in Babel parser output for {file_path}: {e}")
            return self._parse_with_regex(file_path, source, language)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Babel output for {file_path}: {e}")
            return self._parse_with_regex(file_path, source, language)
        except Exception as e:
            logger.error(f"Babel parsing failed for {file_path}: {e}")
            return self._parse_with_regex(file_path, source, language)

    def _babel_component_to_entity(
        self,
        comp: Dict[str, Any],
        file_path: Path
    ) -> Optional[CodeEntity]:
        """Convert Babel component data to CodeEntity."""
        try:
            return CodeEntity(
                type=EntityType.COMPONENT,
                name=comp['name'],
                file_path=file_path,
                start_line=comp['start_line'],
                end_line=comp['end_line'],
                code=comp.get('code', ''),
                metadata={
                    'props_type': comp.get('props_type'),
                    'hooks_used': comp.get('hooks', []),
                    'api_calls': comp.get('api_calls', []),
                    'event_handlers': comp.get('event_handlers', []),
                    'is_exported': comp.get('is_exported', False),
                }
            )
        except Exception as e:
            logger.warning(f"Failed to convert component {comp.get('name')}: {e}")
            return None

    def _babel_route_to_entity(
        self,
        route: Dict[str, Any],
        file_path: Path
    ) -> Optional[CodeEntity]:
        """Convert Babel route data to CodeEntity."""
        try:
            return CodeEntity(
                type=EntityType.ROUTE,
                name=route.get('path', 'unknown'),
                file_path=file_path,
                start_line=route.get('line', 1),
                end_line=route.get('line', 1),
                code='',
                metadata={
                    'path': route.get('path'),
                    'component': route.get('component')
                }
            )
        except Exception as e:
            logger.warning(f"Failed to convert route: {e}")
            return None

    def _parse_with_regex(
        self,
        file_path: Path,
        source: str,
        language: str
    ) -> ParseResult:
        """
        Parse using regex patterns (fallback method).

        WARNING: This is less accurate than Babel parsing and may miss
        complex patterns or give false positives.
        """
        entities = []
        imports = []

        # Extract imports
        imports = self._extract_imports_regex(source)

        # Check if this is a React file
        if not self._is_react_file_regex(source, imports):
            return ParseResult(
                file_path=file_path,
                language=language,
                entities=[],
                imports=imports,
                errors=[]
            )

        # Extract components
        components = self._extract_components_regex(source, file_path)
        entities.extend(components)

        # Extract routes
        routes = self._extract_routes_regex(source, file_path)
        entities.extend(routes)

        return ParseResult(
            file_path=file_path,
            language=language,
            entities=entities,
            imports=imports,
            errors=[]
        )

    def _is_react_file_regex(self, source: str, imports: List[str]) -> bool:
        """Check if file is a React file."""
        # Check imports
        react_imports = ['react', 'React', 'from "react"', "from 'react'"]
        if any(imp in source for imp in react_imports):
            return True

        # Check for JSX syntax
        if re.search(r'<[A-Z]\w+[^>]*>', source):
            return True

        return False

    def _extract_imports_regex(self, source: str) -> List[str]:
        """Extract import statements using regex."""
        imports = []

        # Match: import ... from '...'
        pattern = r'import\s+(?:{[^}]+}|[\w,\s]+)\s+from\s+["\']([^"\']+)["\']'
        matches = re.finditer(pattern, source)

        for match in matches:
            module = match.group(1)
            imports.append(module)

        return imports

    def _extract_components_regex(
        self,
        source: str,
        file_path: Path
    ) -> List[CodeEntity]:
        """Extract React components using regex."""
        components = []

        # Pattern 1: Function components with export
        patterns = [
            # Arrow function components
            r'export\s+(?:const|let|var)\s+([A-Z]\w+)\s*[:=]\s*(?:\([^)]*\)|[^=]+)\s*=>\s*{',
            # Regular function components
            r'export\s+(?:default\s+)?function\s+([A-Z]\w+)\s*\([^)]*\)\s*{',
            # React.FC pattern
            r'(?:const|let|var)\s+([A-Z]\w+)\s*:\s*(?:React\.)?FC(?:<[^>]+>)?\s*=',
        ]

        for pattern in patterns:
            matches = re.finditer(pattern, source, re.MULTILINE)

            for match in matches:
                component_name = match.group(1)

                # Find the component definition
                start_pos = match.start()
                end_pos = self._find_matching_brace(source, start_pos)

                if end_pos == -1:
                    continue

                component_code = source[start_pos:end_pos]

                # Extract component details
                entity = self._create_component_entity(
                    component_name,
                    component_code,
                    source,
                    file_path,
                    start_pos
                )

                if entity:
                    components.append(entity)

        return components

    def _find_matching_brace(self, source: str, start_pos: int) -> int:
        """
        Find matching closing brace.

        WARNING: Naive implementation - doesn't handle strings/comments properly.
        Use Babel parser for accurate results.
        """
        # Find first opening brace after start_pos
        brace_pos = source.find('{', start_pos)
        if brace_pos == -1:
            return -1

        count = 1
        i = brace_pos + 1

        while i < len(source) and count > 0:
            if source[i] == '{':
                count += 1
            elif source[i] == '}':
                count -= 1
            i += 1

        return i if count == 0 else -1

    def _create_component_entity(
        self,
        name: str,
        code: str,
        full_source: str,
        file_path: Path,
        start_pos: int
    ) -> Optional[CodeEntity]:
        """Create CodeEntity for a React component."""
        try:
            # Calculate line numbers
            lines_before = full_source[:start_pos].count('\n')
            lines_in_code = code.count('\n')
            start_line = lines_before + 1
            end_line = start_line + lines_in_code

            # Extract props type
            props_type = self._extract_props_type(code)

            # Extract hooks used
            hooks_used = self._extract_hooks_used(code)

            # Extract API calls
            api_calls = self._extract_api_calls_in_component(code)

            # Extract event handlers
            handlers = self._extract_event_handlers(code)

            return CodeEntity(
                type=EntityType.COMPONENT,
                name=name,
                file_path=file_path,
                start_line=start_line,
                end_line=end_line,
                code=code,
                metadata={
                    'props_type': props_type,
                    'hooks_used': hooks_used,
                    'api_calls': api_calls,
                    'event_handlers': handlers,
                    'is_exported': 'export' in code[:50],
                }
            )

        except Exception as e:
            logger.debug(f"Failed to create component entity for {name}: {e}")
            return None

    def _extract_props_type(self, code: str) -> Optional[str]:
        """Extract props type from component."""
        # Pattern: ComponentName: FC<PropsType>
        match = re.search(r':\s*(?:React\.)?FC<([^>]+)>', code)
        if match:
            return match.group(1)

        # Pattern: (props: PropsType)
        match = re.search(r'\((?:\{[^}]+\}|props)\s*:\s*([^)]+)\)', code)
        if match:
            return match.group(1).strip()

        return None

    def _extract_hooks_used(self, code: str) -> List[Dict[str, Any]]:
        """Extract React hooks used in component."""
        hooks = []

        for hook_name in self.REACT_HOOKS:
            pattern = rf'\b{hook_name}\s*\('
            matches = re.finditer(pattern, code)

            for match in matches:
                hooks.append({
                    'name': hook_name,
                })

        return hooks

    def _extract_api_calls_in_component(self, code: str) -> List[Dict[str, Any]]:
        """Extract API calls within a component."""
        api_calls = []

        # Pattern: fetch('/api/...')
        fetch_pattern = r'fetch\s*\(\s*[`"\']([^`"\']+)[`"\']'
        for match in re.finditer(fetch_pattern, code):
            url = match.group(1)
            api_calls.append({
                'method': 'fetch',
                'url': url,
                'type': 'fetch'
            })

        # Pattern: axios.get('/api/...') or axios.post(...)
        axios_pattern = r'axios\.(\w+)\s*\(\s*[`"\']([^`"\']+)[`"\']'
        for match in re.finditer(axios_pattern, code):
            method = match.group(1).upper()
            url = match.group(2)
            api_calls.append({
                'method': method,
                'url': url,
                'type': 'axios'
            })

        return api_calls

    def _extract_event_handlers(self, code: str) -> List[str]:
        """Extract event handler names."""
        handlers = []

        # Pattern: onClick={handleClick}
        pattern = r'on[A-Z]\w+\s*=\s*\{([^}]+)\}'
        for match in re.finditer(pattern, code):
            handler = match.group(1).strip()
            if handler and not handler.startswith('('):  # Exclude inline functions
                handlers.append(handler)

        return list(set(handlers))

    def _extract_routes_regex(
        self,
        source: str,
        file_path: Path
    ) -> List[CodeEntity]:
        """Extract React Router routes."""
        routes = []

        # Pattern: <Route path="/..." element={<Component />} />
        pattern = r'<Route\s+path\s*=\s*["\']([^"\']+)["\']\s+element\s*=\s*\{<(\w+)'

        matches = re.finditer(pattern, source)

        for match in matches:
            path = match.group(1)
            component = match.group(2)

            # Calculate line number
            lines_before = source[:match.start()].count('\n')
            line_num = lines_before + 1

            entity = CodeEntity(
                type=EntityType.ROUTE,
                name=path,
                file_path=file_path,
                start_line=line_num,
                end_line=line_num,
                code=match.group(0),
                metadata={
                    'path': path,
                    'component': component
                }
            )

            routes.append(entity)

        return routes
