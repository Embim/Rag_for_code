"""
Traceback Analyzer - parses Python tracebacks and finds related code in the knowledge graph.

Usage:
    analyzer = TracebackAnalyzer(neo4j_client, weaviate_indexer)
    result = await analyzer.analyze(traceback_text)
    
    # Result contains:
    # - parsed frames with file, line, function
    # - matched code entities from the graph
    # - LLM-generated explanation and fix suggestions
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from pathlib import Path

from ..logger import get_logger

logger = get_logger(__name__)


@dataclass
class TracebackFrame:
    """Single frame in a traceback."""
    file_path: str
    line_number: int
    function_name: str
    code_line: Optional[str] = None
    # Matched entity from knowledge graph
    matched_entity: Optional[Dict[str, Any]] = None
    matched_file: Optional[Dict[str, Any]] = None


@dataclass
class ParsedTraceback:
    """Parsed traceback with all frames and error info."""
    frames: List[TracebackFrame] = field(default_factory=list)
    exception_type: str = ""
    exception_message: str = ""
    raw_text: str = ""
    
    @property
    def is_valid(self) -> bool:
        return len(self.frames) > 0 and bool(self.exception_type)
    
    @property
    def root_cause_frame(self) -> Optional[TracebackFrame]:
        """Last frame is usually the root cause."""
        return self.frames[-1] if self.frames else None


class TracebackAnalyzer:
    """
    Analyzes Python tracebacks and finds related code in the knowledge graph.
    
    Features:
    - Parses Python traceback format
    - Matches frames to code entities in Neo4j
    - Uses LLM to explain errors and suggest fixes
    - Supports Django, FastAPI, and generic Python tracebacks
    """
    
    # Regex patterns for traceback parsing
    FRAME_PATTERN = re.compile(
        r'File "(?P<file>[^"]+)", line (?P<line>\d+), in (?P<func>\w+)'
    )
    CODE_LINE_PATTERN = re.compile(r'^\s{4}(.+)$')
    EXCEPTION_PATTERN = re.compile(
        r'^(?P<type>[\w.]+):\s*(?P<message>.*)$'
    )
    # Alternative exception pattern (just type, no message)
    EXCEPTION_SIMPLE_PATTERN = re.compile(r'^(?P<type>[\w.]+)$')
    
    def __init__(
        self,
        neo4j_client=None,
        weaviate_indexer=None,
        llm_client=None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the analyzer.
        
        Args:
            neo4j_client: Neo4j client for graph queries
            weaviate_indexer: Weaviate indexer for semantic search
            llm_client: OpenAI-compatible client for explanations
            config: Additional configuration
        """
        self.neo4j = neo4j_client
        self.weaviate = weaviate_indexer
        self.llm = llm_client
        self.config = config or {}
        
    def parse(self, traceback_text: str) -> ParsedTraceback:
        """
        Parse a Python traceback into structured format.
        
        Args:
            traceback_text: Raw traceback text
            
        Returns:
            ParsedTraceback with frames and exception info
        """
        result = ParsedTraceback(raw_text=traceback_text)
        lines = traceback_text.strip().split('\n')
        
        i = 0
        while i < len(lines):
            line = lines[i]
            
            # Try to match a frame
            frame_match = self.FRAME_PATTERN.search(line)
            if frame_match:
                frame = TracebackFrame(
                    file_path=frame_match.group('file'),
                    line_number=int(frame_match.group('line')),
                    function_name=frame_match.group('func')
                )
                
                # Check if next line is the code
                if i + 1 < len(lines):
                    next_line = lines[i + 1]
                    code_match = self.CODE_LINE_PATTERN.match(next_line)
                    if code_match:
                        frame.code_line = code_match.group(1).strip()
                        i += 1
                
                result.frames.append(frame)
            
            # Try to match exception (usually last non-empty line)
            elif line.strip() and not line.startswith(' '):
                exc_match = self.EXCEPTION_PATTERN.match(line)
                if exc_match:
                    result.exception_type = exc_match.group('type')
                    result.exception_message = exc_match.group('message')
                else:
                    # Try simple pattern
                    simple_match = self.EXCEPTION_SIMPLE_PATTERN.match(line)
                    if simple_match and '.' in line:
                        result.exception_type = simple_match.group('type')
            
            i += 1
        
        return result
    
    async def find_code_in_graph(
        self,
        parsed: ParsedTraceback,
        repo_filter: Optional[str] = None
    ) -> ParsedTraceback:
        """
        Find matching code entities in the knowledge graph for each frame.
        
        Args:
            parsed: Parsed traceback
            repo_filter: Optional repository name to filter results
            
        Returns:
            ParsedTraceback with matched_entity and matched_file populated
        """
        if not self.neo4j:
            logger.warning("Neo4j client not available, skipping graph lookup")
            return parsed
        
        for frame in parsed.frames:
            # Normalize file path for matching
            normalized_path = self._normalize_path(frame.file_path)
            
            # Try to find the file in Neo4j
            file_query = """
            MATCH (f:File)
            WHERE f.file_path ENDS WITH $path_suffix
            RETURN f
            LIMIT 1
            """
            
            try:
                file_results = self.neo4j.execute_cypher(
                    file_query,
                    {'path_suffix': normalized_path}
                )
                
                if file_results:
                    frame.matched_file = dict(file_results[0]['f'])
                    
                    # Now find the function in that file
                    func_query = """
                    MATCH (f:File)-[:CONTAINS]->(fn:Function)
                    WHERE f.file_path ENDS WITH $path_suffix
                      AND fn.name = $func_name
                    RETURN fn
                    LIMIT 1
                    """
                    
                    func_results = self.neo4j.execute_cypher(
                        func_query,
                        {
                            'path_suffix': normalized_path,
                            'func_name': frame.function_name
                        }
                    )
                    
                    if func_results:
                        frame.matched_entity = dict(func_results[0]['fn'])
                    else:
                        # Try to find as a method in a class
                        method_query = """
                        MATCH (f:File)-[:CONTAINS]->(c:Class)-[:CONTAINS]->(m:Function)
                        WHERE f.file_path ENDS WITH $path_suffix
                          AND m.name = $func_name
                        RETURN m, c
                        LIMIT 1
                        """
                        
                        method_results = self.neo4j.execute_cypher(
                            method_query,
                            {
                                'path_suffix': normalized_path,
                                'func_name': frame.function_name
                            }
                        )
                        
                        if method_results:
                            frame.matched_entity = dict(method_results[0]['m'])
                            frame.matched_entity['parent_class'] = dict(method_results[0]['c'])
                            
            except Exception as e:
                logger.error(f"Error querying Neo4j for frame {frame}: {e}")
        
        return parsed
    
    async def get_related_code(
        self,
        parsed: ParsedTraceback,
        max_related: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Get code related to the error (callers, imports, etc.)
        
        Args:
            parsed: Parsed traceback with matched entities
            max_related: Maximum number of related entities
            
        Returns:
            List of related code entities
        """
        related = []
        
        if not self.neo4j:
            return related
        
        root_frame = parsed.root_cause_frame
        if not root_frame or not root_frame.matched_entity:
            return related
        
        entity_id = root_frame.matched_entity.get('id')
        if not entity_id:
            return related
        
        # Find who calls this function
        callers_query = """
        MATCH (caller:Function)-[:CALLS]->(target:Function {id: $entity_id})
        RETURN caller
        LIMIT $limit
        """
        
        try:
            callers = self.neo4j.execute_cypher(
                callers_query,
                {'entity_id': entity_id, 'limit': max_related}
            )
            
            for record in callers:
                related.append({
                    'type': 'caller',
                    'entity': dict(record['caller'])
                })
        except Exception as e:
            logger.error(f"Error finding callers: {e}")
        
        # Find what this function calls
        callees_query = """
        MATCH (source:Function {id: $entity_id})-[:CALLS]->(target:Function)
        RETURN target
        LIMIT $limit
        """
        
        try:
            callees = self.neo4j.execute_cypher(
                callees_query,
                {'entity_id': entity_id, 'limit': max_related}
            )
            
            for record in callees:
                related.append({
                    'type': 'calls',
                    'entity': dict(record['target'])
                })
        except Exception as e:
            logger.error(f"Error finding callees: {e}")
        
        return related
    
    async def explain_error(
        self,
        parsed: ParsedTraceback,
        related_code: List[Dict[str, Any]] = None
    ) -> Dict[str, str]:
        """
        Use LLM to explain the error and suggest fixes.
        
        Args:
            parsed: Parsed traceback with matched entities
            related_code: Optional list of related code entities
            
        Returns:
            Dict with 'explanation', 'cause', 'fix_suggestions'
        """
        if not self.llm:
            return {
                'explanation': f"Ошибка {parsed.exception_type}: {parsed.exception_message}",
                'cause': "LLM не настроен для детального анализа",
                'fix_suggestions': []
            }
        
        # Build context from matched entities
        code_context = []
        for frame in parsed.frames:
            if frame.matched_entity:
                entity = frame.matched_entity
                code_context.append(f"""
Файл: {frame.file_path}:{frame.line_number}
Функция: {entity.get('name', frame.function_name)}
Сигнатура: {entity.get('signature', 'N/A')}
Код строки: {frame.code_line or 'N/A'}
""")
        
        related_context = ""
        if related_code:
            related_context = "\n\nСвязанный код:\n"
            for item in related_code[:3]:
                entity = item['entity']
                related_context += f"- {item['type']}: {entity.get('name', 'Unknown')} ({entity.get('signature', '')})\n"
        
        prompt = f"""Проанализируй Python traceback и дай объяснение на русском языке.

TRACEBACK:
{parsed.raw_text}

КОНТЕКСТ КОДА:
{''.join(code_context) if code_context else 'Код не найден в графе знаний'}
{related_context}

ЗАДАЧА:
1. Объясни что произошло простым языком
2. Укажи вероятную причину ошибки
3. Предложи 2-3 способа исправления

Отвечай структурированно:
ОБЪЯСНЕНИЕ: (что произошло)
ПРИЧИНА: (почему это случилось)
ИСПРАВЛЕНИЯ:
- (первый способ)
- (второй способ)
"""

        try:
            response = await self.llm.chat.completions.create(
                model=self.config.get('model', 'deepseek/deepseek-r1:free'),
                messages=[
                    {"role": "system", "content": "Ты эксперт по Python, Django и FastAPI. Анализируешь ошибки и помогаешь их исправить."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1500,
                temperature=0.3
            )
            
            response_text = response.choices[0].message.content
            
            # Parse response
            explanation = ""
            cause = ""
            fixes = []
            
            lines = response_text.split('\n')
            current_section = None
            
            for line in lines:
                line = line.strip()
                if line.startswith('ОБЪЯСНЕНИЕ:'):
                    current_section = 'explanation'
                    explanation = line.replace('ОБЪЯСНЕНИЕ:', '').strip()
                elif line.startswith('ПРИЧИНА:'):
                    current_section = 'cause'
                    cause = line.replace('ПРИЧИНА:', '').strip()
                elif line.startswith('ИСПРАВЛЕНИЯ:'):
                    current_section = 'fixes'
                elif line.startswith('- ') and current_section == 'fixes':
                    fixes.append(line[2:])
                elif current_section == 'explanation' and line:
                    explanation += ' ' + line
                elif current_section == 'cause' and line:
                    cause += ' ' + line
            
            return {
                'explanation': explanation or response_text,
                'cause': cause,
                'fix_suggestions': fixes
            }
            
        except Exception as e:
            logger.error(f"LLM error: {e}")
            return {
                'explanation': f"Ошибка {parsed.exception_type}: {parsed.exception_message}",
                'cause': str(e),
                'fix_suggestions': []
            }
    
    async def analyze(
        self,
        traceback_text: str,
        repo_filter: Optional[str] = None,
        include_related: bool = True,
        explain: bool = True
    ) -> Dict[str, Any]:
        """
        Full analysis of a traceback.
        
        Args:
            traceback_text: Raw traceback text
            repo_filter: Optional repository to filter
            include_related: Whether to find related code
            explain: Whether to generate LLM explanation
            
        Returns:
            Complete analysis result
        """
        # Step 1: Parse traceback
        parsed = self.parse(traceback_text)
        
        if not parsed.is_valid:
            return {
                'success': False,
                'error': 'Could not parse traceback',
                'raw_text': traceback_text
            }
        
        # Step 2: Find code in graph
        parsed = await self.find_code_in_graph(parsed, repo_filter)
        
        # Step 3: Get related code
        related_code = []
        if include_related:
            related_code = await self.get_related_code(parsed)
        
        # Step 4: Explain with LLM
        explanation = {}
        if explain:
            explanation = await self.explain_error(parsed, related_code)
        
        # Build result
        frames_data = []
        for frame in parsed.frames:
            frame_data = {
                'file': frame.file_path,
                'line': frame.line_number,
                'function': frame.function_name,
                'code': frame.code_line,
                'matched': frame.matched_entity is not None,
            }
            
            if frame.matched_entity:
                frame_data['entity'] = {
                    'name': frame.matched_entity.get('name'),
                    'type': frame.matched_entity.get('type', 'Function'),
                    'signature': frame.matched_entity.get('signature'),
                    'docstring': frame.matched_entity.get('docstring'),
                }
                if 'parent_class' in frame.matched_entity:
                    frame_data['entity']['class'] = frame.matched_entity['parent_class'].get('name')
            
            frames_data.append(frame_data)
        
        return {
            'success': True,
            'exception': {
                'type': parsed.exception_type,
                'message': parsed.exception_message,
            },
            'frames': frames_data,
            'frames_matched': sum(1 for f in parsed.frames if f.matched_entity),
            'frames_total': len(parsed.frames),
            'related_code': [
                {
                    'relation': item['type'],
                    'name': item['entity'].get('name'),
                    'signature': item['entity'].get('signature'),
                }
                for item in related_code
            ],
            'explanation': explanation.get('explanation', ''),
            'cause': explanation.get('cause', ''),
            'fix_suggestions': explanation.get('fix_suggestions', []),
        }
    
    def _normalize_path(self, file_path: str) -> str:
        """Normalize file path for matching."""
        # Remove common prefixes
        prefixes_to_remove = [
            '/app/', '/src/', '/code/',
            'app/', 'src/', 'code/',
            './', '../'
        ]
        
        path = file_path
        for prefix in prefixes_to_remove:
            if path.startswith(prefix):
                path = path[len(prefix):]
        
        # Get last 2-3 path components for matching
        parts = Path(path).parts
        if len(parts) > 3:
            path = str(Path(*parts[-3:]))
        
        return path
    
    def format_for_telegram(self, result: Dict[str, Any]) -> str:
        """
        Format analysis result for Telegram message.
        
        Args:
            result: Analysis result from analyze()
            
        Returns:
            Formatted message string
        """
        if not result.get('success'):
            return f"❌ Не удалось разобрать traceback:\n{result.get('error', 'Unknown error')}"
        
        lines = []
        
        # Header
        exc = result['exception']
        lines.append(f"🔍 **Анализ ошибки**\n")
        lines.append(f"**Тип:** `{exc['type']}`")
        if exc['message']:
            lines.append(f"**Сообщение:** {exc['message'][:200]}")
        
        # Stack frames
        lines.append(f"\n📍 **Стек вызовов** ({result['frames_matched']}/{result['frames_total']} найдено):")
        
        for i, frame in enumerate(result['frames'], 1):
            matched = "✅" if frame['matched'] else "❓"
            lines.append(f"{i}. {matched} `{frame['file']}:{frame['line']}` - `{frame['function']}()`")
            
            if frame.get('entity'):
                entity = frame['entity']
                if entity.get('class'):
                    lines.append(f"   └─ Класс: `{entity['class']}`")
        
        # Explanation
        if result.get('explanation'):
            lines.append(f"\n💡 **Объяснение:**")
            lines.append(result['explanation'][:500])
        
        if result.get('cause'):
            lines.append(f"\n🐛 **Причина:**")
            lines.append(result['cause'][:300])
        
        # Fix suggestions
        if result.get('fix_suggestions'):
            lines.append(f"\n🔧 **Как исправить:**")
            for fix in result['fix_suggestions'][:3]:
                lines.append(f"• {fix[:200]}")
        
        # Related code
        if result.get('related_code'):
            lines.append(f"\n📁 **Связанный код:**")
            for item in result['related_code'][:3]:
                lines.append(f"• {item['relation']}: `{item['name']}`")
        
        return '\n'.join(lines)

