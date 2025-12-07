"""
Code Explorer Agent.

An LLM-powered agent that iteratively explores the codebase to answer questions.
Uses tools to gather information, plans exploration strategy, and synthesizes answers.

Workflow:
1. Receive question from user
2. Plan: Decide which tools to use
3. Execute: Call tools and gather information
4. Analyze: Evaluate if enough information gathered
5. Repeat or Synthesize final answer
"""

import asyncio
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import json

from openai import AsyncOpenAI

from .tools import (
    Tool,
    SemanticSearchTool,
    ExactSearchTool,
    GetEntityDetailsTool,
    GetRelatedEntitiesTool,
    ListFilesTool,
    ReadFileTool,
    GrepTool,
    GetGraphPathTool,
)
from ..logger import get_logger


logger = get_logger(__name__)


# Import unified AgentConfig from config module
from ..config.agent import AgentConfig


@dataclass
class AgentMemory:
    """Session memory for the agent."""
    visited_entities: set
    tool_results: List[Dict[str, Any]]
    iteration: int
    start_time: float
    iteration_trace: List[Dict[str, Any]]  # Detailed trace for debugging

    def __init__(self):
        self.visited_entities = set()
        self.tool_results = []
        self.iteration = 0
        self.start_time = time.time()
        self.iteration_trace = []

    def add_result(self, tool_name: str, result: Dict[str, Any], params: Dict[str, Any] = None):
        """Add tool result to memory with detailed tracking."""
        tool_record = {
            'iteration': self.iteration,
            'tool': tool_name,
            'result': result,
            'timestamp': time.time(),
        }

        # Add detailed stats for debugging
        if result.get('success') and result.get('result'):
            result_data = result['result']

            # Count entities found
            entities = result_data.get('entities', [])
            tool_record['stats'] = {
                'success': True,
                'entities_found': len(entities),
                'entity_types': {},
                'repositories': set(),
                'files': set(),
            }

            # Analyze entities
            for entity in entities:
                entity_type = entity.get('type') or entity.get('node_type', 'Unknown')
                tool_record['stats']['entity_types'][entity_type] = \
                    tool_record['stats']['entity_types'].get(entity_type, 0) + 1

                # Track repositories
                repo = entity.get('repository')
                if repo:
                    tool_record['stats']['repositories'].add(repo)

                # Track files
                file_path = entity.get('file') or entity.get('file_path')
                if file_path:
                    tool_record['stats']['files'].add(file_path)

            # Convert sets to lists for JSON serialization
            tool_record['stats']['repositories'] = list(tool_record['stats']['repositories'])
            tool_record['stats']['files'] = list(tool_record['stats']['files'])
        else:
            tool_record['stats'] = {
                'success': False,
                'error': result.get('error', 'Unknown error')
            }

        # Add parameters used
        if params:
            tool_record['params'] = params

        self.tool_results.append(tool_record)

        # Track visited entities
        if result.get('success') and result.get('result'):
            entities = result['result'].get('entities', [])
            for entity in entities:
                if 'id' in entity:
                    self.visited_entities.add(entity['id'])

    def is_exhausted(self, config: AgentConfig) -> bool:
        """Check if budget is exhausted."""
        if self.iteration >= config.max_iterations:
            logger.warning(f"Max iterations reached: {self.iteration}")
            return True

        elapsed = time.time() - self.start_time
        if elapsed >= config.timeout_seconds:
            logger.warning(f"Timeout reached: {elapsed:.1f}s")
            return True

        return False

    def get_trace_summary(self) -> Dict[str, Any]:
        """Get detailed trace summary for debugging."""
        total_entities_found = 0
        total_files_accessed = set()
        total_repositories = set()
        tools_used_count = {}
        entity_types_overall = {}

        for tool_result in self.tool_results:
            if 'stats' in tool_result and tool_result['stats'].get('success'):
                stats = tool_result['stats']
                total_entities_found += stats.get('entities_found', 0)

                # Aggregate files
                for f in stats.get('files', []):
                    total_files_accessed.add(f)

                # Aggregate repositories
                for r in stats.get('repositories', []):
                    total_repositories.add(r)

                # Count entity types
                for etype, count in stats.get('entity_types', {}).items():
                    entity_types_overall[etype] = entity_types_overall.get(etype, 0) + count

            # Count tool usage
            tool_name = tool_result.get('tool')
            tools_used_count[tool_name] = tools_used_count.get(tool_name, 0) + 1

        return {
            'total_iterations': self.iteration,
            'total_tool_calls': len(self.tool_results),
            'total_entities_found': total_entities_found,
            'unique_files_accessed': len(total_files_accessed),
            'repositories_searched': list(total_repositories),
            'tools_used': tools_used_count,
            'entity_types_found': entity_types_overall,
            'duration_seconds': time.time() - self.start_time,
        }

    def get_detailed_trace(self) -> List[Dict[str, Any]]:
        """Get detailed iteration-by-iteration trace."""
        trace = []

        for tool_result in self.tool_results:
            iteration = tool_result.get('iteration')
            tool_name = tool_result.get('tool')
            params = tool_result.get('params', {})
            stats = tool_result.get('stats', {})
            timestamp = tool_result.get('timestamp', 0)

            trace_entry = {
                'iteration': iteration,
                'tool': tool_name,
                'params': params,
                'timestamp': timestamp,
                'elapsed_ms': (timestamp - self.start_time) * 1000,
            }

            if stats.get('success'):
                trace_entry['result'] = {
                    'success': True,
                    'entities_found': stats.get('entities_found', 0),
                    'entity_types': stats.get('entity_types', {}),
                    'files': stats.get('files', []),
                    'repositories': stats.get('repositories', []),
                }
            else:
                trace_entry['result'] = {
                    'success': False,
                    'error': stats.get('error', 'Unknown error')
                }

            trace.append(trace_entry)

        return trace


class CodeExplorerAgent:
    """
    LLM-powered agent for iterative code exploration.

    The agent uses function calling to decide which tools to use,
    executes them, analyzes results, and either continues exploring
    or synthesizes a final answer.
    """

    SYSTEM_PROMPT = """You are a Code Explorer Agent - an expert at investigating codebases.

Your goal is to answer the user's question by iteratively using tools to explore the code.

Available tools:
- semantic_search: Find code by concept or behavior
  Parameters: query (string), scope (optional), top_k (optional, default: 10, recommended: 20), repositories (optional: list of repo names)
  Example: {"tool": "semantic_search", "params": {"query": "authentication logic", "top_k": 20}}
  Example (multi-repo): {"tool": "semantic_search", "params": {"query": "equity trading", "repositories": ["ui", "api"]}}

- exact_search: Find entity by exact name
  Parameters: name (string), entity_type (optional)
  Example: {"tool": "exact_search", "params": {"name": "calculate_total"}}

- get_entity_details: Get full info about specific entity
  Parameters: id or name (string)
  Example: {"tool": "get_entity_details", "params": {"id": "entity-id-123"}}

- get_related_entities: Get entities connected to a given entity
  Parameters: id or name (string), relation_type (optional), direction (optional: "outgoing"/"incoming")
  Example: {"tool": "get_related_entities", "params": {"name": "UserModel", "direction": "outgoing"}}

- list_files: List files in directory (MAY NOT WORK if physical files unavailable)
  Parameters: directory (string), pattern (optional, default: '*')
  Example: {"tool": "list_files", "params": {"directory": "src/api", "pattern": "*.py"}}
  NOTE: If list_files fails, use semantic_search to find files instead

- read_file: Read file content (MAY NOT WORK if physical files unavailable)
  Parameters: path (string), start_line (optional), end_line (optional)
  Example: {"tool": "read_file", "params": {"path": "src/main.py", "start_line": 10, "end_line": 50}}
  NOTE: If read_file fails, use get_entity_details instead - it retrieves code from the knowledge graph

- grep: Search for regex pattern in code
  Parameters: pattern (string), scope (optional)
  Example: {"tool": "grep", "params": {"pattern": "def.*calculate"}}

- get_graph_path: Find how two entities are connected
  Parameters: from_entity (string), to_entity (string), max_depth (optional, default: 5)
  Example: {"tool": "get_graph_path", "params": {"from_entity": "LoginView", "to_entity": "UserModel"}}

Strategy:
1. Start with semantic_search to find relevant entities (use top_k=20 for comprehensive results)
2. Use get_entity_details to examine specific entities (can call multiple in one iteration)
3. Use get_related_entities to trace connections
4. Continue until you have enough information
5. Synthesize a COMPREHENSIVE and DETAILED answer

Guidelines:
- Be efficient: Don't repeat searches, use multiple tool calls per iteration when appropriate
- Be thorough: Gather enough context (e.g., get details for multiple entities at once)
- Be accurate: Only state facts found in code
- If you can't find something, say so clearly

Efficiency tips:
- Use semantic_search with top_k=20 to get more results upfront
- Call get_entity_details for multiple entities in one iteration
- Combine complementary tools (e.g., semantic_search + get_entity_details)

You will be called in a loop. Each iteration:
1. Analyze what you've found so far
2. Decide: continue exploring OR synthesize answer
3. If continuing: call ONE OR MORE tools (max 3 per iteration for efficiency)
4. If done: provide final answer

ANSWER FORMAT REQUIREMENTS:
When providing the final answer (action="answer"), structure it with these sections:

1. **Overview**: Brief summary of what you found (2-3 sentences)

2. **Implementation Details**: Detailed explanation with:
   - File paths and line numbers (e.g., `src/services/trade.py:45-67`)
   - Step-by-step flow with numbered steps
   - Component interactions and data flow
   - Key functions/classes involved with their responsibilities

3. **Code Examples**: Include relevant code snippets with:
   - Proper syntax highlighting (use markdown code blocks)
   - Inline comments explaining key parts
   - Multiple examples if there are different scenarios

4. **Data Flow**: Describe how data moves through the system:
   - Input → Processing → Output
   - Which services/components handle what
   - Database operations if relevant

5. **Additional Context**: Include:
   - Related files that might be useful
   - Dependencies between components
   - Error handling approach
   - Validation/business rules

6. **Example Usage**: Provide concrete examples:
   - API request/response examples
   - Function call examples with parameters
   - UI interaction flow if applicable

Always respond in JSON format:
{
  "thought": "your reasoning about what to do next",
  "action": "continue" or "answer",
  "tool_calls": [{"tool": "tool_name", "params": {...}}, ...] (if action=continue, 1-3 tools),
  "answer": "final answer with ALL sections above" (if action=answer)
}

Examples of efficient multi-tool usage:
{
  "thought": "Found 5 checkout-related entities. Let me get details for the top 3 to understand the flow.",
  "action": "continue",
  "tool_calls": [
    {"tool": "get_entity_details", "params": {"id": "entity-1"}},
    {"tool": "get_entity_details", "params": {"id": "entity-2"}},
    {"tool": "get_entity_details", "params": {"id": "entity-3"}}
  ]
}

Note: For backward compatibility, you can also use "tool_call" (singular) for single tool execution."""

    def __init__(
        self,
        tools: List[Tool],
        api_key: str,
        config: Optional[AgentConfig] = None,
        api_base: str = "https://openrouter.ai/api/v1"
    ):
        """
        Initialize agent.

        Args:
            tools: List of available tools
            api_key: OpenRouter/OpenAI API key
            config: Agent configuration
            api_base: API base URL
        """
        self.tools = {tool.name: tool for tool in tools}
        self.config = config or AgentConfig()

        # Initialize LLM client
        self.llm = AsyncOpenAI(
            api_key=api_key,
            base_url=api_base,
        )

    async def explore(
        self,
        question: str,
        context: Optional[Dict[str, Any]] = None,
        detail_level: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Explore codebase to answer question.

        Args:
            question: User's question
            context: Optional context (repositories, scope hints, etc.)
            detail_level: Answer detail level ("brief", "normal", "detailed")
                         If None, uses config default

        Returns:
            Dict with:
            - answer: Final answer
            - tool_calls: List of tools used
            - iterations: Number of iterations
            - success: Whether answer is complete
        """
        logger.info(f"Code Explorer starting: {question}")

        # Use provided detail_level or config default
        detail_level = detail_level or getattr(self.config, 'detail_level', 'detailed')

        memory = AgentMemory()
        memory.detail_level = detail_level  # Store in memory for later use
        context_str = self._format_context(context) if context else "No additional context."

        while not memory.is_exhausted(self.config):
            memory.iteration += 1
            logger.info(f"Iteration {memory.iteration}/{self.config.max_iterations}")

            # Build prompt for LLM
            prompt = self._build_iteration_prompt(question, context_str, memory)

            # Get LLM decision
            try:
                decision = await self._get_llm_decision(prompt)
            except Exception as e:
                logger.error(f"LLM call failed: {e}")
                # Extract sources even on error
                sources = self._extract_sources_from_memory(memory)
                error_response = self._create_error_response(str(e), memory)
                error_response['sources'] = sources
                return error_response

            logger.info(f"Agent decision: {decision.get('action')} - {decision.get('thought', '')[:100]}")

            # Execute decision
            if decision['action'] == 'answer':
                # Agent has enough information - enrich the answer
                # Extract answer and ensure it's a string
                base_answer = decision.get('answer', '')
                if isinstance(base_answer, dict):
                    base_answer = base_answer.get('answer', base_answer.get('content', str(base_answer)))
                if not isinstance(base_answer, str):
                    base_answer = str(base_answer)

                enriched_answer = await self._enrich_answer(
                    question=question,
                    base_answer=base_answer,
                    memory=memory
                )

                # Extract entities/sources from tool results for downstream use (e.g., visualization)
                sources = self._extract_sources_from_memory(memory)

                # Build response
                response = {
                    'success': True,
                    'answer': enriched_answer,
                    'sources': sources,  # Add sources for visual guide agent
                    'tool_calls': [r['tool'] for r in memory.tool_results],
                    'iterations': memory.iteration,
                    'complete': True,
                }

                # Add debug trace if requested
                if context and context.get('verbose', False):
                    response['debug'] = {
                        'trace_summary': memory.get_trace_summary(),
                        'detailed_trace': memory.get_detailed_trace(),
                    }
                    logger.info(f"📊 Trace summary: {response['debug']['trace_summary']}")

                return response

            elif decision['action'] == 'continue':
                # Handle multiple tool calls (new format) or single tool call (backward compatibility)
                tool_calls = decision.get('tool_calls', [])
                if not tool_calls:
                    # Backward compatibility: check for singular 'tool_call'
                    single_call = decision.get('tool_call')
                    if single_call:
                        tool_calls = [single_call]

                if not tool_calls:
                    logger.warning("No tool calls provided in continue action")
                    continue

                # Limit to 3 tool calls per iteration for efficiency
                tool_calls = tool_calls[:3]

                # Execute all tool calls
                for tool_call in tool_calls:
                    tool_name = tool_call.get('tool')
                    params = tool_call.get('params', {})

                    if not tool_name or tool_name not in self.tools:
                        logger.warning(f"Invalid tool: {tool_name}")
                        continue

                    # Execute tool
                    tool = self.tools[tool_name]
                    result = await tool.execute(**params)

                    # Store result with parameters for debugging
                    memory.add_result(tool_name, result, params)

                    if not result['success']:
                        logger.warning(f"Tool {tool_name} failed: {result.get('error')}")
                    else:
                        # Log summary of what was found
                        if 'stats' in memory.tool_results[-1]:
                            stats = memory.tool_results[-1]['stats']
                            if stats.get('success'):
                                logger.info(f"✅ {tool_name}: found {stats.get('entities_found', 0)} entities "
                                          f"across {len(stats.get('files', []))} files")

            else:
                logger.warning(f"Unknown action: {decision.get('action')}")

        # Budget exhausted
        logger.warning("Agent budget exhausted")

        # Extract sources even for incomplete response
        sources = self._extract_sources_from_memory(memory)

        response = self._create_incomplete_response(memory)
        response['sources'] = sources  # Add sources for visualization fallback
        return response

    def _build_iteration_prompt(self, question: str, context: str, memory: AgentMemory) -> str:
        """Build prompt for current iteration."""
        # Format tool results
        tool_results_str = self._format_tool_results(memory.tool_results)

        prompt = f"""Question: {question}

Context: {context}

Iteration: {memory.iteration}/{self.config.max_iterations}

Tool results so far:
{tool_results_str}

What should you do next? Respond in JSON format."""

        return prompt

    def _format_tool_results(self, results: List[Dict[str, Any]]) -> str:
        """Format tool results for prompt."""
        if not results:
            return "No tools executed yet."

        formatted = []
        for r in results[-5:]:  # Show last 5 results to save tokens
            tool = r['tool']
            success = r['result'].get('success', False)

            if success:
                result_data = r['result'].get('result', {})
                # Summarize result
                summary = f"Tool: {tool}\n"

                if 'entities' in result_data:
                    entities = result_data['entities']
                    summary += f"  Found {len(entities)} entities\n"
                    for entity in entities[:3]:  # Show top 3
                        summary += f"    - {entity.get('name')} ({entity.get('type')})\n"

                elif 'files' in result_data:
                    files = result_data['files']
                    summary += f"  Found {len(files)} files\n"

                elif 'content' in result_data:
                    content = result_data['content']
                    summary += f"  Read {len(content)} characters\n"

                formatted.append(summary)
            else:
                formatted.append(f"Tool: {tool}\n  Error: {r['result'].get('error')}\n")

        return "\n".join(formatted)

    def _format_context(self, context: Dict[str, Any]) -> str:
        """Format context for prompt."""
        parts = []

        if 'repositories' in context:
            repos = context['repositories']
            parts.append(f"Repositories: {', '.join(repos)}")

        if 'scope' in context:
            parts.append(f"Scope: {context['scope']}")

        if 'hints' in context:
            parts.append(f"Hints: {context['hints']}")

        return " | ".join(parts) if parts else "No context"

    async def _get_llm_decision(self, prompt: str) -> Dict[str, Any]:
        """
        Get decision from LLM.

        Returns JSON with action and next step.
        """
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        response = await self.llm.chat.completions.create(
            model=self.config.model,
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens_per_call,
        )

        content = response.choices[0].message.content

        # Parse JSON response with robust handling
        try:
            decision = self._parse_json_response(content)
            return decision
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response after all attempts: {content[:300]}")
            # Try to extract action from text
            if 'answer' in content.lower():
                return {
                    'action': 'answer',
                    'answer': content,
                }
            else:
                return {
                    'action': 'continue',
                    'thought': 'Fallback: trying semantic search',
                    'tool_call': {
                        'tool': 'semantic_search',
                        'params': {'query': prompt[:100]},
                    }
                }

    def _parse_json_response(self, content: str) -> Dict[str, Any]:
        """
        Parse JSON from LLM response, handling various formats.

        Handles:
        - Plain JSON
        - JSON in markdown code blocks (```json ... ```)
        - JSON with text before/after
        - Truncated JSON (try to fix)
        """
        import re

        # Remove markdown code blocks if present
        content = re.sub(r'```json\s*', '', content)
        content = re.sub(r'```\s*', '', content)
        content = content.strip()

        # Try 1: Direct parse
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass

        # Try 2: Find JSON object in text
        # Look for {...} pattern with "action" or "thought" inside
        json_patterns = [
            r'\{[^{}]*"action"[^{}]*\}',  # Simple single-level
            r'\{(?:[^{}]|\{[^{}]*\})*\}',  # Nested objects
        ]

        for pattern in json_patterns:
            matches = re.finditer(pattern, content, re.DOTALL)
            for match in matches:
                try:
                    obj = json.loads(match.group(0))
                    if 'action' in obj:  # Valid decision object
                        return obj
                except json.JSONDecodeError:
                    continue

        # Try 3: Find anything that looks like JSON
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass

        # Try 4: Fix truncated JSON (common issue)
        # If content ends with incomplete structure, try to close it
        if content.strip().endswith(','):
            try:
                return json.loads(content.rstrip(',') + '}')
            except json.JSONDecodeError:
                pass

        # If all fails, raise error
        raise json.JSONDecodeError(f"Could not parse JSON from response", content, 0)

    async def _enrich_answer(
        self,
        question: str,
        base_answer: str,
        memory: AgentMemory
    ) -> str:
        """
        Enrich the base answer with additional details from collected data.

        Args:
            question: Original question
            base_answer: Initial answer from agent
            memory: Agent memory with all tool results

        Returns:
            Enriched answer with more details
        """
        # Check if enrichment is enabled
        if not getattr(self.config, 'enable_enrichment', True):
            logger.info("Answer enrichment disabled, returning base answer")
            return base_answer
        # Collect all entities and their details from memory
        entities_data = []
        code_snippets = []
        file_paths = set()

        for result in memory.tool_results:
            if not result['result'].get('success'):
                continue

            result_data = result['result'].get('result', {})

            # Extract entities
            if 'entities' in result_data:
                for entity in result_data['entities']:
                    entities_data.append({
                        'name': entity.get('name', 'Unknown'),
                        'type': entity.get('type', 'Unknown'),
                        'file': entity.get('file') or entity.get('file_path', 'Unknown'),
                        'code': entity.get('code_snippet', ''),
                        'docstring': entity.get('docstring', ''),
                        'line_start': entity.get('line_start', 0),
                        'line_end': entity.get('line_end', 0),
                    })

                    # Collect file paths
                    file_path = entity.get('file') or entity.get('file_path')
                    if file_path:
                        file_paths.add(file_path)

                    # Collect code snippets
                    code = entity.get('code_snippet', '')
                    if code and len(code) > 50:  # Only substantial snippets
                        code_snippets.append({
                            'name': entity.get('name', 'Unknown'),
                            'code': code,
                            'file': file_path or 'Unknown',
                        })

            # Extract file content
            if 'content' in result_data:
                content = result_data['content']
                if len(content) > 100:
                    code_snippets.append({
                        'name': 'File content',
                        'code': content[:2000],  # Limit to avoid token overflow
                        'file': result_data.get('file', 'Unknown'),
                    })

        # Get detail level from memory
        detail_level = getattr(memory, 'detail_level', 'detailed')

        # Adjust limits based on detail level
        if detail_level == 'brief':
            entity_limit = 5
            snippet_limit = 2
            snippet_length = 300
            file_limit = 10
        elif detail_level == 'normal':
            entity_limit = 10
            snippet_limit = 3
            snippet_length = 500
            file_limit = 15
        else:  # detailed
            entity_limit = 20
            snippet_limit = 7
            snippet_length = 800
            file_limit = 25

        # Build enrichment context
        enrichment_context = []

        if entities_data:
            enrichment_context.append(f"**Found {len(entities_data)} relevant entities:**")
            for entity in entities_data[:entity_limit]:
                location = f"{entity['file']}"
                if entity['line_start'] > 0:
                    location += f":{entity['line_start']}"
                    if entity['line_end'] > entity['line_start']:
                        location += f"-{entity['line_end']}"

                enrichment_context.append(f"- {entity['name']} ({entity['type']}) in {location}")
                if entity['docstring'] and detail_level != 'brief':
                    enrichment_context.append(f"  Doc: {entity['docstring'][:200]}")

        if code_snippets:
            enrichment_context.append(f"\n**Code snippets ({len(code_snippets)} found):**")
            for snippet in code_snippets[:snippet_limit]:
                enrichment_context.append(f"\n`{snippet['file']}`:")
                enrichment_context.append(f"```python\n{snippet['code'][:snippet_length]}\n```")

        if file_paths:
            enrichment_context.append(f"\n**Related files ({len(file_paths)} total):**")
            for fp in sorted(list(file_paths))[:file_limit]:
                enrichment_context.append(f"- {fp}")

        # Create detail level instructions
        detail_instructions = {
            'brief': "Keep the answer concise - focus on the main points with minimal code examples. 2-3 paragraphs max.",
            'normal': "Provide a balanced answer with key details and a few code examples. Include main file paths and important functions.",
            'detailed': "Provide a comprehensive answer with extensive details, multiple code examples, full file paths with line numbers, and thorough explanations."
        }

        # Create enrichment prompt
        enrichment_prompt = f"""You are enhancing a code exploration answer with additional details.

Detail Level: {detail_level.upper()}
{detail_instructions.get(detail_level, detail_instructions['detailed'])}

Original Question: {question}

Base Answer:
{base_answer}

Additional Context Found During Exploration:
{chr(10).join(enrichment_context) if enrichment_context else "No additional context available"}

Your task:
1. Keep the structure from the base answer (Overview, Implementation Details, etc.)
2. Enhance each section with specific details from the context above:
   - Add exact file paths with line numbers (e.g., `src/service.py:45-67`)
   - Include relevant code snippets with syntax highlighting
   - Add concrete examples from the found entities
   - Include docstrings and comments if available
3. Adjust the level of detail according to the Detail Level specified above
4. Make the answer well-organized and use markdown formatting
5. If the base answer already has good structure, preserve it and just add details

Return ONLY the enhanced answer, no meta-commentary."""

        try:
            messages = [
                {"role": "user", "content": enrichment_prompt}
            ]

            # Add timeout to prevent hanging
            enrichment_timeout = getattr(self.config, 'enrichment_timeout', 60.0)

            logger.info(f"Starting answer enrichment (timeout: {enrichment_timeout}s)...")
            start_time = time.time()

            # Wrap the API call in a timeout
            response = await asyncio.wait_for(
                self.llm.chat.completions.create(
                    model=self.config.model,
                    messages=messages,
                    temperature=0.1,  # Low temperature for consistent formatting
                    max_tokens=getattr(self.config, 'max_tokens_enrichment', 8192),
                ),
                timeout=enrichment_timeout
            )

            enriched = response.choices[0].message.content.strip()
            elapsed = time.time() - start_time

            # If enrichment failed or is too short, return base answer
            if len(enriched) < len(base_answer) * 0.5:
                logger.warning(f"Enrichment produced much shorter answer ({len(enriched)} vs {len(base_answer)}), using base answer")
                return base_answer

            logger.info(f"Answer enriched: {len(base_answer)} → {len(enriched)} chars in {elapsed:.1f}s")
            return enriched

        except asyncio.TimeoutError:
            logger.warning(f"Answer enrichment timed out after {enrichment_timeout}s, using fallback enrichment")
            # Use simple fallback enrichment without LLM
            return self._fallback_enrichment(base_answer, entities_data, code_snippets, file_paths)
        except Exception as e:
            logger.error(f"Answer enrichment failed: {e}, using fallback enrichment")
            # Use simple fallback enrichment without LLM
            return self._fallback_enrichment(base_answer, entities_data, code_snippets, file_paths)

    def _extract_sources_from_memory(self, memory: AgentMemory) -> List[Dict[str, Any]]:
        """
        Extract entity sources from tool results for visualization.

        Returns list of entities found during exploration.
        """
        sources = []
        seen_ids = set()

        for tool_result in memory.tool_results:
            if not tool_result.get('result', {}).get('success'):
                continue

            result_data = tool_result['result'].get('result', {})

            # Extract from semantic_search results
            if 'entities' in result_data:
                for entity in result_data['entities']:
                    entity_id = entity.get('id') or entity.get('node_id')
                    if entity_id and entity_id not in seen_ids:
                        sources.append({
                            'id': entity_id,
                            'name': entity.get('name', 'Unknown'),
                            'type': entity.get('type') or entity.get('node_type', 'Unknown'),
                            'file': entity.get('file') or entity.get('file_path', 'Unknown'),
                            'line': entity.get('line') or entity.get('start_line'),
                            'code': entity.get('code') or entity.get('code_snippet', ''),
                        })
                        seen_ids.add(entity_id)

            # Extract from get_entity_details
            elif 'id' in result_data or 'name' in result_data:
                entity_id = result_data.get('id')
                if entity_id and entity_id not in seen_ids:
                    sources.append({
                        'id': entity_id,
                        'name': result_data.get('name', 'Unknown'),
                        'type': result_data.get('type', 'Unknown'),
                        'file': result_data.get('file', 'Unknown'),
                        'line': result_data.get('line'),
                        'code': result_data.get('code', ''),
                    })
                    seen_ids.add(entity_id)

            # Extract from related_entities
            elif 'related_entities' in result_data:
                for entity in result_data['related_entities']:
                    entity_id = entity.get('id')
                    if entity_id and entity_id not in seen_ids:
                        sources.append({
                            'id': entity_id,
                            'name': entity.get('name', 'Unknown'),
                            'type': entity.get('type', 'Unknown'),
                            'file': entity.get('file', 'Unknown'),
                            'line': entity.get('line'),
                            'code': '',  # Related entities don't include full code
                        })
                        seen_ids.add(entity_id)

        logger.info(f"Extracted {len(sources)} sources from {len(memory.tool_results)} tool results")
        return sources[:50]  # Limit to 50 for visualization

    def _fallback_enrichment(
        self,
        base_answer: str,
        entities_data: List[Dict],
        code_snippets: List[Dict],
        file_paths: set
    ) -> str:
        """
        Simple fallback enrichment without LLM.
        Appends found entities and code snippets to base answer.
        """
        # Handle case where base_answer might be a dict instead of string
        if isinstance(base_answer, dict):
            # Try to extract string from common dict structures
            base_answer = base_answer.get('answer', base_answer.get('content', str(base_answer)))

        # Ensure base_answer is a string
        if not isinstance(base_answer, str):
            base_answer = str(base_answer)

        enriched_parts = [base_answer, "\n\n---\n"]

        if entities_data:
            enriched_parts.append(f"\n### 📦 Found {len(entities_data)} relevant entities:\n")
            for entity in entities_data[:10]:  # Limit to 10
                location = f"{entity['file']}"
                if entity['line_start'] > 0:
                    location += f":{entity['line_start']}"
                    if entity['line_end'] > entity['line_start']:
                        location += f"-{entity['line_end']}"
                enriched_parts.append(f"- **{entity['name']}** ({entity['type']}) in `{location}`")
                if entity['docstring']:
                    enriched_parts.append(f"  \n  {entity['docstring'][:150]}...")
                enriched_parts.append("\n")

        if code_snippets:
            enriched_parts.append(f"\n### 💻 Code Examples ({len(code_snippets)} snippets):\n")
            for snippet in code_snippets[:3]:  # Limit to 3
                enriched_parts.append(f"\n**{snippet['file']}**:")
                enriched_parts.append(f"```python\n{snippet['code'][:400]}\n```\n")

        if file_paths:
            enriched_parts.append(f"\n### 📁 Related Files ({len(file_paths)} total):\n")
            for fp in sorted(list(file_paths))[:15]:
                enriched_parts.append(f"- `{fp}`\n")

        return "".join(enriched_parts)

    def _create_error_response(self, error: str, memory: AgentMemory) -> Dict[str, Any]:
        """Create error response."""
        return {
            'success': False,
            'answer': f"Error during exploration: {error}",
            'tool_calls': [r['tool'] for r in memory.tool_results],
            'iterations': memory.iteration,
            'complete': False,
            'error': error,
        }

    def _create_incomplete_response(self, memory: AgentMemory) -> Dict[str, Any]:
        """Create response when budget exhausted."""
        # Synthesize best answer from what we found
        entities_found = []
        for result in memory.tool_results:
            if result['result'].get('success'):
                entities = result['result'].get('result', {}).get('entities', [])
                entities_found.extend(entities)

        answer = "⚠️ Exploration incomplete (budget exhausted).\n\n"

        if entities_found:
            answer += f"Found {len(entities_found)} potentially relevant entities:\n"
            for entity in entities_found[:10]:
                # Support both 'file' and 'file_path' fields
                file_path = entity.get('file') or entity.get('file_path') or 'unknown'
                name = entity.get('name', 'Unknown')
                entity_type = entity.get('type', 'Unknown')

                # Add code snippet preview if available
                code_snippet = entity.get('code_snippet', '')
                if code_snippet:
                    # Show first line of code
                    first_line = code_snippet.split('\n')[0][:60]
                    answer += f"- {name} ({entity_type}) in {file_path}\n  Preview: {first_line}...\n"
                else:
                    answer += f"- {name} ({entity_type}) in {file_path}\n"
        else:
            answer += "No relevant code entities found. Try rephrasing your question."

        return {
            'success': False,
            'answer': answer,
            'tool_calls': [r['tool'] for r in memory.tool_results],
            'iterations': memory.iteration,
            'complete': False,
        }
