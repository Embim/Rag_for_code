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

    def __init__(self):
        self.visited_entities = set()
        self.tool_results = []
        self.iteration = 0
        self.start_time = time.time()

    def add_result(self, tool_name: str, result: Dict[str, Any]):
        """Add tool result to memory."""
        self.tool_results.append({
            'iteration': self.iteration,
            'tool': tool_name,
            'result': result,
        })

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
- semantic_search: Find code by concept or behavior (e.g., "authentication logic")
- exact_search: Find entity by exact name (e.g., "calculate_total")
- get_entity_details: Get full info about specific entity
- get_related_entities: Get entities connected to a given entity
- list_files: List files in directory to understand structure
- read_file: Read file content
- grep: Search for regex pattern in code
- get_graph_path: Find how two entities are connected

Strategy:
1. Start with semantic_search to find relevant entities
2. Use get_entity_details to examine specific entities
3. Use get_related_entities to trace connections
4. Continue until you have enough information
5. Synthesize a comprehensive answer with code examples

Guidelines:
- Be efficient: Don't repeat searches
- Be thorough: Gather enough context
- Be accurate: Only state facts found in code
- If you can't find something, say so clearly

You will be called in a loop. Each iteration:
1. Analyze what you've found so far
2. Decide: continue exploring OR synthesize answer
3. If continuing: call ONE tool
4. If done: provide final answer

Always respond in JSON format:
{
  "thought": "your reasoning about what to do next",
  "action": "continue" or "answer",
  "tool_call": {"tool": "tool_name", "params": {...}} (if action=continue),
  "answer": "final answer" (if action=answer)
}"""

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

    async def explore(self, question: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Explore codebase to answer question.

        Args:
            question: User's question
            context: Optional context (repositories, scope hints, etc.)

        Returns:
            Dict with:
            - answer: Final answer
            - tool_calls: List of tools used
            - iterations: Number of iterations
            - success: Whether answer is complete
        """
        logger.info(f"Code Explorer starting: {question}")

        memory = AgentMemory()
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
                return self._create_error_response(str(e), memory)

            logger.info(f"Agent decision: {decision.get('action')} - {decision.get('thought', '')[:100]}")

            # Execute decision
            if decision['action'] == 'answer':
                # Agent has enough information
                return {
                    'success': True,
                    'answer': decision['answer'],
                    'tool_calls': [r['tool'] for r in memory.tool_results],
                    'iterations': memory.iteration,
                    'complete': True,
                }

            elif decision['action'] == 'continue':
                # Execute tool
                tool_call = decision.get('tool_call', {})
                tool_name = tool_call.get('tool')
                params = tool_call.get('params', {})

                if not tool_name or tool_name not in self.tools:
                    logger.warning(f"Invalid tool: {tool_name}")
                    continue

                # Execute tool
                tool = self.tools[tool_name]
                result = await tool.execute(**params)

                # Store result
                memory.add_result(tool_name, result)

                if not result['success']:
                    logger.warning(f"Tool {tool_name} failed: {result.get('error')}")

            else:
                logger.warning(f"Unknown action: {decision.get('action')}")

        # Budget exhausted
        logger.warning("Agent budget exhausted")
        return self._create_incomplete_response(memory)

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

        # Parse JSON response
        try:
            decision = json.loads(content)
            return decision
        except json.JSONDecodeError:
            logger.error(f"Failed to parse LLM response: {content[:200]}")
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
                answer += f"- {entity.get('name')} ({entity.get('type')}) in {entity.get('file', 'unknown')}\n"
        else:
            answer += "No relevant code entities found. Try rephrasing your question."

        return {
            'success': False,
            'answer': answer,
            'tool_calls': [r['tool'] for r in memory.tool_results],
            'iterations': memory.iteration,
            'complete': False,
        }
