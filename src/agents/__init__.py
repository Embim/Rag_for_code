"""
Agent system for Code RAG.

Provides intelligent LLM-powered agents that can:
- Iteratively explore codebase
- Use tools to gather information
- Plan and adapt exploration strategy
- Synthesize comprehensive answers

Main components:
- CodeExplorerAgent: Iterative code investigation
- QueryOrchestrator: Question classification and routing
- AgentCache: Result caching layer
- Tools: Agent capabilities (search, navigate, analyze)
"""

from .code_explorer import CodeExplorerAgent
from .visual_guide_agent import VisualGuideAgent
from ..config.agent import AgentConfig
from .orchestrator import QueryOrchestrator, QuestionType
from .cache import AgentCache, CacheConfig
from .business_agent import BusinessAgent, TaskType, UIStep, UIWorkflow
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


__all__ = [
    'CodeExplorerAgent',
    'VisualGuideAgent',
    'AgentConfig',
    'QueryOrchestrator',
    'QuestionType',
    'AgentCache',
    'CacheConfig',
    'BusinessAgent',
    'TaskType',
    'UIStep',
    'UIWorkflow',
    'Tool',
    'SemanticSearchTool',
    'ExactSearchTool',
    'GetEntityDetailsTool',
    'GetRelatedEntitiesTool',
    'ListFilesTool',
    'ReadFileTool',
    'GrepTool',
    'GetGraphPathTool',
]
