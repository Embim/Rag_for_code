"""
Example: Using Code Explorer Agent.

This script demonstrates how to use the Code Explorer Agent
to investigate a codebase and answer questions.

Requirements:
- Neo4j and Weaviate running
- Code indexed in knowledge graph
- OPENROUTER_API_KEY environment variable set
"""

import asyncio
import os
from pathlib import Path

# Add parent directory to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents import CodeExplorerAgent
from src.agents.tools import (
    SemanticSearchTool,
    ExactSearchTool,
    GetEntityDetailsTool,
    GetRelatedEntitiesTool,
    ListFilesTool,
    ReadFileTool,
    GrepTool,
    GetGraphPathTool,
)
from src.code_rag.retrieval import CodeRetriever
from src.code_rag.graph import Neo4jClient, WeaviateIndexer
from src.agents.code_explorer import AgentConfig
from src.logger import get_logger


logger = get_logger(__name__)


async def main():
    """Run example agent queries."""

    # Get API key
    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key:
        print("ERROR: Set OPENROUTER_API_KEY environment variable")
        return

    # Initialize clients
    print("Initializing clients...")
    neo4j = Neo4jClient(
        uri="bolt://localhost:7687",
        user="neo4j",
        password=os.getenv('NEO4J_PASSWORD', 'password'),
    )

    weaviate = WeaviateIndexer(url="http://localhost:8080")
    retriever = CodeRetriever(neo4j, weaviate)

    # Initialize tools
    print("Setting up agent tools...")
    repos_dir = Path("data/repos")

    tools = [
        SemanticSearchTool(retriever),
        ExactSearchTool(neo4j),
        GetEntityDetailsTool(neo4j),
        GetRelatedEntitiesTool(neo4j),
        ListFilesTool(repos_dir),
        ReadFileTool(repos_dir),
        GrepTool(neo4j),
        GetGraphPathTool(neo4j),
    ]

    # Create agent
    config = AgentConfig(
        max_iterations=10,
        timeout_seconds=120,
        temperature=0.1,
        model="anthropic/claude-sonnet-4",
    )

    agent = CodeExplorerAgent(
        tools=tools,
        api_key=api_key,
        config=config,
    )

    # Example questions
    questions = [
        "How does user authentication work in this codebase?",
        "Where is the checkout process implemented?",
        "Show me the connection between the ProductCard component and the database",
        "What APIs does the CartButton component call?",
    ]

    # Run agent on each question
    for i, question in enumerate(questions, 1):
        print(f"\n{'='*80}")
        print(f"Question {i}: {question}")
        print('='*80)

        result = await agent.explore(
            question=question,
            context={
                'repositories': ['your-repo-name'],
                'scope': 'hybrid',
            }
        )

        print(f"\nSuccess: {result['success']}")
        print(f"Iterations: {result['iterations']}")
        print(f"Tools used: {', '.join(result['tool_calls'])}")
        print(f"\nAnswer:\n{result['answer']}")

        if not result.get('complete', False):
            print("\n⚠️ Warning: Answer may be incomplete")

    # Cleanup
    neo4j.close()


if __name__ == '__main__':
    print("Code Explorer Agent Example")
    print("="*80)
    asyncio.run(main())
