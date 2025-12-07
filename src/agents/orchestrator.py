"""
Query Orchestrator.

Classifies incoming questions and routes them to the appropriate
agent or component. Coordinates multiple agents for hybrid questions.

Question types:
- Document: Questions about text documents (regulations, documentation)
- Code: Questions about code implementation
- Visual: Questions requiring UI visualization
- Hybrid: Complex questions requiring multiple approaches
"""

import os
from typing import Dict, Any, Optional, AsyncIterator
from enum import Enum
import asyncio
import json

from openai import AsyncOpenAI

from .code_explorer import CodeExplorerAgent
from .visual_guide_agent import VisualGuideAgent
from ..logger import get_logger


logger = get_logger(__name__)


class QuestionType(str, Enum):
    """Types of questions the system can handle."""
    DOCUMENT = "document"
    CODE = "code"
    VISUAL = "visual"
    HYBRID = "hybrid"


class QueryOrchestrator:
    """
    Orchestrates question answering across different agents and components.

    Workflow:
    1. Classify question type
    2. Route to appropriate handler(s)
    3. Execute agents (sequential or parallel)
    4. Aggregate results
    5. Stream progress to user
    """

    # Classification prompt
    CLASSIFICATION_PROMPT = """Classify the following question into one of these categories:

1. DOCUMENT - Question about text documents, regulations, documentation, policies
   Examples: "What are the requirements for loan approval?", "Explain the refund policy"

2. CODE - Question about code implementation, functions, architecture
   Examples: "How is authentication implemented?", "Where is the checkout process?"

3. VISUAL - Question requiring UI screenshots or visual demonstration
   Examples: "Show me how to add a product", "Where is the login button?"

4. HYBRID - Complex question requiring multiple approaches
   Examples: "Explain the complete checkout flow from UI to database", "Show me how payment processing works"

Respond in JSON format:
{
  "type": "DOCUMENT|CODE|VISUAL|HYBRID",
  "confidence": 0.0-1.0,
  "reasoning": "why you chose this classification"
}

Question: {question}"""

    def __init__(
        self,
        code_explorer: CodeExplorerAgent,
        api_key: str,
        model: Optional[str] = None,
        api_base: str = "https://openrouter.ai/api/v1",
        document_rag_pipeline=None,
        visual_agent: Optional[VisualGuideAgent] = None
    ):
        """
        Initialize orchestrator.

        Args:
            code_explorer: Code Explorer Agent instance
            api_key: API key for classification LLM
            model: Model for classification (reads from ORCHESTRATOR_MODEL env if None)
            api_base: API base URL
            document_rag_pipeline: Optional RAGPipeline for document questions
            visual_agent: Optional Visual Guide Agent for visual questions
        """
        self.code_explorer = code_explorer
        self.llm = AsyncOpenAI(api_key=api_key, base_url=api_base)
        self.model = model or os.getenv("ORCHESTRATOR_MODEL", "deepseek/deepseek-r1:free")

        # Document RAG pipeline (from src/retrieval.py)
        self.document_rag = document_rag_pipeline
        self.visual_agent = visual_agent

    async def answer_question(
        self,
        question: str,
        context: Optional[Dict[str, Any]] = None,
        stream: bool = False
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Answer question by routing to appropriate agents.

        Args:
            question: User's question
            context: Optional context (repositories, etc.)
            stream: Whether to stream progress

        Returns:
            Dict with answer and metadata
        """
        logger.info(f"Orchestrator processing: {question}")

        # Step 1: Classify question
        if stream:
            yield {"type": "status", "message": "Classifying question..."}

        classification = await self._classify_question(question)
        question_type = classification['type']
        logger.info(f"Question classified as: {question_type} (confidence: {classification['confidence']:.2f})")

        # Step 2: Route to appropriate handler
        if stream:
            yield {"type": "classification", "data": classification}

        # Route to appropriate handler and collect result
        result = None
        if question_type == QuestionType.CODE:
            # Pure code question - use Code Explorer
            async for item in self._handle_code_question(question, context, stream):
                if stream and item.get("type") == "status":
                    yield item  # Forward status updates
                result = item

        elif question_type == QuestionType.DOCUMENT:
            # Document question - use existing RAG pipeline
            async for item in self._handle_document_question(question, context, stream):
                if stream and item.get("type") == "status":
                    yield item  # Forward status updates
                result = item

        elif question_type == QuestionType.VISUAL:
            # Visual question - needs Visual Agent (TODO: Phase 6.2)
            async for item in self._handle_visual_question(question, context, stream):
                if stream and item.get("type") == "status":
                    yield item  # Forward status updates
                result = item

        elif question_type == QuestionType.HYBRID:
            # Hybrid - use multiple agents
            async for item in self._handle_hybrid_question(question, context, stream):
                if stream and item.get("type") == "status":
                    yield item  # Forward status updates
                result = item

        else:
            result = {
                'success': False,
                'answer': f"Unknown question type: {question_type}",
                'question_type': question_type,
            }

        # Add classification metadata
        result['classification'] = classification

        # Always yield the final result
        yield {"type": "final" if stream else "result", "data": result}

    async def _classify_question(self, question: str) -> Dict[str, Any]:
        """
        Classify question type using LLM.

        Returns:
            Dict with type, confidence, reasoning
        """
        content = None  # Initialize to avoid UnboundLocalError
        try:
            prompt = self.CLASSIFICATION_PROMPT.format(question=question)

            response = await self.llm.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=1024,
            )

            content = response.choices[0].message.content

            # Extract JSON from response (may be in markdown block or have extra text)
            # Try to find JSON object in the response
            import re

            # Remove markdown code blocks if present
            content = re.sub(r'```json\s*', '', content)
            content = re.sub(r'```\s*', '', content)
            content = content.strip()

            # Try to parse JSON directly first
            try:
                classification = json.loads(content)
            except json.JSONDecodeError:
                # If that fails, try to find and extract JSON object
                # Use a more robust regex that handles nested objects
                json_match = re.search(r'\{(?:[^{}]|(?:\{[^{}]*\}))*\}', content, re.DOTALL)
                if json_match:
                    content = json_match.group(0)
                    classification = json.loads(content)
                else:
                    raise json.JSONDecodeError("No valid JSON found", content, 0)

            # Validate
            if classification.get('type') not in [t.value for t in QuestionType]:
                logger.warning(f"Invalid question type: {classification.get('type')}, defaulting to CODE")
                classification['type'] = QuestionType.CODE.value
                classification['confidence'] = classification.get('confidence', 0.6)

            return classification

        except json.JSONDecodeError as e:
            content_preview = content[:200] if content else "N/A"
            logger.error(f"JSON parsing failed: {e}. Content: {content_preview}")
            # Fallback: use keyword-based classification
            return self._fallback_classification(question)
        except Exception as e:
            content_preview = content[:200] if content else "N/A"
            logger.error(f"Classification failed: {e}. Content: {content_preview}")
            # Fallback: use keyword-based classification
            return self._fallback_classification(question)

    def _fallback_classification(self, question: str) -> Dict[str, Any]:
        """Keyword-based fallback classification."""
        question_lower = question.lower()

        # Document keywords
        doc_keywords = ['документ', 'регламент', 'правил', 'инструкц', 'политик',
                        'document', 'regulation', 'policy', 'guideline']
        # Code keywords
        code_keywords = ['код', 'функци', 'класс', 'компонент', 'api', 'реализ',
                         'code', 'function', 'class', 'component', 'implement', 'how does']
        # Visual keywords
        visual_keywords = ['покажи', 'экран', 'кнопк', 'форм', 'ui', 'интерфейс',
                          'show', 'screen', 'button', 'form', 'interface', 'click']

        doc_score = sum(1 for kw in doc_keywords if kw in question_lower)
        code_score = sum(1 for kw in code_keywords if kw in question_lower)
        visual_score = sum(1 for kw in visual_keywords if kw in question_lower)

        # Hybrid if multiple high scores
        if sum(x > 0 for x in [doc_score, code_score, visual_score]) >= 2:
            return {
                'type': QuestionType.HYBRID,
                'confidence': 0.7,
                'reasoning': 'Multiple keyword categories detected (fallback classification)'
            }

        # Highest score
        if code_score >= doc_score and code_score >= visual_score:
            return {
                'type': QuestionType.CODE,
                'confidence': 0.6,
                'reasoning': 'Code keywords detected (fallback classification)'
            }
        elif doc_score > visual_score:
            return {
                'type': QuestionType.DOCUMENT,
                'confidence': 0.6,
                'reasoning': 'Document keywords detected (fallback classification)'
            }
        else:
            return {
                'type': QuestionType.VISUAL,
                'confidence': 0.6,
                'reasoning': 'Visual keywords detected (fallback classification)'
            }

    async def _handle_code_question(
        self,
        question: str,
        context: Optional[Dict[str, Any]],
        stream: bool
    ) -> AsyncIterator[Dict[str, Any]]:
        """Handle code question using Code Explorer Agent."""
        if stream:
            yield {"type": "status", "message": "Exploring codebase..."}

        # Extract detail_level from context
        detail_level = context.get('detail_level') if context else None

        result = await self.code_explorer.explore(
            question=question,
            context=context,
            detail_level=detail_level
        )

        yield {
            'success': result['success'],
            'answer': result['answer'],
            'question_type': QuestionType.CODE,
            'agent_used': 'code_explorer',
            'sources': result.get('sources', []),
            'tool_calls': result.get('tool_calls', []),
            'iterations': result.get('iterations', 0),
        }

    async def _handle_document_question(
        self,
        question: str,
        context: Optional[Dict[str, Any]],
        stream: bool
    ) -> AsyncIterator[Dict[str, Any]]:
        """Handle document question using existing RAG pipeline."""
        if stream:
            yield {"type": "status", "message": "Searching documents..."}

        # Check if Document RAG is available
        if self.document_rag is None:
            yield {
                'success': False,
                'answer': "📄 **Document RAG not configured.**\n\n"
                         "To enable document search:\n"
                         "1. Index documents using `main_pipeline.py build`\n"
                         "2. Restart API server with document index available\n\n"
                         "For code-related questions, use /search or /ask with CODE questions.",
                'question_type': QuestionType.DOCUMENT,
                'agent_used': 'document_rag_unavailable',
            }
            return

        try:
            # Use Document RAG pipeline (src/retrieval.py RAGPipeline)
            if stream:
                yield {"type": "status", "message": "Searching knowledge base..."}

            # Run search (RAGPipeline.search returns dict with results)
            search_result = self.document_rag.search(question)

            # Format answer from results
            if search_result and 'top_docs' in search_result and search_result['top_docs']:
                # Build answer from top documents
                docs = search_result['top_docs']
                answer_parts = [f"📄 **Found {len(docs)} relevant documents:**\n"]

                for i, doc in enumerate(docs[:5], 1):
                    text = doc.get('text', doc.get('clean_text', ''))[:500]
                    source = doc.get('source', doc.get('web_id', 'Unknown'))
                    score = doc.get('rerank_score', doc.get('retrieval_score', 0))

                    answer_parts.append(f"\n**{i}. {source}** (score: {score:.2f})")
                    answer_parts.append(f"```\n{text}...\n```")

                yield {
                    'success': True,
                    'answer': '\n'.join(answer_parts),
                    'question_type': QuestionType.DOCUMENT,
                    'agent_used': 'document_rag',
                    'sources': docs[:5],
                }
            else:
                yield {
                    'success': False,
                    'answer': f"No documents found for: '{question}'\n\n"
                             "Try rephrasing your question or check that documents are indexed.",
                    'question_type': QuestionType.DOCUMENT,
                    'agent_used': 'document_rag',
                }

        except Exception as e:
            logger.error(f"Document RAG error: {e}")
            yield {
                'success': False,
                'answer': f"Error searching documents: {str(e)}",
                'question_type': QuestionType.DOCUMENT,
                'agent_used': 'document_rag',
                'error': str(e),
            }

    async def _handle_visual_question(
        self,
        question: str,
        context: Optional[Dict[str, Any]],
        stream: bool
    ) -> AsyncIterator[Dict[str, Any]]:
        """Handle visual question using Visual Guide Agent."""
        if stream:
            yield {"type": "status", "message": "Generating visualization..."}

        # Check if Visual Agent is available
        if self.visual_agent is None:
            yield {
                'success': False,
                'answer': "📊 **Visual Guide Agent not configured.**\n\n"
                         "Visual diagrams require the Visual Agent to be enabled.\n"
                         "For now, you can use the /visualize command in the Telegram bot "
                         "to generate Mermaid diagrams.",
                'question_type': QuestionType.VISUAL,
                'agent_used': 'visual_agent_unavailable',
            }
            return

        try:
            # Use Visual Guide Agent to create visualization
            result = await self.visual_agent.create_visualization(question, context)

            yield {
                'success': result['success'],
                'answer': result['answer'],
                'question_type': QuestionType.VISUAL,
                'agent_used': 'visual_guide_agent',
                'sources': result.get('sources', []),
                'diagram_code': result.get('diagram_code', ''),
                'diagram_url': result.get('diagram_url', ''),
                'diagram_type': result.get('diagram_type', 'unknown'),
                'complete': result.get('complete', True),
            }

        except Exception as e:
            logger.error(f"Visual Agent error: {e}", exc_info=True)
            yield {
                'success': False,
                'answer': f"❌ **Visualization failed:** {str(e)}\n\n"
                         "Try rephrasing your question or being more specific.",
                'question_type': QuestionType.VISUAL,
                'agent_used': 'visual_guide_agent',
                'error': str(e),
            }

    async def _handle_hybrid_question(
        self,
        question: str,
        context: Optional[Dict[str, Any]],
        stream: bool
    ) -> AsyncIterator[Dict[str, Any]]:
        """Handle hybrid question using multiple agents."""
        if stream:
            yield {"type": "status", "message": "Complex question detected, using multiple agents..."}

        # For hybrid questions, run Code Explorer
        # In future, could run multiple agents in parallel and aggregate
        code_result = None
        async for result in self._handle_code_question(question, context, stream=False):
            code_result = result

        # Future: Add document search, visual generation
        # For now, just yield code result with note
        code_result['answer'] += "\n\n💡 Note: This is a hybrid question. " \
                                  "Additional agents (document search, visualization) " \
                                  "will be integrated in future updates."

        code_result['question_type'] = QuestionType.HYBRID

        yield code_result


async def stream_answer(orchestrator: QueryOrchestrator, question: str, context: Optional[Dict] = None):
    """
    Example of streaming usage.

    Yields progress updates as the orchestrator works.
    """
    async for event in orchestrator.answer_question(question, context, stream=True):
        event_type = event['type']

        if event_type == 'status':
            print(f"⏳ {event['message']}")

        elif event_type == 'classification':
            classification = event['data']
            print(f"📋 Question type: {classification['type']} (confidence: {classification['confidence']:.0%})")

        elif event_type == 'final':
            result = event['data']
            print(f"\n✅ Answer:\n{result['answer']}")
            return result
