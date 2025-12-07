"""
Visual Guide Agent.

An LLM-powered agent that creates visual diagrams and guides to explain code flows.
This agent analyzes questions requiring visualization and generates appropriate diagrams.

Workflow:
1. Receive visual question from user
2. Determine diagram type (sequence, flowchart, component, ER)
3. Use Code Explorer to find relevant entities
4. Generate diagram using MermaidDiagramGenerator
5. Synthesize answer with diagram and explanation
"""

import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import json

from openai import AsyncOpenAI
from typing import TYPE_CHECKING

from .code_explorer import CodeExplorerAgent
from ..config.agent import AgentConfig
from ..logger import get_logger

# Avoid circular import with telegram_bot
if TYPE_CHECKING:
    from ..telegram_bot.visualizer import MermaidDiagramGenerator


logger = get_logger(__name__)


@dataclass
class VisualizationRequest:
    """Parsed visualization request."""
    diagram_type: str  # 'sequence', 'component', 'er', 'flow'
    entities_needed: List[str]  # Entity names to find
    title: str  # Diagram title
    description: str  # What to show


class VisualGuideAgent:
    """
    LLM-powered agent for creating visual code guides.

    Uses Code Explorer to find entities, then generates diagrams
    using MermaidDiagramGenerator.
    """

    SYSTEM_PROMPT = """You are a Visual Guide Agent - an expert at creating visual explanations of code.

Your goal is to create diagrams that help developers understand code flows and architecture.

**Your workflow:**
1. Analyze the user's question to determine what type of diagram is needed
2. Identify which code entities need to be included
3. Plan the diagram structure
4. Return your plan in JSON format

**Diagram types:**
- **sequence**: Shows interaction flow over time (User -> Component -> API -> Database)
  Use for: "show how X works", "explain checkout flow", "trace the request"

- **component**: Shows component dependencies and relationships
  Use for: "show architecture", "how are components connected", "module structure"

- **er**: Entity-relationship diagram for data models
  Use for: "show database schema", "explain models", "data relationships"

- **flow**: Flowchart showing logic flow with decisions
  Use for: "explain algorithm", "show decision logic", "process flow"

**Guidelines:**
- For "show me..." or "how does..." questions → sequence diagram
- For "architecture" or "structure" questions → component diagram
- For "data" or "model" questions → ER diagram
- For "logic" or "algorithm" questions → flow diagram

Always respond in JSON format:
{
  "diagram_type": "sequence|component|er|flow",
  "entities_needed": ["EntityName1", "EntityName2", ...],
  "title": "Diagram Title",
  "description": "What this diagram will show",
  "search_query": "Query to find entities (e.g., 'checkout process', 'user authentication')"
}"""

    def __init__(
        self,
        code_explorer: CodeExplorerAgent,
        diagram_generator: 'MermaidDiagramGenerator',  # String annotation to avoid circular import
        api_key: str,
        config: Optional[AgentConfig] = None,
        api_base: str = "https://openrouter.ai/api/v1"
    ):
        """
        Initialize Visual Guide Agent.

        Args:
            code_explorer: Code Explorer Agent for finding entities
            diagram_generator: Mermaid diagram generator
            api_key: OpenRouter/OpenAI API key
            config: Agent configuration
            api_base: API base URL
        """
        self.code_explorer = code_explorer
        self.diagram_generator = diagram_generator
        self.config = config or AgentConfig()

        self.llm = AsyncOpenAI(
            api_key=api_key,
            base_url=api_base,
        )

    async def create_visualization(
        self,
        question: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create visualization for the question.

        Args:
            question: User's question requiring visualization
            context: Optional context

        Returns:
            Dict with:
            - answer: Text explanation
            - diagram_code: Mermaid code
            - diagram_url: URL to rendered image
            - diagram_type: Type of diagram
            - entities_used: List of entities included
            - success: Whether visualization succeeded
        """
        logger.info(f"Visual Guide Agent starting: {question}")
        start_time = time.time()

        try:
            # Step 1: Analyze question and plan diagram
            plan = await self._plan_visualization(question)
            logger.info(f"Planned diagram type: {plan['diagram_type']}")

            # Step 2: Find entities using Code Explorer
            logger.info(f"Searching for entities: {plan.get('search_query', plan['entities_needed'])}")

            search_query = plan.get('search_query', ' '.join(plan['entities_needed']))
            explorer_result = await self.code_explorer.explore(
                question=search_query,
                context=context
            )

            if not explorer_result.get('success'):
                return self._create_error_response(
                    "Failed to find relevant entities for visualization",
                    plan['diagram_type']
                )

            # Extract entities from explorer result
            # Explorer returns sources with entity info
            entities = self._extract_entities_from_explorer(explorer_result)

            if not entities:
                return self._create_error_response(
                    f"No entities found for: {search_query}",
                    plan['diagram_type']
                )

            logger.info(f"Found {len(entities)} entities for visualization")

            # Step 3: Generate diagram
            diagram_code = self._generate_diagram(
                plan['diagram_type'],
                entities,
                plan['title']
            )

            # Step 4: Render diagram to URL
            diagram_url = self.diagram_generator.render_to_url(diagram_code)

            # Step 5: Generate explanation
            explanation = await self._generate_explanation(
                question=question,
                diagram_type=plan['diagram_type'],
                entities=entities,
                explorer_answer=explorer_result.get('answer', '')
            )

            # Build final answer with diagram
            answer = self._format_answer(
                explanation=explanation,
                diagram_code=diagram_code,
                diagram_url=diagram_url,
                plan=plan
            )

            elapsed_ms = (time.time() - start_time) * 1000

            return {
                'success': True,
                'answer': answer,
                'diagram_code': diagram_code,
                'diagram_url': diagram_url,
                'diagram_type': plan['diagram_type'],
                'entities_used': [e.get('name', 'Unknown') for e in entities],
                'sources': entities,  # Include entities as sources
                'complete': True,
                'took_ms': elapsed_ms,
            }

        except Exception as e:
            logger.error(f"Visual Guide Agent failed: {e}", exc_info=True)
            return self._create_error_response(str(e), "unknown")

    async def _plan_visualization(self, question: str) -> Dict[str, Any]:
        """
        Plan what type of diagram to create and what entities to include.

        Returns:
            Dict with diagram_type, entities_needed, title, description
        """
        try:
            response = await self.llm.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": f"Plan visualization for: {question}"}
                ],
                temperature=0.0,
                max_tokens=1024,
            )

            content = response.choices[0].message.content

            # Parse JSON response
            import re
            # Remove markdown code blocks
            content = re.sub(r'```json\s*', '', content)
            content = re.sub(r'```\s*', '', content)
            content = content.strip()

            # Try to find JSON object
            json_match = re.search(r'\{[^{}]*"diagram_type"[^{}]*\}', content, re.DOTALL)
            if json_match:
                content = json_match.group(0)

            plan = json.loads(content)

            # Validate diagram type
            valid_types = ['sequence', 'component', 'er', 'flow']
            if plan['diagram_type'] not in valid_types:
                logger.warning(f"Invalid diagram type: {plan['diagram_type']}, defaulting to sequence")
                plan['diagram_type'] = 'sequence'

            return plan

        except Exception as e:
            logger.error(f"Planning failed: {e}")
            # Fallback: use keyword-based planning
            return self._fallback_planning(question)

    def _fallback_planning(self, question: str) -> Dict[str, Any]:
        """Keyword-based fallback planning."""
        question_lower = question.lower()

        # Determine diagram type from keywords
        if any(kw in question_lower for kw in ['flow', 'поток', 'trace', 'трейс', 'journey']):
            diagram_type = 'sequence'
        elif any(kw in question_lower for kw in ['architecture', 'архитектура', 'structure', 'структура', 'depends']):
            diagram_type = 'component'
        elif any(kw in question_lower for kw in ['model', 'модел', 'data', 'данн', 'schema', 'схема']):
            diagram_type = 'er'
        elif any(kw in question_lower for kw in ['logic', 'логик', 'algorithm', 'алгоритм', 'decision']):
            diagram_type = 'flow'
        else:
            diagram_type = 'sequence'  # Default

        return {
            'diagram_type': diagram_type,
            'entities_needed': [],
            'title': question[:50],
            'description': f"Visualization of {question}",
            'search_query': question
        }

    def _extract_entities_from_explorer(self, explorer_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract entity information from Code Explorer result."""
        entities = []

        # Explorer result may contain sources
        sources = explorer_result.get('sources', [])
        if sources:
            entities.extend(sources)

        # Also check tool_calls for entity information
        tool_calls = explorer_result.get('tool_calls', [])
        # We could parse tool results, but sources should be sufficient

        return entities

    def _generate_diagram(
        self,
        diagram_type: str,
        entities: List[Dict[str, Any]],
        title: str
    ) -> str:
        """
        Generate Mermaid diagram code.

        Args:
            diagram_type: Type of diagram
            entities: List of entities with metadata
            title: Diagram title

        Returns:
            Mermaid diagram code
        """
        try:
            if diagram_type == 'sequence':
                return self._generate_sequence_diagram(entities, title)
            elif diagram_type == 'component':
                return self._generate_component_diagram(entities, title)
            elif diagram_type == 'er':
                return self._generate_er_diagram(entities, title)
            elif diagram_type == 'flow':
                return self._generate_flow_diagram(entities, title)
            else:
                return self._generate_simple_diagram(entities, title)
        except Exception as e:
            logger.error(f"Diagram generation failed: {e}")
            return self._generate_simple_diagram(entities, title)

    def _generate_sequence_diagram(self, entities: List[Dict[str, Any]], title: str) -> str:
        """Generate sequence diagram showing flow over time."""
        lines = [f"sequenceDiagram"]
        lines.append(f"    title {title}")
        lines.append(f"    actor User")

        # Group entities by type and repository
        ui_entities = []
        api_entities = []
        backend_entities = []
        db_entities = []

        for e in entities:
            # Try different possible keys for entity type
            entity_type = e.get('entity_type') or e.get('node_type') or e.get('type', '')

            # Extract repository from entity_id if not in repository field
            # entity_id format: "repo:ui:path/to/file.jsx"
            repo = e.get('repository', '')
            entity_id = e.get('entity_id', '')
            file_path = e.get('file_path', '')

            if not repo and entity_id:
                if entity_id.startswith('repo:'):
                    parts = entity_id.split(':')
                    if len(parts) >= 2:
                        repo = parts[1]  # Extract "ui" or "***"

                # Also try to extract from entity_id path
                if not repo:
                    if 'ui.' in entity_id or '/ui/' in entity_id:
                        repo = 'ui'
                    elif 'api.' in entity_id or '/api/' in entity_id or '/backend/' in entity_id:
                        repo = 'api'

            # Extract file path from entity_id if not available
            if not file_path and entity_id and ':' in entity_id:
                # Format: repo:***:app/backend/file.py:Function
                parts = entity_id.split(':')
                if len(parts) >= 3:
                    file_path = parts[2]

            # UI layer: Components, Files from ui repos or with UI extensions
            if entity_type in ['Component', 'File'] and 'ui' in repo.lower():
                ui_entities.append(e)
            elif entity_type == 'File' and any(ext in e.get('name', '').lower() for ext in ['.jsx', '.tsx', '.vue']):
                ui_entities.append(e)
            # API layer: Endpoints, Routes
            elif entity_type in ['Endpoint', 'Route']:
                api_entities.append(e)
            # Backend layer: Functions, Methods, Classes from api/backend repos
            elif entity_type in ['Function', 'Method', 'Class'] and ('api' in repo.lower() or 'backend' in repo.lower()):
                backend_entities.append(e)
            # Database layer: Models
            elif entity_type == 'Model':
                db_entities.append(e)
            # GraphNode - categorize by name/path heuristics
            elif entity_type == 'GraphNode':
                name_lower = e.get('name', '').lower()
                if 'model' in name_lower or 'entity' in name_lower:
                    db_entities.append(e)
                elif 'endpoint' in name_lower or 'api' in name_lower or 'route' in name_lower:
                    api_entities.append(e)
                elif 'view' in file_path.lower() or 'endpoint' in file_path.lower():
                    api_entities.append(e)
                else:
                    # Default to backend
                    backend_entities.append(e)

        # Log entity categorization for debugging
        logger.info(f"Entity categorization: UI={len(ui_entities)}, API={len(api_entities)}, "
                   f"Backend={len(backend_entities)}, DB={len(db_entities)}")

        # If no entities categorized, add all to backend as fallback
        if not ui_entities and not api_entities and not backend_entities and not db_entities:
            logger.warning("No entities categorized, using all as backend")
            backend_entities = entities

        # Add participants with actual names (avoiding duplicates)
        participants = []
        seen_names = set()

        # Helper to add participant without duplicates
        def add_participant(layer, entity, limit_chars=30):
            name = entity.get('name', layer)
            # Clean name for display
            display_name = name.replace('.jsx', '').replace('.tsx', '').replace('.py', '')
            # Create safe identifier
            safe_name = display_name.replace('.', '_').replace('-', '_').replace('/', '_')[:limit_chars]

            # Avoid duplicates
            if safe_name in seen_names:
                return None

            seen_names.add(safe_name)
            participants.append((layer, safe_name, display_name))
            lines.append(f"    participant {safe_name} as {display_name}")
            return (layer, safe_name, display_name)

        # UI participants (limit to 2)
        for ui_entity in ui_entities[:2]:
            add_participant('UI', ui_entity, limit_chars=25)

        # API participants (limit to 1)
        for api_entity in api_entities[:1]:
            add_participant('API', api_entity, limit_chars=20)

        # Backend participants (limit to 5 if no API/UI found, otherwise 3)
        backend_limit = 5 if not ui_entities and not api_entities else 3
        for backend_entity in backend_entities[:backend_limit]:
            add_participant('Backend', backend_entity, limit_chars=35)

        # Database participants (limit to 1)
        for db_entity in db_entities[:1]:
            add_participant('DB', db_entity, limit_chars=20)

        # Generate flow with actual participant names
        lines.append(f"")
        if participants:
            ui_parts = [p for p in participants if p[0] == 'UI']
            api_parts = [p for p in participants if p[0] == 'API']
            backend_parts = [p for p in participants if p[0] == 'Backend']
            db_parts = [p for p in participants if p[0] == 'DB']

            # Determine entry point
            entry_point = None
            if ui_parts:
                entry_point = ui_parts[0]
                lines.append(f"    User->>+{entry_point[1]}: Initiate booking")
            elif api_parts:
                entry_point = api_parts[0]
                lines.append(f"    User->>+{entry_point[1]}: API Request")
            elif backend_parts:
                entry_point = backend_parts[0]
                lines.append(f"    User->>+{entry_point[1]}: Process")

            # UI -> API or Backend
            if ui_parts and api_parts:
                lines.append(f"    {ui_parts[0][1]}->>+{api_parts[0][1]}: Submit")
            elif ui_parts and backend_parts:
                lines.append(f"    {ui_parts[0][1]}->>+{backend_parts[0][1]}: Submit")

            # API -> Backend(s)
            if api_parts and backend_parts:
                for i, backend in enumerate(backend_parts[:3]):
                    # Use meaningful action based on function name
                    backend_name = backend[2].lower()
                    if 'validate' in backend_name:
                        action = "Validate"
                    elif 'book' in backend_name or 'create' in backend_name:
                        action = "Book trade"
                    elif 'save' in backend_name or 'persist' in backend_name:
                        action = "Save"
                    else:
                        action = "Process"
                    lines.append(f"    {api_parts[0][1]}->>+{backend[1]}: {action}")

            # Backend chaining (if multiple backend components and no API layer)
            elif not api_parts and len(backend_parts) > 1:
                # Create a proper flow between backend components
                for i in range(len(backend_parts) - 1):
                    # Infer action from target function name
                    target_name = backend_parts[i+1][2].lower()
                    if 'validate' in target_name:
                        action = "Validate"
                    elif 'book' in target_name or 'create' in target_name:
                        action = "Book"
                    elif 'save' in target_name or 'persist' in target_name or 'trade_book' in target_name:
                        action = "Save trade"
                    elif 'upload' in target_name:
                        action = "Upload"
                    else:
                        action = "Call"
                    lines.append(f"    {backend_parts[i][1]}->>+{backend_parts[i+1][1]}: {action}")

            # Backend -> Database
            if backend_parts and db_parts:
                lines.append(f"    {backend_parts[-1][1]}->>+{db_parts[0][1]}: Save")
                lines.append(f"    {db_parts[0][1]}-->>-{backend_parts[-1][1]}: Saved")

            # Return flow (reverse)
            # Backend -> API
            if backend_parts and api_parts:
                for backend in reversed(backend_parts[:3]):
                    lines.append(f"    {backend[1]}-->>-{api_parts[0][1]}: Result")

            # Backend chaining return (if no API)
            elif not api_parts and len(backend_parts) > 1:
                for i in range(len(backend_parts) - 1, 0, -1):
                    lines.append(f"    {backend_parts[i][1]}-->>-{backend_parts[i-1][1]}: Result")

            # API -> UI
            if api_parts and ui_parts:
                lines.append(f"    {api_parts[0][1]}-->>-{ui_parts[0][1]}: Success")
            elif backend_parts and ui_parts and not api_parts:
                lines.append(f"    {backend_parts[0][1]}-->>-{ui_parts[0][1]}: Success")

            # Final return to user
            if entry_point:
                lines.append(f"    {entry_point[1]}-->>-User: Complete")
        else:
            # Fallback: create simple sequential flow if no participants were added
            logger.warning("No flow generated, creating simple sequential diagram")
            if entities:
                lines.append(f"")
                prev_participant = "User"
                for i, entity in enumerate(entities[:5]):
                    name = entity.get('name', f'Step{i}')
                    safe_name = name.replace('.', '_').replace('-', '_')[:30]
                    lines.append(f"    participant {safe_name}")
                    lines.append(f"    {prev_participant}->>+{safe_name}: Process")
                    prev_participant = safe_name
                if prev_participant != "User":
                    lines.append(f"    {prev_participant}-->>-User: Complete")

        return '\n'.join(lines)

    def _generate_component_diagram(self, entities: List[Dict[str, Any]], title: str) -> str:
        """Generate component dependency diagram."""
        lines = [f"graph TD"]
        lines.append(f"    title[\"{title}\"]")

        # Create nodes for each entity
        for i, entity in enumerate(entities[:10]):  # Limit to 10 entities
            name = entity.get('name', f'Entity{i}')
            entity_type = entity.get('entity_type') or entity.get('node_type') or entity.get('type', 'Unknown')
            safe_name = name.replace(' ', '_').replace('-', '_').replace('.', '_')

            lines.append(f"    {safe_name}[\"{name}\\n<{entity_type}>\"]")

        # Add some connections (simplified)
        if len(entities) > 1:
            for i in range(len(entities) - 1):
                from_name = entities[i].get('name', '').replace(' ', '_').replace('-', '_').replace('.', '_')
                to_name = entities[i+1].get('name', '').replace(' ', '_').replace('-', '_').replace('.', '_')
                if from_name and to_name:
                    lines.append(f"    {from_name} --> {to_name}")

        return '\n'.join(lines)

    def _generate_er_diagram(self, entities: List[Dict[str, Any]], title: str) -> str:
        """Generate entity-relationship diagram."""
        lines = [f"erDiagram"]

        # Filter for models only
        models = [e for e in entities if e.get('entity_type') == 'Model' or e.get('node_type') == 'Model' or e.get('type') == 'Model']

        if not models:
            # No models, create generic ER
            lines.append(f"    Entity1 ||--o{{ Entity2 : has")
            lines.append(f"    Entity2 }}o--|| Entity3 : belongs_to")
        else:
            # Create ER from models
            for i, model in enumerate(models[:5]):
                name = model.get('name', f'Model{i}')
                lines.append(f"    {name} {{")
                lines.append(f"        int id PK")
                lines.append(f"        string name")
                lines.append(f"    }}")

        return '\n'.join(lines)

    def _generate_flow_diagram(self, entities: List[Dict[str, Any]], title: str) -> str:
        """Generate flowchart."""
        lines = [f"flowchart TD"]
        lines.append(f"    Start([Start]) --> Action1")

        # Add entities as steps
        for i, entity in enumerate(entities[:5]):
            name = entity.get('name', f'Step{i}')
            lines.append(f"    Action{i+1}[\"{name}\"]")
            if i < len(entities) - 1:
                lines.append(f"    Action{i+1} --> Action{i+2}")

        lines.append(f"    Action{len(entities)} --> End([End])")

        return '\n'.join(lines)

    def _generate_simple_diagram(self, entities: List[Dict[str, Any]], title: str) -> str:
        """Generate simple fallback diagram."""
        lines = [f"graph LR"]
        lines.append(f"    A[\"{title}\"]")

        for i, entity in enumerate(entities[:5]):
            name = entity.get('name', f'Entity{i}')
            lines.append(f"    A --> B{i}[\"{name}\"]")

        return '\n'.join(lines)

    async def _generate_explanation(
        self,
        question: str,
        diagram_type: str,
        entities: List[Dict[str, Any]],
        explorer_answer: str
    ) -> str:
        """Generate text explanation for the diagram."""
        entity_names = [e.get('name', 'Unknown') for e in entities[:5]]
        entity_list = ', '.join(entity_names)

        prompt = f"""Generate a brief explanation (2-3 sentences) for a {diagram_type} diagram.

Question: {question}
Entities included: {entity_list}

Code Explorer found: {explorer_answer[:300]}...

Explain what the diagram shows and how it answers the question."""

        try:
            response = await self.llm.chat.completions.create(
                model=self.config.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=512,
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"Explanation generation failed: {e}")
            return f"This {diagram_type} diagram shows the flow through {entity_list}."

    def _format_answer(
        self,
        explanation: str,
        diagram_code: str,
        diagram_url: str,
        plan: Dict[str, Any]
    ) -> str:
        """Format final answer with diagram."""
        answer_parts = [
            f"# {plan['title']}\n",
            f"{explanation}\n",
            f"\n## Diagram\n",
            f"**Type:** {plan['diagram_type']}\n",
            f"\n**Mermaid Code:**\n```mermaid\n{diagram_code}\n```\n",
        ]

        if diagram_url:
            answer_parts.append(f"\n**Rendered Diagram:** [View Image]({diagram_url})\n")

        answer_parts.append(f"\n{plan['description']}")

        return '\n'.join(answer_parts)

    def _create_error_response(self, error_msg: str, diagram_type: str) -> Dict[str, Any]:
        """Create error response."""
        return {
            'success': False,
            'answer': f"❌ **Visualization Failed**\n\n{error_msg}\n\n"
                     "Try rephrasing your question or being more specific about what you want to visualize.",
            'diagram_code': '',
            'diagram_url': '',
            'diagram_type': diagram_type,
            'entities_used': [],
            'sources': [],
            'complete': False,
        }
