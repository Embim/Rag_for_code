"""
Mermaid Diagram Generator for Code RAG.

Generates visual diagrams showing:
- Code relationships (calls, imports, inherits)
- Data flow (UI -> API -> Database)
- Component dependencies

Uses Mermaid.js for rendering diagrams as images.
"""

import io
import base64
import requests
from typing import List, Dict, Any, Optional

from ..code_rag.retrieval import CodeRetriever, SearchStrategy
from ..logger import get_logger


logger = get_logger(__name__)


class MermaidDiagramGenerator:
    """
    Generates Mermaid diagrams for code visualization.

    Diagram types:
    - Flowchart: Shows execution flow
    - Sequence: Shows interactions over time
    - Graph: Shows relationships
    """

    # Mermaid renderer service (public API)
    MERMAID_INK_URL = "https://mermaid.ink/img/"

    def __init__(self, neo4j_client):
        """
        Initialize diagram generator.

        Args:
            neo4j_client: Neo4j client for graph queries
        """
        self.neo4j = neo4j_client

    async def generate_diagram(self, query: str) -> bytes:
        """
        Generate diagram based on query.

        Args:
            query: What to visualize

        Returns:
            PNG image bytes
        """
        # Determine diagram type from query
        query_lower = query.lower()

        if 'поток' in query_lower or 'flow' in query_lower:
            diagram_code = await self._generate_flow_diagram(query)
        elif 'последовательность' in query_lower or 'sequence' in query_lower:
            diagram_code = await self._generate_sequence_diagram(query)
        else:
            diagram_code = await self._generate_graph_diagram(query)

        # Render diagram
        return self._render_diagram(diagram_code)

    async def _generate_flow_diagram(self, query: str) -> str:
        """
        Generate flowchart diagram.

        Shows: Component -> API -> Endpoint -> Model
        """
        # Extract key entity from query
        # For MVP, use simple keyword extraction
        entity = self._extract_entity(query)

        # Query graph for flow
        cypher = """
        MATCH path = (start:Component {name: $entity})
            -[:SENDS_REQUEST_TO|HANDLES_REQUEST|CALLS|USES_MODEL*1..4]->
            (end:Model)
        RETURN path
        LIMIT 1
        """

        try:
            result = self.neo4j.execute_cypher(cypher, parameters={'entity': entity})

            if not result:
                # Fallback: simple diagram
                return self._generate_simple_diagram(entity)

            # Build Mermaid flowchart from path
            record = list(result)[0]
            path = record['path']

            nodes = []
            edges = []

            for i, node in enumerate(path.nodes):
                node_id = f"N{i}"
                node_name = node.get('name', 'Unknown')
                node_type = node.get('type', 'Unknown')

                # Format node based on type
                if node_type == 'Component':
                    nodes.append(f'{node_id}["{node_name}<br/><i>Component</i>"]')
                elif node_type == 'Endpoint':
                    nodes.append(f'{node_id}{{"{node_name}<br/><i>Endpoint</i>"}}')
                elif node_type == 'Model':
                    nodes.append(f'{node_id}[("{node_name}<br/><i>Model</i>")]')
                else:
                    nodes.append(f'{node_id}["{node_name}<br/><i>{node_type}</i>"]')

            for i, rel in enumerate(path.relationships):
                rel_type = type(rel).__name__
                edges.append(f'N{i} -->|{rel_type}| N{i+1}')

            # Build Mermaid code
            mermaid_code = "flowchart LR\n"
            mermaid_code += "    " + "\n    ".join(nodes) + "\n"
            mermaid_code += "    " + "\n    ".join(edges)

            return mermaid_code

        except Exception as e:
            logger.error(f"Flow diagram generation failed: {e}")
            return self._generate_simple_diagram(entity)

    async def _generate_sequence_diagram(self, query: str) -> str:
        """Generate sequence diagram."""
        entity = self._extract_entity(query)

        # Simplified sequence for MVP
        return f"""
sequenceDiagram
    actor User
    participant Component as {entity}
    participant API
    participant Backend
    participant Database

    User->>Component: Interaction
    Component->>API: Request
    API->>Backend: Process
    Backend->>Database: Query
    Database-->>Backend: Data
    Backend-->>API: Response
    API-->>Component: Update
    Component-->>User: Display
        """

    async def _generate_graph_diagram(self, query: str) -> str:
        """Generate relationship graph."""
        entity = self._extract_entity(query)

        # Query immediate neighbors
        cypher = """
        MATCH (center {name: $entity})-[r]-(neighbor)
        RETURN center, type(r) as rel_type, neighbor
        LIMIT 10
        """

        try:
            result = self.neo4j.execute_cypher(cypher, parameters={'entity': entity})

            nodes = {}
            edges = []

            for record in result:
                center = record['center']
                neighbor = record['neighbor']
                rel_type = record['rel_type']

                # Add center node
                if center['name'] not in nodes:
                    nodes[center['name']] = center.get('type', 'Unknown')

                # Add neighbor node
                if neighbor['name'] not in nodes:
                    nodes[neighbor['name']] = neighbor.get('type', 'Unknown')

                # Add edge
                edges.append(f'{center["name"]} -->|{rel_type}| {neighbor["name"]}')

            # Build Mermaid graph
            mermaid_code = "graph TD\n"

            for name, node_type in nodes.items():
                safe_name = name.replace(' ', '_').replace('-', '_')
                mermaid_code += f'    {safe_name}["{name}<br/><i>{node_type}</i>"]\n'

            for edge in edges:
                # Make names safe
                safe_edge = edge.replace(' ', '_').replace('-', '_')
                mermaid_code += f'    {safe_edge}\n'

            return mermaid_code

        except Exception as e:
            logger.error(f"Graph diagram generation failed: {e}")
            return self._generate_simple_diagram(entity)

    def _generate_simple_diagram(self, entity: str) -> str:
        """Generate simple fallback diagram."""
        return f"""
flowchart LR
    A["{entity}"] --> B["Backend API"]
    B --> C["Database"]
    C --> B
    B --> A

    style A fill:#e1f5ff
    style B fill:#fff3e0
    style C fill:#f3e5f5
        """

    def _extract_entity(self, query: str) -> str:
        """Extract main entity from query."""
        # Simple keyword extraction
        # TODO: Improve with NLP or LLM

        # Remove common words
        stop_words = {
            'как', 'работает', 'покажи', 'связи', 'для', 'где', 'что',
            'how', 'does', 'show', 'connections', 'for', 'where', 'what',
            'поток', 'данных', 'flow', 'data'
        }

        words = query.split()
        entity_words = [w for w in words if w.lower() not in stop_words]

        if entity_words:
            # Try to capitalize (assume PascalCase for components)
            entity = ''.join(w.capitalize() for w in entity_words)
            return entity

        return "Unknown"

    # ========================================================================
    # Public API methods (for REST API endpoints)
    # ========================================================================

    def generate_sequence_diagram(
        self,
        entities: List[Dict[str, Any]],
        title: Optional[str] = None
    ) -> str:
        """
        Generate sequence diagram from entities.

        Args:
            entities: List of entity dicts with 'name', 'type', 'relationships'
            title: Optional diagram title

        Returns:
            Mermaid diagram code
        """
        diagram_title = title or "Sequence Diagram"
        lines = [f"sequenceDiagram"]
        lines.append(f"    title {diagram_title}")

        # Extract participants (unique entities)
        participants = set()
        for entity in entities:
            participants.add(entity.get('name', 'Unknown'))
            # Add related entities from relationships
            for rel in entity.get('relationships', []):
                if 'target' in rel:
                    participants.add(rel['target'])

        # Add participant declarations
        for p in sorted(participants):
            lines.append(f"    participant {p}")

        # Add interactions based on relationships
        for entity in entities:
            source = entity.get('name', 'Unknown')
            for rel in entity.get('relationships', []):
                target = rel.get('target', '')
                rel_type = rel.get('type', 'CALLS')
                if target and rel_type in ['CALLS', 'USES']:
                    lines.append(f"    {source}->>{target}: {rel_type}")

        return '\n'.join(lines)

    def generate_component_diagram(
        self,
        entities: List[Dict[str, Any]],
        title: Optional[str] = None
    ) -> str:
        """
        Generate component/class diagram from entities.

        Args:
            entities: List of entity dicts
            title: Optional diagram title

        Returns:
            Mermaid diagram code
        """
        diagram_title = title or "Component Diagram"
        lines = [f"classDiagram"]
        lines.append(f"    %% {diagram_title}")

        for entity in entities:
            name = entity.get('name', 'Unknown')
            entity_type = entity.get('type', 'Class')

            # Add class declaration
            if entity_type == 'Class':
                lines.append(f"    class {name}")
                # Add methods if available
                if 'methods' in entity:
                    for method in entity['methods'][:5]:  # Limit to 5
                        lines.append(f"        +{method}()")

            # Add relationships
            for rel in entity.get('relationships', []):
                target = rel.get('target', '')
                rel_type = rel.get('type', 'USES')
                if target:
                    if rel_type == 'INHERITS':
                        lines.append(f"    {target} <|-- {name}")
                    elif rel_type == 'USES':
                        lines.append(f"    {name} --> {target}")
                    elif rel_type == 'IMPORTS':
                        lines.append(f"    {name} ..> {target}")

        return '\n'.join(lines)

    def generate_er_diagram(
        self,
        entities: List[Dict[str, Any]],
        title: Optional[str] = None
    ) -> str:
        """
        Generate ER diagram from model entities.

        Args:
            entities: List of model entities
            title: Optional diagram title

        Returns:
            Mermaid diagram code
        """
        diagram_title = title or "Entity Relationship Diagram"
        lines = ["erDiagram"]
        lines.append(f"    %% {diagram_title}")

        for entity in entities:
            name = entity.get('name', 'Unknown')

            # Add entity with fields
            if entity.get('type') == 'Model':
                fields = entity.get('fields', [])
                if fields:
                    lines.append(f"    {name} {{")
                    for field in fields[:10]:  # Limit to 10 fields
                        field_name = field.get('name', 'field')
                        field_type = field.get('type', 'string')
                        lines.append(f"        {field_type} {field_name}")
                    lines.append(f"    }}")

            # Add relationships
            for rel in entity.get('relationships', []):
                target = rel.get('target', '')
                rel_type = rel.get('type', 'USES')
                if target and rel_type in ['USES_MODEL', 'REFERENCES']:
                    lines.append(f"    {name} ||--o{{ {target} : has")

        return '\n'.join(lines)

    def generate_flowchart(
        self,
        entities: List[Dict[str, Any]],
        title: Optional[str] = None
    ) -> str:
        """
        Generate flowchart from entities.

        Args:
            entities: List of entities
            title: Optional diagram title

        Returns:
            Mermaid diagram code
        """
        diagram_title = title or "Flowchart"
        lines = ["flowchart TD"]
        lines.append(f"    %% {diagram_title}")

        # Generate nodes for each entity
        for i, entity in enumerate(entities):
            name = entity.get('name', f'Entity{i}')
            entity_type = entity.get('type', 'Unknown')
            node_id = f"N{i}"

            # Different shapes based on type
            if entity_type == 'Function':
                lines.append(f"    {node_id}[{name}]")
            elif entity_type == 'Class':
                lines.append(f"    {node_id}[/{name}/]")
            elif entity_type == 'Model':
                lines.append(f"    {node_id}[({name})]")
            else:
                lines.append(f"    {node_id}[{name}]")

        # Add relationships as edges
        for i, entity in enumerate(entities):
            node_id = f"N{i}"
            for rel in entity.get('relationships', []):
                target = rel.get('target', '')
                rel_type = rel.get('type', 'CALLS')
                # Find target index
                target_idx = next((j for j, e in enumerate(entities) if e.get('name') == target), None)
                if target_idx is not None:
                    target_id = f"N{target_idx}"
                    lines.append(f"    {node_id} --> |{rel_type}| {target_id}")

        return '\n'.join(lines)

    def _render_diagram(self, mermaid_code: str) -> bytes:
        """
        Render Mermaid diagram to PNG using mermaid.ink API.

        Args:
            mermaid_code: Mermaid diagram code

        Returns:
            PNG image bytes
        """
        try:
            # Encode diagram for URL
            encoded = base64.urlsafe_b64encode(mermaid_code.encode('utf-8')).decode('utf-8')

            # Request image from mermaid.ink
            url = f"{self.MERMAID_INK_URL}{encoded}"

            response = requests.get(url, timeout=10)
            response.raise_for_status()

            return response.content

        except Exception as e:
            logger.error(f"Diagram rendering failed: {e}")

            # Return placeholder image
            return self._generate_placeholder_image(str(e))

    def render_to_url(self, mermaid_code: str) -> str:
        """
        Generate URL for rendering Mermaid diagram via mermaid.ink.

        Args:
            mermaid_code: Mermaid diagram code

        Returns:
            URL to rendered diagram image
        """
        try:
            # Encode diagram for URL
            encoded = base64.urlsafe_b64encode(mermaid_code.encode('utf-8')).decode('utf-8')
            return f"{self.MERMAID_INK_URL}{encoded}"
        except Exception as e:
            logger.error(f"URL generation failed: {e}")
            return ""

    def _generate_placeholder_image(self, error_msg: str) -> bytes:
        """Generate placeholder image on error."""
        # For MVP, return empty bytes
        # TODO: Generate actual error image with PIL
        logger.warning(f"Returning empty image due to error: {error_msg}")
        return b''
