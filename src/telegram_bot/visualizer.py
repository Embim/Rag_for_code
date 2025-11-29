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

    def _generate_placeholder_image(self, error_msg: str) -> bytes:
        """Generate placeholder image on error."""
        # For MVP, return empty bytes
        # TODO: Generate actual error image with PIL
        logger.warning(f"Returning empty image due to error: {error_msg}")
        return b''
