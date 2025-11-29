"""
Visualization endpoint.

Provides:
- POST /visualize - Generate Mermaid diagrams
"""

from typing import List
from fastapi import APIRouter, Depends, HTTPException, status

from ..models import VisualizeRequest, VisualizeResponse
from ..dependencies import get_visualizer, get_neo4j
from ..auth import require_auth, APIKey
from ...telegram_bot.visualizer import MermaidDiagramGenerator
from ...code_rag.graph import Neo4jClient
from ...logger import get_logger


logger = get_logger(__name__)

router = APIRouter(prefix="/api", tags=["visualization"])


# ============================================================================
# Endpoints
# ============================================================================

@router.post(
    "/visualize",
    response_model=VisualizeResponse,
    summary="Generate diagram",
    description="Generate Mermaid diagram showing code relationships and flows"
)
async def visualize(
    request: VisualizeRequest,
    visualizer: MermaidDiagramGenerator = Depends(get_visualizer),
    neo4j: Neo4jClient = Depends(get_neo4j),
    current_key: APIKey = Depends(require_auth),
):
    """
    Generate visualization diagram.

    **Supported diagram types:**
    - `sequence`: Sequence diagram showing call flow
    - `component`: Component dependency diagram
    - `er`: Entity-relationship diagram (for Django models)
    - `flow`: Flowchart showing logic flow

    **Example:**
    ```json
    {
        "diagram_type": "sequence",
        "entities": [
            "repo:frontend/CartButton.tsx:CartButton",
            "repo:backend/api/cart.py:add_to_cart"
        ],
        "title": "Add to Cart Flow"
    }
    ```

    Returns Mermaid diagram code and optional rendered image URL.
    """
    try:
        # Fetch entity details from Neo4j
        entities = []
        for entity_id in request.entities:
            query = """
            MATCH (n:GraphNode {id: $id})
            RETURN n
            """
            results = neo4j.execute_cypher(query, {"id": entity_id})

            if not results:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Entity not found: {entity_id}"
                )

            entities.append(results[0]['n'])

        # Generate diagram based on type
        mermaid_code = ""
        entities_included = len(entities)

        if request.diagram_type == "sequence":
            # Generate sequence diagram
            mermaid_code = visualizer.generate_sequence_diagram(
                entities=entities,
                title=request.title
            )

        elif request.diagram_type == "component":
            # Generate component diagram
            mermaid_code = visualizer.generate_component_diagram(
                entities=entities,
                title=request.title
            )

        elif request.diagram_type == "er":
            # Generate ER diagram (for models)
            mermaid_code = visualizer.generate_er_diagram(
                entities=entities,
                title=request.title
            )

        elif request.diagram_type == "flow":
            # Generate flowchart
            mermaid_code = visualizer.generate_flowchart(
                entities=entities,
                title=request.title
            )

        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported diagram type: {request.diagram_type}"
            )

        # Generate image URL via mermaid.ink
        image_url = visualizer.render_to_url(mermaid_code)

        return VisualizeResponse(
            diagram_type=request.diagram_type,
            mermaid_code=mermaid_code,
            entities_included=entities_included,
            image_url=image_url
        )

    except HTTPException:
        # Re-raise HTTP exceptions
        raise

    except Exception as e:
        logger.error(f"Visualization failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Diagram generation failed: {str(e)}"
        )
