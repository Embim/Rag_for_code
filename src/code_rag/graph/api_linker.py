"""
API Linker - links frontend API calls to backend endpoints.

This is a critical component that establishes SENDS_REQUEST_TO relationships
between frontend components and backend endpoints.

Strategy (as discussed):
1. Primary: Use OpenAPI spec from FastAPI (/openapi.json)
2. Secondary: Parse Django REST Framework endpoints
3. Fallback: Heuristic URL matching with confidence scores
"""

import re
import json
from typing import List, Dict, Optional, Tuple
from urllib.parse import urlparse
from difflib import SequenceMatcher

from .models import GraphRelationship, RelationshipType, ComponentNode, EndpointNode
from ...logger import get_logger


logger = get_logger(__name__)


class APILinker:
    """
    Links frontend API calls to backend endpoints.

    Creates SENDS_REQUEST_TO relationships with confidence scores.
    """

    def __init__(self, use_openapi: bool = True):
        """
        Initialize API linker.

        Args:
            use_openapi: Whether to try using OpenAPI spec
        """
        self.use_openapi = use_openapi
        self.openapi_spec: Optional[Dict] = None

    def load_openapi_spec(self, spec_path_or_dict: any) -> bool:
        """
        Load OpenAPI specification.

        Args:
            spec_path_or_dict: Path to openapi.json or dict

        Returns:
            True if loaded successfully
        """
        try:
            if isinstance(spec_path_or_dict, dict):
                self.openapi_spec = spec_path_or_dict
            else:
                with open(spec_path_or_dict) as f:
                    self.openapi_spec = json.load(f)

            logger.info(f"Loaded OpenAPI spec with {len(self.openapi_spec.get('paths', {}))} paths")
            return True

        except Exception as e:
            logger.warning(f"Failed to load OpenAPI spec: {e}")
            return False

    def link_api_calls(
        self,
        component_nodes: List[ComponentNode],
        endpoint_nodes: List[EndpointNode]
    ) -> List[GraphRelationship]:
        """
        Link frontend API calls to backend endpoints.

        Args:
            component_nodes: List of component nodes with API calls
            endpoint_nodes: List of endpoint nodes

        Returns:
            List of SENDS_REQUEST_TO relationships
        """
        relationships = []

        # Build endpoint index for faster lookup
        endpoint_index = self._build_endpoint_index(endpoint_nodes)

        # Process each component
        for component in component_nodes:
            # Extract API calls from component metadata
            api_calls = component.properties.get('api_calls', [])

            if isinstance(api_calls, str):
                # Parse if stored as string
                try:
                    api_calls = json.loads(api_calls)
                except:
                    api_calls = []

            for api_call in api_calls:
                url = api_call.get('url')
                method = api_call.get('method', 'GET').upper()

                if not url:
                    continue

                # Normalize URL
                normalized_url = self._normalize_url(url)

                # Find matching endpoint
                matches = self._find_matching_endpoints(
                    normalized_url,
                    method,
                    endpoint_index
                )

                # Create relationships for matches above confidence threshold
                for endpoint_id, confidence in matches:
                    if confidence >= 0.7:  # Configurable threshold
                        relationships.append(GraphRelationship(
                            type=RelationshipType.SENDS_REQUEST_TO,
                            source_id=component.id,
                            target_id=endpoint_id,
                            confidence=confidence,
                            properties={
                                'url': url,
                                'method': method,
                                'matched_via': 'api_linker'
                            }
                        ))

        logger.info(f"Created {len(relationships)} API links")
        return relationships

    def _build_endpoint_index(
        self,
        endpoints: List[EndpointNode]
    ) -> Dict[str, List[Tuple[str, str, str]]]:
        """
        Build index of endpoints for faster lookup.

        Returns:
            Dict[method, [(endpoint_id, path, normalized_path)]]
        """
        index = {}

        for endpoint in endpoints:
            method = endpoint.properties.get('http_method', 'GET')
            path = endpoint.properties.get('path', '')

            if method not in index:
                index[method] = []

            normalized_path = self._normalize_url(path)
            index[method].append((endpoint.id, path, normalized_path))

        return index

    def _normalize_url(self, url: str) -> str:
        """
        Normalize URL for comparison.

        Examples:
            /api/products/${id} -> /api/products/{param}
            /api/products/:id -> /api/products/{param}
            /api/products/<int:pk>/ -> /api/products/{param}
            /api/products/{product_id} -> /api/products/{param}

        Args:
            url: Raw URL

        Returns:
            Normalized URL
        """
        # Remove query parameters
        url = url.split('?')[0]

        # Remove trailing slash
        url = url.rstrip('/')

        # Replace all parameter formats with {param}
        patterns = [
            r'\$\{[^}]+\}',      # ${id}, ${productId}
            r':[a-zA-Z_]\w*',    # :id, :productId
            r'<[^>]+>',          # <int:pk>, <str:slug>
            r'\{[^}]+\}',        # {id}, {product_id}
        ]

        for pattern in patterns:
            url = re.sub(pattern, '{param}', url)

        # Normalize multiple consecutive slashes
        url = re.sub(r'/+', '/', url)

        # Remove base URL if present (http://localhost:8000/api -> /api)
        if url.startswith('http'):
            parsed = urlparse(url)
            url = parsed.path

        return url

    def _find_matching_endpoints(
        self,
        normalized_url: str,
        method: str,
        endpoint_index: Dict[str, List[Tuple[str, str, str]]]
    ) -> List[Tuple[str, float]]:
        """
        Find matching endpoints with confidence scores.

        Args:
            normalized_url: Normalized URL from frontend
            method: HTTP method
            endpoint_index: Index of endpoints

        Returns:
            List of (endpoint_id, confidence) tuples
        """
        matches = []

        # Get endpoints with matching method
        candidates = endpoint_index.get(method, [])

        for endpoint_id, original_path, normalized_path in candidates:
            # Exact match
            if normalized_path == normalized_url:
                matches.append((endpoint_id, 1.0))
                continue

            # Partial match with similarity score
            similarity = self._calculate_similarity(normalized_url, normalized_path)

            if similarity > 0.6:
                matches.append((endpoint_id, similarity))

        # Sort by confidence (highest first)
        matches.sort(key=lambda x: x[1], reverse=True)

        return matches

    def _calculate_similarity(self, url1: str, url2: str) -> float:
        """
        Calculate similarity between two URLs.

        Uses multiple heuristics:
        1. Sequence matcher for overall similarity
        2. Path segment matching
        3. Keyword matching

        Args:
            url1: First URL
            url2: Second URL

        Returns:
            Similarity score (0.0 to 1.0)
        """
        # Overall sequence similarity
        seq_similarity = SequenceMatcher(None, url1, url2).ratio()

        # Path segment similarity
        segments1 = [s for s in url1.split('/') if s and s != '{param}']
        segments2 = [s for s in url2.split('/') if s and s != '{param}']

        if not segments1 or not segments2:
            return seq_similarity

        # Count matching segments
        common_segments = set(segments1) & set(segments2)
        segment_similarity = len(common_segments) / max(len(segments1), len(segments2))

        # Combine scores (weighted average)
        final_score = (seq_similarity * 0.3) + (segment_similarity * 0.7)

        return final_score

    def find_orphaned_api_calls(
        self,
        component_nodes: List[ComponentNode],
        endpoint_nodes: List[EndpointNode]
    ) -> List[Dict[str, any]]:
        """
        Find API calls that couldn't be matched to any endpoint.

        Useful for debugging and identifying missing endpoints.

        Args:
            component_nodes: List of component nodes
            endpoint_nodes: List of endpoint nodes

        Returns:
            List of orphaned API calls with details
        """
        orphaned = []

        endpoint_index = self._build_endpoint_index(endpoint_nodes)

        for component in component_nodes:
            api_calls = component.properties.get('api_calls', [])

            if isinstance(api_calls, str):
                try:
                    api_calls = json.loads(api_calls)
                except:
                    api_calls = []

            for api_call in api_calls:
                url = api_call.get('url')
                method = api_call.get('method', 'GET').upper()

                if not url:
                    continue

                normalized_url = self._normalize_url(url)
                matches = self._find_matching_endpoints(normalized_url, method, endpoint_index)

                if not matches or matches[0][1] < 0.7:
                    orphaned.append({
                        'component': component.name,
                        'component_id': component.id,
                        'url': url,
                        'method': method,
                        'normalized_url': normalized_url,
                        'best_match': matches[0] if matches else None
                    })

        logger.warning(f"Found {len(orphaned)} orphaned API calls")
        return orphaned
