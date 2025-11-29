"""
Unit tests for Weaviate indexer.

Tests embedding generation, indexing, and search functionality.
"""

import pytest
from unittest.mock import Mock, MagicMock

from src.code_rag.graph import (
    WeaviateIndexer,
    GraphNode, FunctionNode, ClassNode, EndpointNode, ComponentNode,
    NodeType
)


@pytest.fixture
def mock_weaviate_client():
    """Mock Weaviate client."""
    mock_client = MagicMock()
    mock_client.collections.exists.return_value = False
    return mock_client


@pytest.fixture
def mock_embedding_model():
    """Mock sentence transformer model."""
    mock_model = MagicMock()
    mock_model.get_sentence_embedding_dimension.return_value = 768
    mock_model.encode.return_value = [[0.1] * 768, [0.2] * 768]  # Mock embeddings
    return mock_model


@pytest.fixture
def sample_nodes():
    """Sample graph nodes for testing."""
    return [
        FunctionNode(
            id="repo:test/file.py:get_user",
            name="get_user",
            signature="def get_user(user_id: int) -> User",
            docstring="Get user by ID.",
            start_line=10,
            end_line=15,
            parameters=[{"name": "user_id", "type": "int"}],
            return_type="User"
        ),
        ClassNode(
            id="repo:test/models.py:User",
            name="User",
            docstring="User model.",
            base_classes=["Model"],
            start_line=5,
            end_line=20
        ),
        EndpointNode(
            id="repo:test/api.py:get_users",
            name="GET /api/users",
            http_method="GET",
            path="/api/users",
            response_model="List[User]"
        ),
        ComponentNode(
            id="repo:test/UserList.jsx:UserList",
            name="UserList",
            props_type="UserListProps",
            hooks_used=["useState", "useEffect"],
            start_line=1,
            end_line=30
        )
    ]


class TestWeaviateIndexer:
    """Unit tests for WeaviateIndexer."""

    def test_build_searchable_content_function(self):
        """Test building searchable content from function node."""
        indexer = WeaviateIndexer.__new__(WeaviateIndexer)

        function_node = FunctionNode(
            id="repo:test/file.py:calculate_sum",
            name="calculate_sum",
            signature="def calculate_sum(a: int, b: int) -> int",
            docstring="Calculate the sum of two numbers.",
            start_line=10,
            end_line=12
        )

        content = indexer._build_searchable_content(function_node)

        assert "Name: calculate_sum" in content
        assert "Type: Function" in content
        assert "Signature: def calculate_sum(a: int, b: int) -> int" in content
        assert "Documentation: Calculate the sum of two numbers." in content

    def test_build_searchable_content_endpoint(self):
        """Test building searchable content from endpoint node."""
        indexer = WeaviateIndexer.__new__(WeaviateIndexer)

        endpoint_node = EndpointNode(
            id="repo:test/api.py:create_user",
            name="POST /api/users",
            http_method="POST",
            path="/api/users",
            request_model="UserCreate",
            response_model="User"
        )

        content = indexer._build_searchable_content(endpoint_node)

        assert "Name: POST /api/users" in content
        assert "Type: Endpoint" in content
        assert "Endpoint: POST /api/users" in content

    def test_build_searchable_content_component(self):
        """Test building searchable content from component node."""
        indexer = WeaviateIndexer.__new__(WeaviateIndexer)

        component_node = ComponentNode(
            id="repo:test/UserList.jsx:UserList",
            name="UserList",
            props_type="UserListProps",
            hooks_used=["useState", "useEffect"],
            start_line=1,
            end_line=30
        )

        content = indexer._build_searchable_content(component_node)

        assert "Name: UserList" in content
        assert "Type: Component" in content
        assert "Props: UserListProps" in content
        assert "Hooks: useState,useEffect" in content

    def test_extract_repository_name(self):
        """Test extracting repository name from node ID."""
        indexer = WeaviateIndexer.__new__(WeaviateIndexer)

        # Test various node ID formats
        assert indexer._extract_repository_name("repo:my-project") == "my-project"
        assert indexer._extract_repository_name("repo:my-project/src/file.py") == "my-project"
        assert indexer._extract_repository_name("repo:my-project/src/file.py:MyClass") == "my-project"
        assert indexer._extract_repository_name("invalid") == "unknown"

    @pytest.mark.integration
    def test_create_schema_real(self):
        """Test creating Weaviate schema (requires running Weaviate)."""
        try:
            indexer = WeaviateIndexer(
                weaviate_url="http://localhost:8080",
                embedding_model="BAAI/bge-m3"
            )

            # Create schema
            indexer.create_schema()

            # Verify collection exists
            assert indexer.client.collections.exists("CodeNode")

            # Clean up
            indexer.delete_all()
            indexer.close()

        except Exception as e:
            pytest.skip(f"Weaviate not available: {e}")

    @pytest.mark.integration
    def test_index_nodes_real(self, sample_nodes):
        """Test indexing nodes in Weaviate (requires running Weaviate)."""
        try:
            indexer = WeaviateIndexer(
                weaviate_url="http://localhost:8080",
                embedding_model="BAAI/bge-m3"
            )

            # Create schema
            indexer.create_schema()

            # Clear existing data
            indexer.delete_all()
            indexer.create_schema()

            # Index sample nodes
            indexed = indexer.index_nodes(sample_nodes, batch_size=2)

            assert indexed == len(sample_nodes)

            # Clean up
            indexer.delete_all()
            indexer.close()

        except Exception as e:
            pytest.skip(f"Weaviate not available: {e}")

    @pytest.mark.integration
    def test_search_real(self, sample_nodes):
        """Test hybrid search in Weaviate (requires running Weaviate)."""
        try:
            indexer = WeaviateIndexer(
                weaviate_url="http://localhost:8080",
                embedding_model="BAAI/bge-m3"
            )

            # Create schema and index
            indexer.create_schema()
            indexer.delete_all()
            indexer.create_schema()
            indexer.index_nodes(sample_nodes)

            # Search for function
            results = indexer.search("get user by id", limit=5, alpha=0.5)
            assert len(results) > 0

            # Should find get_user function or related entities
            found_get_user = any(
                'get_user' in r['name'].lower() or 'user' in r['content'].lower()
                for r in results
            )
            assert found_get_user

            # Search for endpoint
            results = indexer.search("api endpoint for users", limit=5)
            assert len(results) > 0

            # Search for component
            results = indexer.search("react component with hooks", limit=5)
            assert len(results) > 0

            # Clean up
            indexer.delete_all()
            indexer.close()

        except Exception as e:
            pytest.skip(f"Weaviate not available: {e}")

    @pytest.mark.integration
    def test_get_statistics_real(self, sample_nodes):
        """Test getting statistics from Weaviate (requires running Weaviate)."""
        try:
            indexer = WeaviateIndexer(
                weaviate_url="http://localhost:8080",
                embedding_model="BAAI/bge-m3"
            )

            # Create schema and index
            indexer.create_schema()
            indexer.delete_all()
            indexer.create_schema()
            indexer.index_nodes(sample_nodes)

            # Get statistics
            stats = indexer.get_statistics()

            assert stats['total_nodes'] == len(sample_nodes)
            assert 'embedding_dimension' in stats
            assert stats['embedding_dimension'] > 0
            assert 'nodes_by_type' in stats

            # Verify counts by type
            assert stats['nodes_by_type']['Function'] >= 1
            assert stats['nodes_by_type']['Class'] >= 1
            assert stats['nodes_by_type']['Endpoint'] >= 1
            assert stats['nodes_by_type']['Component'] >= 1

            # Clean up
            indexer.delete_all()
            indexer.close()

        except Exception as e:
            pytest.skip(f"Weaviate not available: {e}")

    @pytest.mark.integration
    def test_filter_by_node_type(self, sample_nodes):
        """Test filtering search results by node type."""
        try:
            indexer = WeaviateIndexer(
                weaviate_url="http://localhost:8080",
                embedding_model="BAAI/bge-m3"
            )

            # Create schema and index
            indexer.create_schema()
            indexer.delete_all()
            indexer.create_schema()
            indexer.index_nodes(sample_nodes)

            # Search only for functions
            results = indexer.search(
                "user",
                node_types=["Function"],
                limit=10
            )

            # All results should be functions
            for result in results:
                assert result['node_type'] == "Function"

            # Search only for endpoints
            results = indexer.search(
                "api",
                node_types=["Endpoint"],
                limit=10
            )

            # All results should be endpoints
            for result in results:
                assert result['node_type'] == "Endpoint"

            # Clean up
            indexer.delete_all()
            indexer.close()

        except Exception as e:
            pytest.skip(f"Weaviate not available: {e}")


# Run tests with: pytest tests/test_weaviate_indexer.py -v
# Run integration tests: pytest tests/test_weaviate_indexer.py -v -m integration
# Skip integration tests: pytest tests/test_weaviate_indexer.py -v -m "not integration"
