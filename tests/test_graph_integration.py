"""
Integration tests for knowledge graph pipeline.

Tests the complete workflow:
1. Repository loading
2. Code parsing
3. Graph building in Neo4j
4. API linking
5. Weaviate indexing
6. Hybrid search
"""

import pytest
import tempfile
import shutil
from pathlib import Path

from src.code_rag.repo_loader import RepositoryLoader
from src.code_rag.parsers import get_parser
from src.code_rag.graph import (
    Neo4jClient, GraphBuilder, APILinker, WeaviateIndexer,
    NodeType, RelationshipType
)


# Test fixtures
@pytest.fixture(scope="module")
def test_repo_path(tmp_path_factory):
    """
    Create a small test repository with Django backend + React frontend.
    """
    repo_path = tmp_path_factory.mktemp("test_repo")

    # Create directory structure
    (repo_path / "backend").mkdir()
    (repo_path / "backend" / "models.py").write_text('''
"""Django models."""

from django.db import models


class User(models.Model):
    """User model."""
    username = models.CharField(max_length=100)
    email = models.EmailField()
    created_at = models.DateTimeField(auto_now_add=True)


class Post(models.Model):
    """Post model."""
    title = models.CharField(max_length=200)
    content = models.TextField()
    author = models.ForeignKey(User, on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)
''')

    (repo_path / "backend" / "views.py").write_text('''
"""FastAPI views."""

from fastapi import FastAPI, Depends
from pydantic import BaseModel


app = FastAPI()


class UserCreate(BaseModel):
    username: str
    email: str


class UserResponse(BaseModel):
    id: int
    username: str
    email: str


@app.get("/api/users")
async def get_users():
    """Get all users."""
    return []


@app.post("/api/users")
async def create_user(user: UserCreate):
    """Create a new user."""
    return {"id": 1, **user.dict()}


@app.get("/api/users/{user_id}")
async def get_user(user_id: int):
    """Get user by ID."""
    return {"id": user_id, "username": "test"}
''')

    (repo_path / "frontend").mkdir()
    (repo_path / "frontend" / "UserList.jsx").write_text('''
import React, { useState, useEffect } from 'react';

export const UserList = () => {
  const [users, setUsers] = useState([]);

  useEffect(() => {
    // Fetch users from API
    fetch('/api/users')
      .then(res => res.json())
      .then(data => setUsers(data));
  }, []);

  return (
    <div>
      <h1>Users</h1>
      {users.map(user => (
        <div key={user.id}>{user.username}</div>
      ))}
    </div>
  );
};
''')

    (repo_path / "frontend" / "CreateUser.jsx").write_text('''
import React, { useState } from 'react';

export const CreateUser = () => {
  const [username, setUsername] = useState('');
  const [email, setEmail] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();

    // Create user via API
    await fetch('/api/users', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username, email })
    });
  };

  return (
    <form onSubmit={handleSubmit}>
      <input
        value={username}
        onChange={e => setUsername(e.target.value)}
        placeholder="Username"
      />
      <input
        value={email}
        onChange={e => setEmail(e.target.value)}
        placeholder="Email"
      />
      <button type="submit">Create User</button>
    </form>
  );
};
''')

    return repo_path


@pytest.fixture(scope="module")
def neo4j_client():
    """Neo4j client fixture (requires running Neo4j instance)."""
    try:
        client = Neo4jClient(
            uri="bolt://localhost:7687",
            user="neo4j",
            password="password"
        )
        yield client
        client.close()
    except Exception as e:
        pytest.skip(f"Neo4j not available: {e}")


@pytest.fixture(scope="module")
def weaviate_indexer(neo4j_client):
    """Weaviate indexer fixture (requires running Weaviate instance)."""
    try:
        indexer = WeaviateIndexer(
            weaviate_url="http://localhost:8080",
            embedding_model="BAAI/bge-m3",
            neo4j_client=neo4j_client
        )
        yield indexer
        indexer.close()
    except Exception as e:
        pytest.skip(f"Weaviate not available: {e}")


class TestGraphIntegration:
    """Integration tests for complete graph pipeline."""

    def test_repository_loading(self, test_repo_path):
        """Test repository loading."""
        loader = RepositoryLoader()
        repo_info = loader.load(str(test_repo_path), name="test_repo")

        assert repo_info.name == "test_repo"
        assert repo_info.path == test_repo_path
        assert repo_info.project_type in ["backend", "fullstack"]

    def test_code_parsing(self, test_repo_path):
        """Test parsing Python and React files."""
        # Parse Python file
        python_file = test_repo_path / "backend" / "models.py"
        parser = get_parser(python_file)
        assert parser is not None

        result = parser.parse_file(python_file)
        assert result.language == "python"
        assert len(result.entities) >= 2  # User and Post classes

        # Find User and Post models
        entity_names = [e.name for e in result.entities]
        assert "User" in entity_names
        assert "Post" in entity_names

        # Parse React file
        react_file = test_repo_path / "frontend" / "UserList.jsx"
        parser = get_parser(react_file)
        assert parser is not None

        result = parser.parse_file(react_file)
        assert result.language in ["javascript", "typescript"]
        assert len(result.entities) >= 1  # UserList component

    def test_graph_building(self, test_repo_path, neo4j_client):
        """Test building knowledge graph in Neo4j."""
        # Clear existing data
        neo4j_client.clear_database()

        # Load repository
        loader = RepositoryLoader()
        repo_info = loader.load(str(test_repo_path), name="test_repo")

        # Parse all files
        parse_results = []
        for file_path in test_repo_path.rglob("*.py"):
            parser = get_parser(file_path)
            if parser:
                result = parser.parse_file(file_path)
                parse_results.append((file_path, result))

        for file_path in test_repo_path.rglob("*.jsx"):
            parser = get_parser(file_path)
            if parser:
                result = parser.parse_file(file_path)
                parse_results.append((file_path, result))

        # Build graph
        builder = GraphBuilder(neo4j_client)
        stats = builder.build_graph(repo_info, parse_results)

        assert stats['nodes_created'] > 0
        assert stats['relationships_created'] > 0

        # Verify repository node
        repo_node = neo4j_client.get_node(f"repo:test_repo")
        assert repo_node is not None
        assert repo_node['name'] == "test_repo"

        # Verify file nodes
        file_nodes = neo4j_client.find_nodes(node_type=NodeType.FILE)
        assert len(file_nodes) >= 3  # models.py, views.py, UserList.jsx

        # Verify class nodes
        class_nodes = neo4j_client.find_nodes(node_type=NodeType.CLASS)
        assert len(class_nodes) >= 2  # User, Post

        # Verify endpoint nodes
        endpoint_nodes = neo4j_client.find_nodes(node_type=NodeType.ENDPOINT)
        assert len(endpoint_nodes) >= 3  # get_users, create_user, get_user

        # Verify component nodes
        component_nodes = neo4j_client.find_nodes(node_type=NodeType.COMPONENT)
        assert len(component_nodes) >= 1  # UserList

    def test_api_linking(self, test_repo_path, neo4j_client):
        """Test API linking between frontend and backend."""
        # Ensure graph is built
        self.test_graph_building(test_repo_path, neo4j_client)

        # Get components and endpoints
        from src.code_rag.graph.models import ComponentNode, EndpointNode

        component_dicts = neo4j_client.find_nodes(node_type=NodeType.COMPONENT, limit=100)
        endpoint_dicts = neo4j_client.find_nodes(node_type=NodeType.ENDPOINT, limit=100)

        # Convert to proper types
        components = [
            ComponentNode(
                id=c['id'],
                name=c['name'],
                properties=c
            )
            for c in component_dicts
        ]

        endpoints = [
            EndpointNode(
                id=e['id'],
                name=e['name'],
                http_method=e.get('http_method', 'GET'),
                path=e.get('path', '/'),
                properties=e
            )
            for e in endpoint_dicts
        ]

        # Link APIs
        linker = APILinker()
        api_links = linker.link_api_calls(components, endpoints)

        # Should find at least one link (UserList -> GET /api/users)
        assert len(api_links) > 0

        # Verify link properties
        for link in api_links:
            assert link.type == RelationshipType.SENDS_REQUEST_TO
            assert link.confidence >= 0.7
            assert 'url' in link.properties
            assert 'method' in link.properties

    @pytest.mark.slow
    def test_weaviate_indexing(self, test_repo_path, neo4j_client, weaviate_indexer):
        """Test indexing nodes in Weaviate."""
        # Ensure graph is built
        self.test_graph_building(test_repo_path, neo4j_client)

        # Create schema
        weaviate_indexer.create_schema()

        # Delete existing data
        weaviate_indexer.delete_all()

        # Recreate schema
        weaviate_indexer.create_schema()

        # Index nodes from Neo4j
        indexed = weaviate_indexer.index_from_neo4j(
            node_types=[NodeType.CLASS, NodeType.FUNCTION, NodeType.ENDPOINT, NodeType.COMPONENT],
            batch_size=10
        )

        assert indexed > 0

        # Get statistics
        stats = weaviate_indexer.get_statistics()
        assert stats['total_nodes'] > 0
        assert 'embedding_dimension' in stats

    @pytest.mark.slow
    def test_hybrid_search(self, test_repo_path, neo4j_client, weaviate_indexer):
        """Test hybrid search in Weaviate."""
        # Ensure indexing is done
        self.test_weaviate_indexing(test_repo_path, neo4j_client, weaviate_indexer)

        # Search for user-related code
        results = weaviate_indexer.search(
            query="user model with username and email",
            limit=5,
            alpha=0.5  # Balanced hybrid search
        )

        assert len(results) > 0

        # Should find User model or related entities
        found_user = any(
            'user' in r['name'].lower() or 'user' in r['content'].lower()
            for r in results
        )
        assert found_user

    @pytest.mark.slow
    def test_complete_pipeline(self, test_repo_path, neo4j_client, weaviate_indexer):
        """Test complete pipeline from repository to searchable graph."""
        from src.code_rag.graph.build_and_index import GraphPipeline

        # Create pipeline
        pipeline = GraphPipeline(
            neo4j_uri="bolt://localhost:7687",
            neo4j_user="neo4j",
            neo4j_password="password",
            weaviate_url="http://localhost:8080",
            embedding_model="BAAI/bge-m3"
        )

        try:
            # Run complete pipeline
            stats = pipeline.run(
                source=str(test_repo_path),
                name="test_repo_complete",
                clear_existing=True,
                link_apis=True,
                index_weaviate=True
            )

            # Verify statistics
            assert stats['files_parsed'] > 0
            assert stats['entities_found'] > 0
            assert stats['nodes_created'] > 0
            assert stats['relationships_created'] > 0
            assert stats['nodes_indexed'] > 0

            # Verify search works
            results = weaviate_indexer.search(
                query="create user endpoint",
                limit=3
            )
            assert len(results) > 0

        finally:
            pipeline.close()


class TestGraphQueries:
    """Test various graph queries."""

    def test_find_api_endpoints(self, neo4j_client):
        """Test finding all API endpoints."""
        endpoints = neo4j_client.find_nodes(node_type=NodeType.ENDPOINT, limit=100)

        for endpoint in endpoints:
            assert 'http_method' in endpoint
            assert 'path' in endpoint
            assert endpoint['path'].startswith('/')

    def test_find_components_using_api(self, neo4j_client):
        """Test finding components that call APIs."""
        # Find components with SENDS_REQUEST_TO relationships
        query = """
        MATCH (c:Component)-[r:SENDS_REQUEST_TO]->(e:Endpoint)
        RETURN c.name as component, e.path as endpoint, r.method as method
        """

        results = neo4j_client.execute_cypher(query)

        # Should find UserList -> GET /api/users
        assert len(results) > 0

        for result in results:
            assert result['component']
            assert result['endpoint']
            assert result['method']

    def test_find_models_with_relationships(self, neo4j_client):
        """Test finding Django models with ForeignKey relationships."""
        query = """
        MATCH (m1:Model)-[r:FOREIGN_KEY]->(m2:Model)
        RETURN m1.name as source, m2.name as target
        """

        results = neo4j_client.execute_cypher(query)

        # Should find Post -> User relationship
        if results:
            assert any(
                r['source'] == 'Post' and r['target'] == 'User'
                for r in results
            )

    def test_graph_statistics(self, neo4j_client):
        """Test graph statistics."""
        stats = neo4j_client.get_statistics()

        assert 'nodes' in stats
        assert 'relationships' in stats
        assert stats['total_nodes'] > 0

        # Verify node type counts
        for node_type in NodeType:
            assert node_type.value in stats['nodes']


# Run tests with: pytest tests/test_graph_integration.py -v
# Run with slow tests: pytest tests/test_graph_integration.py -v --run-slow
# Skip slow tests: pytest tests/test_graph_integration.py -v -m "not slow"
