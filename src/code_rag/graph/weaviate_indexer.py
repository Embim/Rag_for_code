"""
Weaviate indexer - indexes knowledge graph nodes for vector search.

This component bridges Neo4j knowledge graph and Weaviate vector search:
1. Takes nodes from Neo4j
2. Generates embeddings using BAAI/bge-m3
3. Indexes in Weaviate for semantic search
4. Enables hybrid search (graph structure + semantic similarity)
"""

import json
from typing import List, Dict, Any, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .neo4j_client import Neo4jClient

# Note: sentence_transformers import moved to lazy import to avoid blocking file system operations
# during module import (which blocks event loop in async contexts)

import weaviate
from weaviate.classes.config import Configure, Property, DataType
from weaviate.classes.query import Filter

from .models import GraphNode, NodeType
# Note: Neo4jClient import moved to TYPE_CHECKING to avoid importing neo4j package
# when only AsyncWeaviateIndexer is needed (which doesn't use Neo4j)

from src.logger import get_logger

logger = get_logger(__name__)

# Note: All blocking file I/O operations (like logging to debug.log) were removed from module level
# to avoid blocking the event loop during module import. Any necessary logging should be done
# asynchronously or in separate threads using asyncio.to_thread().


class WeaviateIndexer:
    """
    Indexes knowledge graph nodes in Weaviate for vector search.

    Strategy:
    1. Create Weaviate schema for each node type
    2. Generate embeddings for node content
    3. Batch index nodes with embeddings
    4. Enable hybrid search (BM25 + vector)
    """

    def __init__(
        self,
        weaviate_url: str = "http://localhost:8080",
        embedding_model: str = "BAAI/bge-m3",
        neo4j_client: Optional['Neo4jClient'] = None
    ):
        """
        Initialize Weaviate indexer.

        Args:
            weaviate_url: Weaviate connection URL
            embedding_model: Sentence transformer model name
            neo4j_client: Optional Neo4j client for fetching nodes
        """
        self.weaviate_url = weaviate_url
        self.neo4j_client = neo4j_client
        self._embedding_model_name = embedding_model
        self._embedding_model = None  # Lazy loaded
        self._embedding_dim = None  # Lazy loaded

        # Initialize Weaviate client
        # skip_init_checks=True prevents event loop conflict with LangGraph async worker
        try:
            print(f"[WEAVIATE] Connecting to {weaviate_url}...", flush=True)
            self.client = weaviate.connect_to_local(
                host=weaviate_url.replace("http://", "").replace("https://", "").split(":")[0],
                port=int(weaviate_url.split(":")[-1]) if ":" in weaviate_url else 8080,
                skip_init_checks=True,  # Prevent deadlock in async context (LangGraph)
            )
            # Verify connection manually since we skipped init checks
            if not self.client.is_ready():
                raise ConnectionError(f"Weaviate at {weaviate_url} is not ready")
            logger.info(f"Connected to Weaviate at {weaviate_url}")
            print(f"[WEAVIATE] Connected to {weaviate_url}", flush=True)
        except Exception as e:
            logger.error(f"Failed to connect to Weaviate: {e}")
            print(f"[WEAVIATE] ERROR: {e}", flush=True)
            raise ConnectionError(f"Cannot connect to Weaviate at {weaviate_url}: {e}")

        # Embedding model will be loaded lazily on first use
        # This prevents GPU blocking during uvicorn startup
        logger.info(f"Embedding model will be loaded on first use: {embedding_model}")
        print(f"[WEAVIATE] Embedding model will be loaded on first use: {embedding_model}", flush=True)
    
    @property
    def embedding_model(self):
        """Lazy load embedding model on first use."""
        if self._embedding_model is None:
            import traceback
            # Get caller information for debugging
            stack = traceback.extract_stack()
            caller = stack[-2] if len(stack) >= 2 else None
            caller_info = f"{caller.filename}:{caller.lineno} {caller.name}" if caller else "unknown"
            print(f"[WEAVIATE] Loading embedding model: {self._embedding_model_name} (called from {caller_info})...", flush=True)
            logger.info(f"Loading embedding model: {self._embedding_model_name} (called from {caller_info})")
            try:
                # #region agent log
                import json, time
                with open(r'c:\Users\petrc\OneDrive\Documents\Проекты\ПроетыПоРаботу\Rag_for_code\.cursor\debug.log', 'a', encoding='utf-8') as f:
                    import torch
                    cuda_available = torch.cuda.is_available() if hasattr(torch, 'cuda') else False
                    cuda_device_count = torch.cuda.device_count() if cuda_available else 0
                    f.write(json.dumps({"sessionId":"debug-session","runId":"sync-weaviate-lazy","hypothesisId":"F","location":"weaviate_indexer.py:141","message":"Sync WeaviateIndexer lazy load - before model load","data":{"model_name":self._embedding_model_name,"cuda_available":cuda_available,"cuda_device_count":cuda_device_count},"timestamp":int(time.time()*1000)}) + '\n')
                # #endregion
                
                # Lazy import to avoid blocking file system operations during module import
                from sentence_transformers import SentenceTransformer
                
                self._embedding_model = SentenceTransformer(self._embedding_model_name)
                # #region agent log
                with open(r'c:\Users\petrc\OneDrive\Documents\Проекты\ПроетыПоРаботу\Rag_for_code\.cursor\debug.log', 'a', encoding='utf-8') as f:
                    import torch
                    device = str(self._embedding_model.device) if hasattr(self._embedding_model, 'device') else None
                    cuda_available = torch.cuda.is_available() if hasattr(torch, 'cuda') else False
                    f.write(json.dumps({"sessionId":"debug-session","runId":"sync-weaviate-lazy","hypothesisId":"F","location":"weaviate_indexer.py:149","message":"Sync WeaviateIndexer lazy load - model loaded","data":{"model_device":device,"cuda_available":cuda_available},"timestamp":int(time.time()*1000)}) + '\n')
                # #endregion
                self._embedding_dim = self._embedding_model.get_sentence_embedding_dimension()
                logger.info(f"Embedding model loaded (dimension: {self._embedding_dim})")
                print(f"[WEAVIATE] Embedding model loaded (dimension: {self._embedding_dim})", flush=True)
            except Exception as e:
                print(f"[WEAVIATE] ERROR loading embedding model: {e}", flush=True)
                raise
        return self._embedding_model
    
    @property
    def embedding_dim(self):
        """Get embedding dimension (lazy load model if needed)."""
        if self._embedding_dim is None:
            # Access embedding_model property to trigger lazy loading
            _ = self.embedding_model
        return self._embedding_dim
    
    @property
    def embedding_model_name(self):
        """Get embedding model name."""
        return self._embedding_model_name

    def close(self):
        """Close Weaviate connection."""
        if self.client:
            self.client.close()
            logger.info("Weaviate connection closed")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def create_schema(self):
        """
        Create Weaviate schema for code entities and documents.

        Creates collections for each node type with appropriate properties.
        """
        try:
            # Create CodeNode collection (for code entities)
            if not self.client.collections.exists("CodeNode"):
                self.client.collections.create(
                    name="CodeNode",
                    vectorizer_config=Configure.Vectorizer.none(),  # We provide our own vectors
                    properties=[
                        Property(name="node_id", data_type=DataType.TEXT),
                        Property(name="node_type", data_type=DataType.TEXT),
                        Property(name="name", data_type=DataType.TEXT),
                        Property(name="file_path", data_type=DataType.TEXT),
                        Property(name="repository", data_type=DataType.TEXT),
                        Property(name="content", data_type=DataType.TEXT),  # Searchable content (includes code)
                        Property(name="code", data_type=DataType.TEXT),  # Full source code
                        Property(name="signature", data_type=DataType.TEXT),
                        Property(name="docstring", data_type=DataType.TEXT),
                        Property(name="start_line", data_type=DataType.INT),
                        Property(name="end_line", data_type=DataType.INT),
                        Property(name="metadata", data_type=DataType.TEXT),  # JSON string for additional data
                    ]
                )
                logger.info("Created CodeNode collection in Weaviate")
            else:
                logger.info("CodeNode collection already exists")

            # Create DocumentNode collection (for documentation: SOP, policies, etc.)
            if not self.client.collections.exists("DocumentNode"):
                self.client.collections.create(
                    name="DocumentNode",
                    vectorizer_config=Configure.Vectorizer.none(),  # We provide our own vectors
                    properties=[
                        Property(name="node_id", data_type=DataType.TEXT),
                        Property(name="node_type", data_type=DataType.TEXT),
                        Property(name="name", data_type=DataType.TEXT),  # Document title
                        Property(name="file_path", data_type=DataType.TEXT),
                        Property(name="document_type", data_type=DataType.TEXT),  # SOP, Policy, Manual, etc.
                        Property(name="content", data_type=DataType.TEXT),  # Full searchable text
                        Property(name="author", data_type=DataType.TEXT),
                        Property(name="created_date", data_type=DataType.TEXT),
                        Property(name="modified_date", data_type=DataType.TEXT),
                        Property(name="sections_count", data_type=DataType.INT),
                        Property(name="images_count", data_type=DataType.INT),
                        Property(name="metadata", data_type=DataType.TEXT),  # JSON string for additional data
                    ]
                )
                logger.info("Created DocumentNode collection in Weaviate")
            else:
                logger.info("DocumentNode collection already exists")

        except Exception as e:
            logger.error(f"Failed to create Weaviate schema: {e}")
            raise

    def _build_searchable_content(self, node: GraphNode) -> str:
        """
        Build searchable text content from a node.

        Combines node name, signature, docstring, code, and other relevant text
        into a single searchable string for embedding generation.

        Args:
            node: Graph node

        Returns:
            Searchable text content
        """
        parts = []

        # Node name
        parts.append(f"Name: {node.name}")

        # Node type
        parts.append(f"Type: {node.type.value}")

        # Type-specific content
        if hasattr(node, 'signature') and node.properties.get('signature'):
            parts.append(f"Signature: {node.properties['signature']}")

        if hasattr(node, 'docstring') and node.properties.get('docstring'):
            docstring = node.properties['docstring'].strip()
            if docstring:
                parts.append(f"Documentation: {docstring}")

        # CRITICAL: Include full source code for better semantic search
        # This enables the LLM to understand the actual logic and implementation
        if node.properties.get('code'):
            code = node.properties['code'].strip()
            if code:
                parts.append(f"Code:\n{code}")

        # File path context
        if node.properties.get('file_path'):
            parts.append(f"File: {node.properties['file_path']}")

        # For endpoints, include HTTP method and path
        if node.type == NodeType.ENDPOINT:
            method = node.properties.get('http_method', '')
            path = node.properties.get('path', '')
            parts.append(f"Endpoint: {method} {path}")

        # For components, include props and hooks
        if node.type == NodeType.COMPONENT:
            props = node.properties.get('props_type', '')
            if props:
                parts.append(f"Props: {props}")
            hooks = node.properties.get('hooks_used', '')
            if hooks:
                parts.append(f"Hooks: {hooks}")

        # For models, include fields
        if node.type == NodeType.MODEL:
            model_type = node.properties.get('model_type', '')
            parts.append(f"Model type: {model_type}")

        return "\n".join(parts)

    def _extract_repository_name(self, node_id: str) -> str:
        """
        Extract repository name from node ID.

        Node IDs have format: repo:repository_name/file/path:entity

        Args:
            node_id: Node ID

        Returns:
            Repository name
        """
        if node_id.startswith("repo:"):
            parts = node_id[5:].split("/")
            return parts[0].split(":")[0]
        return "unknown"

    def index_nodes(self, nodes: List[GraphNode], batch_size: int = 100) -> int:
        """
        Index nodes in Weaviate with embeddings.

        Args:
            nodes: List of graph nodes to index
            batch_size: Batch size for indexing

        Returns:
            Number of nodes indexed
        """
        if not nodes:
            logger.warning("No nodes to index")
            return 0

        logger.info(f"Indexing {len(nodes)} nodes in Weaviate...")

        collection = self.client.collections.get("CodeNode")
        indexed_count = 0

        # Process in batches
        for i in range(0, len(nodes), batch_size):
            batch = nodes[i:i + batch_size]

            # Prepare data for batch
            batch_data = []
            batch_contents = []

            for node in batch:
                content = self._build_searchable_content(node)
                batch_contents.append(content)

                # Prepare properties
                properties = {
                    "node_id": node.id,
                    "node_type": node.type.value,
                    "name": node.name,
                    "file_path": node.properties.get('file_path', ''),
                    "repository": self._extract_repository_name(node.id),
                    "content": content,
                    "code": node.properties.get('code', ''),  # Store full source code
                    "signature": node.properties.get('signature', ''),
                    "docstring": node.properties.get('docstring', ''),
                    "start_line": node.properties.get('start_line', 0),
                    "end_line": node.properties.get('end_line', 0),
                    "metadata": json.dumps(node.properties)
                }

                batch_data.append(properties)

            # Generate embeddings in batch
            try:
                embeddings = self.embedding_model.encode(
                    batch_contents,
                    show_progress_bar=False,
                    convert_to_numpy=True
                )

                # Index batch
                with collection.batch.dynamic() as batch_inserter:
                    for properties, vector in zip(batch_data, embeddings):
                        batch_inserter.add_object(
                            properties=properties,
                            vector=vector.tolist()
                        )

                indexed_count += len(batch)
                logger.info(f"Indexed batch {i // batch_size + 1}: {indexed_count}/{len(nodes)} nodes")

            except Exception as e:
                logger.error(f"Failed to index batch {i // batch_size + 1}: {e}")
                continue

        logger.info(f"Successfully indexed {indexed_count} nodes in Weaviate")
        return indexed_count

    def index_from_neo4j(
        self,
        node_types: Optional[List[NodeType]] = None,
        batch_size: int = 100
    ) -> int:
        """
        Fetch nodes from Neo4j and index them in Weaviate.

        Args:
            node_types: Optional list of node types to index (defaults to all)
            batch_size: Batch size for indexing

        Returns:
            Number of nodes indexed
        """
        if not self.neo4j_client:
            raise ValueError("Neo4j client not provided")

        # Default to all node types except Repository (too high-level)
        if node_types is None:
            node_types = [
                NodeType.FILE,
                NodeType.FUNCTION,
                NodeType.CLASS,
                NodeType.COMPONENT,
                NodeType.ENDPOINT,
                NodeType.MODEL,
                NodeType.ROUTE
            ]

        total_indexed = 0

        for node_type in node_types:
            logger.info(f"Fetching {node_type.value} nodes from Neo4j...")

            # Fetch nodes of this type
            # Note: neo4j_client is guaranteed to exist here due to check above
            node_dicts = self.neo4j_client.find_nodes(
                node_type=node_type,
                limit=10000  # Fetch in large batches
            )

            if not node_dicts:
                logger.info(f"No {node_type.value} nodes found")
                continue

            # Convert to GraphNode objects
            nodes = self._convert_to_graph_nodes(node_dicts, node_type)

            # Index in Weaviate
            count = self.index_nodes(nodes, batch_size=batch_size)
            total_indexed += count

        logger.info(f"Total nodes indexed from Neo4j: {total_indexed}")
        return total_indexed

    def _convert_to_graph_nodes(
        self,
        node_dicts: List[Dict[str, Any]],
        node_type: NodeType
    ) -> List[GraphNode]:
        """
        Convert Neo4j node dictionaries to GraphNode objects.

        Args:
            node_dicts: List of node dictionaries from Neo4j
            node_type: Type of nodes

        Returns:
            List of GraphNode objects
        """
        nodes = []

        for node_dict in node_dicts:
            # Create base GraphNode
            node = GraphNode(
                id=node_dict.get('id', ''),
                type=node_type,
                name=node_dict.get('name', ''),
                properties=node_dict
            )
            nodes.append(node)

        return nodes

    def search(
        self,
        query: str,
        node_types: Optional[List[str]] = None,
        limit: int = 10,
        alpha: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Hybrid search in Weaviate (BM25 + vector search).

        Args:
            query: Search query
            node_types: Optional filter by node types
            limit: Maximum number of results
            alpha: Hybrid search alpha (0.0 = BM25 only, 1.0 = vector only, 0.5 = balanced)

        Returns:
            List of search results with scores
        """
        collection = self.client.collections.get("CodeNode")

        # Generate query embedding
        query_vector = self.embedding_model.encode(
            query,
            show_progress_bar=False,
            convert_to_numpy=True
        ).tolist()

        # Build filter if node types specified
        where_filter = None
        if node_types:
            where_filter = Filter.by_property("node_type").contains_any(node_types)

        # Perform hybrid search
        try:
            if where_filter:
                response = collection.query.hybrid(
                    query=query,
                    vector=query_vector,
                    alpha=alpha,
                    limit=limit,
                    filters=where_filter
                )
            else:
                response = collection.query.hybrid(
                    query=query,
                    vector=query_vector,
                    alpha=alpha,
                    limit=limit
                )

            # Format results
            results = []
            for obj in response.objects:
                results.append({
                    'node_id': obj.properties.get('node_id'),
                    'node_type': obj.properties.get('node_type'),
                    'name': obj.properties.get('name'),
                    'file_path': obj.properties.get('file_path'),
                    'repository': obj.properties.get('repository'),
                    'content': obj.properties.get('content'),
                    'code': obj.properties.get('code', ''),  # Include full source code
                    'score': obj.metadata.score if hasattr(obj.metadata, 'score') else 0.0,
                    'metadata': json.loads(obj.properties.get('metadata', '{}'))
                })

            return results

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get indexing statistics.

        Returns:
            Dictionary with statistics
        """
        try:
            collection = self.client.collections.get("CodeNode")

            # Get total count
            result = collection.aggregate.over_all(total_count=True)
            total_count = result.total_count

            # Count by node type
            type_counts = {}
            for node_type in NodeType:
                response = collection.aggregate.over_all(
                    filters=Filter.by_property("node_type").equal(node_type.value),
                    total_count=True
                )
                type_counts[node_type.value] = response.total_count

            return {
                'total_nodes': total_count,
                'nodes_by_type': type_counts,
                'embedding_model': self._embedding_model_name,
                'embedding_dimension': self._embedding_dim if self._embedding_dim is not None else None,
                'model_loaded': self._embedding_model is not None
            }

        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {
                'total_nodes': 0,
                'nodes_by_type': {},
                'error': str(e)
            }

    def delete_all(self):
        """
        Delete all indexed data.

        WARNING: This deletes everything! Use with caution.
        """
        try:
            if self.client.collections.exists("CodeNode"):
                self.client.collections.delete("CodeNode")
                logger.warning("Deleted CodeNode collection - all indexed data removed")

            # Recreate schema
            self.create_schema()

        except Exception as e:
            logger.error(f"Failed to delete collection: {e}")
            raise


class AsyncWeaviateIndexer:
    """
    Async version of WeaviateIndexer for use in async contexts (LangGraph, etc).

    Uses WeaviateAsyncClient to avoid event loop conflicts with gRPC.
    """

    def __init__(
        self,
        weaviate_url: str = "http://localhost:8080",
        embedding_model: str = "BAAI/bge-m3",
    ):
        """
        Initialize async Weaviate indexer.

        Note: Call connect() after creating instance.

        Args:
            weaviate_url: Weaviate connection URL
            embedding_model: Sentence transformer model name
        """
        self.weaviate_url = weaviate_url
        self._client = None
        self._embedding_model_name = embedding_model
        self._embedding_model = None
        self.embedding_dim = None

    def _load_embedding_model_sync(self):
        """Load embedding model synchronously (called from thread).
        
        NOTE: This method is called from asyncio.to_thread() to avoid blocking
        the event loop. All file I/O operations here (including SentenceTransformer
        initialization which may scan directories) run in a separate thread.
        
        PyTorch supports multiple processes sharing the same GPU safely - CUDA contexts
        are process-local, so uvicorn and LangGraph can both use GPU without conflicts.
        """
        # Determine device based on CUDA availability and config
        # Use GPU if available (PyTorch handles multi-process GPU access safely)
        import torch
        from src.config import EMBEDDING_DEVICE
        
        # Use device from config (defaults to 'cuda' if available, 'cpu' otherwise)
        # But override with actual CUDA availability check
        if EMBEDDING_DEVICE == "cuda" and torch.cuda.is_available():
            device = "cuda"
            device_id = torch.cuda.current_device()
            print(f"[ASYNC_WEAVIATE] Loading embedding model: {self._embedding_model_name} (GPU: {device_id})...", flush=True)
            logger.info(f"Loading embedding model: {self._embedding_model_name} (GPU: {device_id})")
        else:
            device = "cpu"
            print(f"[ASYNC_WEAVIATE] Loading embedding model: {self._embedding_model_name} (CPU mode)...", flush=True)
            logger.info(f"Loading embedding model: {self._embedding_model_name} (CPU mode)")
        
        try:
            # Lazy import to avoid blocking file system operations during module import
            from sentence_transformers import SentenceTransformer
            
            # NOTE: SentenceTransformer initialization may scan directories for model files,
            # but this runs in a separate thread via asyncio.to_thread() so it won't block event loop
            # PyTorch handles GPU sharing between processes - no conflicts expected
            model = SentenceTransformer(self._embedding_model_name, device=device)
            
            dim = model.get_sentence_embedding_dimension()
            actual_device = str(model.device) if hasattr(model, 'device') else device
            print(f"[ASYNC_WEAVIATE] Embedding model loaded (dimension: {dim}, device: {actual_device})", flush=True)
            logger.info(f"Embedding model loaded (dimension: {dim}, device: {actual_device})")
            
            return model, dim
        except Exception as e:
            print(f"[ASYNC_WEAVIATE] Error loading embedding model: {e}", flush=True)
            logger.error(f"Error loading embedding model: {e}")
            raise
    
    async def _ensure_embedding_model(self):
        """Lazy load embedding model (async, runs in thread to avoid blocking event loop)."""
        if self._embedding_model is None:
            import asyncio
            
            # Load model in separate thread to avoid blocking event loop
            # All file I/O operations (including SentenceTransformer initialization
            # which may scan directories) run in the separate thread
            print(f"[ASYNC_WEAVIATE] Loading embedding model (lazy load)...", flush=True)
            try:
                self._embedding_model, self.embedding_dim = await asyncio.to_thread(
                    self._load_embedding_model_sync
                )
                print(f"[ASYNC_WEAVIATE] Embedding model loaded successfully (dimension: {self.embedding_dim})", flush=True)
            except Exception as e:
                print(f"[ASYNC_WEAVIATE] Error loading embedding model: {e}", flush=True)
                logger.error(f"Error loading embedding model: {e}")
                raise

    async def connect(self):
        """Connect to Weaviate asynchronously."""
        import asyncio
        
        print(f"[ASYNC_WEAVIATE] Connecting to {self.weaviate_url}...", flush=True)
        logger.info(f"Connecting to Weaviate at {self.weaviate_url}")
        
        host = self.weaviate_url.replace("http://", "").replace("https://", "").split(":")[0]
        port = int(self.weaviate_url.split(":")[-1]) if ":" in self.weaviate_url else 8080

        # Recreate client if it was closed or doesn't exist
        if self._client is not None:
            try:
                # Check if client is actually usable
                if hasattr(self._client, '_closed') and self._client._closed:
                    print(f"[ASYNC_WEAVIATE] Previous client was closed, creating new one", flush=True)
                    logger.info("Previous client was closed, creating new one")
                    self._client = None
                elif hasattr(self._client, 'is_ready'):
                    # Check if client is ready (async method for async client)
                    try:
                        is_ready = await self._client.is_ready()
                        if not is_ready:
                            print(f"[ASYNC_WEAVIATE] Previous client not ready, recreating", flush=True)
                            logger.info("Previous client not ready, recreating")
                            self._client = None
                    except Exception:
                        # Client in bad state, recreate
                        print(f"[ASYNC_WEAVIATE] Previous client in bad state, recreating", flush=True)
                        logger.warning("Previous client in bad state, recreating")
                        self._client = None
            except Exception:
                # Client might be in invalid state, recreate it
                print(f"[ASYNC_WEAVIATE] Previous client invalid, recreating", flush=True)
                logger.warning("Previous client invalid, recreating")
                self._client = None
        
        if self._client is None:
            # Create new async client with skip_init_checks to avoid gRPC blocking event loop
            # gRPC channel creation can block even in async context due to C extensions
            self._client = weaviate.use_async_with_local(
                host=host,
                port=port,
                grpc_port=50051,  # Explicitly set gRPC port (exposed in docker-compose)
                skip_init_checks=True  # Skip blocking init checks that freeze event loop
            )
            print(f"[ASYNC_WEAVIATE] Weaviate client created (type: {type(self._client).__name__})", flush=True)
            logger.info(f"Weaviate async client created for {self.weaviate_url}")
        
        # ALWAYS explicitly connect - use_async_with_local() requires await client.connect()
        # See: https://weaviate.io/developers/weaviate/client-libraries/python/async
        try:
            print(f"[ASYNC_WEAVIATE] Explicitly calling client.connect()...", flush=True)
            await self._client.connect()  # Always async for use_async_with_local()
            print(f"[ASYNC_WEAVIATE] client.connect() completed", flush=True)
            logger.info("Client explicitly connected")
        except Exception as e:
            print(f"[ASYNC_WEAVIATE] client.connect() failed: {e}", flush=True)
            logger.error(f"Client connect() failed: {e}")
            raise  # Connection is critical, don't continue with broken client
        
        # Verify connection
        try:
            is_ready = await self._client.is_ready()  # Async method for async client
            if is_ready:
                print(f"[ASYNC_WEAVIATE] Client is ready", flush=True)
                logger.info("Client is ready")
            else:
                print(f"[ASYNC_WEAVIATE] Client connected but is_ready() returned False", flush=True)
                logger.warning("Client connected but is_ready() returned False - this may indicate Weaviate is not running")
        except Exception as e:
            print(f"[ASYNC_WEAVIATE] Could not verify client readiness: {e}", flush=True)
            logger.warning(f"Could not verify client readiness: {e}")
        
        # Load embedding model (async, in thread to avoid blocking event loop)
        await self._ensure_embedding_model()

        return self

    async def close(self):
        """Close Weaviate connection."""
        if self._client:
            await self._client.close()
            logger.info("Async Weaviate connection closed")

    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

    async def search(
        self,
        query: str,
        node_types: Optional[List[str]] = None,
        limit: int = 10,
        alpha: float = 0.5,
        _retry_count: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Async hybrid search in Weaviate (BM25 + vector search).

        Args:
            query: Search query
            node_types: Optional filter by node types
            limit: Maximum number of results
            alpha: Hybrid search alpha (0.0 = BM25 only, 1.0 = vector only, 0.5 = balanced)

        Returns:
            List of search results with scores
        """
        import asyncio
        
        # Maximum number of retries for reconnection
        max_retries = 2
        
        # Ensure client exists and is connected before operations
        if self._client is None:
            await self.connect()
        else:
            # Check if client is still usable before proceeding
            try:
                is_ready = await self._client.is_ready()  # Async method for async client
                if not is_ready:
                    print(f"[ASYNC_WEAVIATE] Client not ready, reconnecting...", flush=True)
                    logger.warning("Client not ready, reconnecting...")
                    await self.connect()  # Reconnect if not ready
            except Exception as e:
                # Client in bad state, reconnect
                error_msg = str(e)
                if "closed" in error_msg.lower():
                    print(f"[ASYNC_WEAVIATE] Client appears closed, reconnecting: {e}", flush=True)
                    logger.warning(f"Client appears closed, reconnecting: {e}")
                else:
                    print(f"[ASYNC_WEAVIATE] Error checking client state: {e}, reconnecting...", flush=True)
                    logger.warning(f"Error checking client state: {e}, reconnecting...")
                await self.connect()

        await self._ensure_embedding_model()

        # Generate query embedding - wrap in thread to avoid blocking file operations
        # sentence_transformers.encode() can trigger file I/O (ScandirIterator) which blocks event loop
        def encode_sync():
            """Encode query synchronously (called from thread)."""
            return self._embedding_model.encode(
                query,
                show_progress_bar=False,
                convert_to_numpy=True
            ).tolist()
        
        query_vector = await asyncio.to_thread(encode_sync)

        # Build filter if node types specified
        where_filter = None
        if node_types:
            where_filter = Filter.by_property("node_type").contains_any(node_types)

        # Get collection RIGHT BEFORE using it - don't cache it
        # Collection object is bound to client, so we must get it fresh each time
        # to ensure it's bound to the current (connected) client instance
        try:
            collection = self._client.collections.get("CodeNode")
            print(f"[ASYNC_WEAVIATE] Collection 'CodeNode' obtained successfully", flush=True)
        except Exception as e:
            error_msg = str(e)
            print(f"[ASYNC_WEAVIATE] Error getting collection: {error_msg}", flush=True)
            logger.error(f"Error getting collection: {e}")
            # If getting collection fails, reconnect and retry
            if "closed" in error_msg.lower() or "not connected" in error_msg.lower() or "connection" in error_msg.lower():
                if _retry_count < max_retries - 1:
                    print(f"[ASYNC_WEAVIATE] Client closed/not connected during collection.get(), reconnecting and retrying (attempt {_retry_count + 1}/{max_retries - 1})...", flush=True)
                    logger.warning(f"Client closed during collection.get(), reconnecting: {e}")
                    await self.connect()
                    # Retry the entire search - will get collection again
                    return await self.search(query, node_types, limit, alpha, _retry_count=_retry_count + 1)
                else:
                    print(f"[ASYNC_WEAVIATE] Failed to get collection after {max_retries} attempts: {e}", flush=True)
                    logger.error(f"Failed to get collection after {max_retries} attempts: {e}")
                    return []
            else:
                # Other error - return empty results
                print(f"[ASYNC_WEAVIATE] Non-connection error getting collection: {e}", flush=True)
                logger.error(f"Non-connection error getting collection: {e}")
                return []

        # Perform async hybrid search
        # Collection is fresh and bound to connected client at this point
        try:
            if where_filter:
                response = await collection.query.hybrid(
                    query=query,
                    vector=query_vector,
                    alpha=alpha,
                    limit=limit,
                    filters=where_filter
                )
            else:
                response = await collection.query.hybrid(
                    query=query,
                    vector=query_vector,
                    alpha=alpha,
                    limit=limit
                )

            # Format results
            results = []
            for obj in response.objects:
                results.append({
                    'node_id': obj.properties.get('node_id'),
                    'node_type': obj.properties.get('node_type'),
                    'name': obj.properties.get('name'),
                    'file_path': obj.properties.get('file_path'),
                    'repository': obj.properties.get('repository'),
                    'content': obj.properties.get('content'),
                    'code': obj.properties.get('code', ''),  # Include full source code
                    'score': obj.metadata.score if hasattr(obj.metadata, 'score') else 0.0,
                    'metadata': json.loads(obj.properties.get('metadata', '{}'))
                })

            print(f"[ASYNC_WEAVIATE] Search successful, found {len(results)} results", flush=True)
            logger.info(f"Search successful, found {len(results)} results")
            return results

        except Exception as e:
            error_msg = str(e)
            print(f"[ASYNC_WEAVIATE] Search query failed: {error_msg}", flush=True)
            logger.error(f"Async search failed: {e}")
            
            # If query fails due to closed client, reconnect, get collection again, and retry once
            if "closed" in error_msg.lower() or "not connected" in error_msg.lower() or "connection" in error_msg.lower():
                if _retry_count < max_retries - 1:
                    print(f"[ASYNC_WEAVIATE] Search failed due to connection issue, reconnecting and retrying (attempt {_retry_count + 1}/{max_retries - 1})...", flush=True)
                    logger.warning(f"Search failed due to connection issue, reconnecting and retrying (attempt {_retry_count + 1}/{max_retries - 1})...")
                    # Reconnect client
                    await self.connect()
                    # Get collection again - it's now bound to reconnected client
                    try:
                        collection = self._client.collections.get("CodeNode")
                        # Retry query with fresh collection
                        if where_filter:
                            response = await collection.query.hybrid(
                                query=query,
                                vector=query_vector,
                                alpha=alpha,
                                limit=limit,
                                filters=where_filter
                            )
                        else:
                            response = await collection.query.hybrid(
                                query=query,
                                vector=query_vector,
                                alpha=alpha,
                                limit=limit
                            )
                        # Format results
                        results = []
                        for obj in response.objects:
                            results.append({
                                'node_id': obj.properties.get('node_id'),
                                'node_type': obj.properties.get('node_type'),
                                'name': obj.properties.get('name'),
                                'file_path': obj.properties.get('file_path'),
                                'repository': obj.properties.get('repository'),
                                'content': obj.properties.get('content'),
                                'code': obj.properties.get('code', ''),  # Include full source code
                                'score': obj.metadata.score if hasattr(obj.metadata, 'score') else 0.0,
                                'metadata': json.loads(obj.properties.get('metadata', '{}'))
                            })
                        print(f"[ASYNC_WEAVIATE] Retry successful, found {len(results)} results", flush=True)
                        logger.info(f"Retry successful, found {len(results)} results")
                        return results
                    except Exception as retry_error:
                        # Retry also failed, fall through to return empty
                        print(f"[ASYNC_WEAVIATE] Retry also failed: {retry_error}", flush=True)
                        logger.error(f"Retry also failed: {retry_error}")
                        return []
                else:
                    print(f"[ASYNC_WEAVIATE] Search failed after all reconnection retries", flush=True)
                    logger.error("Search failed after all reconnection retries")
                    return []
            else:
                # Other error - return empty results
                print(f"[ASYNC_WEAVIATE] Search failed with non-connection error: {e}", flush=True)
                logger.error(f"Search failed with non-connection error: {e}")
                return []
