"""
Weaviate indexer - indexes knowledge graph nodes for vector search.

This component bridges Neo4j knowledge graph and Weaviate vector search:
1. Takes nodes from Neo4j
2. Generates embeddings using BAAI/bge-m3
3. Indexes in Weaviate for semantic search
4. Enables hybrid search (graph structure + semantic similarity)
"""

import gc
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, TYPE_CHECKING

from tqdm import tqdm

if TYPE_CHECKING:
    from .neo4j_client import Neo4jClient

# Note: sentence_transformers import moved to lazy import to avoid blocking file system operations
# during module import (which blocks event loop in async contexts)

import weaviate
from weaviate.classes.config import Configure, Property, DataType
from weaviate.classes.query import Filter

from .code_loader import read_code
from .models import GraphNode, NodeType
# Note: Neo4jClient import moved to TYPE_CHECKING to avoid importing neo4j package
# when only AsyncWeaviateIndexer is needed (which doesn't use Neo4j)

from src.infra.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Chunking constants
# ---------------------------------------------------------------------------

# Полный размер кода, который мы сохраняем на одну ноду без расщепления.
# ~32 KB ≈ 1000 LOC покрывает 95% реальных функций/классов в проекте.
# Маленькие/средние функции → один chunk = одна нода в Weaviate (как раньше).
DEFAULT_MAX_CHARS = 32_000

# Когда функция > DEFAULT_MAX_CHARS — режем её на sliding-window sub-chunk'и.
# Каждый sub-chunk получает собственный embedding, сфокусированный на своём
# регионе, а не на размытой "первой части". В Weaviate каждый под-chunk —
# отдельный объект с node_id вида ``{original_id}#part{i}``.
SUBCHUNK_WINDOW = 8_000
SUBCHUNK_OVERLAP = 1_000

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
        neo4j_client: Optional['Neo4jClient'] = None,
        repos_dir: Optional[Path] = None,
    ):
        """
        Initialize Weaviate indexer.

        Args:
            weaviate_url: Weaviate connection URL
            embedding_model: Sentence transformer model name
            neo4j_client: Optional Neo4j client for fetching nodes
            repos_dir: Корневая директория репозиториев (для чтения кода с диска
                по ``file_path`` ноды). Если не задан — content при индексации
                пишется без исходного кода.
        """
        self.weaviate_url = weaviate_url
        self.neo4j_client = neo4j_client
        self.repos_dir = Path(repos_dir) if repos_dir else None
        self._embedding_model_name = embedding_model
        self._embedding_model = None  # Lazy loaded
        self._embedding_dim = None  # Lazy loaded

        # Initialize Weaviate client
        # skip_init_checks=True prevents event loop conflict with LangGraph async worker
        try:
            from weaviate.classes.init import AdditionalConfig, Timeout as WeaviateTimeout
            print(f"[WEAVIATE] Connecting to {weaviate_url}...", flush=True)
            host = weaviate_url.replace("http://", "").replace("https://", "").split(":")[0]
            port = int(weaviate_url.split(":")[-1]) if ":" in weaviate_url else 8080
            self.client = weaviate.connect_to_local(
                host=host,
                port=port,
                grpc_port=50051,  # Explicit gRPC port (matches docker-compose)
                skip_init_checks=True,  # Prevent deadlock in async context (LangGraph)
                additional_config=AdditionalConfig(
                    timeout=WeaviateTimeout(init=10, query=60, insert=120)
                ),
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
                # Lazy import to avoid blocking file system operations during module import
                from sentence_transformers import SentenceTransformer

                self._embedding_model = SentenceTransformer(self._embedding_model_name)
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

    def _load_code(
        self,
        node: GraphNode,
        *,
        max_chars: Optional[int] = None,
    ) -> str:
        """
        Read source code for a node from disk via ``code_loader.read_code``.

        По умолчанию возвращает **полный** фрагмент (без обрезки): callers
        сами решают, нужен ли trim — для маленьких функций он не нужен,
        для гигантских применяется sub-chunking (см. ``_split_node_into_subchunks``).

        Возвращает пустую строку, если ``self.repos_dir`` не задан или
        ``file_path`` отсутствует на ноде.
        """
        file_path = node.properties.get('file_path') or ''
        if not file_path:
            return ''
        start = int(node.properties.get('start_line') or 0)
        end = int(node.properties.get('end_line') or 0)

        # Repository name достаём из id (формат "repo:<name>:...")
        repo_root = None
        if self.repos_dir is not None and node.id.startswith('repo:'):
            try:
                repo_name = node.id.split(':', 2)[1]
                repo_root = self.repos_dir / repo_name
            except Exception:
                repo_root = None

        return read_code(file_path, start, end, repo_root=repo_root, max_chars=max_chars)

    def _split_node_into_subchunks(self, node: GraphNode) -> List[Dict[str, Any]]:
        """
        Расщепить ноду на 1+ sub-chunk для индексации.

        - Функции/методы ≤ ``DEFAULT_MAX_CHARS`` → один sub-chunk с оригинальным
          ``node.id``. Поведение совпадает с прошлой реализацией (до уменьшения
          лимита 8000 → 32000).
        - Большие ноды (>32K chars) → N sliding-window sub-chunk'ов. Каждый
          получает суффикс ``#part{i}`` в ``node_id``, общие метаданные
          (name, file_path, signature, docstring) и собственный код-окно
          с overlap'ом. Это даёт каждому региону свой embedding.

        Каждый элемент списка содержит:
            ``node_id``, ``code``, ``part_index`` (0-based), ``total_parts``.
        """
        full_code = self._load_code(node, max_chars=None)

        if len(full_code) <= DEFAULT_MAX_CHARS:
            return [{
                'node_id': node.id,
                'code': full_code,
                'part_index': 0,
                'total_parts': 1,
            }]

        sub_chunks: List[Dict[str, Any]] = []
        pos = 0
        while pos < len(full_code):
            end = min(pos + SUBCHUNK_WINDOW, len(full_code))
            sub_chunks.append({
                'node_id': f'{node.id}#part{len(sub_chunks)}',
                'code': full_code[pos:end],
                'part_index': len(sub_chunks),
                'total_parts': -1,  # обновим в конце
            })
            if end >= len(full_code):
                break
            pos = end - SUBCHUNK_OVERLAP

        total = len(sub_chunks)
        for sc in sub_chunks:
            sc['total_parts'] = total

        logger.info(
            f"[indexer] sub-chunking {node.id}: "
            f"{len(full_code)} chars → {total} parts"
        )
        return sub_chunks

    def _build_searchable_content(
        self,
        node: GraphNode,
        *,
        code_override: Optional[str] = None,
        part_info: Optional[Dict[str, int]] = None,
    ) -> str:
        """
        Build searchable text content from a node.

        Combines node name, signature, docstring, code, and other relevant text
        into a single searchable string for embedding generation.

        Args:
            node: Graph node
            code_override: явное тело кода для этого chunk'а. Используется при
                sub-chunking больших функций — каждый sub-chunk строит свой
                content с собственным окном кода. Если ``None`` — читаем с
                диска через ``_load_code`` (по умолчанию весь фрагмент).
            part_info: ``{'part_index': i, 'total_parts': N}`` для пометки
                "this is part i of N" в content (улучшает retrieval-debug).

        Returns:
            Searchable text content
        """
        parts = []

        # Node name and type
        parts.append(f"Name: {node.name}")
        parts.append(f"Type: {node.type.value}")

        # Module/package context from file path (e.g. "position_keeping.main" from path)
        file_path = node.properties.get('file_path', '')
        if file_path:
            parts.append(f"File: {file_path}")
            # Extract module path for semantic context
            module = file_path.replace('\\', '/').replace('/', '.').removesuffix('.py')
            if module:
                parts.append(f"Module: {module}")

        # Signature
        if node.properties.get('signature'):
            parts.append(f"Signature: {node.properties['signature']}")

        # Docstring — high semantic value, put early
        if node.properties.get('docstring'):
            docstring = node.properties['docstring'].strip()
            if docstring:
                parts.append(f"Documentation: {docstring}")

        # Source code — либо явно переданный (sub-chunk), либо весь фрагмент
        # с диска. Для больших нод splitter передаёт окно по SUBCHUNK_WINDOW.
        code = code_override if code_override is not None else self._load_code(node)
        if code:
            if part_info and part_info.get('total_parts', 1) > 1:
                parts.append(
                    f"Code (part {part_info['part_index'] + 1}"
                    f"/{part_info['total_parts']}):\n{code}"
                )
            else:
                parts.append(f"Code:\n{code}")

        # Type-specific extras
        if node.type == NodeType.ENDPOINT:
            method = node.properties.get('http_method', '')
            path = node.properties.get('path', '')
            if method or path:
                parts.append(f"Endpoint: {method} {path}")

        if node.type == NodeType.COMPONENT:
            props = node.properties.get('props_type', '')
            if props:
                parts.append(f"Props: {props}")
            hooks = node.properties.get('hooks_used', '')
            if hooks:
                parts.append(f"Hooks: {hooks}")

        if node.type == NodeType.MODEL:
            model_type = node.properties.get('model_type', '')
            if model_type:
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

        # Шаг 1: расщепить каждую ноду на sub-chunks (1 для маленьких, N для больших).
        sub_chunks_with_node: List[Tuple[GraphNode, Dict[str, Any]]] = []
        for node in nodes:
            for sc in self._split_node_into_subchunks(node):
                sub_chunks_with_node.append((node, sc))

        if len(sub_chunks_with_node) > len(nodes):
            extra = len(sub_chunks_with_node) - len(nodes)
            logger.info(
                f"Indexing {len(nodes)} nodes → "
                f"{len(sub_chunks_with_node)} sub-chunks (+{extra} from oversized splits)"
            )
        else:
            logger.info(f"Indexing {len(nodes)} nodes in Weaviate...")

        collection = self.client.collections.get("CodeNode")
        indexed_count = 0

        # Шаг 2: батчуем по sub-chunks (а не по нодам напрямую).
        for i in tqdm(
            range(0, len(sub_chunks_with_node), batch_size),
            desc="Indexing batches", unit="batch",
        ):
            batch = sub_chunks_with_node[i:i + batch_size]

            # Prepare data for batch
            batch_data = []
            batch_contents = []

            for node, sc in batch:
                content = self._build_searchable_content(
                    node,
                    code_override=sc['code'],
                    part_info={
                        'part_index': sc['part_index'],
                        'total_parts': sc['total_parts'],
                    },
                )
                batch_contents.append(content)

                # Метаданные включают информацию о sub-chunk'инге для отладки.
                meta = dict(node.properties)
                if sc['total_parts'] > 1:
                    meta['part_index'] = sc['part_index']
                    meta['total_parts'] = sc['total_parts']

                # Prepare properties
                properties = {
                    "node_id": sc['node_id'],
                    "node_type": node.type.value,
                    "name": node.name,
                    "file_path": node.properties.get('file_path', ''),
                    "repository": self._extract_repository_name(node.id),
                    "content": content,
                    "code": sc['code'],
                    "signature": node.properties.get('signature', ''),
                    "docstring": node.properties.get('docstring', ''),
                    "start_line": node.properties.get('start_line', 0),
                    "end_line": node.properties.get('end_line', 0),
                    "metadata": json.dumps(meta),
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
                logger.info(
                    f"Indexed batch {i // batch_size + 1}: "
                    f"{indexed_count}/{len(sub_chunks_with_node)} sub-chunks"
                )

                # Free GPU memory between batches to prevent fragmentation
                gc.collect()
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except ImportError:
                    pass

            except Exception as e:
                logger.error(f"Failed to index batch {i // batch_size + 1}: {e}")
                continue

        logger.info(f"Successfully indexed {indexed_count} sub-chunks in Weaviate")
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

        # По умолчанию индексируем только entity-уровень (Function/Class/etc),
        # File-ноды НЕ индексируем: их content — это весь файл (часто 100KB+),
        # обрезанный до DEFAULT_MAX_CHARS даёт размытый и неинформативный
        # embedding, который засоряет top-k. Function/Method/Class покрывают
        # содержимое файла точечно. Если нужно — пробрось NodeType.FILE явно.
        if node_types is None:
            node_types = [
                NodeType.FUNCTION,
                NodeType.CLASS,
                NodeType.COMPONENT,
                NodeType.ENDPOINT,
                NodeType.MODEL,
                NodeType.ROUTE,
            ]

        total_indexed = 0

        for node_type in tqdm(node_types, desc="Indexing node types", unit="type"):
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

