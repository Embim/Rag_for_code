"""
Agent Result Caching.

Implements multi-level caching for agent results:
1. Query cache - Exact question match
2. Semantic cache - Similar questions (via embeddings)
3. Tool result cache - Expensive tool results
4. LLM response cache - Identical prompts

Backends:
- Memory: In-memory LRU cache (default, no dependencies)
- Redis: Distributed cache (requires redis-py)
"""

import hashlib
import time
from typing import Any, Dict, Optional, List, Tuple
from dataclasses import dataclass
from collections import OrderedDict
import json

from ..logger import get_logger


logger = get_logger(__name__)


@dataclass
class CacheConfig:
    """Cache configuration."""
    enabled: bool = True
    backend: str = "memory"  # "memory" or "redis"

    # TTL settings
    query_ttl: int = 86400  # 24 hours
    tool_result_ttl: int = 3600  # 1 hour
    llm_response_ttl: int = 3600  # 1 hour

    # Semantic cache
    semantic_similarity_threshold: float = 0.95

    # Memory backend settings
    max_memory_entries: int = 1000

    # Redis settings (if backend="redis")
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0


class MemoryCache:
    """
    In-memory LRU cache.

    Simple, no dependencies, good for single-instance deployments.
    """

    def __init__(self, max_entries: int = 1000):
        self.max_entries = max_entries
        self.cache: OrderedDict = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if key in self.cache:
            entry = self.cache[key]

            # Check expiration
            if entry['expires_at'] > time.time():
                # Move to end (LRU)
                self.cache.move_to_end(key)
                self.hits += 1
                return entry['value']
            else:
                # Expired
                del self.cache[key]
                self.misses += 1
                return None

        self.misses += 1
        return None

    def set(self, key: str, value: Any, ttl: int):
        """Set value in cache with TTL."""
        # Remove oldest if at capacity
        if len(self.cache) >= self.max_entries:
            self.cache.popitem(last=False)

        self.cache[key] = {
            'value': value,
            'expires_at': time.time() + ttl,
            'created_at': time.time(),
        }

    def delete(self, key: str):
        """Delete key from cache."""
        if key in self.cache:
            del self.cache[key]

    def clear(self):
        """Clear all cache."""
        self.cache.clear()
        self.hits = 0
        self.misses = 0

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0.0

        return {
            'backend': 'memory',
            'size': len(self.cache),
            'max_size': self.max_entries,
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
        }


class RedisCache:
    """
    Redis-backed cache.

    For distributed deployments with multiple instances.
    """

    def __init__(self, host: str = "localhost", port: int = 6379, db: int = 0):
        try:
            import redis
            self.redis = redis.Redis(host=host, port=port, db=db, decode_responses=True)
            self.redis.ping()  # Test connection
            logger.info(f"Connected to Redis at {host}:{port}")
        except ImportError:
            raise ImportError("redis-py not installed. Install with: pip install redis")
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Redis: {e}")

    def get(self, key: str) -> Optional[Any]:
        """Get value from Redis."""
        value = self.redis.get(key)
        if value:
            return json.loads(value)
        return None

    def set(self, key: str, value: Any, ttl: int):
        """Set value in Redis with TTL."""
        self.redis.setex(key, ttl, json.dumps(value))

    def delete(self, key: str):
        """Delete key from Redis."""
        self.redis.delete(key)

    def clear(self):
        """Clear all cache (careful in production!)"""
        self.redis.flushdb()

    def get_stats(self) -> Dict[str, Any]:
        """Get Redis statistics."""
        info = self.redis.info('stats')

        return {
            'backend': 'redis',
            'keyspace_hits': info.get('keyspace_hits', 0),
            'keyspace_misses': info.get('keyspace_misses', 0),
            'hit_rate': info.get('keyspace_hits', 0) / max(1, info.get('keyspace_hits', 0) + info.get('keyspace_misses', 0)),
        }


class AgentCache:
    """
    Multi-level cache for agent results.

    Provides caching at different levels with appropriate TTLs.
    """

    def __init__(self, config: CacheConfig, embedding_model=None):
        """
        Initialize cache.

        Args:
            config: Cache configuration
            embedding_model: Optional embedding model for semantic cache
        """
        self.config = config
        self.embedding_model = embedding_model

        # Initialize backend
        if not config.enabled:
            logger.info("Cache disabled")
            self.backend = None
        elif config.backend == "redis":
            self.backend = RedisCache(
                host=config.redis_host,
                port=config.redis_port,
                db=config.redis_db,
            )
        else:
            self.backend = MemoryCache(max_entries=config.max_memory_entries)

        # Semantic cache storage (embeddings)
        self.semantic_cache: List[Tuple[str, Any, float]] = []  # (query, embedding, timestamp)

    def _hash_key(self, namespace: str, key: str) -> str:
        """Generate cache key."""
        return f"code_rag:agent:{namespace}:{hashlib.md5(key.encode()).hexdigest()}"

    # Query Cache
    def get_query_result(self, question: str) -> Optional[Dict[str, Any]]:
        """Get cached result for exact question."""
        if not self.config.enabled:
            return None

        key = self._hash_key("query", question)
        result = self.backend.get(key)

        if result:
            logger.info(f"Query cache HIT: {question[:50]}")
        return result

    def set_query_result(self, question: str, result: Dict[str, Any]):
        """Cache result for question."""
        if not self.config.enabled:
            return

        key = self._hash_key("query", question)
        self.backend.set(key, result, self.config.query_ttl)
        logger.info(f"Query cached: {question[:50]}")

    # Semantic Cache
    def get_similar_query(self, question: str) -> Optional[Tuple[str, Dict[str, Any]]]:
        """
        Find similar cached question using embeddings.

        Returns:
            Tuple of (similar_question, result) if found, else None
        """
        if not self.config.enabled or not self.embedding_model:
            return None

        # Get embedding for question
        try:
            question_embedding = self.embedding_model.encode(
                [question],
                show_progress_bar=False,
                convert_to_numpy=True
            )[0]
        except Exception as e:
            logger.warning(f"Failed to encode question for semantic cache: {e}")
            return None

        # Find most similar cached question
        best_similarity = 0.0
        best_match = None

        for cached_question, cached_embedding, timestamp in self.semantic_cache:
            # Check if expired
            if time.time() - timestamp > self.config.query_ttl:
                continue

            # Compute cosine similarity
            similarity = self._cosine_similarity(question_embedding, cached_embedding)

            if similarity > best_similarity and similarity >= self.config.semantic_similarity_threshold:
                best_similarity = similarity
                # Get result from cache
                result = self.get_query_result(cached_question)
                if result:
                    best_match = (cached_question, result)

        if best_match:
            logger.info(f"Semantic cache HIT: {best_match[0][:50]} (similarity: {best_similarity:.2f})")

        return best_match

    def add_to_semantic_cache(self, question: str):
        """Add question to semantic cache index."""
        if not self.config.enabled or not self.embedding_model:
            return

        try:
            embedding = self.embedding_model.encode(
                [question],
                show_progress_bar=False,
                convert_to_numpy=True
            )[0]
            self.semantic_cache.append((question, embedding, time.time()))

            # Limit size
            if len(self.semantic_cache) > 1000:
                # Remove oldest
                self.semantic_cache = self.semantic_cache[-1000:]

        except Exception as e:
            logger.warning(f"Failed to add to semantic cache: {e}")

    def _cosine_similarity(self, a, b) -> float:
        """Compute cosine similarity between vectors."""
        import numpy as np
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    # Tool Result Cache
    def get_tool_result(self, tool_name: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get cached tool result."""
        if not self.config.enabled:
            return None

        # Create key from tool name + params
        params_str = json.dumps(params, sort_keys=True)
        key = self._hash_key("tool", f"{tool_name}:{params_str}")

        result = self.backend.get(key)
        if result:
            logger.info(f"Tool cache HIT: {tool_name}")
        return result

    def set_tool_result(self, tool_name: str, params: Dict[str, Any], result: Dict[str, Any]):
        """Cache tool result."""
        if not self.config.enabled:
            return

        params_str = json.dumps(params, sort_keys=True)
        key = self._hash_key("tool", f"{tool_name}:{params_str}")

        self.backend.set(key, result, self.config.tool_result_ttl)
        logger.debug(f"Tool result cached: {tool_name}")

    # LLM Response Cache
    def get_llm_response(self, prompt: str, model: str) -> Optional[str]:
        """Get cached LLM response."""
        if not self.config.enabled:
            return None

        key = self._hash_key("llm", f"{model}:{prompt}")
        result = self.backend.get(key)

        if result:
            logger.info(f"LLM cache HIT")
        return result

    def set_llm_response(self, prompt: str, model: str, response: str):
        """Cache LLM response."""
        if not self.config.enabled:
            return

        key = self._hash_key("llm", f"{model}:{prompt}")
        self.backend.set(key, response, self.config.llm_response_ttl)
        logger.debug(f"LLM response cached")

    # Invalidation
    def invalidate_repository(self, repo_name: str):
        """
        Invalidate cache for a repository.

        Called when repository is reindexed.
        """
        if not self.config.enabled:
            return

        logger.info(f"Invalidating cache for repository: {repo_name}")

        # For memory backend, we can't efficiently invalidate by repo
        # For Redis, we could use key patterns
        # For MVP, just log warning
        logger.warning("Repository-specific invalidation not implemented. Consider clearing entire cache.")

    def clear(self):
        """Clear all cache."""
        if self.config.enabled and self.backend:
            self.backend.clear()
            self.semantic_cache.clear()
            logger.info("Cache cleared")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if not self.config.enabled or not self.backend:
            return {'enabled': False}

        stats = self.backend.get_stats()
        stats['enabled'] = True
        stats['semantic_cache_size'] = len(self.semantic_cache)
        return stats
