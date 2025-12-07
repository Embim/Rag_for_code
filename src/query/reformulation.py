"""
Query Reformulation for Code RAG.

Uses LLM to improve queries for better code search results.

Methods:
- simple: Basic clarification and specificity
- expanded: Add context and related terms
- multi: Generate multiple query variants
- rephrase: Rephrase for technical terminology
- decompose: Break complex queries into sub-queries
- clarify: Clarify ambiguous parts
"""

import os
import re
import hashlib
from typing import List, Optional, Dict, Any, Literal
from dataclasses import dataclass, field
from pathlib import Path

from ..logger import get_logger

logger = get_logger(__name__)

# Method type
ReformulationMethod = Literal["simple", "expanded", "multi", "rephrase", "decompose", "clarify", "all"]


@dataclass
class ReformulationConfig:
    """Configuration for query reformulation."""
    enabled: bool = True
    method: ReformulationMethod = "simple"
    model: str = field(
        default_factory=lambda: os.getenv(
            "QUERY_REFORMULATION_MODEL",
            "tngtech/tng-r1t-chimera:free"
        )
    )
    max_tokens: int = 500
    temperature: float = 0.3
    use_cache: bool = True
    cache_dir: str = "cache/query_reformulation"


class QueryReformulator:
    """
    LLM-based query reformulation for code search.
    
    Improves search queries by:
    - Adding technical terminology
    - Clarifying ambiguous terms
    - Decomposing complex queries
    - Generating multiple variants
    """
    
    # Prompt templates for different methods
    PROMPTS = {
        "simple": """Переформулируй запрос для поиска по коду.
Сделай его более конкретным и техническим.

Запрос: {query}

Ответь ТОЛЬКО переформулированным запросом, без пояснений.""",

        "expanded": """Расширь запрос для поиска по коду, добавив контекст.

Запрос: {query}

Добавь:
- Технические термины
- Названия файлов/функций которые могут быть связаны
- Уточняющие детали

Ответь ТОЛЬКО расширенным запросом.""",

        "multi": """Сгенерируй 3 варианта поискового запроса по коду.

Оригинал: {query}

Формат:
1. [первый вариант]
2. [второй вариант]
3. [третий вариант]

Только варианты, без пояснений.""",

        "rephrase": """Переформулируй вопрос используя правильную техническую терминологию.

Запрос: {query}

Используй:
- Названия паттернов (если применимо)
- Правильные термины (endpoint вместо url, handler вместо функция обработки)
- Конкретные технологии

Ответь ТОЛЬКО переформулированным запросом.""",

        "decompose": """Разбей сложный вопрос на простые подзапросы.

Вопрос: {query}

Формат:
1. [первый подзапрос]
2. [второй подзапрос]
3. [третий подзапрос]

Каждый подзапрос должен быть самостоятельным для поиска.""",

        "clarify": """Уточни неясные части запроса для поиска по коду.

Запрос: {query}

Если есть неясности:
- Добавь вероятный контекст
- Уточни технологию (React? Django? FastAPI?)
- Конкретизируй тип сущности (компонент? эндпоинт? модель?)

Ответь ТОЛЬКО уточненным запросом.""",
    }
    
    def __init__(
        self,
        config: Optional[ReformulationConfig] = None,
        llm_client=None,
        api_key: Optional[str] = None
    ):
        """
        Initialize reformulator.
        
        Args:
            config: Reformulation configuration
            llm_client: Pre-configured OpenAI client
            api_key: OpenRouter API key (if llm_client not provided)
        """
        self.config = config or ReformulationConfig()
        self._client = llm_client
        self._api_key = api_key
        
        # Cache
        self._cache: Dict[str, str] = {}
        if self.config.use_cache:
            self.cache_dir = Path(self.config.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    @property
    def client(self):
        """Lazy-load OpenAI client."""
        if self._client is None:
            import os
            from openai import OpenAI
            
            api_key = self._api_key or os.getenv("OPENROUTER_API_KEY")
            if not api_key:
                raise ValueError(
                    "OPENROUTER_API_KEY not set. "
                    "Get free key at https://openrouter.ai/keys"
                )
            
            self._client = OpenAI(
                api_key=api_key,
                base_url="https://openrouter.ai/api/v1"
            )
        return self._client
    
    def _get_cache_key(self, query: str, method: str) -> str:
        """Generate cache key."""
        content = f"{method}:{query}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _get_cached(self, query: str, method: str) -> Optional[str]:
        """Get cached result."""
        if not self.config.use_cache:
            return None
        
        key = self._get_cache_key(query, method)
        
        # Memory cache
        if key in self._cache:
            return self._cache[key]
        
        # File cache
        cache_file = self.cache_dir / f"{key}.txt"
        if cache_file.exists():
            result = cache_file.read_text(encoding='utf-8')
            self._cache[key] = result
            return result
        
        return None
    
    def _set_cached(self, query: str, method: str, result: str):
        """Set cached result."""
        if not self.config.use_cache:
            return
        
        key = self._get_cache_key(query, method)
        self._cache[key] = result
        
        cache_file = self.cache_dir / f"{key}.txt"
        cache_file.write_text(result, encoding='utf-8')
    
    def _call_llm(self, prompt: str) -> str:
        """Call LLM API."""
        try:
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": "Ты помощник для улучшения поисковых запросов по коду."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature
            )
            
            text = response.choices[0].message.content
            return self._clean_response(text)
            
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return ""
    
    def _clean_response(self, text: str) -> str:
        """Clean LLM response from reasoning tags."""
        if not text:
            return text

        original_text = text

        # Remove thinking/reasoning tags
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<reasoning>.*?</reasoning>', '', text, flags=re.DOTALL | re.IGNORECASE)

        # Clean whitespace
        text = text.strip()

        # Validation: ensure reformulated query is reasonable
        # If result is too short (< 3 chars) or just punctuation, return empty
        if len(text) < 3 or text.replace('.', '').replace(',', '').strip() == '':
            logger.warning(f"Query reformulation produced invalid result: '{text}' from '{original_text[:100]}'")
            return ""

        return text
    
    def reformulate(
        self,
        query: str,
        method: Optional[ReformulationMethod] = None
    ) -> str:
        """
        Reformulate a single query.
        
        Args:
            query: Original query
            method: Reformulation method
            
        Returns:
            Reformulated query
        """
        if not self.config.enabled:
            return query
        
        method = method or self.config.method
        
        # Check cache
        cached = self._get_cached(query, method)
        if cached:
            logger.debug(f"Cache hit for query: {query[:50]}...")
            return cached
        
        # Get prompt
        prompt_template = self.PROMPTS.get(method, self.PROMPTS["simple"])
        prompt = prompt_template.format(query=query)
        
        # Call LLM
        result = self._call_llm(prompt)
        
        if result:
            self._set_cached(query, method, result)
            return result
        
        return query  # Return original if failed
    
    def reformulate_multi(
        self,
        query: str,
        methods: Optional[List[ReformulationMethod]] = None
    ) -> List[str]:
        """
        Generate multiple reformulations.
        
        Args:
            query: Original query
            methods: List of methods to use
            
        Returns:
            List of reformulated queries
        """
        if not self.config.enabled:
            return [query]
        
        methods = methods or ["simple", "expanded", "rephrase"]
        results = [query]  # Always include original
        
        for method in methods:
            result = self.reformulate(query, method)
            if result and result != query:
                results.append(result)
        
        return list(set(results))  # Remove duplicates
    
    def decompose(self, query: str) -> List[str]:
        """
        Decompose complex query into sub-queries.
        
        Args:
            query: Complex query
            
        Returns:
            List of simpler sub-queries
        """
        result = self.reformulate(query, "decompose")
        
        # Parse numbered list
        sub_queries = []
        for line in result.split('\n'):
            line = line.strip()
            # Remove numbering
            line = re.sub(r'^[\d]+[.\)]\s*', '', line)
            if line:
                sub_queries.append(line)
        
        return sub_queries if sub_queries else [query]
    
    def generate_variants(self, query: str, count: int = 3) -> List[str]:
        """
        Generate query variants.
        
        Args:
            query: Original query
            count: Number of variants
            
        Returns:
            List of variants including original
        """
        result = self.reformulate(query, "multi")
        
        # Parse numbered list
        variants = [query]
        for line in result.split('\n'):
            line = line.strip()
            line = re.sub(r'^[\d]+[.\)]\s*', '', line)
            if line and line != query:
                variants.append(line)
        
        return variants[:count + 1]  # +1 for original


# Convenience function
def reformulate_query(
    query: str,
    method: ReformulationMethod = "simple",
    api_key: Optional[str] = None
) -> str:
    """
    Convenience function to reformulate a query.
    
    Args:
        query: Original query
        method: Reformulation method
        api_key: Optional API key
        
    Returns:
        Reformulated query
    """
    reformulator = QueryReformulator(api_key=api_key)
    return reformulator.reformulate(query, method)

