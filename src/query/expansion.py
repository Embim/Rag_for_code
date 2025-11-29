"""
Query Expansion for Code RAG.

Expands queries with:
- Code-related synonyms
- Programming language mappings
- Framework-specific terms
"""

from typing import List, Dict, Optional, Set
from dataclasses import dataclass, field
import re

from ..logger import get_logger

logger = get_logger(__name__)


@dataclass
class ExpansionConfig:
    """Configuration for query expansion."""
    use_synonyms: bool = True
    use_code_terms: bool = True
    max_expansions: int = 5
    include_original: bool = True


class QueryExpander:
    """
    Expand queries for better code search coverage.
    
    Features:
    - Programming language synonyms
    - Framework-specific term mappings
    - Abbreviation expansion
    - Camel case / snake case variations
    """
    
    # Code-related synonyms
    CODE_SYNONYMS = {
        # Functions/Methods
        "function": ["method", "func", "def", "fn", "procedure"],
        "method": ["function", "member function", "instance method"],
        "handler": ["callback", "listener", "hook", "event handler"],
        
        # Classes/Types
        "class": ["type", "struct", "interface", "model"],
        "interface": ["protocol", "contract", "abstract class"],
        "enum": ["enumeration", "constants", "choices"],
        
        # Components (React)
        "component": ["widget", "element", "view", "ui component"],
        "hook": ["use hook", "custom hook", "react hook"],
        "props": ["properties", "attributes", "parameters"],
        "state": ["local state", "component state", "useState"],
        
        # API/Backend
        "endpoint": ["route", "api endpoint", "url", "path"],
        "controller": ["handler", "view", "api view"],
        "model": ["entity", "schema", "table", "orm model"],
        "serializer": ["schema", "dto", "data transfer object"],
        
        # Database
        "database": ["db", "storage", "persistence"],
        "query": ["sql", "select", "find", "fetch"],
        "migration": ["schema change", "db migration", "alembic"],
        
        # Auth
        "authentication": ["auth", "login", "sign in", "authenticate"],
        "authorization": ["permissions", "access control", "rbac"],
        "token": ["jwt", "access token", "bearer token"],
        
        # Actions
        "create": ["add", "insert", "new", "make", "post"],
        "read": ["get", "fetch", "retrieve", "find", "select"],
        "update": ["edit", "modify", "change", "put", "patch"],
        "delete": ["remove", "destroy", "drop", "erase"],
        
        # Errors
        "error": ["exception", "error handling", "catch", "try"],
        "validation": ["validate", "check", "verify", "sanitize"],
    }
    
    # Framework-specific mappings
    FRAMEWORK_TERMS = {
        "django": {
            "view": ["viewset", "apiview", "generic view"],
            "model": ["django model", "orm model"],
            "form": ["modelform", "django form"],
            "admin": ["django admin", "admin site"],
            "middleware": ["django middleware"],
            "signal": ["django signal", "post_save", "pre_save"],
        },
        "fastapi": {
            "endpoint": ["path operation", "route", "api route"],
            "dependency": ["depends", "fastapi depends"],
            "schema": ["pydantic model", "basemodel"],
            "middleware": ["fastapi middleware"],
        },
        "react": {
            "component": ["functional component", "class component"],
            "hook": ["usestate", "useeffect", "usememo", "usecallback"],
            "context": ["react context", "createcontext", "usecontext"],
            "redux": ["store", "reducer", "action", "dispatch"],
        },
    }
    
    # Abbreviation expansions
    ABBREVIATIONS = {
        "api": "application programming interface",
        "ui": "user interface",
        "ux": "user experience",
        "db": "database",
        "orm": "object relational mapping",
        "jwt": "json web token",
        "dto": "data transfer object",
        "crud": "create read update delete",
        "rest": "representational state transfer",
        "http": "hypertext transfer protocol",
        "url": "uniform resource locator",
        "sql": "structured query language",
        "css": "cascading style sheets",
        "html": "hypertext markup language",
        "json": "javascript object notation",
        "xml": "extensible markup language",
        "tsx": "typescript jsx",
        "jsx": "javascript xml",
    }
    
    def __init__(self, config: Optional[ExpansionConfig] = None):
        """
        Initialize query expander.
        
        Args:
            config: Expansion configuration
        """
        self.config = config or ExpansionConfig()
    
    def expand(
        self,
        query: str,
        framework: Optional[str] = None,
        max_expansions: Optional[int] = None
    ) -> List[str]:
        """
        Expand query with variations.
        
        Args:
            query: Original query
            framework: Optional framework context (django/fastapi/react)
            max_expansions: Maximum number of expansions
            
        Returns:
            List of query variations
        """
        max_exp = max_expansions or self.config.max_expansions
        expansions: Set[str] = set()
        
        if self.config.include_original:
            expansions.add(query)
        
        query_lower = query.lower()
        
        # Apply synonyms
        if self.config.use_synonyms:
            for term, synonyms in self.CODE_SYNONYMS.items():
                if term in query_lower:
                    for syn in synonyms[:3]:  # Limit synonyms per term
                        expanded = query_lower.replace(term, syn)
                        expansions.add(expanded)
        
        # Apply framework-specific terms
        if framework and framework.lower() in self.FRAMEWORK_TERMS:
            fw_terms = self.FRAMEWORK_TERMS[framework.lower()]
            for term, variations in fw_terms.items():
                if term in query_lower:
                    for var in variations[:2]:
                        expanded = query_lower.replace(term, var)
                        expansions.add(expanded)
        
        # Expand abbreviations
        if self.config.use_code_terms:
            for abbr, full in self.ABBREVIATIONS.items():
                if abbr in query_lower.split():
                    expanded = query_lower.replace(abbr, full)
                    expansions.add(expanded)
        
        # Add case variations
        expansions.update(self._case_variations(query))
        
        # Limit results
        result = list(expansions)[:max_exp]
        
        logger.debug(f"Expanded '{query}' to {len(result)} variations")
        
        return result
    
    def _case_variations(self, query: str) -> List[str]:
        """Generate case variations (camelCase, snake_case, etc.)."""
        variations = []
        
        # Extract potential identifiers
        words = re.findall(r'\b\w+\b', query)
        
        for word in words:
            if len(word) < 3:
                continue
            
            # camelCase to snake_case
            if re.search(r'[a-z][A-Z]', word):
                snake = re.sub(r'([a-z])([A-Z])', r'\1_\2', word).lower()
                variations.append(query.replace(word, snake))
            
            # snake_case to camelCase
            if '_' in word:
                parts = word.split('_')
                camel = parts[0] + ''.join(p.title() for p in parts[1:])
                variations.append(query.replace(word, camel))
        
        return variations
    
    def extract_code_terms(self, query: str) -> List[str]:
        """
        Extract potential code identifiers from query.
        
        Args:
            query: Input query
            
        Returns:
            List of potential code terms
        """
        terms = []
        
        # CamelCase words
        camel_pattern = r'\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b'
        terms.extend(re.findall(camel_pattern, query))
        
        # snake_case words
        snake_pattern = r'\b[a-z]+(?:_[a-z]+)+\b'
        terms.extend(re.findall(snake_pattern, query))
        
        # ALL_CAPS constants
        caps_pattern = r'\b[A-Z][A-Z_]+[A-Z]\b'
        terms.extend(re.findall(caps_pattern, query))
        
        return list(set(terms))


# Convenience function
def expand_query(
    query: str,
    framework: Optional[str] = None,
    max_expansions: int = 5
) -> List[str]:
    """
    Convenience function to expand a query.
    
    Args:
        query: Original query
        framework: Optional framework context
        max_expansions: Maximum expansions
        
    Returns:
        List of query variations
    """
    expander = QueryExpander()
    return expander.expand(query, framework, max_expansions)

