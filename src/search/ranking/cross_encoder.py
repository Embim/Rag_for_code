"""
Cross-Encoder Reranker for neural reranking of search results.

100x faster than LLM reranking with comparable quality.
"""

import os
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass, field

from src.infra.logger import get_logger

logger = get_logger(__name__)

# Lazy load to avoid startup cost
_cross_encoder_model = None


@dataclass
class RerankerConfig:
    """Configuration for cross-encoder reranker."""
    model_name: str = field(
        default_factory=lambda: os.getenv(
            "RERANKER_MODEL",
            # bge-reranker-v2-m3 — multilingual, 8K context. Лучше чем
            # mmarco-mMiniLMv2 для кода: видит до ~24K chars вместо 2K
            # (т.е. полное тело крупного метода, а не только init).
            "BAAI/bge-reranker-v2-m3"
        )
    )
    # 1024 — баланс между скоростью и покрытием. Для bge-v2-m3 можно
    # поднять до 8192, но это ×16 latency. 1024 покрывает ~50-80 LOC,
    # этого хватает чтобы увидеть основную формулу метода.
    max_length: int = 1024
    batch_size: int = 32
    device: Optional[str] = None  # auto-detect


class CrossEncoderReranker:
    """
    Cross-Encoder for reranking search results.
    
    Advantages over LLM:
    - Speed: 0.1 sec vs 10 sec per query
    - VRAM: 1-2 GB vs 32 GB
    - Accuracy: comparable or better
    
    Popular models:
    - cross-encoder/ms-marco-MiniLM-L-12-v2: fast, good quality
    - cross-encoder/ms-marco-MiniLM-L-6-v2: very fast
    - cross-encoder/mmarco-mMiniLMv2-L12-H384-v1: multilingual (Russian!)
    """
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        max_length: int = 1024,
        device: Optional[str] = None
    ):
        """
        Initialize reranker.

        Args:
            model_name: HuggingFace model name (reads from RERANKER_MODEL env if None)
            max_length: Maximum sequence length (default 1024 для bge-v2-m3;
                для cross-encoder/mmarco* лимит модели всё равно 512)
            device: Device (cpu/cuda/auto)
        """
        self.model_name = model_name or os.getenv(
            "RERANKER_MODEL",
            "BAAI/bge-reranker-v2-m3"
        )
        self.max_length = max_length
        self.device = device
        self._model = None
    
    @property
    def model(self):
        """Lazy load model on first use."""
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
                logger.info(f"Loading Cross-Encoder: {self.model_name}")
                self._model = CrossEncoder(
                    self.model_name,
                    max_length=self.max_length,
                    device=self.device
                )
                logger.info("✅ Cross-Encoder loaded")
            except ImportError:
                raise ImportError(
                    "sentence-transformers required. Install: pip install sentence-transformers"
                )
        return self._model
    
    def rerank(
        self,
        query: str,
        documents: List[dict],
        top_k: int = 20,
        text_field: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Rerank documents using cross-encoder.
        
        Args:
            query: Search query
            documents: List of documents (dict with text field)
            top_k: Number of top documents to return
            text_field: Field name containing text (auto-detect if None)
            
        Returns:
            DataFrame with reranked documents
        """
        if not documents:
            return pd.DataFrame()
        
        # Auto-detect text field
        if text_field is None:
            for field in ['clean_text', 'text', 'content', 'code', 'docstring']:
                if field in documents[0]:
                    text_field = field
                    break
            else:
                text_field = 'text'
        
        # Create (query, document) pairs.
        # Trim ≈ 4 chars/token: max_length=1024 → ~4000 chars; 8192 → ~32000.
        # Старое значение 2000 chars (≈ 512 tokens) для mmarco-MiniLM было
        # жёстким лимитом самой модели, у bge-v2-m3 лимит 8192 — даём ему
        # видеть больше кода (полная картина метода, а не только init).
        char_budget = max(2000, self.max_length * 4)
        pairs = []
        for doc in documents:
            doc_text = str(doc.get(text_field, ''))[:char_budget]
            pairs.append([query, doc_text])
        
        # Get scores
        logger.debug(f"Reranking {len(pairs)} documents...")
        scores = self.model.predict(pairs, show_progress_bar=False)
        
        # Add scores to documents
        results = []
        for doc, score in zip(documents, scores):
            doc_copy = doc.copy()
            doc_copy['rerank_score'] = float(score)
            results.append(doc_copy)
        
        # Sort by score
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('rerank_score', ascending=False)
        
        # Take top-k
        return results_df.head(top_k).reset_index(drop=True)
    
    def rerank_with_scores(
        self,
        query: str,
        documents: List[dict],
        text_field: Optional[str] = None
    ) -> List[Tuple[dict, float]]:
        """
        Rerank and return (document, score) tuples.
        
        Args:
            query: Search query
            documents: List of documents
            text_field: Field containing text
            
        Returns:
            List of (document, score) tuples sorted by score descending
        """
        if not documents:
            return []
        
        # Auto-detect text field
        if text_field is None:
            for field in ['clean_text', 'text', 'content', 'code']:
                if field in documents[0]:
                    text_field = field
                    break
            else:
                text_field = 'text'
        
        char_budget = max(2000, self.max_length * 4)
        pairs = [
            [query, str(doc.get(text_field, ''))[:char_budget]]
            for doc in documents
        ]
        
        scores = self.model.predict(pairs, show_progress_bar=False)
        
        results = list(zip(documents, scores))
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results
    
    def batch_rerank(
        self,
        queries_and_docs: List[Tuple[str, List[dict]]],
        top_k: int = 20
    ) -> List[pd.DataFrame]:
        """
        Batch reranking for multiple queries.
        
        Args:
            queries_and_docs: List of (query, documents) pairs
            top_k: Top-k for each query
            
        Returns:
            List of DataFrames with results
        """
        return [
            self.rerank(query, docs, top_k=top_k)
            for query, docs in queries_and_docs
        ]


# Backward compatibility alias
def get_reranker(model_name: Optional[str] = None) -> CrossEncoderReranker:
    """
    Get or create singleton reranker instance.

    Args:
        model_name: Model name (reads from RERANKER_MODEL env if None)
    """
    global _cross_encoder_model
    actual_model = model_name or os.getenv(
        "RERANKER_MODEL",
        "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    )
    if _cross_encoder_model is None or _cross_encoder_model.model_name != actual_model:
        _cross_encoder_model = CrossEncoderReranker(actual_model)
    return _cross_encoder_model

