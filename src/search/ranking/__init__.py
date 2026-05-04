"""
Ranking module for Code RAG.

- CrossEncoderReranker: neural reranking with cross-encoder models.
"""

from .cross_encoder import CrossEncoderReranker

__all__ = [
    'CrossEncoderReranker',
]
