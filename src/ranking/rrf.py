"""
Reciprocal Rank Fusion (RRF) for combining search results.

RRF works better than weighted sum because:
- Independent of absolute score values
- Uses only ranks (positions)
- Proven more effective on benchmarks

Formula: RRF_score(d) = Σ 1/(k + rank(d))
where k=60 is constant, rank(d) is document position
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from dataclasses import dataclass

from ..logger import get_logger

logger = get_logger(__name__)


@dataclass
class RRFConfig:
    """Configuration for RRF."""
    k: int = 60  # RRF constant (usually 60, can be 20-100)
    normalize_scores: bool = True
    min_score_threshold: float = 0.0


class ReciprocalRankFusion:
    """
    Reciprocal Rank Fusion for combining results from different retrievers.
    
    Use cases:
    1. Dense + BM25 (hybrid search)
    2. Multiple embedding models
    3. Different query variants (query expansion)
    4. Multi-hop search results
    """
    
    def __init__(self, k: int = 60):
        """
        Initialize RRF.
        
        Args:
            k: RRF constant (default 60)
               Larger k → smaller difference between top results
               Smaller k → more weight to top results
        """
        self.k = k
    
    def compute_rrf_score(self, rank: int) -> float:
        """
        Compute RRF score for given rank.
        
        Args:
            rank: Position in list (starting from 1)
            
        Returns:
            RRF score
        """
        return 1.0 / (self.k + rank)
    
    def _detect_id_column(self, df: pd.DataFrame) -> str:
        """Auto-detect ID column."""
        for col in ['node_id', 'chunk_id', 'id', 'entity_id']:
            if col in df.columns:
                return col
        raise ValueError("Cannot find ID column (node_id, chunk_id, id, or entity_id)")
    
    def fuse_two_results(
        self,
        results1: pd.DataFrame,
        results2: pd.DataFrame,
        score_col1: str = 'retrieval_score',
        score_col2: str = 'retrieval_score',
        id_col: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Combine two result lists using RRF.
        
        Args:
            results1: First result list
            results2: Second result list
            score_col1: Score column in first list
            score_col2: Score column in second list
            id_col: ID column (auto-detect if None)
            
        Returns:
            Combined list with RRF scores
        """
        if results1.empty and results2.empty:
            return pd.DataFrame()
        
        if results1.empty:
            return results2.copy()
        
        if results2.empty:
            return results1.copy()
        
        # Auto-detect ID column
        if id_col is None:
            id_col = self._detect_id_column(results1)
        
        rrf_scores = defaultdict(lambda: {'score': 0.0, 'data': None})
        
        # Process first list
        for rank, (idx, row) in enumerate(results1.iterrows(), start=1):
            doc_id = row[id_col]
            rrf_score = self.compute_rrf_score(rank)
            
            rrf_scores[doc_id]['score'] += rrf_score
            rrf_scores[doc_id]['data'] = row.to_dict()
            rrf_scores[doc_id]['data']['rrf_score'] = rrf_score
            rrf_scores[doc_id]['data']['rank_1'] = rank
            rrf_scores[doc_id]['data']['original_score_1'] = row.get(score_col1, 0.0)
        
        # Process second list
        for rank, (idx, row) in enumerate(results2.iterrows(), start=1):
            doc_id = row[id_col]
            rrf_score = self.compute_rrf_score(rank)
            
            if doc_id in rrf_scores:
                rrf_scores[doc_id]['score'] += rrf_score
                rrf_scores[doc_id]['data']['rank_2'] = rank
                rrf_scores[doc_id]['data']['original_score_2'] = row.get(score_col2, 0.0)
            else:
                rrf_scores[doc_id]['score'] = rrf_score
                rrf_scores[doc_id]['data'] = row.to_dict()
                rrf_scores[doc_id]['data']['rrf_score'] = rrf_score
                rrf_scores[doc_id]['data']['rank_2'] = rank
                rrf_scores[doc_id]['data']['original_score_2'] = row.get(score_col2, 0.0)
        
        # Collect results
        merged_results = []
        for doc_id, info in rrf_scores.items():
            data = info['data'].copy()
            data['rrf_score'] = info['score']
            data['retrieval_score'] = info['score']  # backward compatibility
            merged_results.append(data)
        
        # Sort by RRF score
        results_df = pd.DataFrame(merged_results)
        results_df = results_df.sort_values('rrf_score', ascending=False)
        
        return results_df.reset_index(drop=True)
    
    def fuse_multiple_results(
        self,
        results_list: List[pd.DataFrame],
        score_cols: Optional[List[str]] = None,
        id_col: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Combine multiple result lists using RRF.
        
        Args:
            results_list: List of DataFrames with results
            score_cols: Score column names (default 'retrieval_score')
            id_col: ID column (auto-detect if None)
            
        Returns:
            Combined list with RRF scores
        """
        if not results_list:
            return pd.DataFrame()
        
        # Filter empty DataFrames
        non_empty = [df for df in results_list if not df.empty]
        
        if not non_empty:
            return pd.DataFrame()
        
        if len(non_empty) == 1:
            return non_empty[0].copy()
        
        if score_cols is None:
            score_cols = ['retrieval_score'] * len(non_empty)
        
        # Auto-detect ID column
        if id_col is None:
            id_col = self._detect_id_column(non_empty[0])
        
        rrf_scores = defaultdict(lambda: {'score': 0.0, 'data': None, 'ranks': {}})
        
        # Process each list
        for list_idx, (df, score_col) in enumerate(zip(non_empty, score_cols)):
            for rank, (idx, row) in enumerate(df.iterrows(), start=1):
                doc_id = row[id_col]
                rrf_score = self.compute_rrf_score(rank)
                
                rrf_scores[doc_id]['score'] += rrf_score
                rrf_scores[doc_id]['ranks'][f'rank_{list_idx}'] = rank
                
                if rrf_scores[doc_id]['data'] is None:
                    rrf_scores[doc_id]['data'] = row.to_dict()
        
        # Collect results
        merged_results = []
        for doc_id, info in rrf_scores.items():
            data = info['data'].copy()
            data['rrf_score'] = info['score']
            data['retrieval_score'] = info['score']
            data.update(info['ranks'])
            data['num_sources'] = len(info['ranks'])
            merged_results.append(data)
        
        results_df = pd.DataFrame(merged_results)
        results_df = results_df.sort_values('rrf_score', ascending=False)
        
        return results_df.reset_index(drop=True)
    
    def fuse_dicts(
        self,
        results_list: List[List[dict]],
        id_field: str = 'id'
    ) -> List[dict]:
        """
        Combine multiple result lists (as dicts) using RRF.
        
        Args:
            results_list: List of lists of dicts
            id_field: Field name for document ID
            
        Returns:
            Combined list sorted by RRF score
        """
        if not results_list:
            return []
        
        non_empty = [r for r in results_list if r]
        
        if not non_empty:
            return []
        
        if len(non_empty) == 1:
            return non_empty[0]
        
        rrf_scores = defaultdict(lambda: {'score': 0.0, 'data': None})
        
        for results in non_empty:
            for rank, doc in enumerate(results, start=1):
                doc_id = doc.get(id_field) or doc.get('node_id') or doc.get('chunk_id')
                if doc_id is None:
                    continue
                
                rrf_score = self.compute_rrf_score(rank)
                rrf_scores[doc_id]['score'] += rrf_score
                
                if rrf_scores[doc_id]['data'] is None:
                    rrf_scores[doc_id]['data'] = doc.copy()
        
        # Sort by RRF score
        results = []
        for doc_id, info in rrf_scores.items():
            data = info['data'].copy()
            data['rrf_score'] = info['score']
            results.append(data)
        
        results.sort(key=lambda x: x['rrf_score'], reverse=True)
        
        return results


# Convenience function
def rrf_fuse(
    *result_lists: pd.DataFrame,
    k: int = 60,
    id_col: Optional[str] = None
) -> pd.DataFrame:
    """
    Convenience function for RRF fusion.
    
    Args:
        *result_lists: DataFrames to fuse
        k: RRF constant
        id_col: ID column
        
    Returns:
        Fused DataFrame
    """
    rrf = ReciprocalRankFusion(k=k)
    return rrf.fuse_multiple_results(list(result_lists), id_col=id_col)

