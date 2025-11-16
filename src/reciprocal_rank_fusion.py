"""
Reciprocal Rank Fusion (RRF) для объединения результатов поиска

RRF лучше работает чем weighted sum, потому что:
- Не зависит от абсолютных значений scores
- Использует только ранги (позиции)
- Доказано эффективнее на бенчмарках

Формула: RRF_score(d) = Σ 1/(k + rank(d))
где k=60 - константа, rank(d) - позиция документа в списке
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from collections import defaultdict


class ReciprocalRankFusion:
    """
    Reciprocal Rank Fusion для объединения результатов разных retriever'ов

    Примеры использования:
    1. Dense + BM25 (hybrid search)
    2. Несколько embedding моделей
    3. Разные query варианты (query expansion)
    """

    def __init__(self, k: int = 60):
        """
        Args:
            k: константа для RRF (обычно 60, можно 20-100)
               Большее k → меньше разница между топовыми результатами
               Меньшее k → больше вес топовым результатам
        """
        self.k = k

    def compute_rrf_score(self, rank: int) -> float:
        """
        Вычислить RRF score для заданного ранга

        Args:
            rank: позиция в списке (начиная с 1)

        Returns:
            RRF score
        """
        return 1.0 / (self.k + rank)

    def fuse_two_results(self,
                        results1: pd.DataFrame,
                        results2: pd.DataFrame,
                        score_col1: str = 'retrieval_score',
                        score_col2: str = 'retrieval_score') -> pd.DataFrame:
        """
        Объединение двух списков результатов через RRF

        Args:
            results1: первый список результатов
            results2: второй список результатов
            score_col1: колонка со score в первом списке
            score_col2: колонка со score во втором списке

        Returns:
            объединенный список с RRF scores
        """
        rrf_scores = defaultdict(lambda: {'score': 0.0, 'data': None})

        # Обрабатываем первый список
        for rank, (idx, row) in enumerate(results1.iterrows(), start=1):
            chunk_id = row['chunk_id']
            rrf_score = self.compute_rrf_score(rank)

            rrf_scores[chunk_id]['score'] += rrf_score
            rrf_scores[chunk_id]['data'] = row.to_dict()
            rrf_scores[chunk_id]['data']['rrf_score'] = rrf_score
            rrf_scores[chunk_id]['data']['rank_1'] = rank
            rrf_scores[chunk_id]['data'][f'original_score_1'] = row.get(score_col1, 0.0)

        # Обрабатываем второй список
        for rank, (idx, row) in enumerate(results2.iterrows(), start=1):
            chunk_id = row['chunk_id']
            rrf_score = self.compute_rrf_score(rank)

            if chunk_id in rrf_scores:
                # Документ уже есть в первом списке
                rrf_scores[chunk_id]['score'] += rrf_score
                rrf_scores[chunk_id]['data']['rank_2'] = rank
                rrf_scores[chunk_id]['data'][f'original_score_2'] = row.get(score_col2, 0.0)
            else:
                # Новый документ
                rrf_scores[chunk_id]['score'] = rrf_score
                rrf_scores[chunk_id]['data'] = row.to_dict()
                rrf_scores[chunk_id]['data']['rrf_score'] = rrf_score
                rrf_scores[chunk_id]['data']['rank_2'] = rank
                rrf_scores[chunk_id]['data'][f'original_score_2'] = row.get(score_col2, 0.0)

        # Собираем результаты
        merged_results = []
        for chunk_id, info in rrf_scores.items():
            data = info['data'].copy()
            data['rrf_score'] = info['score']
            data['retrieval_score'] = info['score']  # для совместимости
            merged_results.append(data)

        # Сортируем по RRF score
        results_df = pd.DataFrame(merged_results)
        results_df = results_df.sort_values('rrf_score', ascending=False)

        return results_df.reset_index(drop=True)

    def fuse_multiple_results(self,
                             results_list: List[pd.DataFrame],
                             score_cols: List[str] = None) -> pd.DataFrame:
        """
        Объединение нескольких списков результатов через RRF

        Args:
            results_list: список DataFrame с результатами
            score_cols: названия колонок со scores (если None - 'retrieval_score')

        Returns:
            объединенный список с RRF scores
        """
        if len(results_list) == 0:
            return pd.DataFrame()

        if len(results_list) == 1:
            return results_list[0]

        if score_cols is None:
            score_cols = ['retrieval_score'] * len(results_list)

        rrf_scores = defaultdict(lambda: {'score': 0.0, 'data': None, 'sources': []})

        # Обрабатываем каждый список
        for list_idx, (results, score_col) in enumerate(zip(results_list, score_cols)):
            for rank, (idx, row) in enumerate(results.iterrows(), start=1):
                chunk_id = row['chunk_id']
                rrf_score = self.compute_rrf_score(rank)

                rrf_scores[chunk_id]['score'] += rrf_score
                rrf_scores[chunk_id]['sources'].append(list_idx)

                if rrf_scores[chunk_id]['data'] is None:
                    rrf_scores[chunk_id]['data'] = row.to_dict()

                # Добавляем метаданные
                rrf_scores[chunk_id]['data'][f'rank_{list_idx}'] = rank
                rrf_scores[chunk_id]['data'][f'original_score_{list_idx}'] = row.get(score_col, 0.0)

        # Собираем результаты
        merged_results = []
        for chunk_id, info in rrf_scores.items():
            data = info['data'].copy()
            data['rrf_score'] = info['score']
            data['retrieval_score'] = info['score']  # для совместимости
            data['num_sources'] = len(info['sources'])  # в скольких списках встречался
            merged_results.append(data)

        # Сортируем по RRF score
        results_df = pd.DataFrame(merged_results)
        results_df = results_df.sort_values('rrf_score', ascending=False)

        return results_df.reset_index(drop=True)


def compare_weighted_vs_rrf():
    """Сравнение Weighted Sum vs RRF"""
    print("="*80)
    print("СРАВНЕНИЕ: Weighted Sum vs Reciprocal Rank Fusion")
    print("="*80)

    # Тестовые данные - результаты Dense и BM25 поиска
    dense_results = pd.DataFrame([
        {'chunk_id': 'doc1', 'retrieval_score': 0.95, 'text': 'Документ 1 - очень релевантный'},
        {'chunk_id': 'doc2', 'retrieval_score': 0.85, 'text': 'Документ 2 - релевантный'},
        {'chunk_id': 'doc3', 'retrieval_score': 0.75, 'text': 'Документ 3 - средний'},
        {'chunk_id': 'doc4', 'retrieval_score': 0.65, 'text': 'Документ 4 - слабый'},
        {'chunk_id': 'doc5', 'retrieval_score': 0.55, 'text': 'Документ 5 - очень слабый'},
    ])

    bm25_results = pd.DataFrame([
        {'chunk_id': 'doc3', 'retrieval_score': 15.5, 'text': 'Документ 3 - топ по BM25!'},
        {'chunk_id': 'doc1', 'retrieval_score': 12.3, 'text': 'Документ 1 - хороший'},
        {'chunk_id': 'doc6', 'retrieval_score': 10.1, 'text': 'Документ 6 - только в BM25'},
        {'chunk_id': 'doc2', 'retrieval_score': 8.7, 'text': 'Документ 2 - средний'},
        {'chunk_id': 'doc7', 'retrieval_score': 7.2, 'text': 'Документ 7 - слабый'},
    ])

    print("\n1️⃣  Dense результаты:")
    for idx, row in dense_results.head(3).iterrows():
        print(f"   Rank {idx+1}: {row['chunk_id']} (score: {row['retrieval_score']:.3f})")

    print("\n2️⃣  BM25 результаты:")
    for idx, row in bm25_results.head(3).iterrows():
        print(f"   Rank {idx+1}: {row['chunk_id']} (score: {row['retrieval_score']:.3f})")

    # Weighted Sum (старый метод)
    print("\n3️⃣  Weighted Sum (alpha=0.5):")
    print("   ⚠️  Проблема: нельзя напрямую сложить scores разных шкал!")
    print("   Dense: [0.55-0.95], BM25: [7.2-15.5] - несопоставимые шкалы")

    # RRF
    print("\n4️⃣  Reciprocal Rank Fusion (k=60):")
    rrf = ReciprocalRankFusion(k=60)
    merged = rrf.fuse_two_results(dense_results, bm25_results)

    print("   Топ-5 результатов:")
    for idx, row in merged.head(5).iterrows():
        rank_1 = row.get('rank_1', '-')
        rank_2 = row.get('rank_2', '-')
        print(f"   {idx+1}. {row['chunk_id']}")
        print(f"      RRF score: {row['rrf_score']:.4f}")
        print(f"      Dense rank: {rank_1}, BM25 rank: {rank_2}")

    print("\n" + "="*80)
    print("📊 ВЫВОД:")
    print("   RRF учитывает ранги из обоих источников")
    print("   doc3 и doc1 высоко в обоих → высокий RRF score")
    print("   doc6 и doc7 только в BM25 → ниже в финальном списке")
    print("="*80)


def main():
    """Тест RRF"""
    compare_weighted_vs_rrf()

    # Дополнительный тест
    print("\n\n" + "="*80)
    print("ТЕСТ: Объединение 3 списков (Query Expansion)")
    print("="*80)

    # Симулируем 3 варианта запроса
    query1_results = pd.DataFrame([
        {'chunk_id': 'doc1', 'retrieval_score': 0.9},
        {'chunk_id': 'doc2', 'retrieval_score': 0.8},
        {'chunk_id': 'doc3', 'retrieval_score': 0.7},
    ])

    query2_results = pd.DataFrame([
        {'chunk_id': 'doc2', 'retrieval_score': 0.85},
        {'chunk_id': 'doc1', 'retrieval_score': 0.75},
        {'chunk_id': 'doc4', 'retrieval_score': 0.65},
    ])

    query3_results = pd.DataFrame([
        {'chunk_id': 'doc1', 'retrieval_score': 0.95},
        {'chunk_id': 'doc3', 'retrieval_score': 0.80},
        {'chunk_id': 'doc5', 'retrieval_score': 0.70},
    ])

    rrf = ReciprocalRankFusion(k=60)
    merged = rrf.fuse_multiple_results([query1_results, query2_results, query3_results])

    print("\nРезультаты:")
    for idx, row in merged.iterrows():
        print(f"{idx+1}. {row['chunk_id']}")
        print(f"   RRF: {row['rrf_score']:.4f}, Sources: {row['num_sources']}")
        print(f"   Ranks: [{row.get('rank_0', '-')}, {row.get('rank_1', '-')}, {row.get('rank_2', '-')}]")

    print("\n✅ doc1 высоко во всех 3 → максимальный RRF score")


if __name__ == "__main__":
    main()
