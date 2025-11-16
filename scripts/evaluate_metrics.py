"""
Оценка качества RAG системы на эталонных примерах
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Set
import re

from src.config import EXAMPLES_CSV, WEBSITES_CSV
from main_pipeline import build_knowledge_base
from src.retrieval import RAGPipeline
from src.preprocessing import TextPreprocessor


def extract_web_ids_from_chunks(examples_df: pd.DataFrame,
                                websites_df: pd.DataFrame) -> Dict[str, Set[int]]:
    """
    Извлечение web_id из chunk'ов в эталонных примерах

    Args:
        examples_df: DataFrame с примерами
        websites_df: DataFrame с документами

    Returns:
        словарь {query: set(web_ids)}
    """
    preprocessor = TextPreprocessor()

    # Создаем индекс текстов документов для поиска
    web_id_to_text = {
        row['web_id']: preprocessor.preprocess_document(row['text'], row['title'])
        for _, row in websites_df.iterrows()
    }

    query_to_web_ids = {}

    for idx, row in examples_df.iterrows():
        query = row['query']
        relevant_web_ids = set()

        # Проверяем каждый chunk
        for i in range(1, 6):
            chunk_col = f'chunk_{i}'
            if chunk_col in row and pd.notna(row[chunk_col]):
                chunk_text = str(row[chunk_col])

                # Ищем этот chunk в документах
                chunk_clean = preprocessor.normalize_text(chunk_text)

                for web_id, doc_text in web_id_to_text.items():
                    # Простой поиск подстроки (можно улучшить)
                    if chunk_clean[:100] in doc_text or \
                       any(part in doc_text for part in chunk_clean.split()[:20] if len(part) > 4):
                        relevant_web_ids.add(web_id)
                        break

        query_to_web_ids[query] = relevant_web_ids

    return query_to_web_ids


def calculate_recall_at_k(predicted: List[int], relevant: Set[int], k: int = 5) -> float:
    """
    Расчет Recall@K метрики

    Args:
        predicted: список предсказанных web_id
        relevant: множество релевантных web_id
        k: количество предсказаний

    Returns:
        recall score
    """
    if len(relevant) == 0:
        return 1.0  # если нет релевантных - считаем что все верно

    # Берем только топ-k предсказаний
    predicted_k = set(predicted[:k])

    # Пересечение с релевантными
    found = predicted_k & relevant

    # Recall = найдено / всего релевантных
    recall = len(found) / len(relevant)

    return recall


def evaluate_pipeline(embedding_indexer, bm25_indexer,
                     examples_df: pd.DataFrame,
                     websites_df: pd.DataFrame) -> Dict:
    """
    Оценка пайплайна на эталонных примерах

    Args:
        embedding_indexer: векторный индексер
        bm25_indexer: BM25 индексер
        examples_df: эталонные примеры
        websites_df: документы

    Returns:
        словарь с метриками
    """
    print("="*80)
    print("ОЦЕНКА КАЧЕСТВА НА ЭТАЛОННЫХ ПРИМЕРАХ")
    print("="*80)

    # Извлекаем релевантные web_id
    print("\n1. Извлечение релевантных документов из примеров...")
    query_to_relevant = extract_web_ids_from_chunks(examples_df, websites_df)

    # Создаем пайплайн
    pipeline = RAGPipeline(embedding_indexer, bm25_indexer)

    # Обработка каждого примера
    print("\n2. Обработка примеров...")
    preprocessor = TextPreprocessor()

    recalls = []
    results = []

    for query, relevant_web_ids in query_to_relevant.items():
        # Предобработка запроса
        processed_query = preprocessor.preprocess_query(query)

        # Поиск
        result = pipeline.search(processed_query)
        predicted_web_ids = result['documents_id']

        # Расчет recall
        recall = calculate_recall_at_k(predicted_web_ids, relevant_web_ids, k=5)
        recalls.append(recall)

        results.append({
            'query': query,
            'relevant': list(relevant_web_ids),
            'predicted': predicted_web_ids,
            'recall@5': recall
        })

        print(f"\nQuery: {query[:60]}...")
        print(f"  Relevant: {relevant_web_ids}")
        print(f"  Predicted: {predicted_web_ids}")
        print(f"  Recall@5: {recall:.3f}")

    # Общая статистика
    print("\n" + "="*80)
    print("ИТОГОВЫЕ МЕТРИКИ")
    print("="*80)
    print(f"Примеров обработано: {len(recalls)}")
    print(f"Средний Recall@5: {np.mean(recalls):.3f}")
    print(f"Медианный Recall@5: {np.median(recalls):.3f}")
    print(f"Мин Recall@5: {np.min(recalls):.3f}")
    print(f"Макс Recall@5: {np.max(recalls):.3f}")

    # Распределение по бакетам
    print(f"\nРаспределение Recall@5:")
    print(f"  1.0 (идеально): {sum(1 for r in recalls if r == 1.0)} ({sum(1 for r in recalls if r == 1.0)/len(recalls)*100:.1f}%)")
    print(f"  >= 0.8: {sum(1 for r in recalls if r >= 0.8)} ({sum(1 for r in recalls if r >= 0.8)/len(recalls)*100:.1f}%)")
    print(f"  >= 0.6: {sum(1 for r in recalls if r >= 0.6)} ({sum(1 for r in recalls if r >= 0.6)/len(recalls)*100:.1f}%)")
    print(f"  >= 0.4: {sum(1 for r in recalls if r >= 0.4)} ({sum(1 for r in recalls if r >= 0.4)/len(recalls)*100:.1f}%)")
    print(f"  < 0.4: {sum(1 for r in recalls if r < 0.4)} ({sum(1 for r in recalls if r < 0.4)/len(recalls)*100:.1f}%)")

    return {
        'mean_recall': np.mean(recalls),
        'median_recall': np.median(recalls),
        'results': results
    }


def main():
    """Главная функция"""

    # Загрузка данных
    print("Загрузка данных...")
    examples_df = pd.read_csv(EXAMPLES_CSV)
    websites_df = pd.read_csv(WEBSITES_CSV)

    print(f"Загружено {len(examples_df)} эталонных примеров")
    print(f"Загружено {len(websites_df)} документов")

    # Построение/загрузка индексов
    print("\nЗагрузка индексов...")
    embedding_indexer, bm25_indexer, chunks_df = build_knowledge_base(
        force_rebuild=False
    )

    # Оценка
    metrics = evaluate_pipeline(
        embedding_indexer,
        bm25_indexer,
        examples_df,
        websites_df
    )

    # Сохранение результатов
    results_df = pd.DataFrame(metrics['results'])
    output_path = "outputs/evaluation_results.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\nДетальные результаты сохранены: {output_path}")


if __name__ == "__main__":
    main()
