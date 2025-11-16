"""
Grid Search оптимизация параметров RAG

Интегрирована в main_pipeline.py через флаг --optimize

Оптимизируемые параметры:
- TOP_K_DENSE: количество результатов векторного поиска
- TOP_K_BM25: количество результатов BM25
- TOP_K_RERANK: количество результатов после reranking
- HYBRID_ALPHA: баланс между dense и BM25
"""
import sys
from pathlib import Path
import pandas as pd
from itertools import product
from tqdm import tqdm
from typing import Dict, Tuple
import src.config as config


class GridSearchOptimizer:
    """
    Оптимизация гиперпараметров RAG через grid search
    """

    def __init__(self, retriever, questions_df: pd.DataFrame):
        """
        Args:
            retriever: HybridRetriever или WeaviateIndexer
            questions_df: DataFrame с вопросами для оптимизации
        """
        self.retriever = retriever
        self.questions_df = questions_df

    def define_param_grid(self, mode: str = "quick") -> dict:
        """
        Определение сетки параметров

        Args:
            mode: "quick" (быстрый) или "full" (полный)

        Returns:
            dict с параметрами
        """
        if mode == "quick":
            # Быстрый поиск
            param_grid = {
                "TOP_K_DENSE": [15, 25, 35],
                "TOP_K_BM25": [15, 25, 35],
                "TOP_K_RERANK": [15, 20],
                "HYBRID_ALPHA": [0.4, 0.5, 0.6]
            }
        else:
            # Полный поиск
            param_grid = {
                "TOP_K_DENSE": [10, 15, 20, 25, 30, 35, 40],
                "TOP_K_BM25": [10, 15, 20, 25, 30, 35, 40],
                "TOP_K_RERANK": [10, 15, 20, 25, 30],
                "HYBRID_ALPHA": [0.3, 0.4, 0.5, 0.6, 0.7]
            }

        total_combinations = (
            len(param_grid["TOP_K_DENSE"]) *
            len(param_grid["TOP_K_BM25"]) *
            len(param_grid["TOP_K_RERANK"]) *
            len(param_grid["HYBRID_ALPHA"])
        )

        print(f"\n📊 Grid Search режим: {mode}")
        print(f"   Всего комбинаций: {total_combinations}")

        return param_grid

    def evaluate_params(self, params: Dict) -> float:
        """
        Оценка качества для заданных параметров

        Args:
            params: словарь параметров

        Returns:
            средний score
        """
        # Сохраняем оригинальные параметры
        original_params = {
            "TOP_K_DENSE": config.TOP_K_DENSE,
            "TOP_K_BM25": config.TOP_K_BM25,
            "TOP_K_RERANK": config.TOP_K_RERANK,
            "HYBRID_ALPHA": config.HYBRID_ALPHA,
        }

        try:
            # Устанавливаем новые параметры
            config.TOP_K_DENSE = params["TOP_K_DENSE"]
            config.TOP_K_BM25 = params["TOP_K_BM25"]
            config.TOP_K_RERANK = params["TOP_K_RERANK"]
            config.HYBRID_ALPHA = params["HYBRID_ALPHA"]

            # Оценка на выборке вопросов
            total_score = 0.0

            for idx, row in self.questions_df.iterrows():
                query = row.get('processed_query', row.get('question', ''))

                try:
                    # Поиск
                    results = self.retriever.search(query)

                    # Считаем средний score топ-5
                    if len(results) > 0:
                        top_scores = results.head(5)['final_score'].tolist()
                        total_score += sum(top_scores) / len(top_scores)

                except Exception as e:
                    # Ошибка поиска - score = 0
                    pass

            avg_score = total_score / len(self.questions_df) if len(self.questions_df) > 0 else 0.0

            return avg_score

        finally:
            # Восстанавливаем оригинальные параметры
            config.TOP_K_DENSE = original_params["TOP_K_DENSE"]
            config.TOP_K_BM25 = original_params["TOP_K_BM25"]
            config.TOP_K_RERANK = original_params["TOP_K_RERANK"]
            config.HYBRID_ALPHA = original_params["HYBRID_ALPHA"]

    def search(self, param_grid: dict) -> Tuple[Dict, pd.DataFrame]:
        """
        Запуск grid search

        Args:
            param_grid: сетка параметров

        Returns:
            (best_params, results_df)
        """
        # Генерируем все комбинации
        keys = list(param_grid.keys())
        combinations = list(product(*[param_grid[k] for k in keys]))

        print(f"\n🔍 Запуск Grid Search...")
        print(f"   Комбинаций: {len(combinations)}")
        print(f"   Вопросов в выборке: {len(self.questions_df)}")

        # Результаты
        results = []
        best_score = -1
        best_params = None

        for combo in tqdm(combinations, desc="Grid Search"):
            params = dict(zip(keys, combo))

            # Оценка
            score = self.evaluate_params(params)

            # Сохраняем
            result = {**params, "avg_score": score}
            results.append(result)

            # Обновляем best
            if score > best_score:
                best_score = score
                best_params = params.copy()

        results_df = pd.DataFrame(results).sort_values(by="avg_score", ascending=False)

        return best_params, results_df

    def apply_best_params(self, best_params: Dict):
        """
        Применение лучших параметров к config

        Args:
            best_params: словарь с лучшими параметрами
        """
        print(f"\n⭐ ЛУЧШИЕ ПАРАМЕТРЫ:")
        print(f"   TOP_K_DENSE:   {best_params['TOP_K_DENSE']}")
        print(f"   TOP_K_BM25:    {best_params['TOP_K_BM25']}")
        print(f"   TOP_K_RERANK:  {best_params['TOP_K_RERANK']}")
        print(f"   HYBRID_ALPHA:  {best_params['HYBRID_ALPHA']:.2f}")

        # Применяем к config
        config.TOP_K_DENSE = best_params['TOP_K_DENSE']
        config.TOP_K_BM25 = best_params['TOP_K_BM25']
        config.TOP_K_RERANK = best_params['TOP_K_RERANK']
        config.HYBRID_ALPHA = best_params['HYBRID_ALPHA']

        print(f"\n✅ Параметры применены к config!")


def optimize_rag_params(retriever, questions_df: pd.DataFrame,
                       mode: str = "quick",
                       sample_size: int = 50) -> Dict:
    """
    Удобная функция для оптимизации RAG параметров

    Args:
        retriever: HybridRetriever или WeaviateIndexer
        questions_df: DataFrame с вопросами
        mode: "quick" или "full"
        sample_size: размер выборки для оптимизации

    Returns:
        best_params: словарь с лучшими параметрами
    """
    # Выборка вопросов
    if len(questions_df) > sample_size:
        sample_df = questions_df.sample(n=sample_size, random_state=42)
    else:
        sample_df = questions_df

    print(f"\n{'='*80}")
    print(f"GRID SEARCH ОПТИМИЗАЦИЯ RAG ПАРАМЕТРОВ")
    print(f"{'='*80}")

    # Создаем optimizer
    optimizer = GridSearchOptimizer(retriever, sample_df)

    # Определяем сетку параметров
    param_grid = optimizer.define_param_grid(mode=mode)

    # Запускаем grid search
    best_params, results_df = optimizer.search(param_grid)

    # Показываем результаты
    print(f"\n📊 Топ-5 конфигураций:")
    print(results_df.head(5).to_string())

    # Применяем лучшие параметры
    optimizer.apply_best_params(best_params)

    return best_params


if __name__ == "__main__":
    print("Grid Search Optimizer")
    print("Используйте через main_pipeline.py search --optimize")
