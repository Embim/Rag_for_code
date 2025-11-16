"""
Grid Search оптимизация параметров RAG

Интегрирована в main_pipeline.py через флаг --optimize

Оптимизируемые параметры:
- TOP_K_DENSE: количество результатов векторного поиска
- TOP_K_BM25: количество результатов BM25
- TOP_K_RERANK: количество результатов после reranking
- HYBRID_ALPHA: баланс между dense и BM25

Использует гибридную оценку:
- Косинусное расстояние (semantic similarity)
- LLM as Judge (Context Relevance, Precision, Sufficiency)
"""
import sys
from pathlib import Path
import pandas as pd
from itertools import product
from tqdm import tqdm
from typing import Dict, Tuple
import src.config as config
from src.config import GRID_SEARCH_USE_LLM
from src.logger import get_logger, log_timing


class GridSearchOptimizer:
    """
    Оптимизация гиперпараметров RAG через grid search

    Использует гибридную оценку (cosine + LLM metrics)
    """

    def __init__(self, retriever, questions_df: pd.DataFrame, use_llm_eval: bool = None):
        """
        Args:
            retriever: HybridRetriever или WeaviateIndexer
            questions_df: DataFrame с вопросами для оптимизации
            use_llm_eval: использовать ли LLM для оценки (None = из config.GRID_SEARCH_USE_LLM)
        """
        self.retriever = retriever
        self.questions_df = questions_df

        # Используем значение из config если не передано явно
        if use_llm_eval is None:
            use_llm_eval = GRID_SEARCH_USE_LLM

        self.use_llm_eval = use_llm_eval

        # Инициализация Hybrid Evaluator
        self.evaluator = None
        if use_llm_eval:
            try:
                from src.llm_evaluator import get_hybrid_evaluator
                from src.config import LLM_MODE
                
                # Определяем режим работы (API или локальный)
                use_api = (LLM_MODE == "api")
                
                # Логируем для отладки
                logger = get_logger(__name__)
                logger.info(f"[GridSearch] LLM_MODE из config: {LLM_MODE}")
                logger.info(f"[GridSearch] use_api будет: {use_api}")
                
                # Принудительно сбрасываем singleton если режим изменился
                # (на случай если он был создан ранее с другим режимом)
                import src.llm_evaluator as llm_eval_module
                if hasattr(llm_eval_module, '_evaluator_instance') and llm_eval_module._evaluator_instance is not None:
                    existing_use_api = llm_eval_module._evaluator_instance.use_api
                    if existing_use_api != use_api:
                        logger.info(f"[GridSearch] Сбрасываем singleton evaluator (был {existing_use_api}, нужен {use_api})")
                        llm_eval_module._evaluator_instance = None
                
                self.evaluator = get_hybrid_evaluator(
                    use_llm=True,
                    semantic_weight=0.3,  # 30% косинусное расстояние
                    llm_weight=0.7,       # 70% LLM метрики
                    use_api=use_api
                )
                
                # Проверяем что evaluator действительно в нужном режиме
                actual_mode = "API" if self.evaluator.use_api else "локальный"
                mode_str = "API" if use_api else "локальный"
                logger.info(f"[GridSearch] ✓ Hybrid Evaluator загружен (запрошен: {mode_str}, фактический: {actual_mode}, cosine 30% + LLM 70%)")
                
                if self.evaluator.use_api != use_api:
                    logger.error(f"[GridSearch] ❌ ОШИБКА: evaluator в неправильном режиме! Запрошен: {use_api}, фактический: {self.evaluator.use_api}")
            except Exception as e:
                get_logger(__name__).warning(f"Не удалось загрузить LLM Evaluator: {e}")
                get_logger(__name__).warning("Используется только косинусное расстояние")
                self.use_llm_eval = False

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

        logger = get_logger(__name__)
        logger.info(f"📊 Grid Search режим: {mode}")
        logger.info(f"Всего комбинаций: {total_combinations}")

        return param_grid

    def evaluate_params(self, params: Dict) -> Tuple[float, Dict]:
        """
        Оценка качества для заданных параметров

        Args:
            params: словарь параметров

        Returns:
            (avg_score, detailed_metrics)
            - avg_score: итоговая метрика (hybrid_score если LLM включен, иначе semantic)
            - detailed_metrics: детальные метрики для логирования
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

            # Собираем результаты для всех вопросов
            all_queries = []
            all_results = []

            for idx, row in self.questions_df.iterrows():
                query = row.get('processed_query', row.get('question', ''))

                try:
                    # Поиск
                    results = self.retriever.search(query)

                    if len(results) > 0:
                        all_queries.append(query)
                        all_results.append(results)

                except Exception as e:
                    # Ошибка поиска - пропускаем этот вопрос
                    pass

            # Оценка через Hybrid Evaluator
            if self.use_llm_eval and self.evaluator:
                # LLM-based оценка (медленно, но точно)
                metrics = self.evaluator.evaluate_batch(
                    all_queries,
                    all_results,
                    top_k=params["TOP_K_RERANK"]
                )

                avg_score = metrics['avg_hybrid_score']
                detailed_metrics = metrics

            else:
                # Fallback: только косинусное расстояние (быстро)
                total_score = 0.0
                for results in all_results:
                    if len(results) > 0:
                        top_scores = results.head(5)['final_score'].tolist()
                        total_score += sum(top_scores) / len(top_scores)

                avg_score = total_score / len(all_results) if len(all_results) > 0 else 0.0
                detailed_metrics = {
                    'avg_semantic_score': avg_score,
                    'num_evaluated': len(all_results)
                }

            return avg_score, detailed_metrics

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

        logger = get_logger(__name__)
        logger.info("🔍 Запуск Grid Search...")
        logger.info(f"Комбинаций: {len(combinations)} | Вопросов в выборке: {len(self.questions_df)}")

        # Результаты
        results = []
        best_score = -1
        best_params = None

        for combo in tqdm(combinations, desc="Grid Search"):
            params = dict(zip(keys, combo))

            # Оценка
            logger.debug(f"Оценка: {params}")
            score, detailed_metrics = self.evaluate_params(params)

            # Сохраняем
            result = {
                **params,
                "avg_score": score,
                **{f"metric_{k}": v for k, v in detailed_metrics.items() if k != 'num_evaluated'}
            }
            results.append(result)

            # Логируем промежуточный результат
            if self.use_llm_eval and self.evaluator:
                logger.info(
                    f"  Dense={params['TOP_K_DENSE']}, BM25={params['TOP_K_BM25']}, "
                    f"Rerank={params['TOP_K_RERANK']}, α={params['HYBRID_ALPHA']:.1f} → "
                    f"Hybrid={score:.3f} "
                    f"(semantic={detailed_metrics.get('avg_semantic_score', 0):.3f}, "
                    f"sufficiency={detailed_metrics.get('avg_context_sufficiency', 0):.3f})"
                )
            else:
                logger.info(
                    f"  Dense={params['TOP_K_DENSE']}, BM25={params['TOP_K_BM25']}, "
                    f"Rerank={params['TOP_K_RERANK']}, α={params['HYBRID_ALPHA']:.1f} → "
                    f"Score={score:.3f}"
                )

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
        logger = get_logger(__name__)
        logger.info("⭐ ЛУЧШИЕ ПАРАМЕТРЫ:")
        logger.info(f"TOP_K_DENSE={best_params['TOP_K_DENSE']}, TOP_K_BM25={best_params['TOP_K_BM25']}, TOP_K_RERANK={best_params['TOP_K_RERANK']}, HYBRID_ALPHA={best_params['HYBRID_ALPHA']:.2f}")

        # Применяем к config
        config.TOP_K_DENSE = best_params['TOP_K_DENSE']
        config.TOP_K_BM25 = best_params['TOP_K_BM25']
        config.TOP_K_RERANK = best_params['TOP_K_RERANK']
        config.HYBRID_ALPHA = best_params['HYBRID_ALPHA']

        logger.info("✅ Параметры применены к config")


def optimize_rag_params(retriever, questions_df: pd.DataFrame,
                       mode: str = None,
                       sample_size: int = None,
                       use_llm_eval: bool = None) -> Dict:
    """
    Удобная функция для оптимизации RAG параметров

    Args:
        retriever: HybridRetriever или WeaviateIndexer
        questions_df: DataFrame с вопросами
        mode: "quick" или "full" (None = из config.GRID_SEARCH_MODE)
        sample_size: размер выборки для оптимизации (None = из config.GRID_SEARCH_SAMPLE_SIZE)
        use_llm_eval: использовать ли LLM для оценки (None = из config.GRID_SEARCH_USE_LLM)

    Returns:
        best_params: словарь с лучшими параметрами
    """
    logger = get_logger(__name__)

    # Используем значения из config если не переданы явно
    if mode is None:
        from src.config import GRID_SEARCH_MODE
        mode = GRID_SEARCH_MODE

    if sample_size is None:
        from src.config import GRID_SEARCH_SAMPLE_SIZE
        sample_size = GRID_SEARCH_SAMPLE_SIZE

    if use_llm_eval is None:
        use_llm_eval = GRID_SEARCH_USE_LLM

    # Выборка вопросов
    if len(questions_df) > sample_size:
        sample_df = questions_df.sample(n=sample_size, random_state=42)
    else:
        sample_df = questions_df

    logger.info("="*80)
    logger.info("GRID SEARCH ОПТИМИЗАЦИЯ RAG ПАРАМЕТРОВ")
    logger.info("="*80)
    logger.info(f"Режим: {mode}")
    logger.info(f"Размер выборки: {sample_size} вопросов")
    logger.info(f"Режим оценки: {'LLM + Cosine (гибридный)' if use_llm_eval else 'Только Cosine'}")

    # Создаем optimizer
    optimizer = GridSearchOptimizer(retriever, sample_df, use_llm_eval=use_llm_eval)

    # Определяем сетку параметров
    param_grid = optimizer.define_param_grid(mode=mode)

    # Запускаем grid search
    best_params, results_df = optimizer.search(param_grid)

    # Показываем результаты
    logger.info("📊 Топ-5 конфигураций:")
    logger.info("\n" + results_df.head(5).to_string())

    # Применяем лучшие параметры
    optimizer.apply_best_params(best_params)

    return best_params


if __name__ == "__main__":
    print("Grid Search Optimizer")
    print("Используйте через main_pipeline.py search --optimize")
