"""
Главный скрипт RAG пайплайна

Использование:
    python main_pipeline.py build           # Построить базу знаний
    python main_pipeline.py search          # Обработать вопросы
    python main_pipeline.py all             # Полный цикл (build + search)
    python main_pipeline.py evaluate        # Оценка на примерах
"""
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

from src.config import (
    WEBSITES_CSV,
    QUESTIONS_CSV,
    MODELS_DIR,
    OUTPUTS_DIR,
    PROCESSED_DIR,
    USE_WEAVIATE,
    ENABLE_AGENT_RAG,
    LOG_LEVEL,
    LOG_FILE
)
from src.preprocessing import load_and_preprocess_documents, load_and_preprocess_questions
from src.chunking import create_chunks_from_documents
from src.indexing import WeaviateIndexer
from src.retrieval import RAGPipeline
from src.llm_preprocessing import apply_llm_cleaning
from src.grid_search_optimizer import optimize_rag_params
from src.logger import setup_logging, get_logger, log_timing
import logging
import time

# Проверка доступности Weaviate
try:
    import weaviate
    WEAVIATE_AVAILABLE = True
except ImportError:
    WEAVIATE_AVAILABLE = False
    if USE_WEAVIATE:
        # Логгер будет инициализирован в main()
        pass


def build_knowledge_base(force_rebuild: bool = False, llm_clean: bool = False,
                        min_usefulness: float = 0.3):
    """
    Построение базы знаний (offline этап)

    Args:
        force_rebuild: пересоздать индексы даже если они существуют
        llm_clean: использовать LLM для очистки документов (медленно, но качественно)
        min_usefulness: минимальный порог полезности для LLM фильтрации (0.0-1.0)

    Returns:
        (embedding_indexer, bm25_indexer, chunks_df)
    """
    logger = get_logger(__name__)
    logger.info("="*80)
    logger.info("ЭТАП 1: ПОСТРОЕНИЕ БАЗЫ ЗНАНИЙ (OFFLINE)")
    logger.info("="*80)

    chunks_path = PROCESSED_DIR / "chunks.pkl"

    # В этом проекте используем только Weaviate
    if not (USE_WEAVIATE and WEAVIATE_AVAILABLE):
        raise RuntimeError("Weaviate обязателен. Установите weaviate-client и включите USE_WEAVIATE=True в config.py")

    logger.info("[РЕЖИМ] Используется только Weaviate для векторного поиска (гибридный с BM25)")

    # Проверяем существуют ли чанки
    if not force_rebuild and chunks_path.exists():
        logger.info("Чанки уже существуют. Загружаем...")

        # Загрузка чанков
        chunks_df = pd.read_pickle(chunks_path)
        logger.info(f"Загружено {len(chunks_df)} чанков")

        # Подключаемся к Weaviate
        try:
            weaviate_indexer = WeaviateIndexer()
            # Сохраняем метаданные
            weaviate_indexer.chunk_metadata = chunks_df

            logger.info("✓ Подключено к Weaviate")
            logger.info("Weaviate содержит векторный индекс + BM25")
            logger.info("Для переиндексации используйте --force")

            # Для Weaviate отдельный BM25 не нужен
            return weaviate_indexer, None, chunks_df

        except Exception as e:
            logger.warning(f"Не удалось подключиться к Weaviate: {e}")
            logger.info("Убедитесь что Weaviate запущен: docker-compose up -d")
            raise

    # Строим индексы с нуля
    logger.info("Построение новых индексов...")

    # === ПОТОКОВАЯ ОБРАБОТКА: load → clean → chunk → accumulate/index ===
    # Вместо batch processing (load all → clean all → chunk all)
    # используем streaming (process doc → chunk doc → accumulate/index)
    from src.streaming_builder import build_knowledge_base_streaming

    logger.info("Потоковая обработка документов (по одному документу за раз)...")
    logger.info(f"  - LLM очистка: {'ВКЛ' if llm_clean else 'ВЫКЛ'}")
    if llm_clean:
        logger.info(f"  - Порог полезности: {min_usefulness}")
    logger.info(f"  - Режим: Weaviate (streaming index)")

    with log_timing(logger, "Потоковая обработка документов"):
        # Создаем Weaviate indexer
        weaviate_indexer = WeaviateIndexer()

        # Очищаем если force_rebuild
        if force_rebuild:
            logger.info("Очистка предыдущих данных в Weaviate...")
            weaviate_indexer.delete_all()

        # Потоковая обработка с индексацией в Weaviate
        chunks_df = build_knowledge_base_streaming(
            csv_path=str(WEBSITES_CSV),
            indexer=weaviate_indexer,
            for_weaviate=True,
            llm_clean=llm_clean,
            min_usefulness=min_usefulness,
            chunk_batch_size=500,  # индексируем по 500 чанков
            csv_chunksize=None     # используем CSV_CHUNKSIZE из config.py
        )

    logger.info(f"Всего чанков: {len(chunks_df)}")

    # Сохранение чанков
    chunks_df.to_pickle(chunks_path)
    logger.info(f"Чанки сохранены: {chunks_path}")

    # 3. Завершение индексации (Weaviate уже проиндексирован в streaming режиме выше)
    weaviate_indexer.chunk_metadata = chunks_df
    logger.info("✓ Weaviate индекс построен успешно (streaming mode)!")
    logger.info("Включает: векторный индекс + BM25 (гибридный поиск)")
    return weaviate_indexer, None, chunks_df


def process_questions(embedding_indexer, bm25_indexer,
                     questions_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Обработка вопросов (online этап)

    Args:
        embedding_indexer: векторный индексер
        bm25_indexer: BM25 индексер
        questions_df: DataFrame с вопросами (если None - загружаем из файла)

    Returns:
        DataFrame с результатами
    """
    logger = get_logger(__name__)
    logger.info("="*80)
    logger.info("ЭТАП 2: ОБРАБОТКА ВОПРОСОВ (ONLINE)")
    logger.info("="*80)

    # Загрузка вопросов если не переданы
    if questions_df is None:
        questions_df = load_and_preprocess_questions(
            str(QUESTIONS_CSV),
            apply_lemmatization=False
        )

    # Создание RAG пайплайна
    pipeline = RAGPipeline(embedding_indexer, bm25_indexer)

    # Обработка каждого вопроса
    results = []

    logger.info(f"Обработка {len(questions_df)} вопросов...")

    started_at = time.time()
    last_partial_save = time.time()
    save_every = 50  # каждые N вопросов сохраняем частичный файл
    partial_path = OUTPUTS_DIR / "submission_partial.csv"

    for idx, row in tqdm(questions_df.iterrows(), total=len(questions_df)):
        q_id = row['q_id']
        query = row['processed_query']

        try:
            # Поиск релевантных документов
            t0 = time.time()
            result = pipeline.search(query)
            dt = time.time() - t0

            # Формируем результат
            doc_ids = result['documents_id']

            # Дополняем до 5 документов если нужно
            while len(doc_ids) < 5:
                doc_ids.append(-1)  # заглушка

            results.append({
                'q_id': q_id,
                'web_list': str(doc_ids[:5])
            })

            if (idx + 1) % save_every == 0:
                # Сохраняем частичный результат
                pd.DataFrame(results).to_csv(partial_path, index=False)
                elapsed = time.time() - started_at
                per_q = elapsed / (idx + 1)
                eta = per_q * (len(questions_df) - (idx + 1))
                logger.info(f"Прогресс: {idx + 1}/{len(questions_df)} | {per_q:.2f}s/вопрос | ETA ~ {eta/60:.1f} мин | частичный файл: {partial_path}")

            # Логируем короткую метрику
            logger.debug(f"q_id={q_id} | кандидатов={result.get('num_candidates', 'NA')} | время={dt:.2f}s | docs={doc_ids[:5]}")

        except Exception as e:
            logger.error(f"Ошибка при обработке вопроса {q_id}: {e}")
            # Возвращаем пустой результат
            results.append({
                'q_id': q_id,
                'web_list': '[-1, -1, -1, -1, -1]'
            })

    results_df = pd.DataFrame(results)
    return results_df


def evaluate_on_examples(embedding_indexer, bm25_indexer):
    """
    Оценка качества на эталонных примерах

    Args:
        embedding_indexer: векторный индексер
        bm25_indexer: BM25 индексер

    Returns:
        средняя метрика
    """
    logger = get_logger(__name__)
    logger.info("="*80)
    logger.info("ОЦЕНКА НА ЭТАЛОННЫХ ПРИМЕРАХ")
    logger.info("="*80)

    from src.config import EXAMPLES_CSV

    examples_df = pd.read_csv(EXAMPLES_CSV)
    pipeline = RAGPipeline(embedding_indexer, bm25_indexer)

    # Извлекаем релевантные web_id из chunk'ов
    # (это требует дополнительной логики, упростим)

    logger.info(f"Загружено {len(examples_df)} примеров для валидации")
    logger.info("Детальная оценка на примерах будет реализована отдельно")

    # TODO: Реализовать метрику recall@5
    # Для этого нужно извлечь web_id из chunk'ов в examples

    return None


def cmd_build(args):
    """Команда: построить базу знаний"""
    logger = get_logger(__name__)
    logger.info("="*80)
    logger.info("РЕЖИМ: ПОСТРОЕНИЕ БАЗЫ ЗНАНИЙ")
    logger.info("="*80)

    if args.llm_clean:
        logger.info("[LLM-РЕЖИМ] Включена очистка документов через LLM")
        logger.info(f"[LLM-РЕЖИМ] Минимальный порог полезности: {args.min_usefulness}")
        logger.info("[LLM-РЕЖИМ] Это увеличит время обработки в 10-20 раз!")

    embedding_indexer, bm25_indexer, chunks_df = build_knowledge_base(
        force_rebuild=args.force,
        llm_clean=args.llm_clean,
        min_usefulness=args.min_usefulness
    )

    logger.info("="*80)
    logger.info("[OK] БАЗА ЗНАНИЙ ПОСТРОЕНА УСПЕШНО")
    logger.info("="*80)
    logger.info(f"Всего чанков: {len(chunks_df)}")

    if USE_WEAVIATE and WEAVIATE_AVAILABLE:
        logger.info("Векторный индекс: Weaviate (http://localhost:8080)")
        logger.info("BM25 индекс: встроен в Weaviate (гибридный поиск)")
        try:
            embedding_indexer.close()
        except Exception:
            pass


def cmd_search(args):
    """Команда: обработать вопросы"""
    logger = get_logger(__name__)
    logger.info("="*80)
    logger.info("РЕЖИМ: ОБРАБОТКА ВОПРОСОВ")
    logger.info("="*80)

    # Загрузка существующих индексов
    logger.info("Загрузка базы знаний...")

    chunks_path = PROCESSED_DIR / "chunks.pkl"

    if not chunks_path.exists():
        logger.error("ОШИБКА: База знаний не найдена! Сначала выполните: python main_pipeline.py build")
        return

    # Загрузка чанков
    chunks_df = pd.read_pickle(chunks_path)
    logger.info(f"Загружено {len(chunks_df)} чанков")

    # Загрузка векторного индекса (Weaviate-only)
    logger.info("Используется Weaviate (векторный поиск + BM25)")
    try:
        embedding_indexer = WeaviateIndexer()
        embedding_indexer.chunk_metadata = chunks_df
        bm25_indexer = None  # не нужен для Weaviate
        logger.info("✓ Подключено к Weaviate")
    except Exception as e:
        logger.error(f"Не удалось подключиться к Weaviate: {e}")
        logger.info("Убедитесь что Weaviate запущен: docker-compose up -d")
        return

    # Оптимизация параметров (опционально)
    if args.optimize:
        logger.info("="*80)
        logger.info("GRID SEARCH ОПТИМИЗАЦИЯ ПАРАМЕТРОВ")
        logger.info("="*80)

        # Загружаем вопросы для оптимизации
        optimize_questions_df = load_and_preprocess_questions(
            str(QUESTIONS_CSV),
            apply_lemmatization=False
        )

        # Создаем временный retriever для оптимизации
        from src.retrieval import HybridRetriever
        temp_retriever = HybridRetriever(embedding_indexer, bm25_indexer)

        # Запускаем grid search (используем дефолты из config если не указано)
        try:
            with log_timing(logger, "Grid Search"):
                best_params = optimize_rag_params(
                    retriever=temp_retriever,
                    questions_df=optimize_questions_df,
                    mode=args.optimize_mode,        # None = из config.GRID_SEARCH_MODE
                    sample_size=args.optimize_sample, # None = из config.GRID_SEARCH_SAMPLE_SIZE
                    use_llm_eval=None               # None = из config.GRID_SEARCH_USE_LLM
                )
            logger.info("✅ Параметры оптимизированы! Продолжаем с лучшими параметрами...")

        except Exception as e:
            logger.warning(f"ОШИБКА оптимизации: {e}")
            logger.info("Продолжаем с текущими параметрами из config.py")

    # Обработка вопросов
    if args.limit:
        logger.info(f"Обработка первых {args.limit} вопросов (режим тестирования)")
        questions_df = load_and_preprocess_questions(
            str(QUESTIONS_CSV),
            apply_lemmatization=False
        ).head(args.limit)
    else:
        logger.info("Обработка всех вопросов")
        questions_df = None

    with log_timing(logger, "Обработка всех вопросов"):
        try:
            results_df = process_questions(embedding_indexer, bm25_indexer, questions_df)
        finally:
            try:
                # Закрываем соединение с Weaviate после поиска
                if hasattr(embedding_indexer, 'close'):
                    embedding_indexer.close()
            except Exception:
                pass

    # Сохранение результатов
    output_path = OUTPUTS_DIR / "submission.csv"
    results_df.to_csv(output_path, index=False)

    logger.info("="*80)
    logger.info("[OK] ОБРАБОТКА ЗАВЕРШЕНА")
    logger.info("="*80)
    logger.info(f"Результаты: {output_path}")
    logger.info(f"Обработано вопросов: {len(results_df)}")


def cmd_all(args):
    """Команда: полный цикл (build + search)"""
    logger = get_logger(__name__)
    logger.info("="*80)
    logger.info("РЕЖИМ: ПОЛНЫЙ ЦИКЛ (BUILD + SEARCH)")
    logger.info("="*80)

    if hasattr(args, 'llm_clean') and args.llm_clean:
        logger.info("[LLM-РЕЖИМ] Включена очистка документов через LLM")

    # 1. Построение базы знаний
    logger.info("[1/2] Построение базы знаний...")
    with log_timing(logger, "Полный цикл: build"):
        embedding_indexer, bm25_indexer, chunks_df = build_knowledge_base(
            force_rebuild=args.force,
            llm_clean=getattr(args, 'llm_clean', False),
            min_usefulness=getattr(args, 'min_usefulness', 0.3)
        )

    # 2. Оптимизация параметров (опционально)
    if getattr(args, 'optimize', False):
        logger.info("="*80)
        logger.info("GRID SEARCH ОПТИМИЗАЦИЯ ПАРАМЕТРОВ")
        logger.info("="*80)

        # Загружаем вопросы для оптимизации
        optimize_questions_df = load_and_preprocess_questions(
            str(QUESTIONS_CSV),
            apply_lemmatization=False
        )

        # Создаем временный retriever для оптимизации
        from src.retrieval import HybridRetriever
        temp_retriever = HybridRetriever(embedding_indexer, bm25_indexer)

        # Запускаем grid search (используем дефолты из config если не указано)
        try:
            with log_timing(logger, "Grid Search"):
                best_params = optimize_rag_params(
                    retriever=temp_retriever,
                    questions_df=optimize_questions_df,
                    mode=getattr(args, 'optimize_mode', None),        # None = из config.GRID_SEARCH_MODE
                    sample_size=getattr(args, 'optimize_sample', None), # None = из config.GRID_SEARCH_SAMPLE_SIZE
                    use_llm_eval=None               # None = из config.GRID_SEARCH_USE_LLM
                )
            logger.info("✅ Параметры оптимизированы! Продолжаем с лучшими параметрами...")
        except Exception as e:
            logger.warning(f"⚠️ Ошибка при оптимизации: {e}")
            logger.info("Продолжаем с параметрами по умолчанию...")

    # 3. Обработка вопросов
    if getattr(args, 'optimize', False):
        logger.info("[3/3] Обработка вопросов...")
    else:
        logger.info("[2/2] Обработка вопросов...")

    if args.limit:
        logger.info(f"Обработка первых {args.limit} вопросов (режим тестирования)")
        questions_df = load_and_preprocess_questions(
            str(QUESTIONS_CSV),
            apply_lemmatization=False
        ).head(args.limit)
    else:
        questions_df = None

    with log_timing(logger, "Полный цикл: search"):
        try:
            results_df = process_questions(embedding_indexer, bm25_indexer, questions_df)
        finally:
            try:
                if hasattr(embedding_indexer, 'close'):
                    embedding_indexer.close()
            except Exception:
                pass

    # 4. Сохранение результатов
    output_path = OUTPUTS_DIR / "submission.csv"
    results_df.to_csv(output_path, index=False)

    logger.info("="*80)
    logger.info("[OK] ПОЛНЫЙ ЦИКЛ ЗАВЕРШЕН")
    logger.info("="*80)
    logger.info(f"Результаты: {output_path}")
    logger.info(f"Обработано вопросов: {len(results_df)}")


def cmd_check_env(args):
    """Команда: проверка переменных окружения"""
    import os
    logger = get_logger(__name__)
    
    logger.info("="*80)
    logger.info("ПРОВЕРКА ПЕРЕМЕННЫХ ОКРУЖЕНИЯ И КОНФИГУРАЦИИ")
    logger.info("="*80)
    
    # Импортируем config для получения актуальных значений
    from src import config
    
    def check_env_var(name, value, required=False, sensitive=False):
        """Проверка одной переменной окружения"""
        has_value = bool(value and str(value).strip())
        status = "✅" if (has_value if required else True) else "❌"
        
        if sensitive and has_value:
            # Маскируем чувствительные данные (первые 8 и последние 4 символа)
            value_str = str(value)
            if len(value_str) > 12:
                masked = value_str[:8] + "..." + value_str[-4:]
            else:
                masked = "***"
            display_value = masked
        else:
            display_value = value if has_value else "(не установлено)"
        
        logger.info(f"{status} {name:30s} = {display_value}")
        if required and not has_value:
            logger.warning(f"   ⚠️  ВНИМАНИЕ: {name} обязательна для работы!")
    
    logger.info("\n📋 ОСНОВНЫЕ ПЕРЕМЕННЫЕ ОКРУЖЕНИЯ:\n")
    
    # LLM режим
    logger.info("🤖 LLM НАСТРОЙКИ:")
    check_env_var("LLM_MODE", config.LLM_MODE)
    check_env_var("LLM_API_MODEL", config.LLM_API_MODEL)
    check_env_var("LLM_API_ROUTING", config.LLM_API_ROUTING if config.LLM_API_ROUTING else "(не установлено)")
    check_env_var("OPENROUTER_API_KEY", config.OPENROUTER_API_KEY, required=(config.LLM_MODE == "API"), sensitive=True)
    logger.info(f"   LLM_API_MAX_WORKERS = {config.LLM_API_MAX_WORKERS}")
    logger.info(f"   LLM_API_TIMEOUT = {config.LLM_API_TIMEOUT}s")
    logger.info(f"   LLM_API_RETRIES = {config.LLM_API_RETRIES}")
    
    # Grid Search
    logger.info("\n🔍 GRID SEARCH НАСТРОЙКИ:")
    logger.info(f"   GRID_SEARCH_MODE = {config.GRID_SEARCH_MODE}")
    logger.info(f"   GRID_SEARCH_SAMPLE_SIZE = {config.GRID_SEARCH_SAMPLE_SIZE}")
    logger.info(f"   GRID_SEARCH_USE_LLM = {config.GRID_SEARCH_USE_LLM}")
    
    # Weaviate
    logger.info("\n💾 WEAVIATE НАСТРОЙКИ:")
    logger.info(f"   USE_WEAVIATE = {config.USE_WEAVIATE}")
    logger.info(f"   WEAVIATE_URL = {config.WEAVIATE_URL}")
    
    # Обработка
    logger.info("\n⚙️  ПАРАМЕТРЫ ОБРАБОТКИ:")
    logger.info(f"   CSV_CHUNKSIZE = {config.CSV_CHUNKSIZE}")
    logger.info(f"   LLM_PARALLEL_WORKERS = {config.LLM_PARALLEL_WORKERS}")
    logger.info(f"   FORCE_CPU = {os.environ.get('FORCE_CPU', 'false')}")
    
    # Функциональные флаги
    logger.info("\n🎛️  ФУНКЦИОНАЛЬНЫЕ ФЛАГИ:")
    logger.info(f"   ENABLE_QUERY_EXPANSION = {config.ENABLE_QUERY_EXPANSION}")
    logger.info(f"   ENABLE_RRF = {config.ENABLE_RRF}")
    logger.info(f"   ENABLE_CONTEXT_WINDOW = {config.ENABLE_CONTEXT_WINDOW}")
    logger.info(f"   ENABLE_METADATA_FILTER = {config.ENABLE_METADATA_FILTER}")
    logger.info(f"   ENABLE_USEFULNESS_FILTER = {config.ENABLE_USEFULNESS_FILTER}")
    logger.info(f"   ENABLE_DYNAMIC_TOP_K = {config.ENABLE_DYNAMIC_TOP_K}")
    logger.info(f"   RERANKER_TYPE = {config.RERANKER_TYPE}")
    
    # Логирование
    logger.info("\n📝 ЛОГИРОВАНИЕ:")
    logger.info(f"   LOG_LEVEL = {config.LOG_LEVEL}")
    logger.info(f"   LOG_FILE = {config.LOG_FILE}")
    
    # Проверка критических настроек
    logger.info("\n" + "="*80)
    logger.info("ПРОВЕРКА КРИТИЧЕСКИХ НАСТРОЕК:")
    logger.info("="*80)
    
    issues = []
    if config.LLM_MODE == "API" and not config.OPENROUTER_API_KEY:
        issues.append("❌ OPENROUTER_API_KEY не установлен (обязателен для API режима)")
    
    if config.USE_WEAVIATE:
        try:
            import weaviate
            # Используем v4 API
            if config.WEAVIATE_URL == "http://localhost:8080":
                client = weaviate.connect_to_local()
            else:
                # Для кастомного URL используем connect_to_custom
                from urllib.parse import urlparse
                parsed = urlparse(config.WEAVIATE_URL)
                client = weaviate.connect_to_custom(
                    http_host=parsed.hostname,
                    http_port=parsed.port or 8080,
                    http_secure=parsed.scheme == "https"
                )
            # Проверяем доступность через получение коллекций
            client.collections.list_all()
            logger.info("✅ Weaviate доступен и отвечает")
            client.close()
        except Exception as e:
            issues.append(f"❌ Weaviate недоступен: {e}")
            logger.info("   💡 Запустите: docker-compose up -d")
    
    if config.LLM_MODE == "local":
        model_path = config.MODELS_DIR / config.LLM_MODEL_FILE
        if model_path.exists():
            logger.info(f"✅ Локальная LLM модель найдена: {config.LLM_MODEL_FILE}")
        else:
            issues.append(f"❌ Локальная LLM модель не найдена: {model_path}")
            logger.info("   💡 Скачайте модель: python scripts/download_models.py")
    
    if issues:
        logger.warning("\n⚠️  ОБНАРУЖЕНЫ ПРОБЛЕМЫ:")
        for issue in issues:
            logger.warning(f"   {issue}")
    else:
        logger.info("\n✅ Все критические настройки в порядке!")
    
    logger.info("\n" + "="*80)
    logger.info("💡 ПОДСКАЗКА: Установите переменные окружения перед запуском:")
    logger.info("   export LLM_MODE=api")
    logger.info("   export OPENROUTER_API_KEY=sk-or-v1-...")
    logger.info("   export LLM_API_MODEL=tngtech/deepseek-r1t2-chimera:free")
    logger.info("="*80)


def cmd_evaluate(args):
    """Команда: оценка на примерах"""
    logger = get_logger(__name__)
    logger.info("="*80)
    logger.info("РЕЖИМ: ОЦЕНКА НА ПРИМЕРАХ")
    logger.info("="*80)

    # Загрузка индексов (Weaviate-only)
    chunks_path = PROCESSED_DIR / "chunks.pkl"
    if not chunks_path.exists():
        logger.error("ОШИБКА: База знаний не найдена! Сначала выполните: python main_pipeline.py build")
        return

    chunks_df = pd.read_pickle(chunks_path)
    try:
        embedding_indexer = WeaviateIndexer()
        embedding_indexer.chunk_metadata = chunks_df
        bm25_indexer = None
    except Exception as e:
        logger.error(f"Не удалось подключиться к Weaviate: {e}")
        logger.info("Убедитесь что Weaviate запущен: docker-compose up -d")
        return

    # Оценка
    evaluate_on_examples(embedding_indexer, bm25_indexer)


def main():
    """Главная функция с парсингом аргументов"""
    # Инициализация логирования (до парсинга, чтобы ловить ранние сообщения)
    setup_logging(level=LOG_LEVEL, log_file=LOG_FILE)
    logger = get_logger(__name__)

    parser = argparse.ArgumentParser(
        description="RAG пайплайн для поиска релевантных документов Альфа-Банка",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:

BUILD (создание базы знаний):
  python main_pipeline.py build                           # Построить базу знаний
  python main_pipeline.py build --force                   # Пересоздать базу знаний
  python main_pipeline.py build --llm-clean               # С LLM очисткой документов
  python main_pipeline.py build --llm-clean --min-usefulness 0.5  # С фильтрацией

SEARCH (поиск ответов):
  python main_pipeline.py search                          # Обработать все вопросы
  python main_pipeline.py search --limit 10               # Тест на 10 вопросах
  python main_pipeline.py search --optimize               # С оптимизацией параметров (grid search)
  python main_pipeline.py search --optimize --optimize-mode test  # Тест (5 комбинаций)
  python main_pipeline.py search --optimize --optimize-mode quick  # Быстрая оптимизация (54 комбинации)
  python main_pipeline.py search --optimize --optimize-mode full  # Полная оптимизация (1225 комбинаций)

ALL (полный цикл):
  python main_pipeline.py all                             # Build + Search
  python main_pipeline.py all --llm-clean                 # С LLM очисткой
  python main_pipeline.py all --llm-clean --optimize --optimize-mode test  # С LLM очисткой и оптимизацией (test)
  python main_pipeline.py all --llm-clean --optimize --optimize-mode quick  # С LLM очисткой и оптимизацией (quick)

EVALUATE:
  python main_pipeline.py evaluate                        # Оценка на примерах

CHECK-ENV (проверка конфигурации):
  python main_pipeline.py check-env                      # Проверить все переменные окружения
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Команда для выполнения')

    # Команда: build
    parser_build = subparsers.add_parser(
        'build',
        help='Построить базу знаний (индексация документов)'
    )
    parser_build.add_argument(
        '--force',
        action='store_true',
        help='Пересоздать индексы даже если они существуют'
    )
    parser_build.add_argument(
        '--llm-clean',
        action='store_true',
        help='Использовать LLM для очистки документов (медленно, +качество)'
    )
    parser_build.add_argument(
        '--min-usefulness',
        type=float,
        default=0.3,
        help='Минимальный порог полезности для LLM фильтрации (0.0-1.0, по умолчанию 0.3)'
    )
    parser_build.set_defaults(func=cmd_build)

    # Команда: search
    parser_search = subparsers.add_parser(
        'search',
        help='Обработать вопросы (требует готовую базу знаний)'
    )
    parser_search.add_argument(
        '--limit',
        type=int,
        help='Обработать только первые N вопросов (для тестирования)'
    )
    parser_search.add_argument(
        '--optimize',
        action='store_true',
        help='Запустить grid search для оптимизации параметров перед поиском'
    )
    parser_search.add_argument(
        '--optimize-sample',
        type=int,
        default=None,
        help='Размер выборки для grid search (по умолчанию из config.GRID_SEARCH_SAMPLE_SIZE)'
    )
    parser_search.add_argument(
        '--optimize-mode',
        type=str,
        default=None,
        choices=['test', 'quick', 'full'],
        help='Режим grid search: test (5 комбинаций), quick (54 комбинации) или full (1225 комбинаций) (по умолчанию из config.GRID_SEARCH_MODE)'
    )
    parser_search.set_defaults(func=cmd_search)

    # Команда: all
    parser_all = subparsers.add_parser(
        'all',
        help='Полный цикл: построить базу знаний и обработать вопросы'
    )
    parser_all.add_argument(
        '--force',
        action='store_true',
        help='Пересоздать индексы даже если они существуют'
    )
    parser_all.add_argument(
        '--llm-clean',
        action='store_true',
        help='Использовать LLM для очистки документов (медленно, +качество)'
    )
    parser_all.add_argument(
        '--min-usefulness',
        type=float,
        default=0.3,
        help='Минимальный порог полезности для LLM фильтрации (0.0-1.0, по умолчанию 0.3)'
    )
    parser_all.add_argument(
        '--limit',
        type=int,
        help='Обработать только первые N вопросов (для тестирования)'
    )
    parser_all.add_argument(
        '--optimize',
        action='store_true',
        help='Запустить grid search для оптимизации параметров перед поиском'
    )
    parser_all.add_argument(
        '--optimize-sample',
        type=int,
        default=None,
        help='Размер выборки для grid search (по умолчанию из config.GRID_SEARCH_SAMPLE_SIZE)'
    )
    parser_all.add_argument(
        '--optimize-mode',
        type=str,
        default=None,
        choices=['test', 'quick', 'full'],
        help='Режим grid search: test (5 комбинаций), quick (54 комбинации) или full (1225 комбинаций) (по умолчанию из config.GRID_SEARCH_MODE)'
    )
    parser_all.set_defaults(func=cmd_all)

    # Команда: evaluate
    parser_eval = subparsers.add_parser(
        'evaluate',
        help='Оценка качества на эталонных примерах'
    )
    parser_eval.set_defaults(func=cmd_evaluate)

    # Команда: check-env
    parser_check = subparsers.add_parser(
        'check-env',
        help='Проверить основные переменные окружения и конфигурацию'
    )
    parser_check.set_defaults(func=cmd_check_env)

    # Парсинг аргументов
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Вывод заголовка
    logger.info("="*80)
    logger.info("RAG ПАЙПЛАЙН ДЛЯ ПОИСКА РЕЛЕВАНТНЫХ ДОКУМЕНТОВ АЛЬФА-БАНКА")
    logger.info("="*80)

    if USE_WEAVIATE and WEAVIATE_AVAILABLE:
        logger.info("Используется Weaviate для векторного поиска")
    else:
        logger.warning("Weaviate не доступен или отключен, но проект сконфигурирован как Weaviate-only.")

    if USE_WEAVIATE and not WEAVIATE_AVAILABLE:
        logger.critical("USE_WEAVIATE=true, но weaviate-client не установлен!")

    # Выполнение команды
    args.func(args)

    logger.info("[OK] Готово!")


if __name__ == "__main__":
    main()
