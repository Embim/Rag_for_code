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
    ENABLE_AGENT_RAG
)
from src.preprocessing import load_and_preprocess_documents, load_and_preprocess_questions
from src.chunking import create_chunks_from_documents
from src.indexing import build_indexes, EmbeddingIndexer, BM25Indexer, WeaviateIndexer
from src.retrieval import RAGPipeline
from src.llm_preprocessing import apply_llm_cleaning
from src.grid_search_optimizer import optimize_rag_params

# Проверка доступности Weaviate
try:
    import weaviate
    WEAVIATE_AVAILABLE = True
except ImportError:
    WEAVIATE_AVAILABLE = False
    if USE_WEAVIATE:
        print("[CRITICAL] USE_WEAVIATE=true, но weaviate-client не установлен!")


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
    print("\n" + "="*80)
    print("ЭТАП 1: ПОСТРОЕНИЕ БАЗЫ ЗНАНИЙ (OFFLINE)")
    print("="*80)

    chunks_path = PROCESSED_DIR / "chunks.pkl"
    bm25_path = MODELS_DIR / "bm25.pkl"

    # Определяем режим работы
    use_weaviate = USE_WEAVIATE and WEAVIATE_AVAILABLE

    if use_weaviate:
        print("\n[РЕЖИМ] Используется Weaviate для векторного поиска")

        # Проверяем существуют ли чанки
        if not force_rebuild and chunks_path.exists():
            print("\nЧанки уже существуют. Загружаем...")

            # Загрузка чанков
            chunks_df = pd.read_pickle(chunks_path)
            print(f"Загружено {len(chunks_df)} чанков")

            # Подключаемся к Weaviate
            try:
                weaviate_indexer = WeaviateIndexer()
                # Сохраняем метаданные
                weaviate_indexer.chunk_metadata = chunks_df

                print("✓ Подключено к Weaviate")
                print("  Weaviate содержит векторный индекс + BM25")
                print("  Для переиндексации используйте --force")

                # Для Weaviate BM25 не нужен (встроен в Weaviate)
                return weaviate_indexer, None, chunks_df

            except Exception as e:
                print(f"\n[WARNING] Не удалось подключиться к Weaviate: {e}")
                print("Убедитесь что Weaviate запущен: docker-compose up -d")
                print("Или установите USE_WEAVIATE=false для использования FAISS")
                raise

        # Строим индексы с нуля
        print("\nПостроение новых индексов...")

    else:
        print("\n[РЕЖИМ] Используется FAISS для векторного поиска")

        faiss_path = MODELS_DIR / "faiss.index"

        # Проверяем существуют ли индексы (FAISS режим)
        if not force_rebuild and chunks_path.exists() and faiss_path.exists() and bm25_path.exists():
            print("\nИндексы уже существуют. Загружаем...")

            # Загрузка чанков
            chunks_df = pd.read_pickle(chunks_path)
            print(f"Загружено {len(chunks_df)} чанков")

            # Загрузка индексов
            embedding_indexer = EmbeddingIndexer()
            embedding_indexer.load_index(str(faiss_path))
            embedding_indexer.chunk_metadata = chunks_df

            bm25_indexer = BM25Indexer()
            bm25_indexer.load_index(str(bm25_path))

            return embedding_indexer, bm25_indexer, chunks_df

        print("\nПостроение новых индексов...")

    # === ОБЩАЯ ЧАСТЬ: Предобработка и чанкинг ===

    # 1. Загрузка и предобработка документов
    print("\n1. Предобработка документов...")
    documents_df = load_and_preprocess_documents(
        str(WEBSITES_CSV),
        apply_lemmatization=False  # Отключаем для скорости
    )

    # 1.5. LLM очистка (опционально)
    if llm_clean:
        print("\n1.5. LLM-очистка документов (это может занять несколько часов)...")
        print(f"     Минимальный порог полезности: {min_usefulness}")

        try:
            documents_df = apply_llm_cleaning(
                documents_df,
                min_usefulness=min_usefulness,
                verbose=True
            )

            # Используем clean_text вместо text для дальнейшей обработки
            if 'clean_text' in documents_df.columns:
                documents_df['text'] = documents_df['clean_text']

            print(f"\n✅ LLM-очистка завершена! Документов после фильтрации: {len(documents_df)}")

        except Exception as e:
            print(f"\n⚠️  ОШИБКА LLM-очистки: {e}")
            print("     Продолжаем с исходными документами без LLM обработки")

    # 2. Разбиение на чанки
    print("\n2. Разбиение на чанки...")
    chunks_df = create_chunks_from_documents(documents_df, method='words')

    # Сохранение чанков
    chunks_df.to_pickle(chunks_path)
    print(f"Чанки сохранены: {chunks_path}")

    # 3. Построение векторного индекса
    if use_weaviate:
        print("\n3. Построение Weaviate индекса (с встроенным BM25)...")

        try:
            weaviate_indexer = WeaviateIndexer()

            # Очищаем предыдущие данные если force_rebuild
            if force_rebuild:
                print("Очистка предыдущих данных в Weaviate...")
                weaviate_indexer.delete_all()

            # Индексируем документы (Weaviate автоматически создаст BM25 индекс)
            weaviate_indexer.index_documents(chunks_df, show_progress=True)

            # Сохраняем метаданные
            weaviate_indexer.chunk_metadata = chunks_df

            print("\n✓ Weaviate индекс построен успешно!")
            print("  Включает: векторный индекс + BM25 (гибридный поиск)")

            # Для Weaviate не нужен отдельный BM25
            return weaviate_indexer, None, chunks_df

        except Exception as e:
            print(f"\n[ERROR] Ошибка при построении Weaviate индекса: {e}")
            print("Убедитесь что Weaviate запущен: docker-compose up -d")
            raise

    else:
        print("\n3. Построение BM25 индекса...")
        bm25_indexer = BM25Indexer()
        texts = chunks_df['text'].tolist()
        bm25_indexer.build_index(texts)
        bm25_indexer.save_index(str(bm25_path))

        print("\n4. Построение FAISS индекса...")
        embedding_indexer = EmbeddingIndexer()
        embeddings = embedding_indexer.create_embeddings(texts)
        embedding_indexer.build_faiss_index(embeddings)
        embedding_indexer.chunk_metadata = chunks_df
        embedding_indexer.save_index(str(MODELS_DIR / "faiss.index"))

        print("\nБаза знаний построена успешно!")
        return embedding_indexer, bm25_indexer, chunks_df


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
    print("\n" + "="*80)
    print("ЭТАП 2: ОБРАБОТКА ВОПРОСОВ (ONLINE)")
    print("="*80)

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

    print(f"\nОбработка {len(questions_df)} вопросов...")

    for idx, row in tqdm(questions_df.iterrows(), total=len(questions_df)):
        q_id = row['q_id']
        query = row['processed_query']

        try:
            # Поиск релевантных документов
            result = pipeline.search(query)

            # Формируем результат
            doc_ids = result['documents_id']

            # Дополняем до 5 документов если нужно
            while len(doc_ids) < 5:
                doc_ids.append(-1)  # заглушка

            results.append({
                'q_id': q_id,
                'web_list': str(doc_ids[:5])
            })

        except Exception as e:
            print(f"\nОшибка при обработке вопроса {q_id}: {e}")
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
    print("\n" + "="*80)
    print("ОЦЕНКА НА ЭТАЛОННЫХ ПРИМЕРАХ")
    print("="*80)

    from src.config import EXAMPLES_CSV

    examples_df = pd.read_csv(EXAMPLES_CSV)
    pipeline = RAGPipeline(embedding_indexer, bm25_indexer)

    # Извлекаем релевантные web_id из chunk'ов
    # (это требует дополнительной логики, упростим)

    print(f"\nЗагружено {len(examples_df)} примеров для валидации")
    print("Детальная оценка на примерах будет реализована отдельно")

    # TODO: Реализовать метрику recall@5
    # Для этого нужно извлечь web_id из chunk'ов в examples

    return None


def cmd_build(args):
    """Команда: построить базу знаний"""
    print("\n" + "="*80)
    print("РЕЖИМ: ПОСТРОЕНИЕ БАЗЫ ЗНАНИЙ")
    print("="*80)

    if args.llm_clean:
        print("\n[LLM-РЕЖИМ] Включена очистка документов через LLM")
        print(f"[LLM-РЕЖИМ] Минимальный порог полезности: {args.min_usefulness}")
        print("[LLM-РЕЖИМ] Это увеличит время обработки в 10-20 раз!")

    embedding_indexer, bm25_indexer, chunks_df = build_knowledge_base(
        force_rebuild=args.force,
        llm_clean=args.llm_clean,
        min_usefulness=args.min_usefulness
    )

    print("\n" + "="*80)
    print("[OK] БАЗА ЗНАНИЙ ПОСТРОЕНА УСПЕШНО")
    print("="*80)
    print(f"Всего чанков: {len(chunks_df)}")

    if USE_WEAVIATE and WEAVIATE_AVAILABLE:
        print(f"Векторный индекс: Weaviate (http://localhost:8080)")
        print(f"BM25 индекс: встроен в Weaviate (гибридный поиск)")
    else:
        print(f"Векторный индекс: {MODELS_DIR / 'faiss.index'}")
        print(f"BM25 индекс: {MODELS_DIR / 'bm25.pkl'}")


def cmd_search(args):
    """Команда: обработать вопросы"""
    print("\n" + "="*80)
    print("РЕЖИМ: ОБРАБОТКА ВОПРОСОВ")
    print("="*80)

    # Загрузка существующих индексов
    print("\nЗагрузка базы знаний...")

    chunks_path = PROCESSED_DIR / "chunks.pkl"

    if not chunks_path.exists():
        print("\n[X] ОШИБКА: База знаний не найдена!")
        print("Сначала выполните: python main_pipeline.py build")
        return

    # Определяем режим работы
    use_weaviate = USE_WEAVIATE and WEAVIATE_AVAILABLE

    # Загрузка чанков
    chunks_df = pd.read_pickle(chunks_path)
    print(f"Загружено {len(chunks_df)} чанков")

    # Загрузка векторного индекса
    if use_weaviate:
        print("Используется Weaviate (векторный поиск + BM25)")
        try:
            embedding_indexer = WeaviateIndexer()
            embedding_indexer.chunk_metadata = chunks_df
            bm25_indexer = None  # не нужен для Weaviate
            print("✓ Подключено к Weaviate")
        except Exception as e:
            print(f"\n[ERROR] Не удалось подключиться к Weaviate: {e}")
            print("Убедитесь что Weaviate запущен: docker-compose up -d")
            return
    else:
        print("Используется FAISS для векторного поиска")
        faiss_path = MODELS_DIR / "faiss.index"
        bm25_path = MODELS_DIR / "bm25.pkl"

        if not faiss_path.exists() or not bm25_path.exists():
            print("\n[X] ОШИБКА: FAISS или BM25 индекс не найден!")
            print("Сначала выполните: python main_pipeline.py build")
            return

        # Загрузка BM25
        bm25_indexer = BM25Indexer()
        bm25_indexer.load_index(str(bm25_path))

        # Загрузка FAISS
        embedding_indexer = EmbeddingIndexer()
        embedding_indexer.load_index(str(faiss_path))
        embedding_indexer.chunk_metadata = chunks_df

    # Оптимизация параметров (опционально)
    if args.optimize:
        print("\n" + "="*80)
        print("GRID SEARCH ОПТИМИЗАЦИЯ ПАРАМЕТРОВ")
        print("="*80)

        # Загружаем вопросы для оптимизации
        optimize_questions_df = load_and_preprocess_questions(
            str(QUESTIONS_CSV),
            apply_lemmatization=False
        )

        # Создаем временный retriever для оптимизации
        from src.retrieval import HybridRetriever
        temp_retriever = HybridRetriever(embedding_indexer, bm25_indexer)

        # Запускаем grid search
        try:
            best_params = optimize_rag_params(
                retriever=temp_retriever,
                questions_df=optimize_questions_df,
                mode=args.optimize_mode,
                sample_size=args.optimize_sample
            )
            print("\n✅ Параметры оптимизированы! Продолжаем с лучшими параметрами...")

        except Exception as e:
            print(f"\n⚠️  ОШИБКА оптимизации: {e}")
            print("     Продолжаем с текущими параметрами из config.py")

    # Обработка вопросов
    if args.limit:
        print(f"\nОбработка первых {args.limit} вопросов (режим тестирования)")
        questions_df = load_and_preprocess_questions(
            str(QUESTIONS_CSV),
            apply_lemmatization=False
        ).head(args.limit)
    else:
        print("\nОбработка всех вопросов")
        questions_df = None

    results_df = process_questions(embedding_indexer, bm25_indexer, questions_df)

    # Сохранение результатов
    output_path = OUTPUTS_DIR / "submission.csv"
    results_df.to_csv(output_path, index=False)

    print("\n" + "="*80)
    print("[OK] ОБРАБОТКА ЗАВЕРШЕНА")
    print("="*80)
    print(f"Результаты: {output_path}")
    print(f"Обработано вопросов: {len(results_df)}")


def cmd_all(args):
    """Команда: полный цикл (build + search)"""
    print("\n" + "="*80)
    print("РЕЖИМ: ПОЛНЫЙ ЦИКЛ (BUILD + SEARCH)")
    print("="*80)

    if hasattr(args, 'llm_clean') and args.llm_clean:
        print("\n[LLM-РЕЖИМ] Включена очистка документов через LLM")

    # 1. Построение базы знаний
    print("\n[1/2] Построение базы знаний...")
    embedding_indexer, bm25_indexer, chunks_df = build_knowledge_base(
        force_rebuild=args.force,
        llm_clean=getattr(args, 'llm_clean', False),
        min_usefulness=getattr(args, 'min_usefulness', 0.3)
    )

    # 2. Обработка вопросов
    print("\n[2/2] Обработка вопросов...")

    if args.limit:
        print(f"Обработка первых {args.limit} вопросов (режим тестирования)")
        questions_df = load_and_preprocess_questions(
            str(QUESTIONS_CSV),
            apply_lemmatization=False
        ).head(args.limit)
    else:
        questions_df = None

    results_df = process_questions(embedding_indexer, bm25_indexer, questions_df)

    # 3. Сохранение результатов
    output_path = OUTPUTS_DIR / "submission.csv"
    results_df.to_csv(output_path, index=False)

    print("\n" + "="*80)
    print("[OK] ПОЛНЫЙ ЦИКЛ ЗАВЕРШЕН")
    print("="*80)
    print(f"Результаты: {output_path}")
    print(f"Обработано вопросов: {len(results_df)}")


def cmd_evaluate(args):
    """Команда: оценка на примерах"""
    print("\n" + "="*80)
    print("РЕЖИМ: ОЦЕНКА НА ПРИМЕРАХ")
    print("="*80)

    # Загрузка индексов
    chunks_path = PROCESSED_DIR / "chunks.pkl"
    faiss_path = MODELS_DIR / "faiss.index"
    bm25_path = MODELS_DIR / "bm25.pkl"

    if not chunks_path.exists() or not faiss_path.exists() or not bm25_path.exists():
        print("\n[X] ОШИБКА: База знаний не найдена!")
        print("Сначала выполните: python main_pipeline.py build")
        return

    chunks_df = pd.read_pickle(chunks_path)
    embedding_indexer = EmbeddingIndexer()
    embedding_indexer.load_index(str(faiss_path))
    embedding_indexer.chunk_metadata = chunks_df

    bm25_indexer = BM25Indexer()
    bm25_indexer.load_index(str(bm25_path))

    # Оценка
    evaluate_on_examples(embedding_indexer, bm25_indexer)


def main():
    """Главная функция с парсингом аргументов"""
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
  python main_pipeline.py search --optimize --optimize-mode full  # Полная оптимизация

ALL (полный цикл):
  python main_pipeline.py all                             # Build + Search
  python main_pipeline.py all --llm-clean --optimize      # С LLM очисткой и оптимизацией

EVALUATE:
  python main_pipeline.py evaluate                        # Оценка на примерах
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
        default=50,
        help='Размер выборки для grid search (по умолчанию 50)'
    )
    parser_search.add_argument(
        '--optimize-mode',
        type=str,
        default='quick',
        choices=['quick', 'full'],
        help='Режим grid search: quick (быстрый) или full (полный)'
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
    parser_all.set_defaults(func=cmd_all)

    # Команда: evaluate
    parser_eval = subparsers.add_parser(
        'evaluate',
        help='Оценка качества на эталонных примерах'
    )
    parser_eval.set_defaults(func=cmd_evaluate)

    # Парсинг аргументов
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Вывод заголовка
    print("="*80)
    print("RAG ПАЙПЛАЙН ДЛЯ ПОИСКА РЕЛЕВАНТНЫХ ДОКУМЕНТОВ АЛЬФА-БАНКА")
    print("="*80)

    if USE_WEAVIATE and WEAVIATE_AVAILABLE:
        print("[INFO] Используется Weaviate для векторного поиска")
    else:
        print("[INFO] Используется FAISS для векторного поиска")

    # Выполнение команды
    args.func(args)

    print("\n[OK] Готово!")


if __name__ == "__main__":
    main()
