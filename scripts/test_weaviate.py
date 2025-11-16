"""
Тестовый скрипт для проверки работы Weaviate
Запускать после старта docker-compose up -d
"""
import pandas as pd
from src.indexing import WeaviateIndexer


def test_weaviate_basic():
    """Базовый тест Weaviate"""
    print("=" * 80)
    print("ТЕСТ 1: Базовая индексация и поиск")
    print("=" * 80)

    # Тестовые данные
    test_chunks = pd.DataFrame([
        {
            'chunk_id': '1_0',
            'web_id': 1,
            'title': 'Кредиты',
            'text': 'Альфа-Банк предлагает выгодные кредиты на любые цели'
        },
        {
            'chunk_id': '1_1',
            'web_id': 1,
            'title': 'Кэшбэк',
            'text': 'Кэшбэк на покупки до 10% по карте Альфа-Банка'
        },
        {
            'chunk_id': '2_0',
            'web_id': 2,
            'title': 'ЖКХ',
            'text': 'Оплата коммунальных услуг без комиссии через приложение'
        },
        {
            'chunk_id': '3_0',
            'web_id': 3,
            'title': 'Счета',
            'text': 'Номер счета можно посмотреть в личном кабинете или мобильном приложении'
        },
        {
            'chunk_id': '3_1',
            'web_id': 3,
            'title': 'БИК',
            'text': 'БИК банка: 044525593. Используется для переводов и платежей'
        },
    ])

    try:
        with WeaviateIndexer() as indexer:
            print("\n✓ Подключение к Weaviate успешно")

            # Очистка предыдущих данных
            print("\nОчистка предыдущих данных...")
            indexer.delete_all()

            # Индексация
            print("\nИндексация тестовых документов...")
            indexer.index_documents(test_chunks, show_progress=False)
            print(f"✓ Индексировано {len(test_chunks)} документов")

            # Тестовые запросы
            test_queries = [
                "кэшбэк за покупки",
                "оплата коммунальных услуг",
                "номер счета",
                "БИК банка"
            ]

            print("\n" + "=" * 80)
            print("ТЕСТ 2: Поиск по запросам")
            print("=" * 80)

            for query in test_queries:
                print(f"\n🔍 Запрос: '{query}'")
                print("-" * 80)

                scores, results = indexer.search(query, k=3)

                for i, (score, result) in enumerate(zip(scores, results), 1):
                    print(f"\n{i}. Score: {score:.4f}")
                    print(f"   Web ID: {result['web_id']}")
                    print(f"   Title: {result['title']}")
                    print(f"   Text: {result['text']}")

            print("\n" + "=" * 80)
            print("✓ Все тесты пройдены успешно!")
            print("=" * 80)

    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        print("\nПроверьте, что Weaviate запущен:")
        print("  docker-compose up -d")
        print("  docker-compose logs weaviate")
        raise


def test_weaviate_performance():
    """Тест производительности с большим количеством документов"""
    print("\n" + "=" * 80)
    print("ТЕСТ 3: Производительность")
    print("=" * 80)

    # Генерация большого количества тестовых документов
    num_docs = 1000
    print(f"\nГенерация {num_docs} тестовых документов...")

    test_chunks = []
    for i in range(num_docs):
        web_id = i // 10 + 1  # 10 чанков на документ
        test_chunks.append({
            'chunk_id': f'{web_id}_{i % 10}',
            'web_id': web_id,
            'title': f'Документ {web_id}',
            'text': f'Это тестовый текст документа номер {web_id}, чанк {i % 10}. '
                   f'Содержит информацию о банковских услугах, кредитах, картах и платежах.'
        })

    chunks_df = pd.DataFrame(test_chunks)

    try:
        import time

        with WeaviateIndexer() as indexer:
            # Очистка
            indexer.delete_all()

            # Индексация
            print(f"\nИндексация {num_docs} документов...")
            start_time = time.time()

            indexer.index_documents(chunks_df, show_progress=True)

            index_time = time.time() - start_time
            print(f"\n✓ Индексация завершена за {index_time:.2f} сек")
            print(f"  Скорость: {num_docs / index_time:.1f} док/сек")

            # Поиск
            print("\nТест поиска...")
            queries = [
                "банковские услуги",
                "кредиты и карты",
                "информация о платежах"
            ]

            total_search_time = 0
            num_searches = len(queries)

            for query in queries:
                start_time = time.time()
                scores, results = indexer.search(query, k=10)
                search_time = time.time() - start_time
                total_search_time += search_time

                print(f"  '{query}': {search_time*1000:.1f} мс, найдено {len(results)} результатов")

            avg_search_time = total_search_time / num_searches
            print(f"\n✓ Средняя скорость поиска: {avg_search_time*1000:.1f} мс")

            print("\n" + "=" * 80)
            print("✓ Тест производительности пройден!")
            print("=" * 80)

    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        raise


def test_weaviate_comparison():
    """Сравнение результатов Weaviate и FAISS"""
    print("\n" + "=" * 80)
    print("ТЕСТ 4: Сравнение Weaviate vs FAISS")
    print("=" * 80)

    from src.indexing import EmbeddingIndexer

    test_chunks = pd.DataFrame([
        {
            'chunk_id': '1_0',
            'web_id': 1,
            'title': 'Кэшбэк',
            'text': 'Кэшбэк на покупки до 10% по карте Альфа-Банка'
        },
        {
            'chunk_id': '2_0',
            'web_id': 2,
            'title': 'ЖКХ',
            'text': 'Оплата коммунальных услуг без комиссии'
        },
        {
            'chunk_id': '3_0',
            'web_id': 3,
            'title': 'Счета',
            'text': 'Номер счета можно посмотреть в личном кабинете'
        },
    ])

    query = "кэшбэк за покупки"

    try:
        # Weaviate
        print("\n1. Weaviate:")
        with WeaviateIndexer() as weaviate_idx:
            weaviate_idx.delete_all()
            weaviate_idx.index_documents(test_chunks, show_progress=False)

            w_scores, w_results = weaviate_idx.search(query, k=3)
            for i, (score, result) in enumerate(zip(w_scores, w_results), 1):
                print(f"  {i}. Score: {score:.4f} - Web ID: {result['web_id']} - {result['text'][:50]}...")

        # FAISS
        print("\n2. FAISS:")
        faiss_idx = EmbeddingIndexer()
        texts = test_chunks['text'].tolist()
        embeddings = faiss_idx.create_embeddings(texts, show_progress=False)
        faiss_idx.build_faiss_index(embeddings)
        faiss_idx.chunk_metadata = test_chunks

        query_emb = faiss_idx.model.encode([query], normalize_embeddings=True)[0]
        f_scores, f_indices = faiss_idx.search(query_emb, k=3)

        for i, (score, idx) in enumerate(zip(f_scores, f_indices), 1):
            result = test_chunks.iloc[idx]
            print(f"  {i}. Score: {score:.4f} - Web ID: {result['web_id']} - {result['text'][:50]}...")

        print("\n✓ Оба метода работают корректно!")

    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        raise


if __name__ == "__main__":
    print("\n🚀 Запуск тестов Weaviate")
    print("Убедитесь, что Weaviate запущен: docker-compose up -d\n")

    try:
        # Базовые тесты
        test_weaviate_basic()

        # Тест производительности (опционально)
        user_input = input("\nЗапустить тест производительности? (y/n): ")
        if user_input.lower() == 'y':
            test_weaviate_performance()

        # Сравнение с FAISS
        user_input = input("\nСравнить с FAISS? (y/n): ")
        if user_input.lower() == 'y':
            test_weaviate_comparison()

        print("\n" + "=" * 80)
        print("🎉 ВСЕ ТЕСТЫ ПРОЙДЕНЫ УСПЕШНО!")
        print("=" * 80)
        print("\nWeaviate готов к использованию в вашем проекте.")
        print("Смотрите WEAVIATE_SETUP.md для дополнительной информации.\n")

    except KeyboardInterrupt:
        print("\n\n⚠️  Тесты прерваны пользователем")
    except Exception as e:
        print(f"\n\n❌ Тесты завершились с ошибкой: {e}")
        print("\nПроверьте:")
        print("  1. Weaviate запущен: docker-compose ps")
        print("  2. Логи Weaviate: docker-compose logs weaviate")
        print("  3. Доступность: curl http://localhost:8080/v1/.well-known/ready")
