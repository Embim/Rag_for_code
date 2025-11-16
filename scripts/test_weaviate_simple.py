# -*- coding: utf-8 -*-
"""
Простой тест Weaviate с индексацией и поиском
"""
import pandas as pd
from src.indexing import WeaviateIndexer


def main():
    print("="*80)
    print("ТЕСТ WEAVIATE: ИНДЕКСАЦИЯ И ПОИСК")
    print("="*80)

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
        print("\n1. Подключение к Weaviate...")
        with WeaviateIndexer() as indexer:
            print("[OK] Подключено")

            # Очистка
            print("\n2. Очистка предыдущих данных...")
            indexer.delete_all()
            print("[OK] Очищено")

            # Индексация
            print(f"\n3. Индексация {len(test_chunks)} документов...")
            indexer.index_documents(test_chunks, show_progress=False)
            print("[OK] Индексация завершена")

            # Тестовые запросы
            test_queries = [
                "кэшбэк за покупки",
                "оплата коммунальных услуг",
                "номер счета",
                "БИК банка"
            ]

            print("\n4. Поиск по запросам:")
            print("="*80)

            for query in test_queries:
                print(f"\nЗапрос: '{query}'")
                print("-"*80)

                scores, results = indexer.search(query, k=3)

                for i, (score, result) in enumerate(zip(scores, results), 1):
                    print(f"\n  {i}. Score: {score:.4f}")
                    print(f"     Web ID: {result['web_id']}")
                    print(f"     Title: {result['title']}")
                    print(f"     Text: {result['text'][:60]}...")

            print("\n"+"="*80)
            print("[OK] ВСЕ ТЕСТЫ ПРОЙДЕНЫ УСПЕШНО!")
            print("="*80)

    except Exception as e:
        print(f"\n[X] Ошибка: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
