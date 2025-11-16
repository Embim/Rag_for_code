# -*- coding: utf-8 -*-
"""
Простой тест подключения к Weaviate
"""
try:
    print("Попытка подключения к Weaviate...")

    from src.indexing import WeaviateIndexer

    # Пытаемся создать WeaviateIndexer
    indexer = WeaviateIndexer()

    print("\n[OK] Weaviate подключен успешно!")
    print(f"Класс/коллекция: {indexer.class_name}")
    print(f"Размерность эмбеддингов: {indexer.dimension}")

    # Закрываем соединение
    indexer.close()

    print("\n[OK] Тест пройден!")

except Exception as e:
    print(f"\n[X] Ошибка: {e}")
    import traceback
    traceback.print_exc()
