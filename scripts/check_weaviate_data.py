#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Проверка данных в Weaviate для Code RAG
"""
import weaviate
from weaviate.classes.init import Auth
import os


def check_weaviate_data():
    """Проверить наличие данных в Weaviate."""

    # Подключение к Weaviate
    weaviate_url = os.getenv("WEAVIATE_URL", "http://localhost:8080")

    print("="*80)
    print("ПРОВЕРКА ДАННЫХ В WEAVIATE")
    print("="*80)
    print(f"\nПодключение к: {weaviate_url}")

    try:
        # Подключение
        client = weaviate.connect_to_local(
            host="localhost",
            port=8080
        )

        print("[OK] Подключение успешно\n")

        # Получить список всех коллекций/классов
        print("-"*80)
        print("СПИСОК КОЛЛЕКЦИЙ:")
        print("-"*80)

        collections = client.collections.list_all()

        if not collections:
            print("[!] Нет коллекций в Weaviate")
            print("\nВозможно, данные еще не проиндексированы.")
            print("Запустите индексацию: python -m src.code_rag.graph.build_and_index")
        else:
            for collection_name in collections:
                collection = client.collections.get(collection_name)

                # Подсчитать объекты
                response = collection.aggregate.over_all(total_count=True)
                count = response.total_count

                print(f"\n[+] {collection_name}")
                print(f"   Количество объектов: {count:,}")

                # Показать несколько примеров, если есть данные
                if count > 0:
                    print(f"\n   Примеры данных (первые 3):")
                    results = collection.query.fetch_objects(limit=3)

                    for i, obj in enumerate(results.objects, 1):
                        print(f"\n   {i}. UUID: {obj.uuid}")
                        # Показать первые несколько свойств
                        props = obj.properties
                        for key, value in list(props.items())[:5]:
                            if isinstance(value, str) and len(value) > 100:
                                value = value[:100] + "..."
                            print(f"      {key}: {value}")

        print("\n" + "="*80)
        print("СТАТИСТИКА:")
        print("="*80)
        total_objects = sum(
            client.collections.get(name).aggregate.over_all(total_count=True).total_count
            for name in collections
        )
        print(f"\nВсего коллекций: {len(collections)}")
        print(f"Всего объектов: {total_objects:,}")

        client.close()

    except Exception as e:
        print(f"\n[X] Ошибка: {e}")
        print("\nУбедитесь, что Weaviate запущен:")
        print("  docker-compose up -d weaviate")
        import traceback
        traceback.print_exc()

    print("\n" + "="*80)


if __name__ == "__main__":
    check_weaviate_data()
