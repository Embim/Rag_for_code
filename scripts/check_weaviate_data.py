#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Проверка данных в Weaviate для Code RAG
"""
import weaviate
import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def check_weaviate_data():
    """Проверить наличие данных в Weaviate."""

    # Подключение к Weaviate
    weaviate_url = os.getenv("WEAVIATE_URL", "http://localhost:8080")
    
    # Parse URL
    if weaviate_url.startswith("http://"):
        host = weaviate_url.replace("http://", "").split(":")[0]
        port = int(weaviate_url.split(":")[-1]) if ":" in weaviate_url else 8080
    elif weaviate_url.startswith("https://"):
        host = weaviate_url.replace("https://", "").split(":")[0]
        port = int(weaviate_url.split(":")[-1]) if ":" in weaviate_url else 443
    else:
        host = weaviate_url.split(":")[0]
        port = int(weaviate_url.split(":")[-1]) if ":" in weaviate_url else 8080

    print("="*80)
    print("ПРОВЕРКА ДАННЫХ В WEAVIATE")
    print("="*80)
    print(f"\nПодключение к: {weaviate_url} (host={host}, port={port})")

    try:
        # Подключение
        client = weaviate.connect_to_local(
            host=host,
            port=port
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
            print("Запустите индексацию:")
            print("  python -m src.code_rag.graph.build_and_index <repo_path>")
            print("  или")
            print("  python scripts/reindex_weaviate.py")
        else:
            total_objects = 0
            for collection_name in collections:
                collection = client.collections.get(collection_name)

                # Подсчитать объекты
                try:
                    response = collection.aggregate.over_all(total_count=True)
                    count = response.total_count
                    total_objects += count
                except Exception as e:
                    print(f"\n[!] Ошибка при подсчете объектов в {collection_name}: {e}")
                    count = 0

                print(f"\n[+] {collection_name}")
                print(f"   Количество объектов: {count:,}")

                # Показать несколько примеров, если есть данные
                if count > 0:
                    print(f"\n   Примеры данных (первые 3):")
                    try:
                        results = collection.query.fetch_objects(limit=3)

                        for i, obj in enumerate(results.objects, 1):
                            print(f"\n   {i}. UUID: {obj.uuid}")
                            # Показать первые несколько свойств
                            props = obj.properties
                            for key, value in list(props.items())[:5]:
                                if isinstance(value, str) and len(value) > 100:
                                    value = value[:100] + "..."
                                print(f"      {key}: {value}")
                    except Exception as e:
                        print(f"   [!] Ошибка при получении примеров: {e}")
            
            # Тестовый поиск
            print("\n" + "-"*80)
            print("ТЕСТОВЫЙ ПОИСК:")
            print("-"*80)
            
            # Ищем коллекцию CodeNode (основная коллекция для кода)
            if "CodeNode" in collections:
                try:
                    collection = client.collections.get("CodeNode")
                    print("\nТестовый поиск по запросу 'book trade':")
                    
                    # Простой поиск
                    results = collection.query.near_text(
                        query="book trade",
                        limit=3
                    )
                    
                    if results.objects:
                        print(f"   Найдено результатов: {len(results.objects)}")
                        for i, obj in enumerate(results.objects, 1):
                            props = obj.properties
                            name = props.get("name", "Unknown")
                            node_type = props.get("node_type", "Unknown")
                            print(f"   {i}. {name} ({node_type})")
                    else:
                        print("   [!] Результаты не найдены")
                except Exception as e:
                    print(f"   [!] Ошибка при поиске: {e}")

        print("\n" + "="*80)
        print("СТАТИСТИКА:")
        print("="*80)
        print(f"\nВсего коллекций: {len(collections)}")
        print(f"Всего объектов: {total_objects:,}")
        
        if total_objects == 0:
            print("\n⚠️  ВНИМАНИЕ: Weaviate пуст!")
            print("   Запустите индексацию данных:")
            print("   python scripts/reindex_weaviate.py")

        client.close()

    except Exception as e:
        print(f"\n[X] Ошибка: {e}")
        print("\nУбедитесь, что Weaviate запущен:")
        print("  docker-compose up -d weaviate")
        print("  docker-compose ps")
        import traceback
        traceback.print_exc()

    print("\n" + "="*80)


if __name__ == "__main__":
    check_weaviate_data()
