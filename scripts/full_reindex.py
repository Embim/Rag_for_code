"""
Полная переиндексация графа знаний.

Удаляет все данные из Neo4j и Weaviate, затем переиндексирует репозитории.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.graph.neo4j_client import Neo4jClient
from src.core.graph.weaviate_indexer import WeaviateIndexer
from src.core.graph.build_and_index import build_and_index
from src.infra.logger import get_logger
from src.infra.config.search import get_search_config

logger = get_logger(__name__)


def clear_neo4j():
    """Очистить Neo4j граф."""
    print("\n" + "="*60)
    print("🗑️  ОЧИСТКА NEO4J")
    print("="*60)

    client = Neo4jClient(
        uri="bolt://localhost:7687",
        user="neo4j",
        password="password"
    )

    try:
        print("⏳ Удаление всех нод и связей...")
        client.clear_database(batch_size=10000)
        print("✅ Neo4j очищен!")
    finally:
        client.close()


def clear_weaviate():
    """Очистить Weaviate индекс."""
    print("\n" + "="*60)
    print("🗑️  ОЧИСТКА WEAVIATE")
    print("="*60)

    config = get_search_config()
    indexer = WeaviateIndexer(
        url=config.weaviate_url,
        collection_name=config.weaviate_collection
    )

    try:
        print("⏳ Удаление коллекции...")
        indexer.delete_collection()
        print("✅ Weaviate очищен!")

        print("⏳ Создание новой коллекции...")
        indexer.create_collection()
        print("✅ Коллекция создана!")
    finally:
        indexer.close()


def reindex_repositories():
    """Переиндексировать все репозитории."""
    print("\n" + "="*60)
    print("📚 ИНДЕКСАЦИЯ РЕПОЗИТОРИЕВ")
    print("="*60)

    repos_dir = Path(__file__).parent.parent / "data" / "repos"

    if not repos_dir.exists():
        print(f"❌ Директория с репозиториями не найдена: {repos_dir}")
        return

    print(f"📂 Директория: {repos_dir}")
    print(f"📦 Репозитории:")
    for repo in repos_dir.iterdir():
        if repo.is_dir() and not repo.name.startswith('.'):
            print(f"   - {repo.name}")

    print("\n⏳ Начинаем индексацию...")

    stats = build_and_index(
        repos_dir=str(repos_dir),
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="password",
        weaviate_url="http://localhost:8080",
    )

    print("\n" + "="*60)
    print("📊 СТАТИСТИКА ИНДЕКСАЦИИ")
    print("="*60)
    print(f"📈 Ноды создано: {stats.get('nodes_created', 0)}")
    print(f"🔗 Связей создано: {stats.get('relationships_created', 0)}")
    print(f"🔍 Векторов создано: {stats.get('vectors_indexed', 0)}")
    print("="*60)


def main():
    """Главная функция."""
    print("\n" + "="*60)
    print("🚀 ПОЛНАЯ ПЕРЕИНДЕКСАЦИЯ ГРАФА ЗНАНИЙ")
    print("="*60)
    print("\n⚠️  ВНИМАНИЕ: Это удалит ВСЕ данные из Neo4j и Weaviate!")

    response = input("\n❓ Продолжить? (yes/no): ").strip().lower()

    if response not in ['yes', 'y', 'да', 'д']:
        print("❌ Отменено пользователем")
        return

    try:
        # Шаг 1: Очистить Neo4j
        clear_neo4j()

        # Шаг 2: Очистить Weaviate
        clear_weaviate()

        # Шаг 3: Переиндексировать
        reindex_repositories()

        print("\n" + "="*60)
        print("✅ ПЕРЕИНДЕКСАЦИЯ ЗАВЕРШЕНА!")
        print("="*60)
        print("\n💡 Проверьте результаты:")
        print("   Neo4j:    http://localhost:7474")
        print("   Weaviate: http://localhost:8080")

        print("\n💡 Проверьте количество связей в Neo4j:")
        print("   MATCH ()-[r]->() RETURN type(r), count(*)")

    except Exception as e:
        logger.error(f"Ошибка при переиндексации: {e}", exc_info=True)
        print(f"\n❌ Ошибка: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
