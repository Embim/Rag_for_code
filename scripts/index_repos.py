"""
Индексация репозиториев *** и *** из F:/
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.code_rag.graph.build_and_index import build_and_index

print("🚀 Начинаем индексацию репозиториев...")
print("="*60)

# Пути к репозиториям
repos = [
    ("***", "F:/***"),
    ("***", "F:/***"),
]

total_stats = {
    'nodes_created': 0,
    'relationships_created': 0,
    'vectors_indexed': 0,
}

# Индексируем каждый репозиторий
for name, path in repos:
    print(f"\n📚 Индексация {name}...")
    print(f"📂 Путь: {path}")

    if not Path(path).exists():
        print(f"❌ Репозиторий не найден: {path}")
        continue

    try:
        stats = build_and_index(
            repos_dir=path,
            neo4j_uri="bolt://localhost:7687",
            neo4j_user="neo4j",
            neo4j_password="password",
            weaviate_url="http://localhost:8080"
        )

        # Суммируем статистику
        total_stats['nodes_created'] += stats.get('nodes_created', 0)
        total_stats['relationships_created'] += stats.get('relationships_created', 0)
        total_stats['vectors_indexed'] += stats.get('vectors_indexed', 0)

        print(f"✅ {name} проиндексирован!")
        print(f"   Нод: {stats.get('nodes_created', 0)}")
        print(f"   Связей: {stats.get('relationships_created', 0)}")

    except Exception as e:
        print(f"❌ Ошибка при индексации {name}: {e}")

# Итоговая статистика
print("\n" + "="*60)
print("✅ ИНДЕКСАЦИЯ ЗАВЕРШЕНА")
print("="*60)
print(f"📈 Всего нод создано: {total_stats['nodes_created']}")
print(f"🔗 Всего связей создано: {total_stats['relationships_created']}")
print(f"🔍 Всего векторов: {total_stats['vectors_indexed']}")
print("="*60)

print("\n💡 Проверьте результаты:")
print("   Neo4j: http://localhost:7474")
print("   Запрос: MATCH ()-[r]->() RETURN type(r), count(*)")
