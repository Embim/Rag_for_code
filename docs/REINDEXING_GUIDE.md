# Руководство по переиндексации графа знаний

## 🎯 Когда нужно переиндексировать?

Переиндексация необходима когда:
- ✅ Изменился парсер (добавлены новые поля, извлечение связей)
- ✅ Добавлены новые типы связей (CALLS, IMPORTS)
- ✅ Изменилась структура графа
- ✅ Данные в Neo4j или Weaviate повреждены
- ✅ Нужно начать с чистого листа

---

## 🗑️ Способы очистки

### Вариант 1: Полная очистка (РЕКОМЕНДУЕТСЯ)

**Через скрипт:**
```bash
python scripts/full_reindex.py
```

Скрипт:
1. ❓ Спросит подтверждение
2. 🗑️ Очистит Neo4j (батчами по 10,000 нод)
3. 🗑️ Очистит Weaviate
4. 📚 Переиндексирует все репозитории
5. 📊 Покажет статистику

---

### Вариант 2: Очистка только Neo4j

**Через Python:**
```python
from src.core.graph.neo4j_client import Neo4jClient

client = Neo4jClient(
    uri="bolt://localhost:7687",
    user="neo4j",
    password="password"
)

# Очистить все (батчами)
client.clear_database(batch_size=10000)
client.close()
```

**Через Neo4j Browser** (http://localhost:7474):
```cypher
// Удалить все ноды и связи (батчами)
MATCH (n)
CALL {
  WITH n
  DETACH DELETE n
} IN TRANSACTIONS OF 10000 ROWS
```

**Или:**
```cypher
// Удалить только связи (оставить ноды)
MATCH ()-[r]->()
CALL {
  WITH r
  DELETE r
} IN TRANSACTIONS OF 10000 ROWS
```

---

### Вариант 3: Очистка конкретного репозитория

```cypher
// Удалить только api репозиторий
MATCH (n)
WHERE n.id STARTS WITH 'repo:api'
CALL {
  WITH n
  DETACH DELETE n
} IN TRANSACTIONS OF 5000 ROWS
```

---

## 📚 Переиндексация

### Через скрипт (простой способ)

```bash
# Полная переиндексация (с очисткой)
python scripts/full_reindex.py

# Только индексация (без очистки)
python scripts/reindex_weaviate.py  # Если такой скрипт есть
```

---

### Через Python код

```python
from src.core.graph.build_and_index import build_and_index
from pathlib import Path

# Путь к репозиториям
repos_dir = Path("data/repos")

# Запустить индексацию
stats = build_and_index(
    repos_dir=str(repos_dir),
    neo4j_uri="bolt://localhost:7687",
    neo4j_user="neo4j",
    neo4j_password="password",
    weaviate_url="http://localhost:8080",
)

print(f"Создано нод: {stats['nodes_created']}")
print(f"Создано связей: {stats['relationships_created']}")
```

---

## ⚙️ Оптимизация батчей

### Текущие настройки (после оптимизации)

| Операция | Chunk Size | Описание |
|----------|------------|----------|
| **Удаление** | 10,000 | Батчами для больших графов |
| **Создание нод** | 1,000 | По 1000 нод за раз |
| **Создание связей** | 1,000 | По 1000 связей за раз |

### Можно ли увеличить?

**Да!** Зависит от RAM и количества данных:

| Размер графа | Рекомендуемый chunk_size |
|--------------|--------------------------|
| < 10K нод | 500-1000 ✅ |
| 10K-100K нод | 1000-2000 ✅ |
| > 100K нод | 2000-5000 ⚠️ |
| > 1M нод | 5000-10000 ⚠️ |

**Как изменить:**
```python
# В neo4j_client.py
nodes_created = client.create_nodes_batch(nodes, chunk_size=2000)  # Было 1000
rels_created = client.create_relationships_batch(rels, chunk_size=2000)
```

**Компромисс:**
- 🟢 **Больше chunk_size** = быстрее, но больше памяти
- 🟢 **Меньше chunk_size** = медленнее, но стабильнее

---

## 🧪 Проверка результатов

### 1. Проверить количество нод
```cypher
MATCH (n)
RETURN labels(n) as type, count(*) as count
ORDER BY count DESC
```

**Ожидаемый результат:**
```
File: 500-1000
Function: 2000-5000
Class: 200-500
Method: 3000-8000
...
```

### 2. Проверить количество связей
```cypher
MATCH ()-[r]->()
RETURN type(r) as relationship_type, count(*) as count
ORDER BY count DESC
```

**До исправлений:**
```
CONTAINS: 5993
INHERITS: 240
CALLS: 0        ❌
IMPORTS: 0      ❌
```

**После исправлений (ОЖИДАЕТСЯ):**
```
CONTAINS: 5993
CALLS: 500-1000+    ✅ НОВОЕ!
INHERITS: 240
IMPORTS: 100-300+   ✅ НОВОЕ!
```

### 3. Проверить конкретную функцию
```cypher
// Найти функцию
MATCH (f {name: "blotter_equity"})
RETURN f

// Что она вызывает
MATCH (f {name: "blotter_equity"})-[r:CALLS]->(target)
RETURN f.name, type(r), target.name
LIMIT 10
```

### 4. Проверить Weaviate
```bash
curl http://localhost:8080/v1/schema
```

---

## 📊 Ожидаемое время индексации

| Размер кодовой базы | Количество файлов | Время |
|----------------------|-------------------|--------|
| Маленькая | < 100 | 1-2 мин |
| Средняя | 100-500 | 5-10 мин |
| Большая | 500-1000 | 15-30 мин |
| Очень большая | > 1000 | 30-60 мин |

**Факторы:**
- CPU (парсинг AST)
- RAM (батчи)
- GPU (embeddings)
- Диск (чтение файлов)

---

## ⚠️ Troubleshooting

### Проблема: "Out of memory"

**Решение:**
```python
# Уменьшить chunk_size
client.create_nodes_batch(nodes, chunk_size=500)  # Было 1000
```

### Проблема: "Transaction timeout"

**Решение:**
```cypher
// Увеличить batch_size для удаления
MATCH (n)
CALL {
  WITH n
  DETACH DELETE n
} IN TRANSACTIONS OF 5000 ROWS  -- Было 10000
```

### Проблема: "Connection lost"

**Решение:**
1. Проверить что Neo4j запущен: `http://localhost:7474`
2. Проверить что Weaviate запущен: `http://localhost:8080`
3. Перезапустить сервисы:
   ```bash
   docker-compose restart neo4j weaviate
   ```

### Проблема: "No CALLS relationships found"

**Решение:**
1. Убедиться что переиндексация завершилась
2. Проверить что `python_parser.py` извлекает вызовы:
   ```cypher
   // Проверить есть ли функции с metadata
   MATCH (f:Function)
   WHERE f.code CONTAINS "def "
   RETURN f.name, f.code
   LIMIT 5
   ```

---

## 📝 Чеклист после переиндексации

- [ ] Neo4j показывает ноды
- [ ] Neo4j показывает CALLS связи
- [ ] Neo4j показывает IMPORTS связи
- [ ] Weaviate содержит векторы
- [ ] API `/api/search` работает
- [ ] API `/api/ask` работает
- [ ] `get_entity_details` находит сущности
- [ ] `get_related_entities` находит связи
- [ ] Агент может трейсить потоки

---

## 🚀 Быстрый старт

```bash
# 1. Очистить и переиндексировать все
python scripts/full_reindex.py

# 2. Проверить результаты в Neo4j
# Открыть http://localhost:7474 и выполнить:
# MATCH ()-[r]->() RETURN type(r), count(*)

# 3. Протестировать API
curl -X POST "http://localhost:8000/api/ask" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-key" \
  -d '{
    "question": "What functions does blotter_equity call?",
    "verbose": true
  }'
```

---

## 💡 Советы

1. **Делайте backup** перед очисткой:
   ```bash
   # Neo4j dump
   neo4j-admin dump --database=neo4j --to=backup.dump
   ```

2. **Индексируйте по частям** для больших проектов:
   ```python
   # Сначала один репозиторий
   build_and_index(repos_dir="data/repos/api", ...)

   # Потом второй
   build_and_index(repos_dir="data/repos/ui", ...)
   ```

3. **Мониторьте прогресс** через логи:
   ```bash
   tail -f outputs/pipeline.log
   ```

4. **Используйте профилирование** если медленно:
   ```python
   import cProfile
   cProfile.run('build_and_index(...)', 'stats.prof')
   ```

---

## 📄 Связанные документы

- `docs/CALLS_IMPORTS_IMPLEMENTATION.md` - Реализация CALLS/IMPORTS
- `docs/CRITICAL_FIXES_2025-12-07.md` - Критические исправления
- `docs/LOG_ANALYSIS_2025-12-07.md` - Анализ логов

---

**Дата создания:** 2025-12-07
**Автор:** Claude Code
