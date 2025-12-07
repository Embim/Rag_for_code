# Критические исправления - 2025-12-07

## 🎯 Найденные проблемы

### ❌ Проблема #1: `get_entity_details` ищет по неправильному ID

**Локация:** `src/agents/tools.py:295-304`

**Проблема:**
```python
# ТЕКУЩИЙ КОД (НЕПРАВИЛЬНО):
cypher = """
MATCH (e)
WHERE elementId(e) = $entity_id  # ❌ Ищет по внутреннему Neo4j ID
...
"""
```

**Причина:**
- `semantic_search` возвращает `id` в формате: `repo:api:app/backend/blotters.py:blotter_equity`
- `elementId(e)` - это внутренний Neo4j ID типа `"4:f8a2b1c3:42"`
- Это **разные типы** ID!

**Данные из Neo4j:**
```json
{
  "n.id": "repo:api:app/backend/blotters.py:blotter_equity",
  "n.name": "blotter_equity",
  "labels(n)": ["GraphNode", "Function"]
}
```

Правильное поле - это `n.id`, а не `elementId(e)`.

**Решение:**
```python
# ИСПРАВЛЕННЫЙ КОД:
cypher = """
MATCH (e)
WHERE e.id = $entity_id  # ✅ Ищет по полю id
...
"""
```

**Результат:**
- `get_entity_details` начнет находить сущности ✅
- 0/18 → 18/18 успешных вызовов
- Агент сможет получать детали кода

---

### ❌ Проблема #2: Граф не содержит CALLS/IMPORTS связи

**Локация:** `src/code_rag/graph/graph_builder.py:277-359`

**Найдено в Neo4j:**
```cypher
MATCH ()-[r]->()
RETURN type(r), count(*)

РЕЗУЛЬТАТ:
CONTAINS: 5993  ✅
INHERITS: 240   ✅
CALLS: 0        ❌ НЕТ!
IMPORTS: 0      ❌ НЕТ!
```

**Анализ кода:**

`_create_relationships()` создает ТОЛЬКО:
1. **CONTAINS** (файл → функция, класс → метод) ✅
2. **INHERITS** (класс → базовый класс) ✅
3. **Django relationships** (ForeignKey, ManyToMany, OneToOne)
4. **HANDLES_REQUEST** - закомментировано (`pass`)
5. **RENDERS_AT** - TODO

**НЕ СОЗДАЮТСЯ:**
- ❌ **CALLS** - функция вызывает функцию
- ❌ **IMPORTS** - файл импортирует модуль/класс
- ❌ **USES** - функция использует класс/переменную

**Последствия:**
- `get_related_entities` не может найти связанные функции (0/10 успешных)
- `get_graph_path` не может построить путь между сущностями (failed)
- Невозможно трейсить потоки выполнения (UI → Backend → DB)
- Граф знает структуру, но **не знает как работает код**

---

## 🔧 Исправление #1: get_entity_details (СРОЧНО)

### До:
```python
# src/agents/tools.py:295-314
if entity_id or id:
    # Search by element ID
    cypher = """
    MATCH (e)
    WHERE elementId(e) = $entity_id  # ❌ НЕПРАВИЛЬНО
    OPTIONAL MATCH (e)-[r]->(related)
    RETURN e, labels(e) as types,
           collect({type: type(r), target: related.name}) as relationships
    """
    results = list(self.neo4j.execute_cypher(cypher, parameters={'entity_id': identifier}))
else:
    # Search by name
    cypher = """
    MATCH (e {name: $name})
    OPTIONAL MATCH (e)-[r]->(related)
    RETURN e, labels(e) as types,
           collect({type: type(r), target: related.name}) as relationships
    LIMIT 1
    """
    results = list(self.neo4j.execute_cypher(cypher, parameters={'name': identifier}))
```

### После:
```python
# src/agents/tools.py:295-314
# Try to find by custom ID first (from semantic_search), then by name
cypher = """
MATCH (e)
WHERE e.id = $entity_id  # ✅ Ищем по полю id
OPTIONAL MATCH (e)-[r]->(related)
RETURN e, labels(e) as types,
       collect({type: type(r), target: related.name}) as relationships
LIMIT 1
"""
results = list(self.neo4j.execute_cypher(cypher, parameters={'entity_id': identifier}))

# Fallback: try searching by name if not found
if not results and identifier:
    cypher = """
    MATCH (e {name: $name})
    OPTIONAL MATCH (e)-[r]->(related)
    RETURN e, labels(e) as types,
           collect({type: type(r), target: related.name}) as relationships
    LIMIT 1
    """
    results = list(self.neo4j.execute_cypher(cypher, parameters={'name': identifier}))
```

**Изменения:**
1. Убрали проверку `if entity_id or id` - теперь всегда ищем по `e.id`
2. Добавили fallback на поиск по `name` если не нашли по ID
3. Упростили логику - один путь поиска

---

## 🔧 Исправление #2: Добавление CALLS/IMPORTS (ВАЖНО, но СЛОЖНО)

### Что нужно сделать:

Это **большая задача**, требует парсинга вызовов функций из AST.

#### Вариант 1: Минимальное решение (Quick Fix)
Добавить извлечение вызовов на уровне функций:

```python
# В python_parser.py - добавить метод:
def _extract_function_calls(self, node: ast.FunctionDef, source: str) -> List[str]:
    """Extract names of all functions called in this function."""
    calls = []

    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            if isinstance(child.func, ast.Name):
                # Simple call: foo()
                calls.append(child.func.id)
            elif isinstance(child.func, ast.Attribute):
                # Method call: obj.method()
                calls.append(ast.unparse(child.func))

    return calls
```

Затем в `graph_builder.py._create_relationships()` добавить:
```python
# Extract function calls from metadata
calls = entity.metadata.get('calls', [])
for called_func_name in calls:
    # Try to find called function in graph
    target_id = self._find_function_node(repo_info.name, called_func_name)
    if target_id:
        self.relationships.append(GraphRelationship(
            type=RelationshipType.CALLS,
            source_id=entity_id,
            target_id=target_id
        ))
```

**Проблема:** Резолвинг имен сложен (нужно знать импорты)

#### Вариант 2: Полное решение
1. Парсить все импорты на уровне файла
2. Создавать IMPORTS связи (файл → модуль)
3. Создавать symbol table для резолвинга
4. Парсить вызовы функций с резолвингом
5. Создавать CALLS связи

**Это займет много времени** (несколько часов работы).

### Рекомендация:
1. **Сначала исправить get_entity_details** (Исправление #1) - это 5 минут
2. **Затем протестировать** - агент должен начать работать лучше
3. **Потом решить** нужны ли CALLS/IMPORTS или достаточно CONTAINS

Для многих задач (понять структуру кода, найти функции) достаточно CONTAINS связей.
Для трейсинга потоков (UI → API → DB) нужны CALLS.

---

## 📊 Приоритеты

### P0 - Срочно (сейчас):
- ✅ Исправить `get_entity_details` - ищет по `e.id` вместо `elementId(e)`
- ✅ Исправить `get_related_entities` - та же проблема
- ✅ Исправить `exact_search` - возвращать правильный `id`

### P1 - Скоро (если нужен трейсинг):
- 🔄 Добавить CALLS связи (минимальный вариант)
- 🔄 Добавить IMPORTS связи
- 🔄 Улучшить резолвинг имен

### P2 - Позже:
- 📝 Полный резолвинг с symbol table
- 📝 Поддержка динамических импортов
- 📝 Анализ потока данных

---

## 🧪 Как проверить исправления

### Тест 1: get_entity_details должен работать
```cypher
# В Neo4j:
MATCH (n)
WHERE n.id = "repo:api:app/backend/blotters.py:blotter_equity"
RETURN n

# Должно вернуть 1 запись
```

### Тест 2: Агент должен получать детали
```bash
curl -X POST "http://localhost:8000/api/ask" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-key" \
  -d '{
    "question": "How does blotter_equity work?",
    "verbose": true
  }'

# В debug.detailed_trace проверить:
# - get_entity_details: entities_found > 0 ✅
```

### Тест 3: Проверить relationships
```cypher
# После исправления get_entity_details:
MATCH (n {name: "blotter_equity"})-[r]->(m)
RETURN n, type(r), m
LIMIT 10

# Должно показать CONTAINS связи (если есть)
# CALLS связей пока не будет (нужно Исправление #2)
```

---

## 📝 Резюме

| Проблема | Статус | Приоритет | Время |
|----------|--------|-----------|-------|
| get_entity_details не работает | ✅ ИСПРАВЛЕНО | P0 | 5 мин |
| get_related_entities не работает | ✅ ИСПРАВЛЕНО | P0 | 2 мин |
| exact_search возвращает неправильный ID | ✅ ИСПРАВЛЕНО | P0 | 2 мин |
| Нет CALLS связей | ❌ TODO | P1 | 2-3 часа |
| Нет IMPORTS связей | ❌ TODO | P1 | 1-2 часа |

## ✅ Что исправлено (2025-12-07 23:15)

### Изменения в `src/agents/tools.py`:

1. **GetEntityDetailsTool** (строки 268-342):
   - ✅ Заменил `WHERE elementId(e) = $entity_id` → `WHERE e.id = $entity_id`
   - ✅ Добавил fallback на поиск по `name` если не нашли по ID
   - ✅ Унифицировал формат ответа: `entities_found` + `entities[]`
   - ✅ Использует `file_path` и `start_line` вместо `file` и `line`

2. **GetRelatedEntitiesTool** (строки 368-453):
   - ✅ Заменил все `WHERE elementId(e)` → `WHERE e.id`
   - ✅ Унифицировал формат: `entities_found` + `entities[]`
   - ✅ Возвращает правильный `node.get('id')` вместо `element_id`

3. **ExactSearchTool** (строки 221-241):
   - ✅ Возвращает `node.get('id')` вместо `element_id`
   - ✅ Унифицировал поля: `file_path`, `start_line`

### Ожидаемый результат:

**До исправления:**
```
get_entity_details: 0/18 успешных (0%)  ❌
get_related_entities: 0/10 успешных (0%) ❌
```

**После исправления:**
```
get_entity_details: 18/18 успешных (100%) ✅
get_related_entities: X/10 успешных (зависит от CONTAINS связей) ⚠️
```

**Примечание:** `get_related_entities` теперь будет работать, но найдет только CONTAINS и INHERITS связи, так как CALLS/IMPORTS отсутствуют в графе.

**Следующий шаг:** Протестировать агента с исправлениями.
