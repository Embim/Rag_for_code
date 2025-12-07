# Debug Trace для отслеживания работы AI агента

Новая функция для детального отслеживания того, какие данные AI получает из графовой БД на каждой итерации.

## Зачем это нужно?

При разработке и отладке RAG системы важно понимать:
- ✅ Достаточно ли информации в графовой БД?
- ✅ Какие данные находит AI на каждом шаге?
- ✅ Какие файлы и репозитории используются?
- ✅ Сколько сущностей найдено каждым инструментом?
- ✅ Нужно ли менять конфигурацию поиска?

## Как использовать?

### 1. Включить verbose режим в запросе

Добавьте параметр `verbose=true` в POST запрос `/api/ask`:

```bash
curl -X POST "http://localhost:8000/api/ask" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "question": "How does equity trade booking work?",
    "verbose": true,
    "max_iterations": 10
  }'
```

### 2. Получить детальную трейс-информацию

Ответ будет содержать дополнительное поле `debug` с полной информацией о каждой итерации:

```json
{
  "question": "How does equity trade booking work?",
  "answer": "...",
  "iterations_used": 5,
  "tools_used": ["semantic_search", "get_entity_details", "get_related_entities"],
  "debug": {
    "trace_summary": {
      "total_iterations": 5,
      "total_tool_calls": 8,
      "total_entities_found": 45,
      "unique_files_accessed": 12,
      "repositories_searched": ["api", "ui"],
      "tools_used": {
        "semantic_search": 3,
        "get_entity_details": 4,
        "get_related_entities": 1
      },
      "entity_types_found": {
        "Function": 20,
        "Method": 15,
        "Class": 8,
        "File": 2
      },
      "duration_seconds": 15.23
    },
    "detailed_trace": [
      {
        "iteration": 1,
        "tool": "semantic_search",
        "params": {
          "query": "equity trade booking",
          "top_k": 20
        },
        "timestamp": 1701234567.89,
        "elapsed_ms": 125.5,
        "result": {
          "success": true,
          "entities_found": 15,
          "entity_types": {
            "Function": 8,
            "Method": 5,
            "Class": 2
          },
          "files": [
            "app/backend/middle_office/booking/trade_uploader.py",
            "app/backend/trade_factory.py"
          ],
          "repositories": ["api"]
        }
      },
      {
        "iteration": 2,
        "tool": "get_entity_details",
        "params": {
          "id": "repo:api:app/backend/trade_factory.py:equity_trade_book"
        },
        "timestamp": 1701234568.25,
        "elapsed_ms": 485.2,
        "result": {
          "success": true,
          "entities_found": 1,
          "entity_types": {
            "Function": 1
          },
          "files": ["app/backend/trade_factory.py"],
          "repositories": ["api"]
        }
      }
      // ... остальные итерации
    ]
  }
}
```

## Структура trace_summary

Общая сводка по всем итерациям:

| Поле | Описание |
|------|----------|
| `total_iterations` | Сколько итераций выполнил агент |
| `total_tool_calls` | Сколько раз вызывались инструменты |
| `total_entities_found` | Общее количество найденных сущностей |
| `unique_files_accessed` | Количество уникальных файлов |
| `repositories_searched` | Список проверенных репозиториев |
| `tools_used` | Сколько раз использовался каждый инструмент |
| `entity_types_found` | Распределение сущностей по типам |
| `duration_seconds` | Общее время выполнения |

## Структура detailed_trace

Детальная информация о каждом вызове инструмента:

| Поле | Описание |
|------|----------|
| `iteration` | Номер итерации (1, 2, 3...) |
| `tool` | Имя инструмента |
| `params` | Параметры вызова инструмента |
| `timestamp` | Unix timestamp вызова |
| `elapsed_ms` | Время с начала запроса (мс) |
| `result.success` | Успешность вызова |
| `result.entities_found` | Количество найденных сущностей |
| `result.entity_types` | Типы найденных сущностей |
| `result.files` | Список файлов |
| `result.repositories` | Список репозиториев |

## Примеры использования

### Пример 1: Проверка качества индекса

```python
import requests

response = requests.post(
    "http://localhost:8000/api/ask",
    headers={"X-API-Key": "your-key"},
    json={
        "question": "How does authentication work?",
        "verbose": True
    }
)

data = response.json()
debug = data['debug']

print(f"Найдено сущностей: {debug['trace_summary']['total_entities_found']}")
print(f"Уникальных файлов: {debug['trace_summary']['unique_files_accessed']}")
print(f"Репозитории: {debug['trace_summary']['repositories_searched']}")

# Проверяем достаточно ли данных
if debug['trace_summary']['total_entities_found'] < 5:
    print("⚠️ Слишком мало найдено! Возможно нужно переиндексировать.")
```

### Пример 2: Анализ эффективности инструментов

```python
debug = data['debug']
tools_used = debug['trace_summary']['tools_used']

print("Статистика использования инструментов:")
for tool, count in tools_used.items():
    print(f"  {tool}: {count} раз(а)")

# Смотрим сколько сущностей в среднем находит каждый инструмент
for trace_entry in debug['detailed_trace']:
    tool = trace_entry['tool']
    entities = trace_entry['result'].get('entities_found', 0)
    print(f"{tool} (итерация {trace_entry['iteration']}): {entities} сущностей")
```

### Пример 3: Визуализация процесса поиска

```python
import matplotlib.pyplot as plt

# Собираем данные по итерациям
iterations = []
entities_per_iteration = []

for trace_entry in debug['detailed_trace']:
    iterations.append(trace_entry['iteration'])
    entities_per_iteration.append(
        trace_entry['result'].get('entities_found', 0)
    )

# Строим график
plt.plot(iterations, entities_per_iteration, marker='o')
plt.xlabel('Итерация')
plt.ylabel('Найдено сущностей')
plt.title('Эффективность поиска по итерациям')
plt.grid(True)
plt.show()
```

### Пример 4: Проверка покрытия репозиториев

```python
debug = data['debug']

repositories_searched = set()
files_by_repo = {}

for trace_entry in debug['detailed_trace']:
    repos = trace_entry['result'].get('repositories', [])
    files = trace_entry['result'].get('files', [])

    for repo in repos:
        repositories_searched.add(repo)
        if repo not in files_by_repo:
            files_by_repo[repo] = set()
        files_by_repo[repo].update(files)

print("\nПокрытие по репозиториям:")
for repo in repositories_searched:
    file_count = len(files_by_repo.get(repo, []))
    print(f"  {repo}: {file_count} файлов")
```

## Логирование в консоли

При включенном verbose режиме в логах появляется дополнительная информация:

```
2025-12-07 21:30:15 | INFO | ✅ semantic_search: found 15 entities across 3 files
2025-12-07 21:30:16 | INFO | ✅ get_entity_details: found 1 entities across 1 files
2025-12-07 21:30:17 | INFO | ✅ get_related_entities: found 8 entities across 5 files
2025-12-07 21:30:18 | INFO | 📊 Trace summary: {
    'total_entities_found': 24,
    'unique_files_accessed': 9,
    'repositories_searched': ['api', 'ui']
}
```

## Рекомендации по использованию

### ✅ Когда использовать verbose режим:

- **Разработка** - при настройке и тестировании системы
- **Отладка** - когда ответы кажутся неполными
- **Оптимизация** - для улучшения конфигурации поиска
- **Мониторинг** - проверка качества индекса

### ❌ Когда НЕ использовать:

- **Production** - увеличивает размер ответа и время обработки
- **Обычные запросы** - только для отладки

### 💡 Что проверять:

1. **Достаточно ли данных?**
   - `total_entities_found < 10` → возможно плохой индекс
   - `unique_files_accessed < 3` → слишком узкий поиск

2. **Правильные ли репозитории?**
   - Проверьте `repositories_searched`
   - Убедитесь что поиск идет где нужно

3. **Эффективность инструментов:**
   - Если `semantic_search` возвращает 0 результатов → проблема с embedding
   - Если `get_entity_details` часто падает → проблема с ID

4. **Распределение типов:**
   - Проверьте `entity_types_found`
   - Убедитесь что находятся нужные типы (Function, Class, etc.)

## Performance Impact

| Режим | Response Size | Processing Time |
|-------|---------------|-----------------|
| Normal | ~5-10 KB | Baseline |
| Verbose | ~50-100 KB | +5-10% |

Verbose режим добавляет ~40-90 KB к размеру ответа и незначительно увеличивает время обработки.

## Troubleshooting

### Проблема: Мало найдено сущностей

```python
if debug['trace_summary']['total_entities_found'] < 5:
    print("Возможные причины:")
    print("1. Индекс не содержит нужные данные")
    print("2. Запрос слишком специфичный")
    print("3. Проблема с embedding моделью")
```

**Решение:**
1. Переиндексируйте репозитории
2. Проверьте настройки `top_k` (увеличьте до 20)
3. Используйте более общие запросы

### Проблема: Не те репозитории

```python
expected_repos = {'api', 'ui'}
actual_repos = set(debug['trace_summary']['repositories_searched'])

if not expected_repos.issubset(actual_repos):
    missing = expected_repos - actual_repos
    print(f"Отсутствуют репозитории: {missing}")
```

**Решение:**
1. Добавьте `context.repositories` в запрос
2. Проверьте что репозитории проиндексированы

### Проблема: Только один тип сущностей

```python
entity_types = debug['trace_summary']['entity_types_found']
if len(entity_types) == 1:
    print(f"Найден только тип: {list(entity_types.keys())[0]}")
    print("Возможно нужно расширить поиск")
```

**Решение:**
1. Используйте `get_related_entities` для расширения
2. Проверьте что все типы нод проиндексированы

---

## Резюме

Debug trace позволяет:
- 🔍 Видеть что именно находит AI в БД
- 📊 Анализировать эффективность каждого шага
- ⚙️ Оптимизировать конфигурацию системы
- 🐛 Быстро находить проблемы с индексом

Используйте `verbose=true` при разработке и отладке для полного понимания работы агента!
