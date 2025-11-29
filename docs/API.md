# Code RAG API

## Запуск

```bash
# Запуск БД
docker-compose up -d

# Переменные
export NEO4J_PASSWORD="password"
export OPENROUTER_API_KEY="sk-or-..."

# Запуск API
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

**Документация:** http://localhost:8000/docs

---

## Аутентификация

API использует ключи в заголовке `X-API-Key`.

### Первый запуск

При старте создаётся admin-ключ (сохраните его!):

```
🔑 INITIAL ADMIN API KEY CREATED
   API Key: sk-rag-abc123xyz789...
```

### Создание ключа

```bash
curl -X POST http://localhost:8000/api/keys \
  -H "X-API-Key: <admin_key>" \
  -H "Content-Type: application/json" \
  -d '{"name": "Bot Key", "role": "user", "expires_in_days": 90}'
```

### Роли

| Роль | Права |
|------|-------|
| `admin` | Всё + управление ключами |
| `user` | Поиск, вопросы, добавление репо |
| `readonly` | Только чтение |

---

## Эндпоинты

### Поиск

```bash
POST /api/search

curl -X POST http://localhost:8000/api/search \
  -H "X-API-Key: <key>" \
  -H "Content-Type: application/json" \
  -d '{"query": "authentication", "strategy": "hybrid", "limit": 10}'
```

**Стратегии:** `semantic`, `hybrid`, `bm25`, `ui_to_database`, `database_to_ui`

### Вопрос агенту

```bash
POST /api/ask

curl -X POST http://localhost:8000/api/ask \
  -H "X-API-Key: <key>" \
  -H "Content-Type: application/json" \
  -d '{"question": "как работает checkout"}'
```

### Репозитории

```bash
# Список
GET /api/repos

# Добавить
POST /api/repos
{"source": "https://github.com/org/repo.git", "name": "repo", "type": "backend"}

# Статус индексации
GET /api/repos/{name}/status

# Переиндексация
POST /api/repos/{name}/reindex

# Удалить (только admin)
DELETE /api/repos/{name}
```

### Визуализация

```bash
POST /api/visualize
{"diagram_type": "sequence", "entities": ["...", "..."], "title": "Flow"}
```

**Типы:** `sequence`, `component`, `er`, `flow`

### Служебные

```bash
GET /api/health    # Статус системы
GET /api/stats     # Статистика
GET /api/keys/me   # Информация о текущем ключе
```

---

## Python клиент

```python
import requests

class CodeRAGClient:
    def __init__(self, url="http://localhost:8000", api_key=None):
        self.url = url
        self.headers = {"X-API-Key": api_key} if api_key else {}

    def search(self, query, limit=10):
        r = requests.post(f"{self.url}/api/search", 
            headers=self.headers, json={"query": query, "limit": limit})
        return r.json()

    def ask(self, question):
        r = requests.post(f"{self.url}/api/ask",
            headers=self.headers, json={"question": question})
        return r.json()

# Использование
client = CodeRAGClient(api_key="sk-rag-...")
results = client.search("authentication")
answer = client.ask("как работает авторизация")
```

---

## Управление ключами (admin)

```bash
# Список ключей
GET /api/keys

# Создать ключ
POST /api/keys
{"name": "App Key", "role": "user", "expires_in_days": 90}

# Отозвать ключ
POST /api/keys/{key_id}/revoke

# Удалить ключ
DELETE /api/keys/{key_id}
```

---

## Ошибки

| Код | Причина | Решение |
|-----|---------|---------|
| 401 | Нет ключа / невалидный | Добавить `X-API-Key` |
| 403 | Недостаточно прав | Использовать ключ с нужной ролью |
| 404 | Не найдено | Проверить путь/параметры |
| 500 | Ошибка сервера | Проверить логи |

---

## Хранение ключей

Файл: `data/api_keys.json`

- Хранятся только SHA-256 хэши
- Ключ показывается только при создании
- Рекомендуется `chmod 600 data/api_keys.json`

