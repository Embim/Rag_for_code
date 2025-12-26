# Code RAG - Быстрый старт

## Требования

- Python 3.10+
- Docker
- OpenRouter API ключ (бесплатно: https://openrouter.ai/keys)

---

## 1. Установка

```bash
git clone <repo_url> rag-for-code
cd rag-for-code

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

---

## 2. Конфигурация

Создайте `.env`:

```bash
NEO4J_PASSWORD=your_password
OPENROUTER_API_KEY=sk-or-v1-...
TELEGRAM_BOT_TOKEN=123456:ABC...  # опционально

# LangSmith для мониторинга LangGraph (опционально, но рекомендуется)
# Получите бесплатный ключ: https://smith.langchain.com/
LANGSMITH_API_KEY=lsv2_pt_...
```

---

## 3. Запуск инфраструктуры

```bash
docker-compose up -d

# Проверка
docker-compose ps
# Neo4j: http://localhost:7474
# Weaviate: http://localhost:8080/v1/meta
```

---

## 4. Запуск API

```bash
source .venv/bin/activate
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

**Документация:** http://localhost:8000/docs

При первом запуске будет создан admin API-ключ — **сохраните его!**

---

## 5. Добавление репозитория
python -m src.code_rag.graph.build_and_index G:\ui.bo  --clear

python -m src.code_rag.graph.build_and_index G:\api.bo 
```bash
# GitHub URL
curl -X POST "http://localhost:8000/api/repos" \
  -H "X-API-Key: <admin_key>" \
  -H "Content-Type: application/json" \
  -d '{"source": "https://github.com/org/repo.git", "name": "repo", "type": "backend"}'

# Локальный путь
curl -X POST "http://localhost:8000/api/repos" \
  -H "X-API-Key: <admin_key>" \
  -H "Content-Type: application/json" \
  -d '{"source": "C:/Projects/myapp", "name": "myapp", "type": "frontend"}'

# Проверить статус
curl "http://localhost:8000/api/repos/<name>/status" -H "X-API-Key: <key>"
```

---

## 6. Использование

### Поиск

```bash
curl -X POST "http://localhost:8000/api/search" \
  -H "X-API-Key: <key>" \
  -H "Content-Type: application/json" \
  -d '{"query": "authentication", "strategy": "hybrid", "limit": 10}'
```

**Стратегии:** `semantic`, `hybrid`, `bm25`, `ui_to_database`, `database_to_ui`

### Вопрос агенту

```bash
curl -X POST "http://localhost:8000/api/ask" \
  -H "X-API-Key: <key>" \
  -H "Content-Type: application/json" \
  -d '{"question": "как работает авторизация"}'
```

---

## 7. Telegram бот

```bash
export TELEGRAM_BOT_TOKEN=your_token
python -m src.telegram_bot.bot
```

**Команды:**
- `/ask <вопрос>` — вопрос по коду
- `/analyze <traceback>` — анализ ошибки
- `/repos` — список репозиториев

---

## 8. LangGraph Server + Langfuse

Агентный RAG с визуализацией и мониторингом.

### Архитектура

```
┌─────────────────────────────────────────────────────────┐
│  LangGraph Server (localhost:2024)                      │
│                                                         │
│  context_collector → quality_checker → [decision]       │
│         ▲                                  │            │
│         │              ┌───────────────────┤            │
│         │              │                   │            │
│         │         score < 0.6         score >= 0.6      │
│         │              │                   │            │
│         └── query_rewriter            answer_generator  │
│                                            │            │
│                                           END           │
└─────────────────────────────────────────────────────────┘
                         │ traces
                         ▼
┌─────────────────────────────────────────────────────────┐
│  Langfuse (cloud.langfuse.com)                          │
│  📊 Метрики  │  🔍 Трейсы  │  💰 Стоимость токенов     │
└─────────────────────────────────────────────────────────┘
```

### Настройка Langfuse

1. Зарегистрируйтесь: https://cloud.langfuse.com
2. Создайте проект и получите ключи
3. Добавьте в `.env`:

```bash
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com
```

### Запуск LangGraph Server

```bash
# Установка
pip install langgraph-cli langfuse

# Запуск сервера
cd src/langgraph_server
langgraph dev
```

Сервер: http://127.0.0.1:2024

### Использование

**Python SDK:**
```python
from langgraph_sdk import get_client

client = get_client(url="http://127.0.0.1:2024")

# Создать запуск
result = await client.runs.create(
    assistant_id="rag",
    input={"query": "Как работает аутентификация?"}
)
print(result)
```

**Напрямую из кода:**
```python
from src.langgraph_server import run_rag

result = run_rag("Как работает checkout?")
print(result["answer"])
print(result["sources"])
print(f"Итераций: {result['iterations']}, Качество: {result['quality_score']}")
```

**cURL:**
```bash
curl -X POST http://127.0.0.1:2024/runs \
  -H "Content-Type: application/json" \
  -d '{"assistant_id": "rag", "input": {"query": "authentication flow"}}'
```

### Мониторинг в Langfuse

После запросов откройте https://cloud.langfuse.com:

- **Traces** — полная цепочка выполнения
- **Generations** — каждый вызов LLM
- **Metrics** — латентность, токены, стоимость
- **Scores** — качество контекста (quality_score)

---

## 9. Troubleshooting

### Neo4j не запускается

```bash
docker-compose logs neo4j
docker-compose restart neo4j
```

### Агенты не работают

Проверьте `OPENROUTER_API_KEY`:
```bash
echo $OPENROUTER_API_KEY
```

### Медленный поиск

Используйте `strategy: "semantic"` вместо `"hybrid"` или уменьшите `limit`.

---

## Полезные команды

```bash
docker-compose ps          # статус контейнеров
docker-compose logs -f     # логи
docker-compose down -v     # удалить всё (включая данные!)

curl http://localhost:8000/api/health  # проверка API
```

---

**Документация API:** http://localhost:8000/docs

