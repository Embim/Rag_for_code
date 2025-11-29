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

python3 -m venv .venv
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

## 8. Troubleshooting

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

