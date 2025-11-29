# Code RAG

Система поиска и анализа кода на основе RAG (Retrieval-Augmented Generation) с поддержкой Python, Django, FastAPI и React.

## Возможности

- Индексация нескольких репозиториев (Git URL или локальный путь)
- Граф знаний кода (Neo4j) + векторный поиск (Weaviate)
- Парсинг Python/Django/FastAPI/React с извлечением сущностей
- LLM-агент для ответов на вопросы по коду
- Анализ traceback ошибок
- REST API + Telegram бот
- Автоматическая переиндексация при изменениях

## Требования

- Python 3.10+
- Docker (Neo4j, Weaviate)
- OpenRouter API ключ

## Быстрый старт

```bash
# 1. Зависимости
pip install -r requirements.txt

# 2. Переменные окружения
cp .env.example .env
# Заполнить: OPENROUTER_API_KEY, NEO4J_PASSWORD

# 3. Запуск БД
docker-compose up -d

# 4. Запуск API
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

## Структура

```
src/
├── api/                 # REST API (FastAPI)
├── agents/              # LLM агенты
├── code_rag/
│   ├── graph/           # Neo4j + Weaviate
│   ├── parsers/         # Python/Django/FastAPI/React
│   └── retrieval/       # Поиск и ранжирование
├── config/              # Конфигурации
├── ranking/             # Reranking, RRF
├── query/               # Query expansion/reformulation
├── telegram_bot/        # Telegram бот
└── visualization/       # Скриншоты UI
```

## API

### Добавление репозитория

```bash
# Git URL
curl -X POST "http://localhost:8000/api/repos" \
  -H "Content-Type: application/json" \
  -d '{"source": "https://github.com/org/repo.git", "name": "repo", "type": "backend"}'

# Локальный путь
curl -X POST "http://localhost:8000/api/repos" \
  -H "Content-Type: application/json" \
  -d '{"source": "C:/Projects/myapp", "name": "myapp", "type": "frontend"}'
```

### Поиск

```bash
curl -X POST "http://localhost:8000/api/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "как работает авторизация", "strategy": "hybrid", "top_k": 10}'
```

### Вопрос агенту

```bash
curl -X POST "http://localhost:8000/api/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "объясни связь между UserModel и AuthService"}'
```

## Telegram бот

```bash
# Запуск
python -m src.telegram_bot.bot

# Команды
/ask <вопрос>        # Вопрос по коду
/analyze <traceback> # Анализ ошибки
/guide <действие>    # Инструкция для пользователя
/repos               # Список репозиториев
```

## Конфигурация

### .env

```bash
# LLM
OPENROUTER_API_KEY=sk-or-v1-xxx
LLM_API_MODEL=anthropic/claude-3-haiku

# Neo4j
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password

# Weaviate
WEAVIATE_URL=http://localhost:8080

# Telegram
TELEGRAM_BOT_TOKEN=xxx
```

## Стратегии поиска

| Стратегия | Описание |
|-----------|----------|
| `semantic` | Векторный поиск по embeddings |
| `bm25` | Лексический поиск |
| `hybrid` | Комбинация semantic + bm25 + RRF |
| `multi_hop` | Граф-поиск по связям |

## Автопереиндексация

```bash
# Webhook для CI/CD
curl -X POST "http://localhost:8000/api/reindex/webhook" \
  -d '{"repository": "org/repo", "action": "push"}'

# Запуск планировщика (каждые 24 часа)
curl -X POST "http://localhost:8000/api/reindex/scheduler/start?interval_hours=24"
```

## Документация

- [docs/QUICKSTART.md](docs/QUICKSTART.md) - Полное руководство
- [docs/API.md](docs/API.md) - API и аутентификация

## Лицензия

MIT
