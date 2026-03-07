# Code RAG -- Быстрый старт

## Требования

Для работы проекта нужны Python 3.10 или выше, Docker и API-ключ OpenRouter (получить бесплатно можно на https://openrouter.ai/keys).


## Установка

Клонируйте репозиторий, создайте виртуальное окружение и установите зависимости:

```bash
git clone <repo_url> rag-for-code
cd rag-for-code

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```


## Конфигурация

Создайте файл `.env` в корне проекта со следующими переменными:

```
NEO4J_PASSWORD=your_password
OPENROUTER_API_KEY=sk-or-v1-...
TELEGRAM_BOT_TOKEN=123456:ABC...
```

Telegram-токен указывать не обязательно -- он нужен только если вы планируете использовать Telegram-бота.

Для трейсинга через Langfuse (рекомендуется) добавьте:

```
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com   # или http://localhost:3000 для self-hosted
```

Получить ключи: https://cloud.langfuse.com (бесплатный тариф есть)


## Запуск инфраструктуры

Поднимите Neo4j и Weaviate через Docker:

```bash
docker-compose up -d
docker-compose ps
```

После запуска Neo4j будет доступен на http://localhost:7474, Weaviate -- на http://localhost:8080/v1/meta.


## Запуск API

```bash
source .venv/bin/activate
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

Swagger-документация будет доступна по адресу http://localhost:8000/docs. При первом запуске сервер создаст admin API-ключ -- сохраните его, он понадобится для всех последующих запросов.


## Индексация репозитория

Есть два способа добавить репозиторий для анализа.

Через CLI напрямую (рекомендуется для локальных проектов):

```bash
python -m src.code_rag.graph.build_and_index /path/to/repo --clear
```

Флаг `--clear` очищает предыдущие данные перед индексацией. Без него новые данные добавятся к существующим.

Через API:

```bash
curl -X POST "http://localhost:8000/api/repos" \
  -H "X-API-Key: <admin_key>" \
  -H "Content-Type: application/json" \
  -d '{"source": "https://github.com/org/repo.git", "name": "repo", "type": "backend"}'
```

Вместо GitHub URL можно указать локальный путь. Поле type принимает значения backend, frontend и другие -- это метка для удобства, на логику не влияет.

Проверить статус индексации:

```bash
curl "http://localhost:8000/api/repos/<name>/status" -H "X-API-Key: <key>"
```


## Использование

Поиск по коду:

```bash
curl -X POST "http://localhost:8000/api/search" \
  -H "X-API-Key: <key>" \
  -H "Content-Type: application/json" \
  -d '{"query": "authentication", "strategy": "hybrid", "limit": 10}'
```

Доступные стратегии поиска: semantic, hybrid, bm25, ui_to_database, database_to_ui.

RAG-пайплайн с итеративным улучшением (рекомендуется):

```bash
curl -X POST "http://localhost:8000/api/ask/langgraph" \
  -H "X-API-Key: <key>" \
  -H "Content-Type: application/json" \
  -d '{"question": "как работает авторизация", "max_iterations": 3}'
```

Вопрос агенту (глубокий анализ с несколькими инструментами):

```bash
curl -X POST "http://localhost:8000/api/ask" \
  -H "X-API-Key: <key>" \
  -H "Content-Type: application/json" \
  -d '{"question": "как работает авторизация"}'
```


## Telegram-бот

Если в `.env` указан TELEGRAM_BOT_TOKEN, можно запустить бота:

```bash
python -m src.telegram_bot.bot
```

Команды бота: `/ask <вопрос>` -- задать вопрос по коду, `/analyze <traceback>` -- проанализировать ошибку, `/repos` -- посмотреть список проиндексированных репозиториев.


## RAG Pipeline

RAG-пайплайн работает прямо внутри API-сервера -- отдельный сервер не нужен. Логика: собирает контекст из базы, оценивает качество с помощью LLM, и если оценка ниже порога 0.6, переписывает запрос и повторяет поиск. Когда качество достаточное, генерирует финальный ответ.

Использование напрямую из кода:

```python
from src.langgraph_server import run_rag

result = run_rag("Как работает checkout?", max_iterations=3)
print(result["answer"])
print(result["sources"])
print(f"Iterations: {result['iterations']}, Quality: {result['quality_score']:.2f}")
```


## Трейсинг через Langfuse

Langfuse позволяет видеть в дашборде каждый RAG-запрос: что нашёл поиск, что передали в LLM и что получили в ответ. Для каждого запроса через `/ask/langgraph` в Langfuse появляется трейс с вложенными спанами:

```
Trace: rag_pipeline  (input: query)
├── Span: context_collector  — что нашёл векторный поиск
├── Generation: quality_check  — оценка качества контекста
├── Generation: query_rewrite  — (только если нужен rewrite)
└── Generation: answer_generator  — финальный ответ
```

**Вариант 1 — облако** (быстрый старт, бесплатный тариф):

1. Зарегистрируйтесь на https://cloud.langfuse.com
2. Создайте проект, скопируйте ключи
3. Добавьте в `.env`:

```
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com
```

**Вариант 2 — self-hosted** (данные остаются локально):

```bash
# Добавьте сервисы в docker-compose.yml или запустите отдельно:
docker run --name langfuse \
  -e DATABASE_URL=postgresql://... \
  -e NEXTAUTH_SECRET=<random> \
  -e SALT=<random> \
  -p 3000:3000 \
  langfuse/langfuse:latest
```

Либо используйте официальный docker-compose из репозитория Langfuse:

```bash
git clone https://github.com/langfuse/langfuse.git
cd langfuse
docker-compose up -d
```

Дашборд откроется на http://localhost:3000. Добавьте в `.env`:

```
LANGFUSE_HOST=http://localhost:3000
```

После настройки ключей все запросы через `/ask/langgraph` автоматически появляются в Langfuse Dashboard.


## Решение проблем

Если Neo4j не запускается, проверьте логи контейнера командой `docker-compose logs neo4j` и попробуйте перезапустить: `docker-compose restart neo4j`.

Если агенты возвращают ошибки, убедитесь что переменная OPENROUTER_API_KEY задана и ключ валидный.

Если поиск работает медленно, попробуйте стратегию semantic вместо hybrid или уменьшите параметр limit.


## Полезные команды

```bash
docker-compose ps            # статус контейнеров
docker-compose logs -f       # логи в реальном времени
docker-compose down -v       # остановить и удалить все данные
curl http://localhost:8000/api/health   # проверка работоспособности API
```
