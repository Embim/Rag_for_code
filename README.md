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
- **ripgrep** — опционально (ускоряет `GrepEnrichTask` ~10×); при отсутствии работает Python fallback. См. [GrepEnrichTask и ripgrep](#grepenrichtask-и-ripgrep).

## Быстрый старт

```bash
# 1. Зависимости
pip install -r requirements.txt

# 2. Переменные окружения
cp .env.example .env
# Заполнить: OPENROUTER_API_KEY, NEO4J_PASSWORD
# Опционально: LANGFUSE_PUBLIC_KEY + LANGFUSE_SECRET_KEY (для трассировки)

# 3. Запуск БД
docker-compose up -d

# 4. Запуск API
uvicorn src.interfaces.api.main:app --host 0.0.0.0 --port 8000
```

## Структура

```
src/
├── infra/                  # Логгер, конфиг, runtime cache
│   ├── logger.py
│   ├── cache.py
│   └── config/             # AgentConfig, SearchConfig, Neo4jConfig, ...
├── core/                   # Build-time-нейтральный домен
│   ├── parsers/            # Python/Django/FastAPI/React + js_parser.js
│   ├── graph/              # Модели нод, Neo4jClient, GraphBuilder, APILinker, code_loader
│   ├── repo_loader.py
│   └── project_detector.py
├── indexing/               # Build-time pipelines (executor/task/service)
│   ├── auto_reindex.py
│   └── pipeline/           # full_index / graph_only / api_links_only
├── search/                 # Query-time
│   ├── retrieval/          # CodeRetriever + scope_detector
│   ├── ranking/            # cross_encoder, RRF
│   ├── preprocessing/      # QueryExpander, QueryReformulator
│   ├── pipeline/           # RAG executor + RagControllerTask + Cypher chain
│   └── agents/             # CodeExplorer, Business, VisualGuide, Orchestrator
├── interfaces/             # Точки входа
│   ├── api/                # FastAPI + routes
│   └── telegram_bot/       # Telegram bot
└── visualization/          # screenshot_service
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
python -m src.interfaces.telegram_bot.bot

# Команды
/ask <вопрос>        # Вопрос по коду
/analyze <traceback> # Анализ ошибки
/guide <действие>    # Инструкция для пользователя
/repos               # Список репозиториев
```

## Запуск RAG pipeline напрямую

RAG‑пайплайн работает внутри API‑сервера — отдельный сервер не нужен.
Можно вызывать и из кода:

```python
from src.search.pipeline.pipelines.rag_search import run as run_rag

result = run_rag("Как работает checkout?", max_iterations=3)
print(result["answer"])
print(result["sources"])
print(f"Iterations: {result['iterations']}, Quality: {result['quality_score']:.2f}")
```

## Трассировка через Langfuse

Все этапы RAG‑pipeline'а оборачиваются в Langfuse spans автоматически —
без правок в коде задач/сервисов. Видно: какой запрос пришёл, какой
контекст собрал retriever, какая оценка качества, что генерировал
LLM, какие были failover'ы между primary и fallback моделями.

В `.env`:

```
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com   # или self-hosted
```

Если ключи не заданы — трассировка автоматически отключается (silent
no‑op, без шума в логах). Реализация — [`LangfuseHooks`](src/search/pipeline/hooks/langfuse_hooks.py).

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

| Стратегия | Что делает |
|-----------|----------|
| `semantic_only` | Только Weaviate (vector + BM25 через `hybrid_alpha`). Default. |
| `ui_to_database` | Graph expansion: Component → ApiCall → Endpoint → Function → Model. |
| `database_to_ui` | Обратный путь: Model → Function → Endpoint → ApiCall → Component. |

Cypher‑паттерны выполняются через рёбра `MAKES_CALL | CALLS_ENDPOINT |
HANDLES_REQUEST | CALLS | USES_MODEL` (длина пути 1..6). Стратегия
определяется автоматически в `StrategyService.detect()` либо передаётся
явно через `?strategy=`.

Дополнительно в `RagControllerTask` встроен **NL→Cypher chain**: LLM сам
генерирует cypher по NL-вопросу пользователя, исполняется как чистое
READ через `Neo4jClient.execute_cypher`, результат идёт третьим блоком
контекста рядом с `primary` и `graph`. Это даёт точные ответы на
структурные вопросы ("сколько endpoint'ов в репо", "какие компоненты
дёргают /api/users/", "что наследуется от APIView").

**Устаревшие алиасы** (`bm25`, `hybrid`, `vector`, `multi_hop`,
`impact_analysis`, `pattern_search`) сохранены в API маппинге и
автоматически деградируют до `semantic_only`. Старые клиенты не
сломаются.

## GrepEnrichTask и ripgrep

В RAG-loop'е между `CollectContextTask` (Weaviate) и `CypherEnrichTask`
(NL→Cypher) встроен `GrepEnrichTask` — точный текстовый поиск по локальным
репозиториям. Он извлекает из запроса идентификаторы (snake_case,
camelCase, CONST_CASE — мин. 4 символа), грепает их по `RAG_REPOS_DIR` и
добавляет ±10 строк контекста как chunks `source='grep'`. Особенно
полезно для вопросов *«как считается net_pos»*, *«где определяется
OrderService.createOrder»* — vector search на таких токенах часто
промахивается, а grep гарантирован.

**Реализация кросс-платформенная** (Windows / macOS / Linux). Резолвинг
исполняемого `rg`:

1. env `RAG_RG_PATH=/полный/путь/к/rg` — явный override номер 1.
2. `shutil.which('rg')` — стандартный `PATH`.
3. Bundled-локации (VSCode и Cursor встраивают свой `rg`).
4. Если ничего не нашли — Python-fallback (медленнее в ~10×, но
   работает на любой ОС без зависимостей).

### Что нужно от пользователя на каждой ОС

| ОС | Что сделать |
|----|-------------|
| **Windows** | Ничего: bundled-`rg.exe` от Cursor/VSCode автоматически найдётся. Альтернативно — `scoop install ripgrep` / `cargo install ripgrep` или `RAG_RG_PATH=C:\путь\к\rg.exe`. |
| **Linux** | `apt install ripgrep` (Debian/Ubuntu) / `dnf install ripgrep` (Fedora) / `pacman -S ripgrep` (Arch). Если установлен VSCode/Cursor — bundled-rg тоже подхватится. Иначе работает Python-fallback (~10× медленнее). |
| **macOS** | `brew install ripgrep`. Опционально — bundled из VSCode/Cursor. |

### Вывод из строя / тонкая настройка

- Найти rg вручную: `where rg` (Win) / `which rg` (Linux/Mac). Скопировать
  путь и положить в `.env`: `RAG_RG_PATH=...`.
- Лимиты в `GrepEnrichTask` (см. `src/search/pipeline/tasks/grep_enrich.py`):
  `MAX_TOKENS=5` (сколько идентификаторов извлекаем), `MAX_HITS_PER_TOKEN=20`,
  `MAX_COUNT_PER_FILE=3`, `CONTEXT_LINES=10`. Round-robin
  диверсификация по файлам — лимит распределяется по разным источникам,
  не съедается одним «болтливым» файлом.

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
