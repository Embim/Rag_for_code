# 🔧 Настройка моделей через .env

## Обзор

Все модели в проекте теперь настраиваются через переменные окружения в `.env` файле. Это позволяет легко менять модели без изменения кода.

---

## 🎯 Доступные переменные

### 1. **EMBEDDING_MODEL** - Модель для векторизации

```bash
EMBEDDING_MODEL=BAAI/bge-m3
```

**Используется для:**
- Генерация векторных представлений кода
- Индексация в Weaviate
- Семантический поиск

**Рекомендуемые модели:**
- `BAAI/bge-m3` (по умолчанию) - multilingual, 1024-dim, отличное качество
- `BAAI/bge-large-en-v1.5` - только английский, 1024-dim
- `sentence-transformers/all-MiniLM-L6-v2` - быстрая, 384-dim

**Где используется:**
- `src/code_rag/graph/weaviate_indexer.py`
- `src/code_rag/graph/build_and_index.py`
- `src/config/database.py`

---

### 2. **RERANKER_MODEL** - Модель для reranking

```bash
RERANKER_MODEL=cross-encoder/mmarco-mMiniLMv2-L12-H384-v1
```

**Используется для:**
- Перерангирование результатов поиска
- Повышение точности top-k результатов

**Рекомендуемые модели:**
- `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` (по умолчанию) - **multilingual, русский** ⭐
- `cross-encoder/ms-marco-MiniLM-L-12-v2` - английский, хорошее качество
- `cross-encoder/ms-marco-MiniLM-L-6-v2` - английский, очень быстрая

**Где используется:**
- `src/ranking/cross_encoder.py`
- `src/config/search.py`
- `src/code_rag/retrieval/code_retriever.py` (через SearchConfig)

---

### 3. **CODE_EXPLORER_MODEL** - Code Explorer Agent

```bash
CODE_EXPLORER_MODEL=qwen/qwen3-coder:free
```

**Используется для:**
- `/ask` команда в Telegram боте
- Итеративное исследование кода
- Agent-powered Q&A через API

**Рекомендуемые модели:**
- `qwen/qwen3-coder:free` (по умолчанию) - **262k context, специализирована для кода, бесплатная** ⭐
- `openai/gpt-oss-20b:free` - 131k context, 21B параметров, бесплатная
- `anthropic/claude-sonnet-4` - 200k context, **ПЛАТНАЯ** ($3-15/1M tokens)

**Где используется:**
- `src/agents/code_explorer.py` (через AgentConfig)
- `src/config/agent.py`
- `src/api/config.py`
- `src/telegram_bot/bot.py`

---

### 4. **ORCHESTRATOR_MODEL** - Query Orchestrator

```bash
ORCHESTRATOR_MODEL=deepseek/deepseek-r1:free
```

**Используется для:**
- Классификация типов вопросов (CODE/DOCUMENT/VISUAL/HYBRID)
- Маршрутизация к соответствующим агентам

**Рекомендуемые модели:**
- `deepseek/deepseek-r1:free` (по умолчанию) - быстрая, детерминистическая ⭐
- `openai/gpt-oss-20b:free` - альтернатива
- Любая быстрая бесплатная модель

**Где используется:**
- `src/agents/orchestrator.py`
- `src/config/agent.py`
- `src/telegram_bot/bot.py`

---

### 5. **ANALYSIS_MODEL** - Traceback & Business Agent

```bash
ANALYSIS_MODEL=tngtech/tng-r1t-chimera:free
```

**Используется для:**
- `/analyze` команда - анализ Python traceback
- `/guide` команда - инструкции для бизнес-пользователей
- Traceback analysis через API

**Рекомендуемые модели:**
- `tngtech/tng-r1t-chimera:free` (по умолчанию) - **enhanced tool-calling, 163k context** ⭐
- `qwen/qwen3-coder:free` - альтернатива для кода
- `deepseek/deepseek-r1:free` - базовая версия

**Где используется:**
- `src/analyzers/traceback_analyzer.py`
- `src/agents/business_agent.py`
- `src/telegram_bot/bot.py`

---

### 6. **QUERY_REFORMULATION_MODEL** - Query Reformulation

```bash
QUERY_REFORMULATION_MODEL=tngtech/tng-r1t-chimera:free
```

**Используется для:**
- Переформулирование запросов пользователя
- Улучшение качества поиска

**Рекомендуемые модели:**
- `tngtech/tng-r1t-chimera:free` (по умолчанию) - хороший reasoning ⭐
- `qwen/qwen3-coder:free` - для технических запросов
- Любая LLM модель через OpenRouter

**Где используется:**
- `src/query/reformulation.py`

---

## 📝 Полный пример .env

```bash
# ========================================
# МОДЕЛИ
# ========================================

# Embedding модель (векторизация)
EMBEDDING_MODEL=BAAI/bge-m3

# Reranker модель (перерангирование)
RERANKER_TYPE=cross_encoder
RERANKER_MODEL=cross-encoder/mmarco-mMiniLMv2-L12-H384-v1

# === AGENT MODELS ===

# Code Explorer Agent - итеративное исследование кода
CODE_EXPLORER_MODEL=qwen/qwen3-coder:free

# Query Orchestrator - классификация вопросов
ORCHESTRATOR_MODEL=deepseek/deepseek-r1:free

# Analysis Model - traceback & business инструкции
ANALYSIS_MODEL=tngtech/tng-r1t-chimera:free

# Query Reformulation - переформулирование запросов
QUERY_REFORMULATION_MODEL=tngtech/tng-r1t-chimera:free
```

---

## 🚀 Как применить изменения

### 1. Скопируйте .env.example
```bash
cp .env.example .env
```

### 2. Отредактируйте .env
```bash
nano .env  # или любой редактор
```

### 3. Измените нужные модели
```bash
# Например, использовать Claude для агента:
CODE_EXPLORER_MODEL=anthropic/claude-sonnet-4

# Или использовать другой reranker:
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
```

### 4. Перезапустите сервисы
```bash
# Telegram бот
python -m src.interfaces.telegram_bot.bot

# Или API
uvicorn src.interfaces.api.main:app --reload
```

---

## 🔍 Как узнать, какая модель используется

### Проверка через логи

При запуске бота/API в логах будет:

```
✅ Code Explorer Agent initialized with model: qwen/qwen3-coder:free
✅ Orchestrator initialized with model: deepseek/deepseek-r1:free
✅ Traceback analyzer initialized with model: tngtech/tng-r1t-chimera:free
```

### Проверка в коде

Все конфигурационные классы теперь читают из env:

```python
# src/config/agent.py
code_explorer_model: str = field(
    default_factory=lambda: os.getenv("CODE_EXPLORER_MODEL", "qwen/qwen3-coder:free")
)
```

---

## 💡 Сценарии использования

### Сценарий 1: Максимальная экономия (всё бесплатно)

```bash
# Используйте бесплатные модели
CODE_EXPLORER_MODEL=qwen/qwen3-coder:free
ORCHESTRATOR_MODEL=deepseek/deepseek-r1:free
ANALYSIS_MODEL=tngtech/tng-r1t-chimera:free
QUERY_REFORMULATION_MODEL=tngtech/tng-r1t-chimera:free
RERANKER_MODEL=cross-encoder/mmarco-mMiniLMv2-L12-H384-v1
EMBEDDING_MODEL=BAAI/bge-m3
```

**Стоимость:** $0/месяц

---

### Сценарий 2: Максимальное качество (премиум модели)

```bash
# Claude для агента (лучший reasoning)
CODE_EXPLORER_MODEL=anthropic/claude-sonnet-4

# Остальное бесплатное
ORCHESTRATOR_MODEL=deepseek/deepseek-r1:free
ANALYSIS_MODEL=tngtech/tng-r1t-chimera:free
QUERY_REFORMULATION_MODEL=qwen/qwen3-coder:free
RERANKER_MODEL=cross-encoder/mmarco-mMiniLMv2-L12-H384-v1
EMBEDDING_MODEL=BAAI/bge-m3
```

**Стоимость:** ~$50-200/месяц (в зависимости от нагрузки)

---

### Сценарий 3: Только для кода (специализация)

```bash
# Qwen3-Coder везде где нужен код
CODE_EXPLORER_MODEL=qwen/qwen3-coder:free
ANALYSIS_MODEL=qwen/qwen3-coder:free
QUERY_REFORMULATION_MODEL=qwen/qwen3-coder:free

# Остальное стандартно
ORCHESTRATOR_MODEL=deepseek/deepseek-r1:free
RERANKER_MODEL=cross-encoder/mmarco-mMiniLMv2-L12-H384-v1
EMBEDDING_MODEL=BAAI/bge-m3
```

**Стоимость:** $0/месяц
**Преимущество:** Лучшее понимание кода

---

### Сценарий 4: Быстрый reranker (производительность)

```bash
# Быстрый reranker (жертвуем multilingual)
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2

# Остальное по умолчанию
CODE_EXPLORER_MODEL=qwen/qwen3-coder:free
ORCHESTRATOR_MODEL=deepseek/deepseek-r1:free
ANALYSIS_MODEL=tngtech/tng-r1t-chimera:free
EMBEDDING_MODEL=BAAI/bge-m3
```

**Преимущество:** Быстрее на 30-40%
**Недостаток:** Хуже работает с русским языком

---

## 🧪 Тестирование моделей

### 1. Проверка через Telegram бота

```bash
# Запустите бота
python -m src.interfaces.telegram_bot.bot

# Тестируйте команды:
/ask как работает авторизация
/analyze [вставьте traceback]
/search ProductCard
```

### 2. Проверка через API

```bash
# Запустите API
uvicorn src.interfaces.api.main:app --reload

# Тест Code Explorer
curl -X POST http://localhost:8000/api/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Как реализована корзина?"}'

# Тест Search
curl -X POST http://localhost:8000/api/search \
  -H "Content-Type: application/json" \
  -d '{"query": "ProductCard", "strategy": "semantic"}'
```

### 3. Проверка логов

```bash
tail -f outputs/pipeline.log | grep -i "model"
```

---

## ⚠️ Важные замечания

### 1. OpenRouter API Key обязателен

Для **всех LLM моделей** (CODE_EXPLORER, ORCHESTRATOR, ANALYSIS, QUERY_REFORMULATION) нужен ключ:

```bash
OPENROUTER_API_KEY=sk-or-v1-...
```

Получить ключ: https://openrouter.ai/keys

### 2. Embedding модель - локальная

`EMBEDDING_MODEL` скачивается и работает локально (через HuggingFace):

```bash
# Модель скачается автоматически при первом использовании
# Или скачайте заранее:
python scripts/download_models.py
```

### 3. Reranker модель - локальная

`RERANKER_MODEL` тоже локальная (через sentence-transformers):

```bash
# Скачается автоматически
# Размер: ~200-500 MB в зависимости от модели
```

### 4. Кэширование моделей

Модели кэшируются в:
- `~/.cache/huggingface/` - embedding модели
- `~/.cache/torch/sentence_transformers/` - reranker модели

Для очистки:
```bash
rm -rf ~/.cache/huggingface
rm -rf ~/.cache/torch/sentence_transformers
```

---

## 🆘 Troubleshooting

### Проблема: "Model not found"

**Решение:**
1. Проверьте, что модель существует на OpenRouter: https://openrouter.ai/models
2. Проверьте правильность названия модели
3. Для локальных моделей - проверьте интернет-соединение

### Проблема: "API key invalid"

**Решение:**
1. Проверьте `OPENROUTER_API_KEY` в .env
2. Убедитесь, что ключ активен: https://openrouter.ai/keys

### Проблема: Медленная работа reranker

**Решение:**
1. Используйте более быструю модель:
   ```bash
   RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
   ```
2. Или отключите reranking:
   ```bash
   RERANKER_TYPE=none
   ```

### Проблема: Out of memory при embedding

**Решение:**
1. Используйте меньшую модель:
   ```bash
   EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
   ```
2. Уменьшите batch_size в config/base.yaml

---

## 📚 Дополнительные ресурсы

- **OpenRouter Models:** https://openrouter.ai/models
- **HuggingFace Models:** https://huggingface.co/models
- **Cross-Encoder Models:** https://www.sbert.net/docs/pretrained-models/ce-msmarco.html
- **Embedding Models:** https://www.sbert.net/docs/pretrained_models.html

---

## 📊 Таблица совместимости

| Переменная | Источник | Тип | Требует API Key |
|------------|----------|-----|-----------------|
| EMBEDDING_MODEL | HuggingFace | Локальная | ❌ Нет |
| RERANKER_MODEL | HuggingFace | Локальная | ❌ Нет |
| CODE_EXPLORER_MODEL | OpenRouter | API | ✅ Да |
| ORCHESTRATOR_MODEL | OpenRouter | API | ✅ Да |
| ANALYSIS_MODEL | OpenRouter | API | ✅ Да |
| QUERY_REFORMULATION_MODEL | OpenRouter | API | ✅ Да |

---

**Дата обновления:** 2025-01-02
**Версия:** 2.0
**Статус:** Все модели настраиваются через .env ✅
