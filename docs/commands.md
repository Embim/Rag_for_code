# Команды запуска

Все команды выполняются из корня проекта.

## Режимы работы LLM

### Локальный режим (по умолчанию)
Использует локальную модель через llama-cpp-python:
```bash
export LLM_MODE=local  # или не устанавливать (по умолчанию)
python main_pipeline.py build --force --llm-clean
```

### API режим (OpenRouter) — РЕКОМЕНДУЕТСЯ
Использует OpenRouter API для ускорения (в 5-20 раз быстрее):
```bash
# Настройка (API ключ ОБЯЗАТЕЛЕН даже для бесплатных моделей)
export LLM_MODE=api
export LLM_API_MODEL=tngtech/deepseek-r1t2-chimera:free
export OPENROUTER_API_KEY=sk-or-v1-...  # получи БЕСПЛАТНЫЙ ключ на https://openrouter.ai/keys
export LLM_API_MAX_WORKERS=10  # параллельных запросов (по умолчанию 10)
python main_pipeline.py build --force --llm-clean
```

**Важно:** OpenRouter требует API ключ даже для бесплатных моделей. Получите бесплатный ключ на https://openrouter.ai/keys

**Преимущества API режима:**
- ⚡ Ускорение в 5-20 раз (параллельные запросы)
- 💰 Бесплатные модели доступны (DeepSeek R1T2 Chimera)
- 🚀 Не занимает VRAM (освобождает GPU)
- 📈 Масштабируемость (обработка больших объемов)

**Другие модели OpenRouter:**
```bash
# Бесплатные модели
export LLM_API_MODEL=tngtech/deepseek-r1t2-chimera:free  # по умолчанию
export LLM_API_MODEL=meta-llama/llama-3.2-3b-instruct:free
export LLM_API_MODEL=google/gemma-2-2b-it:free
export LLM_API_MODEL=openrouter/sherlock-think-alpha  # бесплатно, 1.8M контекст, reasoning модель

```
Полный каталог: https://openrouter.ai/models

**Пример использования Sherlock Think Alpha:**
```bash
export LLM_MODE=api
export LLM_API_MODEL=openrouter/sherlock-think-alpha  # reasoning модель с 1.8M контекстом
export OPENROUTER_API_KEY=sk-or-v1-...
export LLM_API_MAX_WORKERS=10

python main_pipeline.py build --force --llm-clean
```

python main_pipeline.py build --force --llm-clean

## Build базы знаний
```bash
# Без LLM-clean (быстро)
python main_pipeline.py build --force

# С LLM очисткой (локальный режим, медленно)
python main_pipeline.py build --force --llm-clean  # min-usefulness по умолчанию 0.3
python main_pipeline.py build --force --llm-clean --min-usefulness 0.5  # более строгая фильтрация

# С LLM очисткой (API режим, быстро!)
export LLM_MODE=api
export LLM_API_MODEL=tngtech/deepseek-r1t2-chimera:free
python main_pipeline.py build --force --llm-clean  # min-usefulness по умолчанию 0.3
python main_pipeline.py build --force --llm-clean --min-usefulness 0.5  # более строгая фильтрация
```

Аргументы:
- `--force` — очистить Weaviate и пересоздать индекс.
- `--llm-clean` — включить LLM-очистку документов.
- `--min-usefulness` — порог фильтрации в LLM-clean (0.0–1.0, по умолчанию: 0.3).
  - Можно не указывать — будет использовано значение по умолчанию (0.3).
  - 0.0–0.3: мусор (навигация, реклама) — фильтруется
  - 0.4–0.6: частично полезно — сохраняется
  - 0.7–1.0: очень полезно — сохраняется

Переменные окружения (см. `src/config.py`):
- `LLM_MODE=local|api` — режим работы LLM (local = локальная модель, api = OpenRouter API).
- `LLM_API_MODEL` — модель для API (по умолчанию: `tngtech/deepseek-r1t2-chimera:free`).
- `LLM_API_ROUTING` — провайдер для роутинга (опционально, например: "grok", "openai", "anthropic").
- `OPENROUTER_API_KEY` — API ключ OpenRouter (ОБЯЗАТЕЛЕН, получите бесплатный на https://openrouter.ai/keys).
- `LLM_API_MAX_WORKERS` — количество параллельных запросов к API (по умолчанию: 10).
- `USE_WEAVIATE=true` — включен по умолчанию.
- `LOG_LEVEL=INFO|DEBUG` — уровень логирования.
- `LOG_FILE=custom.log` — имя файла логов в `outputs/`.

## Поиск ответов
```bash
# Все вопросы из questions_clean.csv
python main_pipeline.py search

# Первые N вопросов (тестовый прогон)
python main_pipeline.py search --limit 20

# С оптимизацией параметров (grid search)
python main_pipeline.py search --optimize --optimize-mode quick --optimize-sample 50

# С LLM reranking через API (быстро!)
export LLM_MODE=api
export LLM_API_MODEL=tngtech/deepseek-r1t2-chimera:free
export RERANKER_TYPE=llm  # использовать LLM reranker
python main_pipeline.py search
```

## Полный цикл
```bash
# Локальный режим
python main_pipeline.py all --llm-clean --limit 20

# API режим (рекомендуется - быстрее!)
export LLM_MODE=api
export LLM_API_MODEL=tngtech/deepseek-r1t2-chimera:free
export LLM_API_MAX_WORKERS=10
python main_pipeline.py all --llm-clean --limit 20
```

## Оценка (заглушка для будущей метрики)
```bash
python main_pipeline.py evaluate
```

## Grid Search (скрипт)
```bash
python scripts/run_grid_search.py --mode quick --sample 30
python scripts/run_grid_search.py --mode full  --sample 100
python scripts/run_grid_search.py --mode quick --sample 30 --no-llm
```

## Тестовый прогон на маленьком наборе данных

Быстрый сценарий, чтобы проверить, что всё работает на небольшом объёме данных:

```bash
# 1) Перестроить базу знаний только по небольшой выборке документов (если поддерживается флагом)
python main_pipeline.py build --force --limit 100

# 2) Запустить поиск ответов только по первым 20 вопросам
python main_pipeline.py search --limit 20

# 3) (опционально) полный мини-цикл "build + search" одной командой
python main_pipeline.py all --llm-clean --limit 20

# 4) С API режимом (быстрее для тестирования!)
export LLM_MODE=api
export LLM_API_MODEL=tngtech/deepseek-r1t2-chimera:free
python main_pipeline.py all --llm-clean --limit 20
```

## Примеры использования API режима

### Быстрый build с LLM очисткой (500 документов)
```bash
export LLM_MODE=api
export LLM_API_MODEL=tngtech/deepseek-r1t2-chimera:free
export LLM_API_MAX_WORKERS=10
export OPENROUTER_API_KEY=sk-or-v1-...  # опционально

# С порогом по умолчанию (0.3) - сохраняет больше документов
python main_pipeline.py build --force --llm-clean

# С более строгим порогом (0.5) - фильтрует больше мусора
python main_pipeline.py build --force --llm-clean --min-usefulness 0.5

# Ожидаемое время: ~15-40 минут (вместо ~4.7 часов локально)
```

### Поиск с LLM reranking через API
```bash
export LLM_MODE=api
export LLM_API_MODEL=tngtech/deepseek-r1t2-chimera:free
export RERANKER_TYPE=llm  # использовать LLM reranker

python main_pipeline.py search
```

### Полный цикл с оптимизацией через API
```bash
export LLM_MODE=api
export LLM_API_MODEL=tngtech/deepseek-r1t2-chimera:free
export LLM_API_MAX_WORKERS=10

python main_pipeline.py all --llm-clean --optimize --optimize-mode quick
```
