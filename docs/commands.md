# Команды запуска

Все команды выполняются из корня проекта.

## Build базы знаний
```bash
# Без LLM-clean (быстро)
python main_pipeline.py build --force

# С LLM очисткой (дольше)
python main_pipeline.py build --force --llm-clean --min-usefulness 0.5
```

Аргументы:
- `--force` — очистить Weaviate и пересоздать индекс.
- `--llm-clean` — включить LLM-очистку документов.
- `--min-usefulness` — порог фильтрации в LLM-clean (0.0–1.0).

Переменные окружения (см. `src/config.py`):
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
```

## Полный цикл
```bash
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
```
