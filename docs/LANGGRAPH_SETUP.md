# Настройка LangGraph и LangSmith

## Обзор

LangGraph используется для создания агентных RAG пайплайнов с качественной обратной связью. LangSmith предоставляет мониторинг и отладку для LangGraph приложений.

## Быстрый старт

### 1. Установка зависимостей

LangGraph уже включен в `requirements.txt`. Если нужно установить отдельно:

```bash
pip install langgraph langgraph-cli
```

### 2. Настройка LangSmith (рекомендуется)

LangSmith предоставляет:
- Мониторинг выполнения графов
- Отладку узлов и состояний
- Трассировку запросов
- Анализ производительности

#### Получение API ключа

1. Зарегистрируйтесь на [https://smith.langchain.com/](https://smith.langchain.com/)
2. Перейдите в Settings → API Keys
3. Создайте новый API ключ
4. Скопируйте ключ (начинается с `lsv2_pt_...`)

#### Добавление ключа в проект

Добавьте в `.env` файл в **корне проекта** (без пробелов вокруг `=`):

```bash
LANGSMITH_API_KEY=lsv2_pt_your_key_here
```

**Важно:** 
- НЕ используйте пробелы вокруг `=`: `LANGSMITH_API_KEY = ...` ❌
- Правильно: `LANGSMITH_API_KEY=...` ✅
- Файл `.env` должен быть в корне проекта (там же где `README.md`)

**Примечание:** LangSmith опционален. LangGraph будет работать без него, но мониторинг очень полезен для отладки.

### 3. Запуск LangGraph Dev сервера

```bash
cd src/langgraph_server
langgraph dev
```

Сервер запустится на `http://localhost:8123` (по умолчанию).

#### Если видите предупреждение о LangSmith

Если при запуске появляется предупреждение:
```
It looks like your LangSmith API key is missing. 
Please make sure to add LANGSMITH_API_KEY to your local server's .env file.
```

Это не критично - LangGraph будет работать, но без мониторинга. Чтобы убрать предупреждение:
1. Добавьте `LANGSMITH_API_KEY` в `.env` файл в корне проекта
2. Перезапустите `langgraph dev`

### 4. Использование LangGraph RAG

LangGraph RAG доступен через API эндпоинт:

```bash
POST /api/ask/langgraph
```

Пример запроса:

```json
{
  "question": "How to book equity trade?",
  "max_iterations": 10,
  "context": {
    "repositories": ["ui.bo", "api.bo"]
  }
}
```

## Структура LangGraph проекта

```
src/langgraph_server/
├── langgraph.json      # Конфигурация LangGraph сервера
├── rag_graph.py        # Определение графа RAG
├── nodes.py            # Узлы графа (context_collector, quality_checker, etc.)
└── state.py            # Определение состояния RAGState
```

## Конфигурация

### langgraph.json

```json
{
  "$schema": "https://langgra.ph/schema.json",
  "graphs": {
    "rag": "./rag_graph.py:graph"
  },
  "env": "../../.env",
  "python_version": "3.11",
  "dependencies": ["../.."]
}
```

Файл указывает:
- `graphs`: какие графы экспортировать
- `env`: путь к `.env` файлу (относительно `langgraph.json`)
- `python_version`: версия Python
- `dependencies`: зависимости проекта

### Переменные окружения

LangGraph автоматически загружает переменные из `.env` файла (указан в `langgraph.json`).

Основные переменные:
- `OPENROUTER_API_KEY` - для вызовов LLM
- `LANGSMITH_API_KEY` - для мониторинга (опционально)
- `RAG_ANSWER_MODEL` - модель для генерации ответов
- `RAG_QUALITY_MODEL` - модель для оценки качества
- `RAG_REWRITE_MODEL` - модель для переписывания запросов

## Мониторинг в LangSmith

После настройки `LANGSMITH_API_KEY` вы сможете:

1. **Просматривать трассировки** на [https://smith.langchain.com/](https://smith.langchain.com/)
2. **Отлаживать узлы** - видеть входы/выходы каждого узла
3. **Анализировать производительность** - время выполнения, использование токенов
4. **Отслеживать ошибки** - автоматическое логирование исключений

## Устранение проблем

### Предупреждение о LangSmith API ключе

**Проблема:** При запуске `langgraph dev` появляется предупреждение о отсутствующем ключе.

**Важно:** LangGraph ищет `.env` файл в директории запуска (`src/langgraph_server/`), а не в корне проекта. Сообщение "your local server's .env file" означает, что нужен `.env` в `src/langgraph_server/`.

**Решение (выберите один способ):**

#### Способ 1: Автоматический (рекомендуется)

Используйте скрипт для автоматического создания `.env` в нужной директории:

```bash
python scripts/setup_langgraph_env.py
```

Скрипт скопирует `LANGSMITH_API_KEY` из корневого `.env` в `src/langgraph_server/.env`.

#### Способ 2: Вручную

Создайте файл `src/langgraph_server/.env` вручную:

```bash
cd src/langgraph_server
echo LANGSMITH_API_KEY=lsv2_pt_your_key_here > .env
```

**Важно:** БЕЗ пробелов вокруг `=`:
- ❌ Неправильно: `LANGSMITH_API_KEY = lsv2_pt_...`
- ✅ Правильно: `LANGSMITH_API_KEY=lsv2_pt_...`

#### Способ 3: Использовать скрипт запуска (рекомендуется, если .env не работает)

Используйте готовый скрипт, который автоматически устанавливает переменную:

**Windows (PowerShell):**
```powershell
.\scripts\run_langgraph_dev.ps1
```

**Windows (CMD):**
```cmd
scripts\run_langgraph_dev.bat
```

Скрипт автоматически читает `LANGSMITH_API_KEY` из корневого `.env` и устанавливает как переменную окружения перед запуском.

#### Способ 4: Переменная окружения системы вручную

Установите переменную перед запуском:

```powershell
# Windows PowerShell
$env:LANGSMITH_API_KEY="lsv2_pt_your_key_here"
cd src/langgraph_server
langgraph dev
```

**После изменений:**
1. Полностью остановите `langgraph dev` (Ctrl+C)
2. Запустите заново: `cd src/langgraph_server && langgraph dev`

**Если все еще не работает:**
- Это может быть просто предупреждение, которое можно игнорировать
- LangGraph будет работать без мониторинга LangSmith
- Попробуйте установить переменную в системных переменных окружения Windows (через Панель управления)

**Если все еще не работает:**

LangGraph может искать `.env` в директории запуска, а не по пути из `langgraph.json`. Попробуйте:

1. **Создать `.env` также в `src/langgraph_server/`** (где запускается `langgraph dev`):
   ```bash
   cd src/langgraph_server
   echo LANGSMITH_API_KEY=lsv2_pt_your_key_here > .env
   ```
   Или скопируйте только строку с `LANGSMITH_API_KEY` из корневого `.env` в `src/langgraph_server/.env`

2. **Использовать переменную окружения системы** (Windows):
   ```powershell
   $env:LANGSMITH_API_KEY="lsv2_pt_your_key_here"
   langgraph dev
   ```

3. **Проверить формат в `.env`**:
   - Убедитесь, что нет пробелов: `LANGSMITH_API_KEY=lsv2_pt_...` (не `LANGSMITH_API_KEY = ...`)
   - Убедитесь, что нет кавычек вокруг значения
   - Убедитесь, что нет пробелов в начале/конце строки

**Альтернатива:** Можно проигнорировать предупреждение - LangGraph будет работать без мониторинга.

### LangGraph не находит .env файл

**Проблема:** LangGraph не загружает переменные окружения.

**Решение:**
1. Проверьте путь в `langgraph.json`: `"env": "../../.env"`
2. Убедитесь, что `.env` файл существует в корне проекта
3. Проверьте, что вы запускаете `langgraph dev` из `src/langgraph_server/`

## Дополнительные ресурсы

- [LangGraph документация](https://langchain-ai.github.io/langgraph/)
- [LangSmith документация](https://docs.smith.langchain.com/)
- [LangGraph Server](https://langchain-ai.github.io/langgraph/how-tos/langgraph-server/)

