# 📥 Скачивание моделей

## Обзор

Code RAG использует несколько типов моделей:

### Локальные модели (скачиваются на ваш компьютер):
- ✅ **Embedding модель** - векторизация кода (обязательно)
- ✅ **Reranker модель** - перерангирование результатов (обязательно)
- ⚙️ **Локальная LLM** - для работы без API (опционально)

### API модели (через OpenRouter, ничего скачивать не нужно):
- 🌐 **Code Explorer** - agent для исследования кода
- 🌐 **Orchestrator** - классификация вопросов
- 🌐 **Analysis** - traceback и business agent
- 🌐 **Query Reformulation** - переформулирование запросов

---

## 🚀 Быстрый старт

### Вариант 1: Автоматическое скачивание (рекомендуется)

Модели скачаются **автоматически** при первом использовании. Ничего делать не нужно!

```bash
# Просто запустите бота или API
python -m src.interfaces.telegram_bot.bot

# Модели скачаются при первом запуске
```

**Плюсы:** Ничего не нужно делать заранее
**Минусы:** Первый запуск будет дольше (~5-10 минут для скачивания)

---

### Вариант 2: Скачать заранее (ручное управление)

```bash
# Установите huggingface_hub (если еще не установлен)
pip install huggingface_hub

# Запустите скрипт скачивания
python scripts/download_models.py
```

**Плюсы:** Можете выбрать какие модели скачивать, контроль процесса
**Минусы:** Требует дополнительных действий

---

## 📦 Какие модели скачиваются

### Обязательные модели (всегда нужны):

#### 1. Embedding модель: `BAAI/bge-m3`
- **Размер:** ~2 GB
- **Назначение:** Векторизация кода для семантического поиска
- **Где хранится:** `~/.cache/huggingface/hub/models--BAAI--bge-m3/`
- **Как используется:** Каждый раз при индексации репозитория и при поиске

#### 2. Reranker модель: `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`
- **Размер:** ~500 MB
- **Назначение:** Улучшение качества поиска через перерангирование
- **Где хранится:** `~/.cache/torch/sentence_transformers/`
- **Как используется:** При каждом поиске (если `ENABLE_RERANKING=true`)

**Итого обязательных:** ~2.5 GB

---

### Опциональные модели (только для локального режима):

#### 3. Локальная LLM: `Qwen3-32B-IQ4_NL.gguf`
- **Размер:** ~20 GB
- **Назначение:** Замена OpenRouter API (если нет интернета или хотите полностью локальный режим)
- **Где хранится:** `models/Qwen3-32B-IQ4_NL.gguf`
- **Как используется:** Только если установлен `LLM_MODE=local` в .env

**Нужна только если:**
- Нет доступа к интернету
- Не хотите использовать OpenRouter API
- Хотите полную приватность данных

---

## 🎯 Интерактивное скачивание

```bash
python scripts/download_models.py
```

Скрипт спросит:

```
Что скачивать?
1. Только обязательные (рекомендуется, если используете OpenRouter API)
2. Все модели (если планируете использовать локальный режим)
3. Выбрать вручную
```

**Рекомендуем вариант 1** для большинства случаев (используйте API модели через OpenRouter).

---

## 📋 Примеры использования

### Пример 1: Минимальная установка (с OpenRouter API)

```bash
# 1. Скачайте только обязательные модели
python scripts/download_models.py
# Выберите: 1 (только обязательные)

# 2. Настройте .env
cp .env.example .env
nano .env
# Добавьте OPENROUTER_API_KEY

# 3. Готово! Используйте API модели
```

**Результат:** ~2.5 GB скачано, все LLM работают через API

---

### Пример 2: Полностью локальная установка

```bash
# 1. Скачайте все модели
python scripts/download_models.py
# Выберите: 2 (все модели)

# 2. Настройте .env для локального режима
cp .env.example .env
nano .env
# Установите: LLM_MODE=local

# 3. Готово! Всё работает локально
```

**Результат:** ~22.5 GB скачано, никакой зависимости от API

---

### Пример 3: Ручной выбор

```bash
python scripts/download_models.py
# Выберите: 3 (выбрать вручную)

# Скрипт спросит для каждой модели:
# Скачать embedding? (y/n): y
# Скачать reranker? (y/n): y
# Скачать llm_local? (y/n): n
```

---

## 🔧 Настройка моделей

### Изменить embedding модель

```bash
# В .env
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

Скачается автоматически при первом использовании, или через:
```bash
python scripts/download_models.py
```

### Изменить reranker модель

```bash
# В .env
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
```

---

## 📊 Требования к диску

| Конфигурация | Размер | Модели |
|--------------|--------|--------|
| **Минимальная** (API режим) | ~2.5 GB | Embedding + Reranker |
| **Полная** (локальный режим) | ~22.5 GB | + Локальная LLM |

---

## ⚡ Скорость скачивания

При хорошем интернете (~100 Mbps):
- Embedding (~2 GB) - ~3-5 минут
- Reranker (~500 MB) - ~1-2 минуты
- Локальная LLM (~20 GB) - ~30-40 минут

**Итого (минимальная установка):** ~5-7 минут

---

## 🔍 Проверка скачанных моделей

### Вариант 1: Через Python

```python
from sentence_transformers import SentenceTransformer, CrossEncoder

# Проверка embedding модели
model = SentenceTransformer("BAAI/bge-m3")
print("✅ Embedding модель загружена")

# Проверка reranker модели
reranker = CrossEncoder("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
print("✅ Reranker модель загружена")
```

### Вариант 2: Через файловую систему

```bash
# Embedding модель
ls ~/.cache/huggingface/hub/ | grep bge-m3

# Reranker модель
ls ~/.cache/torch/sentence_transformers/ | grep mmarco

# Локальная LLM
ls models/ | grep Qwen3
```

---

## ❓ FAQ

### Q: Модели скачиваются каждый раз при запуске?

**A:** Нет! Модели кэшируются и скачиваются только один раз.

---

### Q: Можно ли использовать другие модели?

**A:** Да! Измените `EMBEDDING_MODEL` и `RERANKER_MODEL` в `.env`. Модели скачаются автоматически.

---

### Q: Что если скачивание прервалось?

**A:** Просто запустите скрипт снова:
```bash
python scripts/download_models.py
```
HuggingFace CLI автоматически продолжит с места обрыва.

---

### Q: Нужен ли OpenRouter API Key для embedding?

**A:** Нет! Embedding и reranker модели - локальные, работают без интернета после скачивания. API ключ нужен только для LLM моделей (агентов).

---

### Q: Как удалить скачанные модели?

**A:**
```bash
# Embedding модели
rm -rf ~/.cache/huggingface/

# Reranker модели
rm -rf ~/.cache/torch/sentence_transformers/

# Локальная LLM
rm -rf models/
```

---

### Q: Автоматическое скачивание не работает

**A:** Проверьте:
1. Есть ли интернет-соединение
2. Установлен ли `sentence-transformers`: `pip install sentence-transformers`
3. Есть ли свободное место на диске (~2.5 GB минимум)

Попробуйте ручное скачивание:
```bash
python scripts/download_models.py
```

---

## 🚨 Troubleshooting

### Ошибка: "No space left on device"

**Решение:** Очистите место на диске или используйте другой диск:
```bash
# Укажите другую директорию для кэша
export HF_HOME=/path/to/large/disk/.cache/huggingface
export TRANSFORMERS_CACHE=/path/to/large/disk/.cache/transformers
```

---

### Ошибка: "Connection timeout"

**Решение:** Используйте VPN или настройте proxy:
```bash
# Через переменные окружения
export HTTP_PROXY=http://proxy:port
export HTTPS_PROXY=http://proxy:port

# Или в коде
python scripts/download_models.py
```

---

### Ошибка: "ImportError: No module named 'sentence_transformers'"

**Решение:**
```bash
pip install sentence-transformers
# или
pip install -r requirements.txt
```

---

## 📚 Дополнительные ресурсы

- **HuggingFace Hub:** https://huggingface.co/models
- **Sentence Transformers:** https://www.sbert.net/
- **Настройка моделей:** `docs/MODEL_CONFIGURATION.md`
- **Рекомендации:** `docs/MODEL_RECOMMENDATIONS.md`

---

**Дата создания:** 2025-01-02
**Версия:** 1.0
**Автор:** Code RAG Team
