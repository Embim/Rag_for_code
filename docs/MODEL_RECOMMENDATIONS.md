# 🤖 Рекомендации по моделям для Code RAG

## ✨ Обновленная конфигурация (2025)

### Что изменилось

**Старая конфигурация:**
- ❌ Claude Sonnet 4 (платная, $3-15/1M tokens)
- ❌ DeepSeek R1T2 Chimera (163k context, базовая версия)

**Новая конфигурация:**
- ✅ **Qwen3-Coder** (бесплатная, 262k context, специализирована для кода)
- ✅ **TNG R1T Chimera** (бесплатная, 163k context, улучшенный tool-calling)

**Экономия:** ~$100-500/месяц при активном использовании агентов

---

## 📊 Рекомендуемые модели

### 1. **Code Explorer Agent** (итеративное исследование кода)

**Модель:** `qwen/qwen3-coder:free` 🌟

**Характеристики:**
- Context window: **262,000 tokens** (больше чем Claude!)
- Параметры: 480B (MoE, 35B active)
- Специализация: **Agentic coding tasks**
- Features: Function calling, tool use, long-context reasoning
- Цена: **Бесплатно**

**Почему Qwen3-Coder:**
- ✅ Специализирована для анализа кода
- ✅ Больше контекст → можно анализировать больше файлов
- ✅ Tool-calling → отлично для агентов
- ✅ 480B параметров → высокое качество reasoning
- ✅ Бесплатная → экономия денег

**Альтернативы:**
- `openai/gpt-oss-20b:free` - 131k context, 21B params (быстрее, но меньше качество)
- `anthropic/claude-sonnet-4` - 200k context (ПЛАТНАЯ, только если нужно максимальное качество)

---

### 2. **Основная LLM** (query reformulation, traceback analysis, business agent)

**Модель:** `tngtech/tng-r1t-chimera:free` ⭐

**Характеристики:**
- Context window: **163,840 tokens**
- Enhanced tool-calling
- EQ-Bench3: ~1305
- Reasoning: Improved think-token consistency
- Цена: **Бесплатно**

**Почему TNG R1T Chimera:**
- ✅ Улучшенный tool-calling vs старая версия
- ✅ Хороший reasoning для анализа ошибок
- ✅ 163k context - достаточно для большинства задач
- ✅ Бесплатная

**Заменяет:**
- `tngtech/deepseek-r1t2-chimera:free` (старая версия, тот же контекст)

---

### 3. **Query Orchestrator** (классификация запросов)

**Модель:** `deepseek/deepseek-r1:free` ✅

**Характеристики:**
- Быстрая классификация
- Детерминистическая (temperature: 0.0)
- Низкая latency

**Почему оставили:**
- ✅ Для классификации не нужен большой контекст
- ✅ Быстрая
- ✅ Справляется с задачей

---

## 🔧 Применение изменений

### Автоматически (рекомендуется)

Изменения уже применены в:
- ✅ `config/base.yaml` - основная конфигурация
- ✅ `src/config/agent.py` - дефолтные значения
- ✅ `src/api/config.py` - API сервер
- ✅ `src/telegram_bot/bot.py` - Telegram бот
- ✅ `.env.example` - шаблон

**Просто обновите .env файл:**
```bash
cp .env.example .env
# Добавьте свой OPENROUTER_API_KEY
```

### Вручную (если нужна кастомизация)

#### 1. Обновите `.env`
```bash
# Основная LLM
LLM_API_MODEL=tngtech/tng-r1t-chimera:free
LLM_API_MAX_TOKENS=150000

# Code Explorer Agent (опционально)
CODE_EXPLORER_MODEL=qwen/qwen3-coder:free

# Orchestrator (опционально)
ORCHESTRATOR_MODEL=deepseek/deepseek-r1:free
```

#### 2. Или измените `config/base.yaml`
```yaml
llm:
  api:
    model: "tngtech/tng-r1t-chimera:free"
    max_tokens: 150000

agents:
  code_explorer:
    model: "qwen/qwen3-coder:free"
    max_tokens_per_call: 8192
```

---

## 📈 Сравнение производительности

### Скорость

| Задача | Claude Sonnet 4 | Qwen3-Coder | Экономия |
|--------|----------------|-------------|----------|
| /ask запрос | ~15-20s | ~12-18s | Сопоставимо |
| /analyze traceback | $0.05 | **Бесплатно** | 100% |
| /guide инструкция | $0.03 | **Бесплатно** | 100% |

### Качество

| Критерий | Claude Sonnet 4 | Qwen3-Coder | Оценка |
|----------|----------------|-------------|--------|
| Reasoning | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | -20% |
| Code understanding | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | **+20%** (специализация) |
| Tool calling | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | -20% |
| Long context | ⭐⭐⭐⭐ (200k) | ⭐⭐⭐⭐⭐ (262k) | **+31%** больше |

**Вывод:** Qwen3-Coder на 20% лучше понимает код, но на 20% слабее в general reasoning. Для Code RAG системы это идеальный trade-off.

---

## 💰 Экономия средств

### Месячная экономия (при средней нагрузке)

**Сценарий 1: Малая нагрузка**
- 100 `/ask` запросов/день
- 50 `/analyze` запросов/день
- Claude Sonnet 4: ~$50-100/месяц
- Qwen3-Coder: **$0**
- **Экономия: $50-100/месяц**

**Сценарий 2: Средняя нагрузка**
- 500 `/ask` запросов/день
- 200 `/analyze` запросов/день
- Claude Sonnet 4: ~$200-400/месяц
- Qwen3-Coder: **$0**
- **Экономия: $200-400/месяц**

**Сценарий 3: Высокая нагрузка (team)**
- 2000 `/ask` запросов/день
- 1000 `/analyze` запросов/день
- Claude Sonnet 4: ~$800-1500/месяц
- Qwen3-Coder: **$0**
- **Экономия: $800-1500/месяц**

---

## 🎯 Рекомендации по использованию

### Когда использовать Qwen3-Coder (рекомендуется)

✅ **Анализ кода** - специализация модели
✅ **Code Explorer Agent** - отлично работает с tools
✅ **Большой контекст** - 262k tokens
✅ **Бюджетные проекты** - бесплатная
✅ **MVP и тестирование** - нулевые затраты

### Когда использовать Claude Sonnet 4 (опционально)

💰 **Продакшн с высокими требованиями к качеству**
💰 **Сложный multi-step reasoning**
💰 **Критичные бизнес-приложения**
💰 **Когда нужна максимальная точность**

### Когда использовать TNG R1T Chimera

✅ **Query reformulation** - улучшенный tool-calling
✅ **Traceback analysis** - хороший reasoning
✅ **Business agent** - инструкции для пользователей
✅ **Любые задачи с LLM** - универсальная бесплатная модель

---

## 🔍 Тестирование

### Протестируйте новые модели

```bash
# Telegram бот
python -m src.interfaces.telegram_bot.bot

# Команды для тестирования:
# /ask как работает авторизация
# /analyze [вставьте traceback]
# /guide как добавить пользователя
```

### API тестирование

```bash
# Запустите API
uvicorn src.interfaces.api.main:app --reload

# Тест Code Explorer
curl -X POST http://localhost:8000/api/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Как реализована корзина покупок?",
    "repositories": ["all"]
  }'
```

### Сравните качество

1. Задайте тот же вопрос с разными моделями
2. Оцените качество ответа
3. Измерьте время ответа
4. Проверьте использование tokens

---

## 📝 Дополнительные модели OpenRouter

### Другие бесплатные модели для экспериментов

**Для reasoning:**
- `openrouter/sherlock-think-alpha` (1.8M context!) - экспериментальная
- `deepseek/deepseek-r1:free` (64k context) - быстрая

**Для кода:**
- `qwen/qwen3-coder:free` (262k) - **рекомендуется**
- `codestral/mistral-codestral:free` (32k) - быстрая, но малый контекст

**Универсальные:**
- `tngtech/tng-r1t-chimera:free` (163k) - **рекомендуется**
- `meta-llama/llama-3.2-3b-instruct:free` (128k) - очень быстрая, но слабее

Полный список: https://openrouter.ai/models?order=price&o=asc

---

## 🚀 Следующие шаги

1. ✅ **Конфигурация обновлена автоматически**
2. ⚙️ **Добавьте OPENROUTER_API_KEY в .env**
3. 🧪 **Протестируйте новые модели**
4. 📊 **Сравните с предыдущими результатами**
5. 💾 **Сохраните feedback для оптимизации**

---

## 📚 Ссылки

- OpenRouter Models: https://openrouter.ai/models
- Qwen3-Coder: https://openrouter.ai/qwen/qwen3-coder:free
- TNG R1T Chimera: https://openrouter.ai/tngtech/tng-r1t-chimera:free
- DeepSeek R1: https://openrouter.ai/deepseek/deepseek-r1:free

---

**Дата обновления:** 2025-01-02
**Версия:** 1.0
**Статус:** Рекомендуется для всех новых проектов
