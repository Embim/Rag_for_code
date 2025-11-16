"""
Query Reformulation - улучшение запросов через LLM

Концепция:
Пользовательские запросы часто нечеткие, неполные или используют разговорную речь.
LLM переформулирует запрос для лучшего поиска в базе знаний.

Примеры:
- "можно ли платить телефон" → "Как оплатить мобильную связь через Альфа-Банк"
- "кэшбэк по карте" → "Какой размер кэшбэка по дебетовым картам Альфа-Банка"
- "открыть счет" → "Инструкция по открытию расчетного счета в Альфа-Банке"

Преимущества:
- +8-12% accuracy
- Более конкретные запросы
- Добавление банковских терминов
- Расширение контекста
"""
from typing import List, Optional, Dict
import hashlib
import pickle
from pathlib import Path

# Условный импорт llama_cpp
try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False


class QueryReformulator:
    """
    LLM-based переформулирование запросов для улучшения поиска

    Стратегии:
    1. Конкретизация - добавление деталей
    2. Профессионализация - банковские термины
    3. Расширение - добавление контекста
    4. Нормализация - исправление опечаток и сленга
    """

    def __init__(self, llm_model_path: str = None,
                 use_cache: bool = True,
                 cache_dir: str = "cache/query_reformulation",
                 use_api: bool = None):
        """
        Args:
            llm_model_path: путь к LLM модели (для локального режима)
            use_cache: использовать кэш (ускоряет повторные запросы)
            cache_dir: директория для кэша
            use_api: использовать ли API (если None - определяется из LLM_MODE)
        """
        from src.config import (
            LLM_MODE, LLM_API_MODEL, LLM_API_MAX_TOKENS, LLM_API_ROUTING, OPENROUTER_API_KEY,
            LLM_CONTEXT_SIZE, LLM_GPU_LAYERS, MODELS_DIR, LLM_MODEL_FILE
        )
        
        # Определяем режим работы
        if use_api is None:
            use_api = (LLM_MODE == "api")
        
        self.use_api = use_api
        
        if use_api:
            # API режим (OpenRouter)
            print(f"[QueryReformulator] Инициализация (API режим, модель: {LLM_API_MODEL})")
            try:
                from openai import OpenAI
                base_url = "https://openrouter.ai/api/v1"
                
                if not OPENROUTER_API_KEY:
                    raise ValueError(
                        "OPENROUTER_API_KEY не установлен!\n"
                        "Получите бесплатный ключ на https://openrouter.ai/keys\n"
                        "Установите: export OPENROUTER_API_KEY=sk-or-v1-..."
                    )
                
                default_headers = {
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "HTTP-Referer": "https://github.com/your-repo",
                    "X-Title": "AlfaBank RAG Pipeline"
                }
                
                if LLM_API_ROUTING:
                    default_headers["X-OpenRouter-Provider"] = LLM_API_ROUTING
                
                self.client = OpenAI(
                    base_url=base_url,
                    api_key=OPENROUTER_API_KEY,
                    timeout=60,
                    default_headers=default_headers
                )
                self.model_name = LLM_API_MODEL
                self.max_tokens = LLM_API_MAX_TOKENS
                self.llm = None
                print(f"[QueryReformulator] Инициализирован (API)")
            except ImportError:
                raise ImportError("Установите openai: pip install openai")
        else:
            # Локальный режим
            if not LLAMA_CPP_AVAILABLE:
                raise ImportError(
                    "llama-cpp-python не установлен. "
                    "Установите: pip install llama-cpp-python"
                )
            
            if llm_model_path is None:
                llm_model_path = str(MODELS_DIR / LLM_MODEL_FILE)
            
            print(f"[QueryReformulator] Загрузка LLM: {llm_model_path}")

            self.llm = Llama(
                model_path=llm_model_path,
                n_ctx=LLM_CONTEXT_SIZE,
                n_gpu_layers=LLM_GPU_LAYERS,
                n_batch=512,
                verbose=False
            )
            self.client = None
            print(f"[QueryReformulator] Инициализирован (локальный)")

        self.use_cache = use_cache
        if use_cache:
            self.cache_dir = Path(cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            print(f"[QueryReformulator] Кэш включен: {self.cache_dir}")

    def _extract_final_answer(self, text: str) -> str:
        """
        Извлечение финального ответа из reasoning моделей
        
        Reasoning модели (sherlock-think-alpha, deepseek-r1 и т.д.) возвращают
        reasoning процесс в тегах <think>, <think> и т.д.
        Нужно извлечь только финальный ответ.
        """
        import re
        
        if not text or not isinstance(text, str):
            return text
        
        original_text = text
        
        # Удаляем reasoning теги и их содержимое (разные варианты)
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<reasoning>.*?</reasoning>', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<think>.*', '', text, flags=re.DOTALL | re.IGNORECASE)  # незакрытый тег
        
        # Удаляем строки, начинающиеся с reasoning-маркеров
        lines = text.split('\n')
        cleaned_lines = []
        skip_reasoning = True
        reasoning_markers = [
            'хорошо', 'давайте', 'сначала', 'нужно', 'возможно', 'итак', 'теперь',
            'well', 'let', 'first', 'need', 'maybe', 'so', 'now', 'then'
        ]
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Пропускаем reasoning строки
            if skip_reasoning:
                line_lower = line.lower()
                # Проверяем начало строки
                if any(line_lower.startswith(marker) for marker in reasoning_markers):
                    continue
                # Проверяем если строка содержит только reasoning-текст
                if len(line) < 30 and any(marker in line_lower for marker in reasoning_markers):
                    continue
                # Если нашли что-то похожее на финальный ответ, начинаем собирать
                if len(line) > 15 and not line.startswith('<') and not any(marker in line_lower[:20] for marker in reasoning_markers):
                    skip_reasoning = False
            
            if not skip_reasoning:
                # Пропускаем строки, которые явно являются reasoning
                line_lower = line.lower()
                if len(line) < 30 and any(marker in line_lower for marker in reasoning_markers):
                    continue
                cleaned_lines.append(line)
        
        result = ' '.join(cleaned_lines).strip()
        
        # Если ничего не осталось или результат слишком короткий, пробуем другой подход
        if not result or len(result) < 10:
            # Пробуем взять последнюю значимую строку
            lines = original_text.split('\n')
            for line in reversed(lines):
                line = line.strip()
                if line and len(line) > 15:
                    line_lower = line.lower()
                    if not any(line_lower.startswith(marker) for marker in reasoning_markers):
                        result = line
                        break
        
        # Если все еще ничего, возвращаем оригинал (на случай если это не reasoning модель)
        if not result or len(result) < 5:
            result = original_text.strip()
        
        return result

    def _get_cache_key(self, query: str, method: str) -> str:
        """Генерация ключа кэша"""
        combined = f"{query}_{method}"
        return hashlib.md5(combined.encode()).hexdigest()

    def _load_from_cache(self, cache_key: str) -> Optional[str]:
        """Загрузка из кэша"""
        if not self.use_cache:
            return None

        cache_file = self.cache_dir / f"{cache_key}.pkl"
        if cache_file.exists():
            import logging
            logger = logging.getLogger(__name__)
            logger.debug(f"[QueryReformulator] Использован кэш для ключа {cache_key[:16]}...")
            with open(cache_file, 'rb') as f:
                cached_result = pickle.load(f)
                # Применяем извлечение финального ответа даже для кэшированных результатов
                # (на случай если кэш был создан до исправления парсинга)
                if isinstance(cached_result, str):
                    cleaned = self._extract_final_answer(cached_result)
                    # Если результат изменился после очистки, обновляем кэш
                    if cleaned != cached_result:
                        logger.debug(f"[QueryReformulator] Очистка кэшированного результата от reasoning")
                        self._save_to_cache(cache_key, cleaned)
                    return cleaned
                return cached_result
        return None

    def _save_to_cache(self, cache_key: str, result: str):
        """Сохранение в кэш"""
        if not self.use_cache:
            return

        cache_file = self.cache_dir / f"{cache_key}.pkl"
        with open(cache_file, 'wb') as f:
            pickle.dump(result, f)

    def reformulate_simple(self, query: str) -> str:
        """
        Простое переформулирование - конкретизация и добавление банковских терминов

        Args:
            query: исходный запрос

        Returns:
            переформулированный запрос
        """
        # Проверяем кэш
        cache_key = self._get_cache_key(query, "simple")
        cached = self._load_from_cache(cache_key)
        if cached:
            return cached

        prompt = f"""<|im_start|>system
Ты - эксперт по банковским запросам. Переформулируй вопрос пользователя для поиска в базе знаний Альфа-Банка.

Требования:
- Сделай запрос более конкретным
- Добавь банковские термины где уместно
- Сохрани смысл и основную тему
- Ответ должен быть один вопрос (не список)
- Максимум 15-20 слов<|im_end|>
<|im_start|>user
Исходный запрос: {query}

Переформулированный запрос:<|im_end|>
<|im_start|>assistant
"""

        try:
            if self.use_api:
                # API режим
                request_params = {
                    "model": self.model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.3,
                    "max_tokens": self.max_tokens  # используем LLM_API_MAX_TOKENS (нужно для reasoning моделей)
                }
                
                from src.config import LLM_API_ROUTING
                if LLM_API_ROUTING:
                    request_params["extra_headers"] = {"X-OpenRouter-Provider": LLM_API_ROUTING}
                
                # Логируем запрос
                import logging
                logger = logging.getLogger(__name__)
                logger.info(f"[QueryReformulator API] → Запрос к {self.model_name}")
                
                response = self.client.chat.completions.create(**request_params)
                raw_response = response.choices[0].message.content.strip()
                
                # Извлекаем финальный ответ из reasoning моделей
                reformulated = self._extract_final_answer(raw_response)
                
                logger.info(f"[QueryReformulator API] ← Ответ получен: {reformulated[:80]}...")
            else:
                # Локальный режим
                response = self.llm(
                    prompt,
                    max_tokens=100,
                    temperature=0.3,
                    stop=["<|im_end|>", "\n\n"],
                    echo=False
                )
                reformulated = response['choices'][0]['text'].strip()

            # Очистка от артефактов
            if reformulated.startswith('"') and reformulated.endswith('"'):
                reformulated = reformulated[1:-1]

            # Сохраняем в кэш
            self._save_to_cache(cache_key, reformulated)

            return reformulated

        except Exception as e:
            print(f"[QueryReformulator] Ошибка: {e}")
            return query  # fallback на исходный

    def reformulate_expanded(self, query: str) -> str:
        """
        Расширенное переформулирование - добавление контекста и деталей

        Args:
            query: исходный запрос

        Returns:
            расширенный запрос
        """
        cache_key = self._get_cache_key(query, "expanded")
        cached = self._load_from_cache(cache_key)
        if cached:
            return cached

        prompt = f"""<|im_start|>system
Ты - эксперт по банковским услугам. Расширь вопрос пользователя, добавив важный контекст для поиска.

Требования:
- Добавь контекст (например, "через мобильное приложение", "для физических лиц")
- Используй профессиональные термины
- Сохрани исходный смысл
- Сделай запрос более информативным
- Максимум 25-30 слов<|im_end|>
<|im_start|>user
Исходный запрос: {query}

Расширенный запрос:<|im_end|>
<|im_start|>assistant
"""

        try:
            if self.use_api:
                # API режим
                request_params = {
                    "model": self.model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.4,
                    "max_tokens": self.max_tokens  # используем LLM_API_MAX_TOKENS
                }
                
                from src.config import LLM_API_ROUTING
                if LLM_API_ROUTING:
                    request_params["extra_headers"] = {"X-OpenRouter-Provider": LLM_API_ROUTING}
                
                import logging
                logger = logging.getLogger(__name__)
                logger.info(f"[QueryReformulator API] → Запрос (expanded) к {self.model_name}")
                
                response = self.client.chat.completions.create(**request_params)
                raw_response = response.choices[0].message.content.strip()
                reformulated = self._extract_final_answer(raw_response)
                
                logger.info(f"[QueryReformulator API] ← Ответ (expanded): {reformulated[:80]}...")
            else:
                # Локальный режим
                response = self.llm(
                    prompt,
                    max_tokens=150,
                    temperature=0.4,
                    stop=["<|im_end|>", "\n\n"],
                    echo=False
                )
                reformulated = response['choices'][0]['text'].strip()

            if reformulated.startswith('"') and reformulated.endswith('"'):
                reformulated = reformulated[1:-1]

            self._save_to_cache(cache_key, reformulated)

            return reformulated

        except Exception as e:
            print(f"[QueryReformulator] Ошибка: {e}")
            return query

    def reformulate_multi_variant(self, query: str, num_variants: int = 3) -> List[str]:
        """
        Генерация нескольких вариантов переформулирования

        Args:
            query: исходный запрос
            num_variants: количество вариантов (обычно 2-3)

        Returns:
            список вариантов запроса (включая исходный)
        """
        cache_key = self._get_cache_key(query, f"multi_{num_variants}")
        cached = self._load_from_cache(cache_key)
        if cached:
            return cached

        prompt = f"""<|im_start|>system
Ты - эксперт по банковским запросам. Создай {num_variants} альтернативных формулировки вопроса для поиска в базе знаний банка.

Требования:
- Каждый вариант должен быть уникальным
- Используй разные банковские термины
- Сохрани основной смысл
- По одному варианту на строку без нумерации<|im_end|>
<|im_start|>user
Исходный запрос: {query}

Варианты:<|im_end|>
<|im_start|>assistant
"""

        try:
            if self.use_api:
                # API режим
                request_params = {
                    "model": self.model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.5,  # выше для разнообразия
                    "max_tokens": self.max_tokens  # используем LLM_API_MAX_TOKENS
                }
                
                from src.config import LLM_API_ROUTING
                if LLM_API_ROUTING:
                    request_params["extra_headers"] = {"X-OpenRouter-Provider": LLM_API_ROUTING}
                
                import logging
                logger = logging.getLogger(__name__)
                logger.info(f"[QueryReformulator API] → Запрос (multi) к {self.model_name}")
                
                response = self.client.chat.completions.create(**request_params)
                raw_response = response.choices[0].message.content.strip()
                result_text = self._extract_final_answer(raw_response)
                
                logger.info(f"[QueryReformulator API] ← Ответ (multi): {result_text[:80]}...")
            else:
                # Локальный режим
                response = self.llm(
                    prompt,
                    max_tokens=200,
                    temperature=0.5,  # выше для разнообразия
                    stop=["<|im_end|>"],
                    echo=False
                )
                result_text = response['choices'][0]['text'].strip()

            # Парсим варианты
            variants = [query]  # Всегда включаем исходный
            for line in result_text.split('\n'):
                line = line.strip()
                if line and len(line) > 5:
                    # Убираем нумерацию
                    if line[0].isdigit() and '.' in line:
                        line = line.split('.', 1)[-1].strip()

                    # Убираем кавычки
                    if line.startswith('"') and line.endswith('"'):
                        line = line[1:-1]

                    variants.append(line)

            # Ограничиваем количество
            variants = variants[:num_variants + 1]  # +1 для исходного

            # Убираем дубликаты
            variants = list(dict.fromkeys(variants))

            self._save_to_cache(cache_key, variants)

            return variants

        except Exception as e:
            print(f"[QueryReformulator] Ошибка: {e}")
            return [query]

    def reformulate(self, query: str, method: str = "simple") -> List[str]:
        """
        Универсальный метод переформулирования

        Args:
            query: исходный запрос
            method: метод переформулирования
                - "simple": простое (1 вариант)
                - "expanded": расширенное (1 вариант)
                - "multi": несколько вариантов (3 варианта)
                - "all": все методы (4+ вариантов)

        Returns:
            список вариантов запроса
        """
        if method == "simple":
            reformulated = self.reformulate_simple(query)
            return [query, reformulated]

        elif method == "expanded":
            reformulated = self.reformulate_expanded(query)
            return [query, reformulated]

        elif method == "multi":
            return self.reformulate_multi_variant(query, num_variants=2)

        elif method == "all":
            # Все варианты
            simple = self.reformulate_simple(query)
            expanded = self.reformulate_expanded(query)
            multi = self.reformulate_multi_variant(query, num_variants=2)

            # Объединяем и убираем дубликаты
            all_variants = [query, simple, expanded] + multi
            return list(dict.fromkeys(all_variants))

        else:
            print(f"[QueryReformulator] Неизвестный метод: {method}, используется 'simple'")
            return self.reformulate(query, method="simple")


def demonstrate_reformulation():
    """Демонстрация работы Query Reformulation"""
    print("="*80)
    print("ДЕМОНСТРАЦИЯ QUERY REFORMULATION")
    print("="*80)

    from src.config import MODELS_DIR, LLM_MODEL_FILE
    import sys

    llm_path = str(MODELS_DIR / LLM_MODEL_FILE)

    if not (MODELS_DIR / LLM_MODEL_FILE).exists():
        print(f"❌ LLM модель не найдена: {llm_path}")
        print("   Query Reformulation требует LLM")
        sys.exit(1)

    reformulator = QueryReformulator(llm_path, use_cache=True)

    test_queries = [
        "можно ли платить телефон",
        "кэшбэк по карте",
        "открыть счет",
        "перевод без комиссии",
        "как подключить онлайн банк",
    ]

    for query in test_queries:
        print(f"\n{'='*80}")
        print(f"📝 Исходный: {query}")
        print(f"{'='*80}")

        # Simple
        simple = reformulator.reformulate_simple(query)
        print(f"\n1️⃣  Simple (конкретизация):")
        print(f"   {simple}")

        # Expanded
        expanded = reformulator.reformulate_expanded(query)
        print(f"\n2️⃣  Expanded (с контекстом):")
        print(f"   {expanded}")

        # Multi-variant
        multi = reformulator.reformulate_multi_variant(query, num_variants=2)
        print(f"\n3️⃣  Multi-variant ({len(multi)} варианта):")
        for i, variant in enumerate(multi, 1):
            marker = "📌" if variant == query else "  "
            print(f"   {marker} {i}. {variant}")

    print(f"\n{'='*80}")
    print("✅ Query Reformulation готов!")
    print(f"{'='*80}")

    # Статистика кэша
    if reformulator.use_cache:
        cache_files = list(reformulator.cache_dir.glob("*.pkl"))
        print(f"\n💾 Кэш: {len(cache_files)} записей в {reformulator.cache_dir}")


def main():
    """Тест Query Reformulation"""
    demonstrate_reformulation()


if __name__ == "__main__":
    main()
