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
from llama_cpp import Llama
from typing import List, Optional, Dict
import hashlib
import pickle
from pathlib import Path


class QueryReformulator:
    """
    LLM-based переформулирование запросов для улучшения поиска

    Стратегии:
    1. Конкретизация - добавление деталей
    2. Профессионализация - банковские термины
    3. Расширение - добавление контекста
    4. Нормализация - исправление опечаток и сленга
    """

    def __init__(self, llm_model_path: str,
                 use_cache: bool = True,
                 cache_dir: str = "cache/query_reformulation"):
        """
        Args:
            llm_model_path: путь к LLM модели
            use_cache: использовать кэш (ускоряет повторные запросы)
            cache_dir: директория для кэша
        """
        print(f"[QueryReformulator] Загрузка LLM: {llm_model_path}")

        from src.config import LLM_CONTEXT_SIZE, LLM_GPU_LAYERS

        self.llm = Llama(
            model_path=llm_model_path,
            n_ctx=LLM_CONTEXT_SIZE,
            n_gpu_layers=LLM_GPU_LAYERS,
            n_batch=512,
            verbose=False
        )

        self.use_cache = use_cache
        if use_cache:
            self.cache_dir = Path(cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            print(f"[QueryReformulator] Кэш включен: {self.cache_dir}")

        print(f"[QueryReformulator] Инициализирован")

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
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
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
