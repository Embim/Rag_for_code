"""
Расширение запроса через LLM и сущности
"""
from llama_cpp import Llama
from pathlib import Path
from typing import List

from src.config import (
    LLM_MODEL_FILE,
    LLM_CONTEXT_SIZE,
    LLM_GPU_LAYERS,
    MODELS_DIR
)


class QueryExpander:
    """Расширение запроса для улучшения поиска"""

    # Словарь синонимов для банковских терминов
    BANK_SYNONYMS = {
        "кредитная карта": ["кредитка", "КК", "кред карта", "карта с кредитным лимитом"],
        "дебетовая карта": ["дебетка", "ДК", "зарплатная карта"],
        "кэшбэк": ["кешбек", "возврат денег", "бонусы", "cashback"],
        "жкх": ["коммунальные услуги", "коммуналка", "квартплата", "ком. услуги"],
        "перевод": ["отправить деньги", "перекинуть", "передать деньги", "транзакция"],
        "комиссия": ["процент", "плата", "сбор", "комса"],
        "онлайн-банк": ["мобильное приложение", "приложение банка", "альфа-онлайн"],
        "счет": ["расчетный счет", "р/с", "банковский счет", "счёт"],
    }

    def __init__(self, model_path: str = None, use_llm: bool = True):
        """
        Инициализация

        Args:
            model_path: путь к LLM модели
            use_llm: использовать ли LLM для расширения (или только словарь)
        """
        self.use_llm = use_llm

        if use_llm:
            if model_path is None:
                model_path = str(MODELS_DIR / LLM_MODEL_FILE)

            print(f"Загрузка LLM для Query Expansion: {model_path}")

            self.llm = Llama(
                model_path=model_path,
                n_ctx=LLM_CONTEXT_SIZE,
                n_gpu_layers=LLM_GPU_LAYERS,
                n_batch=512,
                verbose=False
            )

            print("  LLM загружена успешно")
        else:
            self.llm = None
            print("Query Expansion: только словарь синонимов")

    def expand_with_synonyms(self, query: str) -> List[str]:
        """
        Расширение через словарь синонимов

        Args:
            query: исходный запрос

        Returns:
            список вариантов запроса
        """
        query_lower = query.lower()
        expanded = [query]  # Исходный запрос

        # Ищем совпадения в словаре
        for term, synonyms in self.BANK_SYNONYMS.items():
            if term in query_lower:
                # Добавляем варианты с синонимами
                for syn in synonyms:
                    expanded.append(query_lower.replace(term, syn))

        return list(set(expanded))  # Убираем дубли

    def expand_with_llm(self, query: str) -> List[str]:
        """
        Расширение через LLM

        Args:
            query: исходный запрос

        Returns:
            список вариантов запроса
        """
        if not self.use_llm or self.llm is None:
            return [query]

        prompt = f"""<|im_start|>system
Ты - эксперт по банковским запросам. Перефразируй вопрос 3 способами, используя разные формулировки и синонимы.<|im_end|>
<|im_start|>user
Вопрос: {query}

Варианты (по одному на строку):
1.<|im_end|>
<|im_start|>assistant
1. """

        try:
            response = self.llm(
                prompt,
                max_tokens=200,
                temperature=0.3,
                stop=["<|im_end|>"],
                echo=False
            )

            result_text = response['choices'][0]['text'].strip()

            # Парсим варианты
            variants = [query]  # Исходный
            for line in result_text.split('\n'):
                line = line.strip()
                # Убираем нумерацию
                if line and line[0].isdigit():
                    variant = line.split('.', 1)[-1].strip()
                    if variant:
                        variants.append(variant)

            return variants[:4]  # Макс 4 варианта

        except Exception as e:
            print(f"  Ошибка LLM expansion: {e}")
            return [query]

    def expand_query(self, query: str, method: str = "hybrid") -> List[str]:
        """
        Полное расширение запроса

        Args:
            query: исходный запрос
            method: способ расширения
                - "synonyms": только словарь
                - "llm": только LLM
                - "hybrid": оба способа (рекомендуется)

        Returns:
            список вариантов запроса
        """
        expanded = set([query])  # Исходный запрос

        if method in ["synonyms", "hybrid"]:
            # Расширение через словарь
            syn_variants = self.expand_with_synonyms(query)
            expanded.update(syn_variants)

        if method in ["llm", "hybrid"] and self.use_llm:
            # Расширение через LLM
            llm_variants = self.expand_with_llm(query)
            expanded.update(llm_variants)

        return list(expanded)


def main():
    """Тест Query Expansion"""
    expander = QueryExpander(use_llm=True)

    test_queries = [
        "Когда смогу пользоваться кредитной картой?",
        "Как оплатить ЖКХ без комиссии?",
        "Где посмотреть реквизиты счета?",
    ]

    for query in test_queries:
        print("\n" + "="*80)
        print(f"Исходный запрос: {query}")
        print("="*80)

        # Только словарь
        syn_variants = expander.expand_query(query, method="synonyms")
        print(f"\nС словарем синонимов ({len(syn_variants)}):")
        for i, v in enumerate(syn_variants, 1):
            print(f"  {i}. {v}")

        # С LLM
        all_variants = expander.expand_query(query, method="hybrid")
        print(f"\nС LLM ({len(all_variants)}):")
        for i, v in enumerate(all_variants, 1):
            print(f"  {i}. {v}")


if __name__ == "__main__":
    main()
