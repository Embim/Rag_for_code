"""
Фильтрация результатов поиска по метаданным
Использует products, topics, conditions извлеченные при предочистке
"""
import pandas as pd
import json
import re
from typing import List, Optional


class MetadataFilter:
    """Фильтрация документов по метаданным"""

    # Словарь типов продуктов/услуг
    PRODUCT_KEYWORDS = {
        "кредитные_карты": ["кредитная карта", "кредитка", "кк", "альфа-карта", "100 дней"],
        "дебетовые_карты": ["дебетовая карта", "дебетка", "дк", "зарплатная карта"],
        "счета": ["расчетный счет", "р/с", "счет", "счёт", "реквизиты"],
        "переводы": ["перевод", "отправить деньги", "перекинуть"],
        "жкх": ["жкх", "коммунальные услуги", "коммуналка", "квартплата"],
        "кэшбэк": ["кэшбэк", "кешбек", "возврат", "бонусы", "cashback"],
        "ипотека": ["ипотека", "кредит на жилье", "квартира в кредит"],
        "вклады": ["вклад", "депозит", "накопительный счет"],
        "альфа_онлайн": ["альфа-онлайн", "мобильное приложение", "приложение"],
    }

    # Ключевые слова для определения intent
    INTENT_KEYWORDS = {
        "how_to": ["как", "каким образом", "способ", "инструкция", "порядок"],
        "conditions": ["условия", "требования", "комиссия", "процент", "лимит", "тариф"],
        "comparison": ["сравнить", "отличие", "разница", "лучше", "выгоднее"],
        "availability": ["можно", "доступно", "есть ли", "возможно ли"],
        "troubleshooting": ["не работает", "ошибка", "проблема", "не получается"],
    }

    def __init__(self):
        pass

    def extract_products_from_query(self, query: str) -> List[str]:
        """
        Извлекает упоминания продуктов из запроса

        Args:
            query: текст запроса

        Returns:
            список названий продуктов
        """
        query_lower = query.lower()
        found_products = []

        for product_type, keywords in self.PRODUCT_KEYWORDS.items():
            for keyword in keywords:
                if keyword in query_lower:
                    found_products.append(product_type)
                    break  # один продукт найден, переходим к следующему

        return list(set(found_products))

    def detect_intent(self, query: str) -> str:
        """
        Определяет intent (намерение) запроса

        Args:
            query: текст запроса

        Returns:
            тип intent или "general"
        """
        query_lower = query.lower()

        for intent_type, keywords in self.INTENT_KEYWORDS.items():
            if any(kw in query_lower for kw in keywords):
                return intent_type

        return "general"

    def filter_by_products(self, results_df: pd.DataFrame,
                          required_products: List[str],
                          min_overlap: int = 1) -> pd.DataFrame:
        """
        Фильтрация по продуктам

        Args:
            results_df: DataFrame с результатами поиска
            required_products: список требуемых продуктов
            min_overlap: минимальное количество совпадающих продуктов

        Returns:
            отфильтрованный DataFrame
        """
        if not required_products or 'products' not in results_df.columns:
            return results_df

        def check_products(products_json: str) -> bool:
            try:
                if pd.isna(products_json):
                    return False

                products = json.loads(products_json) if isinstance(products_json, str) else products_json

                if not products:
                    return False

                # Нормализуем
                products_lower = [p.lower() for p in products]

                # Проверяем overlap
                matches = sum(1 for req in required_products
                            if any(req.lower() in p for p in products_lower))

                return matches >= min_overlap

            except:
                return False

        mask = results_df['products'].apply(check_products)
        filtered = results_df[mask].copy()

        return filtered if len(filtered) > 0 else results_df  # fallback к исходным

    def filter_by_topics(self, results_df: pd.DataFrame,
                        required_topics: List[str],
                        min_overlap: int = 1) -> pd.DataFrame:
        """
        Фильтрация по темам

        Args:
            results_df: DataFrame с результатами поиска
            required_topics: список требуемых тем
            min_overlap: минимальное количество совпадающих тем

        Returns:
            отфильтрованный DataFrame
        """
        if not required_topics or 'topics' not in results_df.columns:
            return results_df

        def check_topics(topics_json: str) -> bool:
            try:
                if pd.isna(topics_json):
                    return False

                topics = json.loads(topics_json) if isinstance(topics_json, str) else topics_json

                if not topics:
                    return False

                # Проверяем overlap
                matches = sum(1 for req in required_topics if req.lower() in [t.lower() for t in topics])
                return matches >= min_overlap

            except:
                return False

        mask = results_df['topics'].apply(check_topics)
        filtered = results_df[mask].copy()

        return filtered if len(filtered) > 0 else results_df

    def auto_filter(self, query: str, results_df: pd.DataFrame,
                   boost_score: float = 1.2) -> pd.DataFrame:
        """
        Автоматическая фильтрация на основе запроса

        Args:
            query: текст запроса
            results_df: результаты поиска
            boost_score: коэффициент усиления score для совпавших документов

        Returns:
            отфильтрованный и переранкированный DataFrame
        """
        # Извлекаем продукты из запроса
        query_products = self.extract_products_from_query(query)

        if not query_products:
            return results_df

        # Boost документы с нужными продуктами
        def boost_if_match(row):
            try:
                if pd.isna(row.get('products')):
                    return row['retrieval_score']

                products = json.loads(row['products']) if isinstance(row['products'], str) else row['products']
                products_lower = [p.lower() for p in products]

                # Проверяем совпадение
                has_match = any(
                    qp.lower() in p for qp in query_products for p in products_lower
                )

                return row['retrieval_score'] * boost_score if has_match else row['retrieval_score']

            except:
                return row['retrieval_score']

        if 'retrieval_score' in results_df.columns and 'products' in results_df.columns:
            results_df = results_df.copy()
            results_df['retrieval_score'] = results_df.apply(boost_if_match, axis=1)
            results_df = results_df.sort_values('retrieval_score', ascending=False)

        return results_df


def main():
    """Тест фильтрации по метаданным"""
    # Тестовые данные
    test_data = {
        'chunk_id': [1, 2, 3, 4, 5],
        'text': [
            'Альфа-Карта дает 2% кэшбэк на все покупки',
            'Как оплатить ЖКХ без комиссии в приложении',
            'Условия получения кредитной карты',
            'Процентные ставки по вкладам',
            'Инструкция по переводу денег'
        ],
        'products': [
            '["Альфа-Карта"]',
            '["Альфа-Онлайн"]',
            '["Кредитная карта"]',
            '["Вклады"]',
            '[]'
        ],
        'topics': [
            '["кэшбэк", "кредитные_карты"]',
            '["жкх", "мобильное_приложение"]',
            '["кредитные_карты", "условия"]',
            '["вклады", "проценты"]',
            '["переводы"]'
        ],
        'retrieval_score': [0.9, 0.8, 0.85, 0.7, 0.75]
    }

    results_df = pd.DataFrame(test_data)

    filter = MetadataFilter()

    # Тест 1: Извлечение продуктов
    print("=" * 80)
    print("ТЕСТ 1: Извлечение продуктов из запроса")
    print("=" * 80)

    test_queries = [
        "Какой кэшбэк по Альфа-Карте?",
        "Как оплатить ЖКХ?",
        "Условия кредитной карты"
    ]

    for query in test_queries:
        products = filter.extract_products_from_query(query)
        print(f"\nЗапрос: {query}")
        print(f"Найденные продукты: {products}")

    # Тест 2: Фильтрация по продуктам
    print("\n" + "=" * 80)
    print("ТЕСТ 2: Фильтрация по продуктам")
    print("=" * 80)

    filtered = filter.filter_by_products(results_df, ["кредитная карта"])
    print(f"\nИсходно документов: {len(results_df)}")
    print(f"После фильтрации (кредитная карта): {len(filtered)}")
    print(filtered[['text', 'products']])

    # Тест 3: Auto-filter с boost
    print("\n" + "=" * 80)
    print("ТЕСТ 3: Auto-filter с boost score")
    print("=" * 80)

    query = "Какой кэшбэк по Альфа-Карте?"
    print(f"\nЗапрос: {query}")
    print("\nДо auto-filter:")
    print(results_df[['text', 'retrieval_score']])

    boosted = filter.auto_filter(query, results_df, boost_score=1.5)
    print("\nПосле auto-filter:")
    print(boosted[['text', 'retrieval_score']])


if __name__ == "__main__":
    main()
