"""
Классификация запросов и динамический выбор TOP_K
"""
from typing import Tuple, Dict
import re


class QueryClassifier:
    """Классификация типа и сложности запроса"""

    # Типы запросов и ключевые слова
    QUERY_TYPES = {
        "product_info": {
            "keywords": ["что такое", "описание", "особенности", "про", "информация о"],
            "alpha": 0.7,  # больше вес embeddings
            "top_k": 20
        },
        "how_to": {
            "keywords": ["как", "каким образом", "способ", "инструкция", "порядок", "пошагово"],
            "alpha": 0.4,  # больше вес BM25
            "top_k": 25
        },
        "comparison": {
            "keywords": ["сравнить", "отличие", "разница", "лучше", "выгоднее", "какая карта"],
            "alpha": 0.5,
            "top_k": 30  # нужно больше результатов
        },
        "conditions": {
            "keywords": ["условия", "требования", "комиссия", "процент", "лимит", "тариф", "ставка"],
            "alpha": 0.6,
            "top_k": 20
        },
        "availability": {
            "keywords": ["можно", "доступно", "есть ли", "возможно ли", "могу ли"],
            "alpha": 0.5,
            "top_k": 15  # обычно короткий ответ
        },
        "troubleshooting": {
            "keywords": ["не работает", "ошибка", "проблема", "не получается", "почему не"],
            "alpha": 0.4,
            "top_k": 25
        },
        "yes_no": {
            "keywords": ["да или нет", "правда ли", "верно ли"],
            "alpha": 0.6,
            "top_k": 15  # простой вопрос
        }
    }

    # Маркеры сложности запроса
    COMPLEXITY_MARKERS = {
        "simple": {
            "keywords": ["можно", "есть ли", "сколько стоит", "когда"],
            "score": 1
        },
        "medium": {
            "keywords": ["как", "где", "какой", "какие"],
            "score": 2
        },
        "complex": {
            "keywords": [
                "сравнить", "в чем разница", "что выгоднее",
                "при каких условиях", "зависит ли", "какие факторы"
            ],
            "score": 3
        }
    }

    def __init__(self):
        pass

    def classify_type(self, query: str) -> Tuple[str, float, int]:
        """
        Определяет тип запроса

        Args:
            query: текст запроса

        Returns:
            (тип, alpha, top_k)
        """
        query_lower = query.lower()

        # Ищем совпадения с типами
        for query_type, config in self.QUERY_TYPES.items():
            for keyword in config["keywords"]:
                if keyword in query_lower:
                    return query_type, config["alpha"], config["top_k"]

        # По умолчанию - general
        return "general", 0.5, 25

    def classify_complexity(self, query: str) -> str:
        """
        Определяет сложность запроса

        Args:
            query: текст запроса

        Returns:
            "simple", "medium", "complex"
        """
        query_lower = query.lower()

        # Подсчитываем score
        total_score = 0
        matches = 0

        for complexity, config in self.COMPLEXITY_MARKERS.items():
            for keyword in config["keywords"]:
                if keyword in query_lower:
                    total_score += config["score"]
                    matches += 1
                    break  # один match per complexity level

        # Определяем сложность
        if matches == 0:
            return "medium"  # по умолчанию

        avg_score = total_score / matches

        if avg_score <= 1.5:
            return "simple"
        elif avg_score <= 2.5:
            return "medium"
        else:
            return "complex"

    def get_dynamic_top_k(self, query: str, default_top_k: int = 25) -> Dict[str, int]:
        """
        Динамически определяет TOP_K на основе типа и сложности запроса

        Args:
            query: текст запроса
            default_top_k: TOP_K по умолчанию

        Returns:
            dict с k_dense, k_bm25, k_rerank
        """
        # Классификация
        query_type, alpha, suggested_top_k = self.classify_type(query)
        complexity = self.classify_complexity(query)

        # Базовый TOP_K из типа запроса
        base_k = suggested_top_k

        # Корректировка на основе сложности
        if complexity == "simple":
            k_dense = int(base_k * 0.7)  # меньше результатов
            k_bm25 = int(base_k * 0.7)
            k_rerank = int(base_k * 0.6)
        elif complexity == "complex":
            k_dense = int(base_k * 1.3)  # больше результатов
            k_bm25 = int(base_k * 1.3)
            k_rerank = int(base_k * 1.2)
        else:
            # medium
            k_dense = base_k
            k_bm25 = base_k
            k_rerank = int(base_k * 0.8)

        return {
            "k_dense": max(10, min(50, k_dense)),  # ограничиваем 10-50
            "k_bm25": max(10, min(50, k_bm25)),
            "k_rerank": max(10, min(30, k_rerank)),
            "alpha": alpha,
            "query_type": query_type,
            "complexity": complexity
        }

    def analyze_query(self, query: str) -> Dict:
        """
        Полный анализ запроса

        Args:
            query: текст запроса

        Returns:
            dict с полной информацией
        """
        query_type, alpha, suggested_top_k = self.classify_type(query)
        complexity = self.classify_complexity(query)
        dynamic_k = self.get_dynamic_top_k(query)

        # Дополнительные признаки
        word_count = len(query.split())
        has_question_mark = '?' in query
        has_numbers = bool(re.search(r'\d+', query))

        return {
            "query": query,
            "query_type": query_type,
            "complexity": complexity,
            "word_count": word_count,
            "has_question_mark": has_question_mark,
            "has_numbers": has_numbers,
            "suggested_alpha": alpha,
            "suggested_top_k": suggested_top_k,
            **dynamic_k
        }


def main():
    """Тест классификатора"""
    classifier = QueryClassifier()

    test_queries = [
        "Какой кэшбэк по Альфа-Карте?",  # product_info, simple
        "Как оплатить ЖКХ без комиссии?",  # how_to, medium
        "В чем разница между дебетовой и кредитной картой?",  # comparison, complex
        "Можно ли снять наличные?",  # availability, simple
        "Какие условия для получения ипотеки?",  # conditions, medium
        "Не работает перевод, что делать?",  # troubleshooting, medium
        "Что такое А-Клуб?",  # product_info, simple
    ]

    print("=" * 80)
    print("ТЕСТ КЛАССИФИКАТОРА ЗАПРОСОВ")
    print("=" * 80)

    for query in test_queries:
        print(f"\n{'─' * 80}")
        print(f"Запрос: {query}")
        print(f"{'─' * 80}")

        analysis = classifier.analyze_query(query)

        print(f"Тип: {analysis['query_type']}")
        print(f"Сложность: {analysis['complexity']}")
        print(f"Слов: {analysis['word_count']}")
        print(f"Рекомендуемые параметры:")
        print(f"  TOP_K_DENSE: {analysis['k_dense']}")
        print(f"  TOP_K_BM25: {analysis['k_bm25']}")
        print(f"  TOP_K_RERANK: {analysis['k_rerank']}")
        print(f"  HYBRID_ALPHA: {analysis['alpha']:.2f}")

    # Статистика
    print(f"\n{'=' * 80}")
    print("СТАТИСТИКА")
    print("=" * 80)

    types_count = {}
    complexity_count = {}

    for query in test_queries:
        analysis = classifier.analyze_query(query)
        types_count[analysis['query_type']] = types_count.get(analysis['query_type'], 0) + 1
        complexity_count[analysis['complexity']] = complexity_count.get(analysis['complexity'], 0) + 1

    print("\nТипы запросов:")
    for qtype, count in types_count.items():
        print(f"  {qtype}: {count}")

    print("\nСложность:")
    for complexity, count in complexity_count.items():
        print(f"  {complexity}: {count}")


if __name__ == "__main__":
    main()
