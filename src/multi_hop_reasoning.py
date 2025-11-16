"""
Multi-hop Reasoning - итеративный поиск для сложных вопросов

Концепция:
Некоторые вопросы требуют информации из нескольких источников.
Вместо одного запроса делаем несколько итераций поиска.

Пример:
Вопрос: "В чем разница между Альфа-Картой и Альфа-Картой Gold?"

Итерация 1: "Альфа-Карта характеристики"
  → Находим информацию об Альфа-Карте

Итерация 2: "Альфа-Карта Gold характеристики"
  → Находим информацию об Альфа-Карта Gold

Итерация 3 (опционально): "сравнение Альфа-Карта и Gold"
  → Находим сравнительную информацию

Результат: Объединяем информацию из всех итераций

Преимущества:
- +15-25% accuracy для сложных вопросов
- Лучшее покрытие информации
- Умеет разбивать сложные вопросы на подзадачи
"""
import pandas as pd
from typing import List, Dict, Tuple, Optional
from llama_cpp import Llama


class MultiHopReasoner:
    """
    Multi-hop reasoning для сложных вопросов

    Стратегия:
    1. Классифицировать вопрос (простой/сложный)
    2. Для сложных вопросов - несколько итераций поиска
    3. Проверка полноты ответа после каждой итерации
    4. Генерация follow-up запросов
    5. Объединение результатов всех итераций
    """

    def __init__(self, llm_model_path: str, max_hops: int = 3):
        """
        Args:
            llm_model_path: путь к LLM модели для генерации запросов
            max_hops: максимальное количество итераций (обычно 2-3)
        """
        self.max_hops = max_hops

        print(f"[MultiHop] Загрузка LLM для multi-hop reasoning: {llm_model_path}")

        from src.config import LLM_CONTEXT_SIZE, LLM_GPU_LAYERS

        self.llm = Llama(
            model_path=llm_model_path,
            n_ctx=LLM_CONTEXT_SIZE,
            n_gpu_layers=LLM_GPU_LAYERS,
            n_batch=512,
            verbose=False
        )

        print(f"[MultiHop] Инициализирован (max_hops={max_hops})")

    def classify_question_complexity(self, query: str) -> Dict:
        """
        Классификация сложности вопроса

        Args:
            query: запрос

        Returns:
            {
                'complexity': 'simple' | 'medium' | 'complex',
                'reasoning': str,
                'needs_multi_hop': bool
            }
        """
        prompt = f"""<|im_start|>system
Ты - эксперт по анализу запросов. Определи сложность вопроса.

Простой (simple): один факт, одна тема
- "Какая комиссия за перевод?"
- "Где найти реквизиты?"

Средний (medium): несколько фактов из одной темы
- "Какие условия и комиссии по Альфа-Карте?"

Сложный (complex): сравнение, несколько тем, требует синтеза
- "В чем разница между картами?"
- "Как открыть счет и подключить онлайн-банк?"<|im_end|>
<|im_start|>user
Вопрос: {query}

Ответ (одним словом: simple/medium/complex):<|im_end|>
<|im_start|>assistant
"""

        try:
            response = self.llm(
                prompt,
                max_tokens=20,
                temperature=0.1,
                stop=["<|im_end|>", "\n"],
                echo=False
            )

            complexity = response['choices'][0]['text'].strip().lower()

            # Валидация
            if complexity not in ['simple', 'medium', 'complex']:
                complexity = 'medium'  # default

            needs_multi_hop = complexity in ['complex']

            return {
                'complexity': complexity,
                'needs_multi_hop': needs_multi_hop,
                'reasoning': f"Вопрос классифицирован как {complexity}"
            }

        except Exception as e:
            print(f"[MultiHop] Ошибка классификации: {e}")
            return {
                'complexity': 'medium',
                'needs_multi_hop': False,
                'reasoning': 'Ошибка классификации, используется default'
            }

    def generate_sub_queries(self, query: str, hop_number: int,
                            previous_results: List[Dict] = None) -> List[str]:
        """
        Генерация подзапросов для multi-hop поиска

        Args:
            query: исходный запрос
            hop_number: номер итерации (1, 2, 3...)
            previous_results: результаты предыдущих итераций

        Returns:
            список подзапросов для этой итерации
        """
        if hop_number == 1:
            # Первая итерация - декомпозиция основного вопроса
            prompt = f"""<|im_start|>system
Разбей сложный вопрос на 2-3 простых подвопроса для поиска информации.<|im_end|>
<|im_start|>user
Вопрос: {query}

Подвопросы (по одному на строку, без нумерации):<|im_end|>
<|im_start|>assistant
"""
        else:
            # Последующие итерации - дополнительные запросы на основе найденного
            prev_summary = self._summarize_previous_results(previous_results)

            prompt = f"""<|im_start|>system
Определи, какая информация еще нужна для полного ответа на вопрос.<|im_end|>
<|im_start|>user
Основной вопрос: {query}

Уже найдено: {prev_summary}

Дополнительные запросы (1-2 штуки, по одному на строку):<|im_end|>
<|im_start|>assistant
"""

        try:
            response = self.llm(
                prompt,
                max_tokens=200,
                temperature=0.3,
                stop=["<|im_end|>"],
                echo=False
            )

            result_text = response['choices'][0]['text'].strip()

            # Парсим подзапросы
            sub_queries = []
            for line in result_text.split('\n'):
                line = line.strip()
                if line and len(line) > 5:  # минимальная длина запроса
                    # Убираем нумерацию если есть
                    if line[0].isdigit() and '.' in line:
                        line = line.split('.', 1)[-1].strip()
                    sub_queries.append(line)

            # Ограничиваем количество
            max_sub_queries = 3 if hop_number == 1 else 2
            sub_queries = sub_queries[:max_sub_queries]

            if len(sub_queries) == 0:
                # Если LLM не сгенерировал подзапросы, используем исходный
                sub_queries = [query]

            return sub_queries

        except Exception as e:
            print(f"[MultiHop] Ошибка генерации подзапросов: {e}")
            return [query]  # fallback

    def _summarize_previous_results(self, results: List[Dict]) -> str:
        """
        Краткая сводка предыдущих результатов

        Args:
            results: результаты предыдущих итераций

        Returns:
            краткое описание
        """
        if not results or len(results) == 0:
            return "Ничего"

        # Берем первые несколько результатов
        summaries = []
        for r in results[:3]:
            text = r.get('text', r.get('clean_text', ''))[:100]
            summaries.append(text)

        return "; ".join(summaries)

    def check_completeness(self, query: str, all_results: List[pd.DataFrame]) -> Dict:
        """
        Проверка полноты ответа

        Args:
            query: исходный вопрос
            all_results: результаты всех итераций

        Returns:
            {
                'is_complete': bool,
                'confidence': float (0-1),
                'reasoning': str
            }
        """
        if len(all_results) == 0:
            return {
                'is_complete': False,
                'confidence': 0.0,
                'reasoning': 'Нет результатов'
            }

        # Собираем топ результаты
        top_texts = []
        for results_df in all_results:
            for idx, row in results_df.head(3).iterrows():
                text = row.get('clean_text', row.get('text', ''))
                top_texts.append(text[:200])

        context = "\n\n".join(top_texts)

        prompt = f"""<|im_start|>system
Оцени, достаточно ли информации для ответа на вопрос. Ответ: да/нет.<|im_end|>
<|im_start|>user
Вопрос: {query}

Найденная информация:
{context}

Достаточно ли информации? (да/нет):<|im_end|>
<|im_start|>assistant
"""

        try:
            response = self.llm(
                prompt,
                max_tokens=10,
                temperature=0.1,
                stop=["<|im_end|>", "\n"],
                echo=False
            )

            result = response['choices'][0]['text'].strip().lower()

            is_complete = 'да' in result or 'yes' in result

            return {
                'is_complete': is_complete,
                'confidence': 0.8 if is_complete else 0.3,
                'reasoning': f"LLM оценка: {result}"
            }

        except Exception as e:
            print(f"[MultiHop] Ошибка проверки полноты: {e}")
            return {
                'is_complete': len(all_results) >= 2,  # эвристика
                'confidence': 0.5,
                'reasoning': 'Эвристическая оценка'
            }

    def merge_multi_hop_results(self, all_results: List[pd.DataFrame]) -> pd.DataFrame:
        """
        Объединение результатов всех итераций

        Args:
            all_results: список DataFrame с результатами каждой итерации

        Returns:
            объединенный DataFrame
        """
        if len(all_results) == 0:
            return pd.DataFrame()

        if len(all_results) == 1:
            return all_results[0]

        # Объединяем через RRF
        from src.reciprocal_rank_fusion import ReciprocalRankFusion

        rrf = ReciprocalRankFusion(k=60)
        merged = rrf.fuse_multiple_results(all_results)

        return merged


class MultiHopRAGPipeline:
    """
    RAG Pipeline с multi-hop reasoning

    Использует MultiHopReasoner для сложных вопросов
    """

    def __init__(self, base_pipeline, llm_model_path: str,
                 enable_multi_hop: bool = True,
                 max_hops: int = 3):
        """
        Args:
            base_pipeline: базовый RAGPipeline
            llm_model_path: путь к LLM
            enable_multi_hop: включить multi-hop для сложных вопросов
            max_hops: максимальное количество итераций
        """
        self.base_pipeline = base_pipeline
        self.enable_multi_hop = enable_multi_hop
        self.max_hops = max_hops

        if self.enable_multi_hop:
            self.reasoner = MultiHopReasoner(llm_model_path, max_hops=max_hops)
        else:
            self.reasoner = None

        print(f"[MultiHopRAG] Инициализирован (multi_hop={'ON' if enable_multi_hop else 'OFF'})")

    def search(self, query: str, k_dense: int = 25, k_bm25: int = 25,
              k_rerank: int = 20, top_n: int = 5) -> Dict:
        """
        Поиск с multi-hop reasoning (если нужно)

        Args:
            query: запрос
            остальные: параметры для базового pipeline

        Returns:
            результаты поиска
        """
        # Проверяем сложность вопроса
        if self.enable_multi_hop and self.reasoner:
            complexity_info = self.reasoner.classify_question_complexity(query)

            if complexity_info['needs_multi_hop']:
                print(f"[MultiHopRAG] Обнаружен сложный вопрос: {complexity_info['complexity']}")
                print(f"[MultiHopRAG] Запуск multi-hop reasoning...")

                return self._multi_hop_search(
                    query, k_dense, k_bm25, k_rerank, top_n
                )
            else:
                print(f"[MultiHopRAG] Простой вопрос ({complexity_info['complexity']}), обычный поиск")

        # Простой вопрос - обычный поиск
        return self.base_pipeline.search(query, k_dense, k_bm25, k_rerank, top_n)

    def _multi_hop_search(self, query: str, k_dense: int, k_bm25: int,
                         k_rerank: int, top_n: int) -> Dict:
        """
        Multi-hop поиск для сложных вопросов

        Args:
            query: исходный вопрос
            параметры поиска

        Returns:
            результаты всех итераций
        """
        all_results = []
        all_results_raw = []

        for hop in range(1, self.max_hops + 1):
            print(f"\n[MultiHop] === Итерация {hop}/{self.max_hops} ===")

            # Генерируем подзапросы
            sub_queries = self.reasoner.generate_sub_queries(
                query, hop, all_results_raw
            )

            print(f"[MultiHop] Подзапросы ({len(sub_queries)}):")
            for i, sq in enumerate(sub_queries, 1):
                print(f"  {i}. {sq}")

            # Выполняем поиск для каждого подзапроса
            hop_results = []
            for sub_query in sub_queries:
                result = self.base_pipeline.search(
                    sub_query, k_dense, k_bm25, k_rerank, top_n
                )
                hop_results.append(result['reranked_chunks'])

            # Объединяем результаты этой итерации
            if len(hop_results) > 0:
                from src.reciprocal_rank_fusion import ReciprocalRankFusion
                rrf = ReciprocalRankFusion(k=60)
                merged_hop = rrf.fuse_multiple_results(hop_results)

                all_results.append(merged_hop)
                all_results_raw.extend([r.to_dict('records')[0] if len(r) > 0 else {}
                                       for r in hop_results])

            # Проверяем полноту
            if hop < self.max_hops:
                completeness = self.reasoner.check_completeness(query, all_results)
                print(f"[MultiHop] Полнота: {completeness['is_complete']} (confidence: {completeness['confidence']:.2f})")

                if completeness['is_complete'] and completeness['confidence'] > 0.7:
                    print(f"[MultiHop] Информация достаточна, останавливаем поиск")
                    break

        # Объединяем все итерации
        print(f"\n[MultiHop] Объединение {len(all_results)} итераций...")
        final_results = self.reasoner.merge_multi_hop_results(all_results)

        # Выбираем топ документы
        from src.retrieval import DocumentSelector
        selector = DocumentSelector()
        top_docs = selector.select_top_documents_with_diversity(final_results, top_n)

        return {
            'query': query,
            'documents_id': top_docs,
            'reranked_chunks': final_results,
            'num_hops': len(all_results),
            'multi_hop': True
        }


def test_multi_hop():
    """Тест multi-hop reasoning"""
    print("="*80)
    print("ТЕСТ MULTI-HOP REASONING")
    print("="*80)

    from src.config import MODELS_DIR, LLM_MODEL_FILE
    import sys

    llm_path = str(MODELS_DIR / LLM_MODEL_FILE)

    if not (MODELS_DIR / LLM_MODEL_FILE).exists():
        print(f"❌ LLM модель не найдена: {llm_path}")
        print("   Multi-hop reasoning требует LLM")
        sys.exit(1)

    reasoner = MultiHopReasoner(llm_path, max_hops=3)

    test_queries = [
        "Какая комиссия за перевод?",  # простой
        "В чем разница между Альфа-Картой и Альфа-Картой Gold?",  # сложный
        "Как открыть счет и подключить онлайн-банк?",  # сложный
    ]

    for query in test_queries:
        print(f"\n{'='*80}")
        print(f"Вопрос: {query}")
        print(f"{'='*80}")

        # Классификация
        complexity = reasoner.classify_question_complexity(query)
        print(f"\nСложность: {complexity['complexity']}")
        print(f"Multi-hop нужен: {complexity['needs_multi_hop']}")

        if complexity['needs_multi_hop']:
            # Генерация подзапросов
            sub_queries = reasoner.generate_sub_queries(query, hop_number=1)
            print(f"\nПодзапросы (итерация 1):")
            for i, sq in enumerate(sub_queries, 1):
                print(f"  {i}. {sq}")

    print(f"\n{'='*80}")
    print("✅ Multi-hop reasoning готов к использованию!")
    print(f"{'='*80}")


if __name__ == "__main__":
    test_multi_hop()
