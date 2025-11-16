"""
Hybrid LLM-based evaluator для оценки качества RAG системы

Комбинирует:
1. Косинусное расстояние (semantic similarity) - базовая быстрая метрика
2. LLM as Judge (RAGAS-style) - глубокая оценка релевантности

Используется для grid search оптимизации на выборке 50-70 вопросов
"""
import json
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import os

# Фикс для CUDA путей перед импортом llama_cpp
if os.name == 'nt':  # Windows
    cuda_path = os.environ.get('CUDA_PATH', '')

    if cuda_path and not os.path.exists(cuda_path):
        cuda_base = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
        if os.path.exists(cuda_base):
            versions = sorted([d for d in os.listdir(cuda_base) if d.startswith('v')])
            if versions:
                correct_path = os.path.join(cuda_base, versions[0])
                os.environ['CUDA_PATH'] = correct_path
            else:
                del os.environ['CUDA_PATH']
        else:
            del os.environ['CUDA_PATH']

try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except Exception as e:
    LLAMA_CPP_AVAILABLE = False
    print(f"Warning: llama-cpp-python не загружен: {e}")

from src.config import (
    LLM_MODEL_FILE,
    LLM_CONTEXT_SIZE,
    LLM_TEMPERATURE,
    LLM_GPU_LAYERS,
    LLM_MODE,
    LLM_API_MODEL,
    LLM_API_ROUTING,
    OPENROUTER_API_KEY,
    MODELS_DIR
)
from src.logger import get_logger


class HybridRAGEvaluator:
    """
    Гибридный evaluator для оценки качества retrieval

    Комбинирует косинусное расстояние + LLM as Judge

    Метрики:
    1. Semantic Score (косинусное расстояние) - avg similarity scores
    2. Context Relevance (LLM) - средняя релевантность чанков
    3. Context Precision (LLM) - доля релевантных чанков
    4. Context Sufficiency (LLM) - достаточность для ответа
    5. Hybrid Score - взвешенная комбинация всех метрик
    """

    def __init__(
        self,
        llm_model_path: Optional[str] = None,
        use_llm: bool = True,
        semantic_weight: float = 0.3,
        llm_weight: float = 0.7,
        use_api: bool = None
    ):
        """
        Args:
            llm_model_path: путь к LLM модели для оценки (для локального режима)
            use_llm: использовать ли LLM Judge (если False - только косинусное)
            semantic_weight: вес косинусного расстояния в итоговой метрике
            llm_weight: вес LLM метрик в итоговой метрике
            use_api: использовать ли API (если None - определяется из LLM_MODE)
        """
        self.logger = get_logger(__name__)
        
        # Определяем режим работы
        if use_api is None:
            use_api = (LLM_MODE == "api")
        
        self.use_api = use_api
        
        # Логируем режим работы
        self.logger.info(f"[HybridRAGEvaluator] Режим работы: {'API' if use_api else 'ЛОКАЛЬНЫЙ'}")
        if use_api:
            self.logger.info(f"[HybridRAGEvaluator] API модель: {LLM_API_MODEL}")
        else:
            self.logger.info(f"[HybridRAGEvaluator] Локальная модель: {LLM_MODEL_FILE}")
        
        self.use_llm = use_llm and (LLAMA_CPP_AVAILABLE or use_api)
        self.semantic_weight = semantic_weight
        self.llm_weight = llm_weight

        # Инициализация LLM Judge (если включено)
        self.llm = None
        if self.use_llm:
            try:
                if use_api:
                    # API режим (OpenRouter)
                    self.logger.info(f"Инициализация LLM Evaluator (API режим, модель: {LLM_API_MODEL})")
                    try:
                        from openai import OpenAI
                        base_url = "https://openrouter.ai/api/v1"
                        
                        # OpenRouter требует API ключ даже для бесплатных моделей
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
                        
                        # Добавляем провайдера для роутинга (если указан)
                        if LLM_API_ROUTING:
                            default_headers["X-OpenRouter-Provider"] = LLM_API_ROUTING
                        
                        self.client = OpenAI(
                            base_url=base_url,
                            api_key=OPENROUTER_API_KEY,
                            timeout=60,
                            default_headers=default_headers
                        )
                        self.model_name = LLM_API_MODEL
                        self.logger.info(f"LLM Evaluator (API) инициализирован")
                    except ImportError:
                        raise ImportError("Установите openai: pip install openai")
                else:
                    # Локальный режим
                    if llm_model_path is None:
                        llm_model_path = MODELS_DIR / LLM_MODEL_FILE

                    llm_model_path = Path(llm_model_path)

                    if not llm_model_path.exists():
                        self.logger.warning(f"LLM модель не найдена: {llm_model_path}")
                        self.logger.warning("LLM Judge отключен, используется только косинусное расстояние")
                        self.use_llm = False
                    else:
                        self.logger.info(f"Загрузка LLM модели для оценки: {llm_model_path}")
                        self.llm = Llama(
                            model_path=str(llm_model_path),
                            n_ctx=LLM_CONTEXT_SIZE,
                            n_gpu_layers=LLM_GPU_LAYERS,
                            verbose=False
                        )
                        self.logger.info(f"LLM Evaluator загружен (GPU layers: {LLM_GPU_LAYERS})")
            except Exception as e:
                self.logger.warning(f"Ошибка загрузки LLM Judge: {e}")
                self.logger.warning("LLM Judge отключен")
                self.use_llm = False

        self.logger.info(f"Hybrid Evaluator: semantic_weight={semantic_weight}, llm_weight={llm_weight}, api={use_api}")

    def evaluate_retrieval(
        self,
        query: str,
        retrieved_results: pd.DataFrame,
        top_k: int = 20
    ) -> Dict[str, float]:
        """
        Комплексная оценка качества retrieval

        Args:
            query: вопрос пользователя
            retrieved_results: DataFrame с результатами (должен содержать 'text' и 'final_score')
            top_k: количество топовых документов для оценки

        Returns:
            dict с метриками:
            {
                'semantic_score': float (0-1) - средний cosine similarity,
                'context_relevance': float (0-1) - LLM оценка релевантности,
                'context_precision': float (0-1) - LLM доля релевантных,
                'context_sufficiency': float (0-1) - LLM достаточность,
                'hybrid_score': float (0-1) - итоговая комбинированная метрика
            }
        """
        # Берем топ-K
        top_results = retrieved_results.head(top_k)

        # 1. Semantic Score (из final_score - косинусное расстояние)
        if 'final_score' in top_results.columns:
            # Нормализуем к 0-1 (если это косинус, он уже 0-1)
            semantic_scores = top_results['final_score'].tolist()
            semantic_score = np.mean(semantic_scores) if semantic_scores else 0.0
        else:
            self.logger.warning("Колонка 'final_score' не найдена, semantic_score=0")
            semantic_score = 0.0

        # 2. LLM-based метрики (если включено)
        llm_metrics = {
            'context_relevance': 0.0,
            'context_precision': 0.0,
            'context_sufficiency': 0.0
        }

        if self.use_llm and self.llm:
            chunks = top_results['text'].tolist()
            llm_metrics = self._evaluate_with_llm(query, chunks)

        # 3. Hybrid Score - взвешенная комбинация
        if self.use_llm:
            # Комбинируем semantic + LLM метрики
            llm_combined = (
                0.5 * llm_metrics['context_sufficiency'] +
                0.3 * llm_metrics['context_relevance'] +
                0.2 * llm_metrics['context_precision']
            )
            hybrid_score = (
                self.semantic_weight * semantic_score +
                self.llm_weight * llm_combined
            )
        else:
            # Только semantic
            hybrid_score = semantic_score

        return {
            'semantic_score': semantic_score,
            'context_relevance': llm_metrics['context_relevance'],
            'context_precision': llm_metrics['context_precision'],
            'context_sufficiency': llm_metrics['context_sufficiency'],
            'hybrid_score': hybrid_score
        }

    def _evaluate_with_llm(
        self,
        query: str,
        chunks: List[str]
    ) -> Dict[str, float]:
        """
        LLM-based оценка контекста (RAGAS-style с Structured Output)
        """
        # Формируем промпт
        prompt = self._build_llm_prompt(query, chunks)

        try:
            if self.use_api:
                # API режим
                if self.client is None:
                    self.logger.error("[HybridRAGEvaluator] ❌ API клиент не инициализирован, но use_api=True!")
                    raise RuntimeError("API клиент не инициализирован")
                
                self.logger.debug(f"[HybridRAGEvaluator] → API запрос к {self.model_name}")
                request_params = {
                    "model": self.model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.1,
                    "max_tokens": 1024
                }
                
                # Добавляем провайдера через extra_headers если указан
                if LLM_API_ROUTING:
                    request_params["extra_headers"] = {"X-OpenRouter-Provider": LLM_API_ROUTING}
                
                response = self.client.chat.completions.create(**request_params)
                content = response.choices[0].message.content
                self.logger.debug(f"[HybridRAGEvaluator] ← API ответ получен")
            else:
                # Локальный режим
                if self.llm is None:
                    self.logger.error("[HybridRAGEvaluator] ❌ Локальная модель не загружена, но use_api=False!")
                    raise RuntimeError("Локальная модель не загружена")
                
                self.logger.debug(f"[HybridRAGEvaluator] → Локальный LLM запрос")
                response = self.llm.create_chat_completion(
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=1024
                )
                content = response['choices'][0]['message']['content']
                self.logger.debug(f"[HybridRAGEvaluator] ← Локальный LLM ответ получен")

            # Парсим JSON
            metrics = self._parse_llm_response(content, len(chunks))
            return metrics

        except Exception as e:
            self.logger.error(f"Ошибка LLM оценки: {e}")
            return {
                'context_relevance': 0.5,
                'context_precision': 0.5,
                'context_sufficiency': 0.5
            }

    def _build_llm_prompt(self, query: str, chunks: List[str]) -> str:
        """Формирует промпт для LLM оценки (Structured Output)"""
        chunks_text = "\n\n".join([
            f"[Чанк {i+1}]\n{chunk[:500]}"  # обрезаем до 500 символов для экономии токенов
            for i, chunk in enumerate(chunks)
        ])

        prompt = f"""Ты - эксперт по оценке качества контекста для вопросов клиентов банка.

ВОПРОС КЛИЕНТА:
{query}

ИЗВЛЕЧЕННЫЕ ФРАГМЕНТЫ ({len(chunks)} шт.):
{chunks_text}

ЗАДАЧА:
Оцени качество контекста по трем критериям:

1. **Релевантность каждого чанка** (relevance_scores):
   - Для КАЖДОГО из {len(chunks)} чанков оцени от 0.0 до 1.0
   - 1.0 = полностью релевантен, содержит прямой ответ
   - 0.5 = частично релевантен
   - 0.0 = нерелевантен

2. **Context Sufficiency** (достаточность):
   - Достаточно ли информации для ПОЛНОГО ответа? (0.0-1.0)
   - 1.0 = вся информация есть
   - 0.5 = частичная информация
   - 0.0 = недостаточно

Формат ответа (СТРОГО JSON):
{{
    "relevance_scores": [score1, score2, ...],
    "context_sufficiency": 0.0-1.0,
    "brief_reason": "краткое объяснение (1 предложение)"
}}

ВАЖНО: relevance_scores должен содержать ровно {len(chunks)} чисел!

Ответ (JSON):"""

        return prompt

    def _parse_llm_response(
        self,
        response_text: str,
        num_chunks: int
    ) -> Dict[str, float]:
        """Парсит ответ LLM и вычисляет метрики"""
        try:
            # Извлекаем JSON
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1

            if start_idx == -1 or end_idx <= start_idx:
                raise ValueError("JSON не найден")

            json_str = response_text[start_idx:end_idx]
            data = json.loads(json_str)

            # Relevance scores
            relevance_scores = data.get('relevance_scores', [])

            # Корректируем длину если нужно
            if len(relevance_scores) != num_chunks:
                if len(relevance_scores) < num_chunks:
                    relevance_scores.extend([0.5] * (num_chunks - len(relevance_scores)))
                else:
                    relevance_scores = relevance_scores[:num_chunks]

            # Нормализуем к 0-1
            relevance_scores = [max(0.0, min(1.0, float(s))) for s in relevance_scores]

            # Context Relevance = среднее
            context_relevance = np.mean(relevance_scores) if relevance_scores else 0.0

            # Context Precision = доля >= 0.5
            relevant_count = sum(1 for s in relevance_scores if s >= 0.5)
            context_precision = relevant_count / num_chunks if num_chunks > 0 else 0.0

            # Context Sufficiency
            context_sufficiency = max(0.0, min(1.0, float(data.get('context_sufficiency', 0.5))))

            return {
                'context_relevance': context_relevance,
                'context_precision': context_precision,
                'context_sufficiency': context_sufficiency
            }

        except Exception as e:
            self.logger.error(f"Ошибка парсинга LLM ответа: {e}")
            self.logger.debug(f"Ответ: {response_text[:200]}")

            return {
                'context_relevance': 0.5,
                'context_precision': 0.5,
                'context_sufficiency': 0.5
            }

    def evaluate_batch(
        self,
        queries: List[str],
        results_list: List[pd.DataFrame],
        top_k: int = 20
    ) -> Dict[str, float]:
        """
        Оценка качества для батча запросов

        Args:
            queries: список запросов
            results_list: список DataFrame с результатами
            top_k: количество топовых документов

        Returns:
            dict с усредненными метриками
        """
        all_metrics = []

        for query, results in zip(queries, results_list):
            metrics = self.evaluate_retrieval(query, results, top_k)
            all_metrics.append(metrics)

        if len(all_metrics) == 0:
            return {
                'avg_semantic_score': 0.0,
                'avg_context_relevance': 0.0,
                'avg_context_precision': 0.0,
                'avg_context_sufficiency': 0.0,
                'avg_hybrid_score': 0.0,
                'num_evaluated': 0
            }

        # Усредняем
        return {
            'avg_semantic_score': np.mean([m['semantic_score'] for m in all_metrics]),
            'avg_context_relevance': np.mean([m['context_relevance'] for m in all_metrics]),
            'avg_context_precision': np.mean([m['context_precision'] for m in all_metrics]),
            'avg_context_sufficiency': np.mean([m['context_sufficiency'] for m in all_metrics]),
            'avg_hybrid_score': np.mean([m['hybrid_score'] for m in all_metrics]),
            'num_evaluated': len(all_metrics)
        }

    def __del__(self):
        """Освобождаем ресурсы"""
        if hasattr(self, 'llm') and self.llm:
            del self.llm


# Singleton
_evaluator_instance = None


def get_hybrid_evaluator(
    llm_model_path: Optional[str] = None,
    use_llm: bool = True,
    semantic_weight: float = 0.3,
    llm_weight: float = 0.7,
    use_api: bool = None
) -> HybridRAGEvaluator:
    """
    Получить singleton экземпляр HybridRAGEvaluator

    Args:
        llm_model_path: путь к LLM модели
        use_llm: использовать ли LLM Judge
        semantic_weight: вес косинусного расстояния (0-1)
        llm_weight: вес LLM метрик (0-1)
        use_api: использовать ли API (если None - определяется из LLM_MODE)

    Returns:
        HybridRAGEvaluator instance
    """
    global _evaluator_instance
    
    # Определяем режим работы если не передан явно
    if use_api is None:
        from src.config import LLM_MODE
        use_api = (LLM_MODE == "api")
    
    # Если singleton уже создан, проверяем совместимость режима
    if _evaluator_instance is not None:
        # Если режим изменился (API <-> локальный), пересоздаем
        if _evaluator_instance.use_api != use_api:
            get_logger(__name__).info(f"Режим изменился ({_evaluator_instance.use_api} -> {use_api}), пересоздаем evaluator")
            _evaluator_instance = None
    
    # Создаем новый экземпляр если нужно
    if _evaluator_instance is None:
        _evaluator_instance = HybridRAGEvaluator(
            llm_model_path,
            use_llm,
            semantic_weight,
            llm_weight,
            use_api
        )

    return _evaluator_instance


if __name__ == "__main__":
    # Тестирование
    import sys
    import io

    # Фикс кодировки для Windows
    if sys.platform == 'win32':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    print("="*80)
    print("ТЕСТИРОВАНИЕ HYBRID RAG EVALUATOR")
    print("="*80)

    try:
        # Создаем evaluator (без LLM для быстрого теста)
        evaluator = get_hybrid_evaluator(use_llm=False)

        # Тестовые данные
        test_query = "Почему не начисляется кэшбэк за оплату коммунальных услуг?"

        test_results = pd.DataFrame([
            {'text': 'Кэшбэк начисляется на покупки в магазинах и ресторанах.', 'final_score': 0.75},
            {'text': 'На оплату ЖКХ кэшбэк не распространяется.', 'final_score': 0.92},
            {'text': 'Условия программы лояльности в приложении.', 'final_score': 0.68},
            {'text': 'Активация карты по телефону.', 'final_score': 0.45},
            {'text': 'Оплата ЖКХ по QR-коду.', 'final_score': 0.55}
        ])

        print(f"\nВопрос: {test_query}")
        print(f"Результатов: {len(test_results)}\n")

        # Оцениваем
        metrics = evaluator.evaluate_retrieval(test_query, test_results, top_k=5)

        print("МЕТРИКИ:")
        print(f"  Semantic Score (cosine): {metrics['semantic_score']:.3f}")
        print(f"  Context Relevance (LLM): {metrics['context_relevance']:.3f}")
        print(f"  Context Precision (LLM): {metrics['context_precision']:.3f}")
        print(f"  Context Sufficiency (LLM): {metrics['context_sufficiency']:.3f}")
        print(f"  → HYBRID SCORE: {metrics['hybrid_score']:.3f}")

        print("\n" + "="*80)
        print("✓ Тест завершен успешно!")
        print("\nДля полного теста с LLM запустите:")
        print("  evaluator = get_hybrid_evaluator(use_llm=True)")

    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
