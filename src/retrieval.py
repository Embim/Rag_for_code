"""
Гибридный ретривер и LLM-based reranker
"""
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
import re
import os

# Фикс для CUDA путей перед импортом llama_cpp
if os.name == 'nt':  # Windows
    cuda_path = os.environ.get('CUDA_PATH', '')

    # Если CUDA_PATH установлен но не существует - пытаемся исправить
    if cuda_path and not os.path.exists(cuda_path):
        cuda_base = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
        if os.path.exists(cuda_base):
            # Находим все установленные версии CUDA
            versions = sorted([d for d in os.listdir(cuda_base) if d.startswith('v')])
            if versions:
                # Берем первую найденную версию
                correct_path = os.path.join(cuda_base, versions[0])
                os.environ['CUDA_PATH'] = correct_path
                print(f"[INFO] CUDA_PATH исправлен: {cuda_path} -> {correct_path}")
            else:
                # CUDA папка есть, но версий нет - удаляем CUDA_PATH
                print(f"[WARNING] CUDA Toolkit не найден, удаляем CUDA_PATH: {cuda_path}")
                del os.environ['CUDA_PATH']
        else:
            # CUDA вообще не установлена - удаляем переменную
            print(f"[WARNING] CUDA Toolkit не установлен, удаляем CUDA_PATH: {cuda_path}")
            del os.environ['CUDA_PATH']
    elif cuda_path:
        print(f"[INFO] CUDA_PATH: {cuda_path}")

try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except Exception as e:
    LLAMA_CPP_AVAILABLE = False
    print(f"[WARNING] llama_cpp не загружен: {e}")
    print("[WARNING] LLM реранкинг будет недоступен")

from src.config import (
    TOP_K_DENSE,
    TOP_K_BM25,
    TOP_K_RERANK,
    HYBRID_ALPHA,
    TOP_N_DOCUMENTS,
    LLM_MODEL_FILE,
    ENABLE_AGENT_RAG,
    ENABLE_QUERY_EXPANSION,
    QUERY_EXPANSION_METHOD,
    ENABLE_METADATA_FILTER,
    METADATA_BOOST_SCORE,
    ENABLE_USEFULNESS_FILTER,
    MIN_USEFULNESS_SCORE,
    ENABLE_DYNAMIC_TOP_K,
    LLM_CONTEXT_SIZE,
    LLM_TEMPERATURE,
    LLM_GPU_LAYERS,
    MODELS_DIR,
    USE_TRANSFORMER_RERANKER,
    RERANKER_MODEL_PATH,
    RERANKER_BATCH_SIZE,
    RERANKER_MAX_LENGTH,
    RERANKER_TYPE,
    CROSS_ENCODER_MODEL,
    ENABLE_RRF,
    RRF_K,
    ENABLE_CONTEXT_WINDOW,
    CONTEXT_WINDOW_SIZE,
    CONTEXT_MERGE_MODE,
    ENABLE_QUERY_REFORMULATION,
    QUERY_REFORMULATION_METHOD,
    QUERY_REFORMULATION_CACHE
)


class HybridRetriever:
    """Гибридный ретривер объединяющий dense и BM25 поиск"""

    def __init__(self, embedding_indexer, bm25_indexer,
                 alpha: float = HYBRID_ALPHA,
                 use_query_expansion: bool = ENABLE_QUERY_EXPANSION):
        """
        Args:
            embedding_indexer: векторный индексер (WeaviateIndexer или EmbeddingIndexer)
            bm25_indexer: BM25 индексер
            alpha: вес для dense поиска (1-alpha для BM25)
            use_query_expansion: использовать ли расширение запроса
        """
        self.embedding_indexer = embedding_indexer
        self.bm25_indexer = bm25_indexer
        self.alpha = alpha
        self.use_query_expansion = use_query_expansion

        # Определяем тип индексера
        self.is_weaviate = hasattr(embedding_indexer, 'collection')

        # Инициализация Query Expander (если нужно)
        self.query_expander = None
        if self.use_query_expansion:
            try:
                from src.query_expansion import QueryExpander
                # Используем только словарь синонимов (без LLM для скорости)
                use_llm = QUERY_EXPANSION_METHOD in ["llm", "hybrid"]
                self.query_expander = QueryExpander(use_llm=use_llm)
                print(f"[INFO] Query Expansion включен (метод: {QUERY_EXPANSION_METHOD})")
            except Exception as e:
                print(f"[WARNING] Query Expansion не загружен: {e}")
                self.use_query_expansion = False

        # Инициализация Metadata Filter (если нужно)
        self.metadata_filter = None
        if ENABLE_METADATA_FILTER:
            try:
                from src.metadata_filter import MetadataFilter
                self.metadata_filter = MetadataFilter()
                print(f"[INFO] Metadata Filtering включен (boost: {METADATA_BOOST_SCORE})")
            except Exception as e:
                print(f"[WARNING] Metadata Filter не загружен: {e}")

        # Инициализация Query Classifier (для Dynamic TOP_K)
        self.query_classifier = None
        if ENABLE_DYNAMIC_TOP_K:
            try:
                from src.query_classifier import QueryClassifier
                self.query_classifier = QueryClassifier()
                print(f"[INFO] Dynamic TOP_K включен")
            except Exception as e:
                print(f"[WARNING] Query Classifier не загружен: {e}")

        # Инициализация Query Reformulator (для улучшения запросов)
        self.query_reformulator = None
        if ENABLE_QUERY_REFORMULATION and LLAMA_CPP_AVAILABLE:
            try:
                from src.query_reformulation import QueryReformulator
                llm_path = str(MODELS_DIR / LLM_MODEL_FILE)
                self.query_reformulator = QueryReformulator(
                    llm_path,
                    use_cache=QUERY_REFORMULATION_CACHE
                )
                print(f"[INFO] Query Reformulation включен (метод: {QUERY_REFORMULATION_METHOD})")
            except Exception as e:
                print(f"[WARNING] Query Reformulator не загружен: {e}")

    def search(self, query: str,
              k_dense: int = TOP_K_DENSE,
              k_bm25: int = TOP_K_BM25) -> pd.DataFrame:
        """
        Гибридный поиск

        Args:
            query: поисковый запрос
            k_dense: топ-k для векторного поиска
            k_bm25: топ-k для BM25 (игнорируется для Weaviate)

        Returns:
            DataFrame с результатами поиска
        """
        # 0. Dynamic TOP_K (если включено)
        if self.query_classifier:
            dynamic_params = self.query_classifier.get_dynamic_top_k(query)
            k_dense = dynamic_params['k_dense']
            k_bm25 = dynamic_params['k_bm25']
            print(f"[INFO] Dynamic TOP_K: {dynamic_params['query_type']} ({dynamic_params['complexity']}) → k_dense={k_dense}, k_bm25={k_bm25}")

        # 0.5. Query Reformulation (если включено)
        working_query = query  # рабочий запрос для дальнейшей обработки
        if self.query_reformulator:
            try:
                # Переформулируем запрос
                reformulated_variants = self.query_reformulator.reformulate(
                    query,
                    method=QUERY_REFORMULATION_METHOD
                )

                if len(reformulated_variants) > 1:
                    print(f"[INFO] Query Reformulation: {len(reformulated_variants)} вариантов")
                    print(f"      Исходный: {query}")
                    print(f"      Переформулированный: {reformulated_variants[1]}")

                    # Используем переформулированный как основной (первый после исходного)
                    working_query = reformulated_variants[1]

            except Exception as e:
                print(f"[WARNING] Query Reformulation ошибка: {e}")

        # 1. Query Expansion (если включено)
        queries = [working_query]  # по умолчанию используем (возможно переформулированный) запрос
        if self.use_query_expansion and self.query_expander:
            try:
                expanded = self.query_expander.expand_query(working_query, method=QUERY_EXPANSION_METHOD)
                # Берем первые 3 варианта для экономии времени
                queries = expanded[:3]
                if len(queries) > 1:
                    print(f"[INFO] Query expansion: {len(queries)} вариантов запроса")
            except Exception as e:
                print(f"[WARNING] Query expansion ошибка: {e}")
                queries = [working_query]

        # Выполняем поиск для каждого варианта запроса и объединяем результаты
        all_results = []
        for q in queries:
            # 1. Векторный/Гибридный поиск
            if self.is_weaviate:
                # Weaviate уже выполняет гибридный поиск (BM25 + vector) внутри
                # Используем alpha из конфига для баланса
                k = max(k_dense, k_bm25)  # берем максимум из двух
                scores, results = self.embedding_indexer.search(
                    q,  # используем текущий вариант запроса
                    k=k,
                    alpha=self.alpha  # баланс между vector и BM25
                )

                # Преобразуем в DataFrame
                results_df = pd.DataFrame(results)
                results_df['retrieval_score'] = scores
                all_results.append(results_df)

            else:
                # FAISS режим - делаем как раньше
                # 1. Векторный поиск
                query_embedding = self.embedding_indexer.model.encode([q])[0]
                dense_scores, dense_indices = self.embedding_indexer.search(
                    query_embedding, k=k_dense
                )

                # 2. BM25 поиск
                bm25_scores, bm25_indices = self.bm25_indexer.search(q, k=k_bm25)

                # 3. Объединение результатов
                results = self._merge_results(
                    dense_scores, dense_indices,
                    bm25_scores, bm25_indices
                )

                all_results.append(results)

        # Объединяем результаты от разных вариантов запроса
        if len(all_results) == 1:
            final_results = all_results[0]
        else:
            # Объединяем результаты от разных queries
            combined = pd.concat(all_results, ignore_index=True)
            # Группируем по chunk_id и берем максимальный score
            combined = combined.sort_values('retrieval_score', ascending=False)
            final_results = combined.drop_duplicates(subset=['chunk_id'], keep='first')

        # Применяем фильтры (metadata + usefulness)
        final_results = self._apply_filters(query, final_results)

        return final_results.reset_index(drop=True)

    def _apply_filters(self, query: str, results_df: pd.DataFrame) -> pd.DataFrame:
        """
        Применение всех фильтров к результатам поиска

        Args:
            query: исходный запрос
            results_df: результаты поиска

        Returns:
            отфильтрованные результаты
        """
        if len(results_df) == 0:
            return results_df

        # 0. Negative Mining (исключаем нерелевантные паттерны)
        anti_patterns = [
            "© 2001-2025",
            "все права защищены",
            "политика конфиденциальности",
            "согласие на обработку",
            "подписаться на рассылку",
            "cookie",
            "cookies",
            "поделиться в",
            "следите за нами",
            "скачать приложение в app store",
        ]

        if 'text' in results_df.columns or 'clean_text' in results_df.columns:
            text_column = 'clean_text' if 'clean_text' in results_df.columns else 'text'

            def has_anti_pattern(text: str) -> bool:
                if pd.isna(text):
                    return True  # фильтруем пустые
                text_lower = str(text).lower()
                return any(pattern in text_lower for pattern in anti_patterns)

            before_count = len(results_df)
            results_df = results_df[~results_df[text_column].apply(has_anti_pattern)].copy()
            after_count = len(results_df)

            if before_count > after_count:
                print(f"[INFO] Negative Mining: {before_count} → {after_count} документов")

            # Если отфильтровали все - возвращаем хотя бы топ-5 исходных
            if len(results_df) == 0:
                print(f"[WARNING] Все документы отфильтрованы negative mining, возвращаем топ-5")
                results_df = results_df.nlargest(5, 'retrieval_score')

        # 1. Usefulness Score Filtering
        if ENABLE_USEFULNESS_FILTER and 'usefulness_score' in results_df.columns:
            before_count = len(results_df)
            results_df = results_df[results_df['usefulness_score'] >= MIN_USEFULNESS_SCORE].copy()
            after_count = len(results_df)

            if before_count > after_count:
                print(f"[INFO] Usefulness filter: {before_count} → {after_count} документов (порог: {MIN_USEFULNESS_SCORE})")

            # Если отфильтровали все - возвращаем хотя бы топ-3 исходных
            if len(results_df) == 0:
                print(f"[WARNING] Все документы отфильтрованы, возвращаем топ-3 исходных")
                results_df = results_df.nlargest(3, 'retrieval_score')

        # 2. Metadata Filtering (boost)
        if self.metadata_filter and 'products' in results_df.columns:
            results_df = self.metadata_filter.auto_filter(
                query,
                results_df,
                boost_score=METADATA_BOOST_SCORE
            )

        return results_df

    def _merge_results(self, dense_scores, dense_indices,
                      bm25_scores, bm25_indices) -> pd.DataFrame:
        """
        Объединение результатов dense и BM25 поиска

        Методы:
        - RRF (Reciprocal Rank Fusion) - рекомендуется, лучше работает
        - Weighted Sum - старый метод

        Args:
            dense_scores: скоры векторного поиска
            dense_indices: индексы векторного поиска
            bm25_scores: скоры BM25
            bm25_indices: индексы BM25

        Returns:
            DataFrame с объединенными результатами
        """
        chunks_metadata = self.embedding_indexer.chunk_metadata

        if ENABLE_RRF:
            # RRF метод (лучше!)
            from src.reciprocal_rank_fusion import ReciprocalRankFusion

            # Создаем DataFrame для dense результатов
            dense_df = chunks_metadata.iloc[dense_indices].copy()
            dense_df['retrieval_score'] = dense_scores

            # Создаем DataFrame для BM25 результатов
            bm25_df = chunks_metadata.iloc[bm25_indices].copy()
            bm25_df['retrieval_score'] = bm25_scores

            # Объединяем через RRF
            rrf = ReciprocalRankFusion(k=RRF_K)
            results_df = rrf.fuse_two_results(dense_df, bm25_df)

            return results_df

        else:
            # Weighted Sum (старый метод)
            # Нормализация скоров
            dense_scores_norm = self._normalize_scores(dense_scores)
            bm25_scores_norm = self._normalize_scores(bm25_scores)

            # Создаем словарь для объединения
            combined_scores = {}

            # Добавляем dense результаты
            for idx, score in zip(dense_indices, dense_scores_norm):
                combined_scores[idx] = self.alpha * score

            # Добавляем BM25 результаты
            for idx, score in zip(bm25_indices, bm25_scores_norm):
                if idx in combined_scores:
                    combined_scores[idx] += (1 - self.alpha) * score
                else:
                    combined_scores[idx] = (1 - self.alpha) * score

            # Сортируем по скору
            sorted_results = sorted(combined_scores.items(),
                                   key=lambda x: x[1], reverse=True)

            # Формируем DataFrame
            indices = [idx for idx, _ in sorted_results]
            scores = [score for _, score in sorted_results]

            results_df = chunks_metadata.iloc[indices].copy()
            results_df['retrieval_score'] = scores
            results_df = results_df.reset_index(drop=True)

            return results_df

    def _merge_results_weaviate(self, dense_scores, dense_df,
                               bm25_scores, bm25_indices) -> pd.DataFrame:
        """
        Объединение результатов для Weaviate (когда dense_df уже содержит данные)

        Args:
            dense_scores: скоры векторного поиска
            dense_df: DataFrame с результатами векторного поиска
            bm25_scores: скоры BM25
            bm25_indices: индексы BM25

        Returns:
            DataFrame с объединенными результатами
        """
        # Нормализация скоров
        dense_scores_norm = self._normalize_scores(np.array(dense_scores))
        bm25_scores_norm = self._normalize_scores(bm25_scores)

        # Добавляем скоры к dense результатам
        dense_df = dense_df.copy()
        dense_df['retrieval_score'] = self.alpha * dense_scores_norm

        # Получаем BM25 результаты из метаданных BM25
        # (предполагаем что у нас есть общий chunk_metadata)
        if hasattr(self.embedding_indexer, 'chunk_metadata'):
            chunks_metadata = self.embedding_indexer.chunk_metadata
        else:
            # Используем dense_df как базу
            chunks_metadata = dense_df

        # Создаем словарь для объединения по chunk_id
        combined_scores = {}

        # Добавляем dense результаты
        for idx, row in dense_df.iterrows():
            chunk_id = row.get('chunk_id', idx)
            combined_scores[chunk_id] = {
                'data': row.to_dict(),
                'score': row['retrieval_score']
            }

        # Добавляем BM25 результаты
        for idx, score in zip(bm25_indices, bm25_scores_norm):
            if idx < len(chunks_metadata):
                row = chunks_metadata.iloc[idx]
                chunk_id = row.get('chunk_id', idx)
                bm25_contribution = (1 - self.alpha) * score

                if chunk_id in combined_scores:
                    combined_scores[chunk_id]['score'] += bm25_contribution
                else:
                    combined_scores[chunk_id] = {
                        'data': row.to_dict(),
                        'score': bm25_contribution
                    }

        # Сортируем по скору
        sorted_results = sorted(combined_scores.items(),
                               key=lambda x: x[1]['score'], reverse=True)

        # Формируем DataFrame
        results_data = [item[1]['data'] for item in sorted_results]
        scores = [item[1]['score'] for item in sorted_results]

        results_df = pd.DataFrame(results_data)
        results_df['retrieval_score'] = scores
        results_df = results_df.reset_index(drop=True)

        return results_df

    @staticmethod
    def _normalize_scores(scores: np.ndarray) -> np.ndarray:
        """Нормализация скоров в диапазон [0, 1]"""
        if len(scores) == 0:
            return scores

        # Преобразуем в numpy array если это список
        if isinstance(scores, list):
            scores = np.array(scores)

        min_score = scores.min()
        max_score = scores.max()

        if max_score == min_score:
            return np.ones_like(scores)

        return (scores - min_score) / (max_score - min_score)


class LLMReranker:
    """LLM-based реранкер для оценки релевантности документов"""

    def __init__(self, model_path: str = None):
        """
        Args:
            model_path: путь к GGUF модели
        """
        if not LLAMA_CPP_AVAILABLE:
            raise ImportError(
                "llama-cpp-python не доступен. "
                "Проверьте установку: pip list | grep llama"
            )

        if model_path is None:
            model_path = str(MODELS_DIR / LLM_MODEL_FILE)

        print(f"Загрузка LLM reranker модели: {model_path}")

        self.model = Llama(
            model_path=model_path,
            n_ctx=LLM_CONTEXT_SIZE,
            n_gpu_layers=LLM_GPU_LAYERS,
            n_batch=512,  # Размер батча для обработки токенов
            n_threads=8,  # Количество CPU потоков
            use_mlock=True,  # Блокировка памяти для скорости
            verbose=False
        )
        print(f"  LLM загружена успешно (GPU layers: {LLM_GPU_LAYERS})")

    def _create_rerank_prompt(self, query: str, passage: str) -> str:
        """Создание промпта для оценки релевантности"""
        return f"""<|im_start|>system
Вы - эксперт по оценке релевантности документов. Оцените насколько следующий фрагмент текста релевантен вопросу пользователя по шкале от 0 до 10, где:
- 0: совершенно нерелевантен
- 5: частично релевантен
- 10: полностью релевантен и отвечает на вопрос

Ответьте ТОЛЬКО числом от 0 до 10.<|im_end|>
<|im_start|>user
Вопрос: {query}

Фрагмент текста: {passage[:500]}

Оценка релевантности (0-10):<|im_end|>
<|im_start|>assistant
"""

    def _extract_score(self, response: str) -> float:
        """Извлечение числовой оценки из ответа LLM"""
        # Ищем первое число в ответе
        match = re.search(r'(\d+(?:\.\d+)?)', response)
        if match:
            score = float(match.group(1))
            # Нормализуем в диапазон 0-1
            return min(max(score / 10.0, 0.0), 1.0)
        return 0.0

    def rerank(self, query: str, candidates_df: pd.DataFrame,
              top_k: int = TOP_K_RERANK) -> pd.DataFrame:
        """
        Переранжирование кандидатов с помощью LLM

        Args:
            query: запрос
            candidates_df: DataFrame с кандидатами
            top_k: количество лучших результатов

        Returns:
            DataFrame с переранжированными результатами
        """
        if len(candidates_df) == 0:
            return candidates_df

        print(f"LLM Reranking {len(candidates_df)} кандидатов...")

        rerank_scores = []

        for idx, row in candidates_df.iterrows():
            text = row['text']

            # Создаем промпт
            prompt = self._create_rerank_prompt(query, text)

            # Получаем оценку от LLM
            try:
                response = self.model(
                    prompt,
                    max_tokens=10,  # Нужно только число
                    temperature=LLM_TEMPERATURE,
                    stop=["<|im_end|>", "\n"],
                    echo=False,
                    top_p=0.9,  # Nucleus sampling для скорости
                    top_k=40,   # Ограничение словаря для скорости
                    repeat_penalty=1.0  # Без penalty для скорости
                )

                # Извлекаем оценку
                score_text = response['choices'][0]['text'].strip()
                score = self._extract_score(score_text)
                rerank_scores.append(score)

            except Exception as e:
                print(f"  Ошибка при оценке документа {idx}: {e}")
                rerank_scores.append(0.0)

        # Добавляем скоры и сортируем
        candidates_df = candidates_df.copy()
        candidates_df['rerank_score'] = rerank_scores

        # Сортируем по rerank_score
        reranked_df = candidates_df.sort_values(
            'rerank_score', ascending=False
        ).head(top_k).reset_index(drop=True)

        print(f"  Топ-3 оценки: {reranked_df['rerank_score'].head(3).tolist()}")

        return reranked_df


class TransformerReranker:
    """Быстрый reranker на основе Qwen3-Reranker-0.6B"""

    def __init__(self, model_path: str = None):
        """
        Инициализация transformer reranker

        Args:
            model_path: путь к модели
        """
        if model_path is None:
            model_path = str(RERANKER_MODEL_PATH)

        print(f"Загрузка Transformer Reranker: {model_path}")

        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            import torch

            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)

            # Добавляем pad_token если его нет (обязательно для батчинга)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

            # Устанавливаем padding_side
            self.tokenizer.padding_side = "right"

            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
            self.model.to(self.device)
            self.model.eval()

            print(f"  Reranker загружен успешно (device: {self.device})")

        except Exception as e:
            raise ImportError(
                f"Не удалось загрузить Transformer Reranker: {e}\n"
                f"Установите: pip install transformers torch"
            )

    def rerank(self, query: str, candidates_df: pd.DataFrame,
              top_k: int = TOP_K_RERANK) -> pd.DataFrame:
        """
        Переранжирование кандидатов с помощью Transformer Reranker

        Args:
            query: запрос
            candidates_df: DataFrame с кандидатами
            top_k: количество лучших результатов

        Returns:
            DataFrame с переранжированными результатами
        """
        import torch

        if len(candidates_df) == 0:
            return candidates_df

        print(f"Transformer Reranking {len(candidates_df)} кандидатов...")

        # Подготовка пар (query, passage)
        pairs = []
        for _, row in candidates_df.iterrows():
            text = row['text'][:RERANKER_MAX_LENGTH]  # Ограничиваем длину
            pairs.append([query, text])

        # Батчинг для ускорения
        all_scores = []
        batch_size = RERANKER_BATCH_SIZE

        with torch.no_grad():
            for i in range(0, len(pairs), batch_size):
                batch = pairs[i:i + batch_size]

                # Токенизация
                inputs = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=RERANKER_MAX_LENGTH,
                    return_tensors="pt"
                ).to(self.device)

                # Inference
                outputs = self.model(**inputs)
                scores = outputs.logits.squeeze(-1).cpu().numpy()

                # Для моделей с двумя классами берем softmax
                if len(scores.shape) > 1:
                    import numpy as np
                    scores = torch.softmax(torch.tensor(scores), dim=-1)[:, 1].numpy()

                all_scores.extend(scores.tolist() if hasattr(scores, 'tolist') else [scores])

        # Добавляем скоры и сортируем
        candidates_df = candidates_df.copy()
        candidates_df['rerank_score'] = all_scores

        reranked_df = candidates_df.sort_values(
            'rerank_score', ascending=False
        ).head(top_k).reset_index(drop=True)

        print(f"  Топ-3 оценки: {reranked_df['rerank_score'].head(3).tolist()}")

        return reranked_df


# Алиас для обратной совместимости
Reranker = LLMReranker


class DocumentSelector:
    """Выбор топ-N документов из ранжированных чанков"""

    @staticmethod
    def select_top_documents(reranked_chunks: pd.DataFrame,
                           top_n: int = TOP_N_DOCUMENTS) -> List[int]:
        """
        Выбор топ-N уникальных документов

        Args:
            reranked_chunks: DataFrame с ранжированными чанками
            top_n: количество документов для выбора

        Returns:
            список web_id топ-N документов
        """
        if len(reranked_chunks) == 0:
            return []

        # Группируем по web_id и берем максимальный rerank_score
        doc_scores = reranked_chunks.groupby('web_id')['rerank_score'].max()

        # Сортируем и выбираем топ-N
        top_docs = doc_scores.sort_values(ascending=False).head(top_n)

        return top_docs.index.tolist()

    @staticmethod
    def select_top_documents_with_diversity(reranked_chunks: pd.DataFrame,
                                           top_n: int = TOP_N_DOCUMENTS) -> List[int]:
        """
        Выбор топ-N документов с учетом разнообразия

        Проходим по ранжированным чанкам и добавляем документы,
        которые еще не были выбраны

        Args:
            reranked_chunks: DataFrame с ранжированными чанками
            top_n: количество документов

        Returns:
            список web_id
        """
        selected_docs = []
        seen_web_ids = set()

        for _, row in reranked_chunks.iterrows():
            web_id = row['web_id']

            if web_id not in seen_web_ids:
                selected_docs.append(web_id)
                seen_web_ids.add(web_id)

            if len(selected_docs) >= top_n:
                break

        # Дополняем до top_n если нужно (хотя обычно не требуется)
        while len(selected_docs) < top_n:
            # Берем документы с самым высоким скором, даже если они повторяются
            for _, row in reranked_chunks.iterrows():
                web_id = row['web_id']
                if web_id not in selected_docs:
                    selected_docs.append(web_id)
                    if len(selected_docs) >= top_n:
                        break
            break  # Если все документы уже добавлены

        return selected_docs


class RAGPipeline:
    """Полный пайплайн RAG"""

    def __init__(self, embedding_indexer, bm25_indexer,
                 alpha: float = HYBRID_ALPHA):
        """
        Args:
            embedding_indexer: векторный индексер
            bm25_indexer: BM25 индексер
            alpha: вес для гибридного поиска
        """
        self.retriever = HybridRetriever(embedding_indexer, bm25_indexer, alpha)

        # Выбираем reranker в зависимости от конфигурации
        if RERANKER_TYPE == "cross_encoder":
            print(f"[INFO] Используется Cross-Encoder Reranker (быстрый и качественный)")
            print(f"       Модель: {CROSS_ENCODER_MODEL}")
            from src.cross_encoder_reranker import CrossEncoderReranker
            self.reranker = CrossEncoderReranker(model_name=CROSS_ENCODER_MODEL)
        elif RERANKER_TYPE == "transformer" or USE_TRANSFORMER_RERANKER:
            print("[INFO] Используется Transformer Reranker (Qwen3-Reranker)")
            self.reranker = TransformerReranker()
        elif RERANKER_TYPE == "llm":
            print("[INFO] Используется LLM Reranker (медленный, очень качественный)")
            self.reranker = LLMReranker()
        elif RERANKER_TYPE == "none":
            print("[WARNING] Reranking отключен (RERANKER_TYPE=none)")
            self.reranker = None
        else:
            # Fallback на LLM если тип неизвестен
            print(f"[WARNING] Неизвестный RERANKER_TYPE: {RERANKER_TYPE}, используется LLM")
            self.reranker = LLMReranker()

        self.selector = DocumentSelector()

    def search(self, query: str,
              k_dense: int = TOP_K_DENSE,
              k_bm25: int = TOP_K_BM25,
              k_rerank: int = TOP_K_RERANK,
              top_n: int = TOP_N_DOCUMENTS) -> Dict:
        """
        Полный поисковый пайплайн

        Args:
            query: запрос
            k_dense: топ-k для dense поиска
            k_bm25: топ-k для BM25
            k_rerank: топ-k после reranking
            top_n: количество финальных документов

        Returns:
            словарь с результатами
        """
        # 1. Гибридный поиск
        candidates = self.retriever.search(query, k_dense, k_bm25)

        # 2. Reranking
        if self.reranker is not None:
            reranked = self.reranker.rerank(query, candidates, k_rerank)
        else:
            # Если reranking отключен - просто берем топ-k кандидатов
            reranked = candidates.head(k_rerank).copy()
            reranked['rerank_score'] = reranked.get('retrieval_score', 0.0)

        # 2.5. Context Window Expansion (добавление соседних чанков)
        if ENABLE_CONTEXT_WINDOW:
            from src.context_window import ContextWindowExpander

            # Получаем все чанки для поиска соседей
            chunks_df = self.retriever.embedding_indexer.chunk_metadata

            expander = ContextWindowExpander(window_size=CONTEXT_WINDOW_SIZE)
            reranked = expander.expand_with_neighbors(
                chunks_df,
                reranked,
                preserve_scores=True
            )

            if CONTEXT_MERGE_MODE == "merged":
                # Объединяем тексты соседних чанков
                reranked = expander.merge_neighbors_text(reranked)

            print(f"[INFO] Context Window: добавлено {len(reranked)} чанков (включая соседей)")

        # 3. Выбор топ-N документов
        top_docs = self.selector.select_top_documents_with_diversity(
            reranked, top_n
        )

        # Дополняем до top_n если нужно
        while len(top_docs) < top_n and len(reranked) > 0:
            # Берем любые оставшиеся документы
            remaining = [web_id for web_id in reranked['web_id'].unique()
                        if web_id not in top_docs]
            if remaining:
                top_docs.extend(remaining[:top_n - len(top_docs)])
            else:
                break

        return {
            'query': query,
            'documents_id': top_docs,
            'reranked_chunks': reranked,
            'num_candidates': len(candidates)
        }


if __name__ == "__main__":
    print("Модуль retrieval.py готов к использованию")
