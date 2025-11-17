"""
Построение векторного и BM25 индексов
"""
import os
import numpy as np
import pandas as pd
import pickle
from typing import List, Tuple, Dict
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from tqdm import tqdm
import logging

from src.config import (
    EMBEDDING_MODEL,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_CHUNK_SIZE,
    EMBEDDING_DEVICE,
    MODELS_DIR,
    WEAVIATE_URL,
    WEAVIATE_CLASS_NAME
)

try:
    import weaviate
    from weaviate.classes.config import Configure, Property, DataType
    WEAVIATE_AVAILABLE = True
except ImportError:
    WEAVIATE_AVAILABLE = False
    logging.getLogger(__name__).warning("weaviate-client not installed. WeaviateIndexer will not be available.")





class WeaviateIndexer:
    """Класс для работы с Weaviate векторной базой данных"""

    def __init__(self,
                 url: str = WEAVIATE_URL,
                 class_name: str = WEAVIATE_CLASS_NAME,
                 embedding_model: str = EMBEDDING_MODEL,
                 device: str = EMBEDDING_DEVICE):
        """
        Args:
            url: URL Weaviate сервера
            class_name: имя класса/коллекции в Weaviate
            embedding_model: модель для эмбеддингов
            device: устройство для вычислений (cpu/cuda)
        """
        if not WEAVIATE_AVAILABLE:
            raise ImportError("weaviate-client не установлен. Установите: pip install weaviate-client")

        logging.getLogger(__name__).info(f"Подключение к Weaviate: {url}")

        # Подключение к Weaviate
        try:
            self.client = weaviate.connect_to_local(
                host="localhost",
                port=8080,
                grpc_port=50051
            )
            logging.getLogger(__name__).info("Успешно подключено к Weaviate")
        except Exception as e:
            logging.getLogger(__name__).error(f"Ошибка подключения к Weaviate: {e}")
            logging.getLogger(__name__).info("Убедитесь, что Weaviate запущен: docker-compose up -d")
            raise

        self.class_name = class_name

        logging.getLogger(__name__).info(f"Загрузка embedding модели: {embedding_model}")

        # Специальная обработка для Qwen моделей
        if "Qwen" in embedding_model:
            logging.getLogger(__name__).info("Используется Qwen модель, загружаем с trust_remote_code=True")
            self.model = SentenceTransformer(embedding_model, device=device, trust_remote_code=True)
        else:
            self.model = SentenceTransformer(embedding_model, device=device)

        self.dimension = self.model.get_sentence_embedding_dimension()

        self._create_schema()

    def _create_schema(self):
        """Создание схемы класса в Weaviate"""
        try:
            # Проверяем, существует ли уже класс
            if self.client.collections.exists(self.class_name):
                logging.getLogger(__name__).info(f"Класс {self.class_name} уже существует")
                self.collection = self.client.collections.get(self.class_name)
                return

            # Создаем новый класс
            logging.getLogger(__name__).info(f"Создание класса {self.class_name}...")

            self.collection = self.client.collections.create(
                name=self.class_name,
                # Используем простую схему без векторайзера
                properties=[
                    Property(name="chunk_id", data_type=DataType.TEXT),
                    Property(name="web_id", data_type=DataType.INT),
                    Property(name="title", data_type=DataType.TEXT),
                    Property(name="text", data_type=DataType.TEXT),
                    # Дополнительные метаданные из websites.csv и LLM-clean
                    Property(name="url", data_type=DataType.TEXT),
                    Property(name="kind", data_type=DataType.TEXT),
                    Property(name="entities", data_type=DataType.TEXT),
                    Property(name="topics", data_type=DataType.TEXT),
                    Property(name="word_count", data_type=DataType.INT),
                    Property(name="char_count", data_type=DataType.INT),
                    # Информация о чанке
                    Property(name="chunk_index", data_type=DataType.INT),  # номер чанка в документе
                ]
            )
            logging.getLogger(__name__).info(f"Класс {self.class_name} создан успешно")

        except Exception as e:
            logging.getLogger(__name__).error(f"Ошибка при создании схемы: {e}")
            raise

    def create_embeddings(self, texts: List[str],
                         batch_size: int = EMBEDDING_BATCH_SIZE,
                         show_progress: bool = True) -> np.ndarray:
        """
        Создание эмбеддингов для списка текстов

        Args:
            texts: список текстов
            batch_size: размер батча
            show_progress: показывать прогресс-бар

        Returns:
            массив эмбеддингов
        """
        logging.getLogger(__name__).info(f"Генерация эмбеддингов для {len(texts)} текстов...")

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        logging.getLogger(__name__).info(f"Создано эмбеддингов: {embeddings.shape}")
        return embeddings

    def index_documents(self, chunks_df: pd.DataFrame,
                       batch_size: int = 100,
                       show_progress: bool = True,
                       embedding_chunk_size: int = None):
        """
        Индексация документов в Weaviate (итеративно по батчам)

        Args:
            chunks_df: DataFrame с чанками (chunk_id, web_id, title, text)
            batch_size: размер батча для загрузки в Weaviate
            show_progress: показывать прогресс
            embedding_chunk_size: количество документов для обработки за раз (для экономии GPU памяти)
        """
        import torch
        import gc

        # Используем значение из config если не передано
        if embedding_chunk_size is None:
            embedding_chunk_size = EMBEDDING_CHUNK_SIZE

        logging.getLogger(__name__).info(f"Индексация {len(chunks_df)} документов в Weaviate...")
        logging.getLogger(__name__).info(f"Обработка по {embedding_chunk_size} документов за раз для экономии памяти GPU")

        total_docs = len(chunks_df)
        num_chunks = (total_docs + embedding_chunk_size - 1) // embedding_chunk_size

        # Обрабатываем документы итеративно по частям
        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * embedding_chunk_size
            end_idx = min(start_idx + embedding_chunk_size, total_docs)

            logging.getLogger(__name__).info(f"[Батч {chunk_idx + 1}/{num_chunks}] Обработка документов {start_idx}-{end_idx}...")

            # Получаем подмножество документов
            chunk_df = chunks_df.iloc[start_idx:end_idx]
            texts = chunk_df['text'].tolist()

            # Генерируем эмбеддинги только для этой части
            embeddings = self.create_embeddings(texts, show_progress=show_progress)

            # Загружаем в Weaviate
            with self.collection.batch.dynamic() as batch:
                iterator = enumerate(zip(chunk_df.iterrows(), embeddings))
                if show_progress:
                    iterator = tqdm(iterator, total=len(chunk_df), desc=f"Загрузка батча {chunk_idx + 1}/{num_chunks}")

                for idx, ((_, row), embedding) in iterator:
                    # Извлекаем номер чанка из chunk_id (формат: "web_id_chunk_index")
                    chunk_id_str = str(row['chunk_id'])
                    chunk_index = 0
                    if '_' in chunk_id_str:
                        try:
                            chunk_index = int(chunk_id_str.split('_')[-1])
                        except ValueError:
                            chunk_index = 0

                    properties = {
                        "chunk_id": chunk_id_str,
                        "web_id": int(row['web_id']),
                        "title": str(row.get('title', '')),
                        "text": str(row['text']),
                        # Дополнительные метаданные
                        "url": str(row.get('url', '')),
                        "kind": str(row.get('kind', '')),
                        "chunk_index": chunk_index,
                    }

                    # Опциональные дополнительные свойства
                    if 'entities' in row and pd.notna(row['entities']):
                        properties["entities"] = str(row['entities'])
                    if 'topics' in row and pd.notna(row['topics']):
                        properties["topics"] = str(row['topics'])
                    if 'word_count' in row and pd.notna(row['word_count']):
                        properties["word_count"] = int(row['word_count'])
                    if 'char_count' in row and pd.notna(row['char_count']):
                        properties["char_count"] = int(row['char_count'])

                    batch.add_object(
                        properties=properties,
                        vector=embedding.tolist()
                    )

            # Очистка памяти после каждого батча
            del embeddings
            del texts
            del chunk_df
            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logging.getLogger(__name__).info("GPU память очищена")

        logging.getLogger(__name__).info(f"Индексация завершена. Всего документов: {total_docs}")

    def search(self, query: str, k: int = 10, alpha: float = 0.5) -> Tuple[List[float], List[Dict]]:
        """
        Поиск похожих документов (поддерживает как векторный, так и гибридный поиск)

        Args:
            query: текст запроса
            k: количество результатов
            alpha: вес для векторного поиска (0-1), если None - только векторный поиск
                   alpha=1.0 - только векторный, alpha=0.0 - только BM25, alpha=0.5 - баланс

        Returns:
            (scores, results) - скоры и найденные документы
        """
        # Генерируем эмбеддинг запроса
        query_embedding = self.model.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]

        # Выполняем гибридный поиск (BM25 + векторный)
        response = self.collection.query.hybrid(
            query=query,  # для BM25
            vector=query_embedding.tolist(),  # для векторного поиска
            alpha=alpha,  # баланс между BM25 и векторным (0=BM25, 1=vector)
            limit=k,
            return_metadata=["score"]
        )

        # Извлекаем результаты
        results = []
        scores = []

        for obj in response.objects:
            # Score уже нормализован Weaviate
            score = obj.metadata.score if hasattr(obj.metadata, 'score') else 1.0
            scores.append(score)

            results.append({
                'chunk_id': obj.properties['chunk_id'],
                'web_id': obj.properties['web_id'],
                'title': obj.properties['title'],
                'text': obj.properties['text'],
                'url': obj.properties.get('url', ''),
                'kind': obj.properties.get('kind', ''),
                'chunk_index': obj.properties.get('chunk_index', 0),
                'entities': obj.properties.get('entities', ''),
                'topics': obj.properties.get('topics', ''),
                'word_count': obj.properties.get('word_count', None),
                'char_count': obj.properties.get('char_count', None),
            })

        return scores, results

    def search_vector_only(self, query: str, k: int = 10) -> Tuple[List[float], List[Dict]]:
        """
        Только векторный поиск (без BM25)

        Args:
            query: текст запроса
            k: количество результатов

        Returns:
            (scores, results) - скоры и найденные документы
        """
        # Генерируем эмбеддинг запроса
        query_embedding = self.model.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]

        # Выполняем поиск
        response = self.collection.query.near_vector(
            near_vector=query_embedding.tolist(),
            limit=k,
            return_metadata=["distance"]
        )

        # Извлекаем результаты
        results = []
        scores = []

        for obj in response.objects:
            # Преобразуем distance в similarity score (чем меньше distance, тем выше score)
            score = 1.0 / (1.0 + obj.metadata.distance)
            scores.append(score)

            results.append({
                'chunk_id': obj.properties['chunk_id'],
                'web_id': obj.properties['web_id'],
                'title': obj.properties['title'],
                'text': obj.properties['text'],
                'url': obj.properties.get('url', ''),
                'kind': obj.properties.get('kind', ''),
                'chunk_index': obj.properties.get('chunk_index', 0),
                'entities': obj.properties.get('entities', ''),
                'topics': obj.properties.get('topics', ''),
                'word_count': obj.properties.get('word_count', None),
                'char_count': obj.properties.get('char_count', None),
            })

        return scores, results

    def delete_all(self):
        """Удаление всех документов из коллекции"""
        try:
            self.client.collections.delete(self.class_name)
            logging.getLogger(__name__).info(f"Коллекция {self.class_name} удалена")
            self._create_schema()
        except Exception as e:
            logging.getLogger(__name__).error(f"Ошибка при удалении: {e}")

    def close(self):
        """Закрытие соединения с Weaviate"""
        if hasattr(self, 'client'):
            self.client.close()
            logging.getLogger(__name__).info("Соединение с Weaviate закрыто")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class BM25Indexer:
    """Класс для создания и работы с BM25 индексом"""

    def __init__(self):
        self.bm25 = None
        self.tokenized_corpus = None

    def tokenize(self, text: str) -> List[str]:
        """Простая токенизация по пробелам"""
        return text.lower().split()

    def build_index(self, texts: List[str]):
        """
        Построение BM25 индекса

        Args:
            texts: список текстов для индексации
        """
        logging.getLogger(__name__).info(f"Построение BM25 индекса для {len(texts)} текстов...")

        self.tokenized_corpus = [self.tokenize(text) for text in tqdm(texts)]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

        logging.getLogger(__name__).info("BM25 индекс построен")

    def search(self, query: str, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Поиск по BM25

        Args:
            query: запрос
            k: количество результатов

        Returns:
            (scores, indices)
        """
        if self.bm25 is None:
            raise ValueError("Индекс не построен")

        tokenized_query = self.tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        # Получаем топ-k индексов
        top_k_indices = np.argsort(scores)[::-1][:k]
        top_k_scores = scores[top_k_indices]

        return top_k_scores, top_k_indices

    def save_index(self, filepath: str):
        """Сохранение BM25 индекса"""
        filepath = str(filepath)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        with open(filepath, 'wb') as f:
            pickle.dump({
                'bm25': self.bm25,
                'tokenized_corpus': self.tokenized_corpus
            }, f)
        logging.getLogger(__name__).info(f"BM25 индекс сохранен: {filepath}")

    def load_index(self, filepath: str):
        """Загрузка BM25 индекса"""
        filepath = str(filepath)
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.bm25 = data['bm25']
            self.tokenized_corpus = data['tokenized_corpus']
        logging.getLogger(__name__).info(f"BM25 индекс загружен: {filepath}")


def build_indexes(*args, **kwargs):
    raise NotImplementedError("FAISS удалён. Используйте WeaviateIndexer для индексации.")


if __name__ == "__main__":
    print("Weaviate-only: используйте WeaviateIndexer. FAISS и EmbeddingIndexer удалены.")
