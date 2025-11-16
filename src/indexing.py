"""
Построение векторного и BM25 индексов
"""
import os
import numpy as np
import pandas as pd
import faiss
import pickle
from typing import List, Tuple, Dict
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from tqdm import tqdm

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
    print("Warning: weaviate-client not installed. WeaviateIndexer will not be available.")


class EmbeddingIndexer:
    """Класс для создания и работы с векторным индексом"""

    def __init__(self, model_name: str = EMBEDDING_MODEL,
                 device: str = EMBEDDING_DEVICE):
        """
        Args:
            model_name: название модели из SentenceTransformers или HuggingFace
            device: устройство для вычислений (cpu/cuda)
        """
        print(f"Загрузка embedding модели: {model_name}")

        # Специальная обработка для Qwen моделей
        if "Qwen" in model_name:
            print(f"  Используется Qwen модель, загружаем с trust_remote_code=True")
            self.model = SentenceTransformer(model_name, device=device, trust_remote_code=True)
        else:
            self.model = SentenceTransformer(model_name, device=device)

        self.dimension = self.model.get_sentence_embedding_dimension()
        self.index = None
        self.chunk_metadata = None

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
        print(f"Генерация эмбеддингов для {len(texts)} текстов...")

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True  # для косинусного расстояния
        )

        print(f"Создано эмбеддингов: {embeddings.shape}")
        return embeddings

    def build_faiss_index(self, embeddings: np.ndarray,
                         use_gpu: bool = False) -> faiss.Index:
        """
        Построение FAISS индекса

        Args:
            embeddings: массив эмбеддингов
            use_gpu: использовать GPU

        Returns:
            FAISS индекс
        """
        print(f"Построение FAISS индекса размерности {self.dimension}...")

        # Используем IndexFlatIP для косинусного расстояния (Inner Product)
        # т.к. векторы нормализованы
        index = faiss.IndexFlatIP(self.dimension)

        if use_gpu:
            try:
                res = faiss.StandardGpuResources()
                index = faiss.index_cpu_to_gpu(res, 0, index)
                print("FAISS индекс создан на GPU")
            except:
                print("GPU недоступен, используем CPU")

        index.add(embeddings.astype('float32'))
        print(f"Добавлено {index.ntotal} векторов в индекс")

        self.index = index
        return index

    def search(self, query_embedding: np.ndarray,
              k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Поиск k ближайших векторов

        Args:
            query_embedding: вектор запроса
            k: количество результатов

        Returns:
            (scores, indices) - скоры и индексы найденных векторов
        """
        if self.index is None:
            raise ValueError("Индекс не построен")

        # Нормализация запроса
        query_embedding = query_embedding.astype('float32')
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)

        scores, indices = self.index.search(query_embedding, k)
        return scores[0], indices[0]

    def save_index(self, filepath: str):
        """Сохранение индекса на диск"""
        if self.index is None:
            raise ValueError("Индекс не построен")

        # Убедимся что путь - строка, не Path объект
        filepath = str(filepath)

        # Создаем директорию если не существует
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # FAISS не поддерживает кириллицу, используем сериализацию через bytes
        try:
            # Сериализуем индекс в bytes
            index_bytes = faiss.serialize_index(self.index)

            # Сохраняем bytes в файл (Python поддерживает Unicode пути)
            with open(filepath, 'wb') as f:
                f.write(index_bytes)

            print(f"Индекс сохранен: {filepath}")
        except Exception as e:
            # Fallback на прямое сохранение (для путей без кириллицы)
            print(f"[WARNING] Ошибка сериализации: {e}")
            print("[INFO] Попытка прямого сохранения...")
            faiss.write_index(self.index, filepath)
            print(f"Индекс сохранен: {filepath}")

    def load_index(self, filepath: str):
        """Загрузка индекса с диска"""
        filepath = str(filepath)

        # Загружаем bytes из файла
        with open(filepath, 'rb') as f:
            index_bytes = f.read()

        # Пробуем десериализацию (новый формат)
        try:
            # Конвертируем bytes в numpy array для десериализации
            import numpy as np
            index_array = np.frombuffer(index_bytes, dtype=np.uint8)
            self.index = faiss.deserialize_index(index_array)
            print(f"Индекс загружен: {filepath}")
        except Exception as e:
            # Fallback: файл в старом формате, используем прямую загрузку
            # Создаем временный файл без кириллицы
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix='.index') as tmp:
                tmp.write(index_bytes)
                tmp_path = tmp.name

            try:
                self.index = faiss.read_index(tmp_path)
                print(f"Индекс загружен: {filepath} (через временный файл)")
            finally:
                # Удаляем временный файл
                os.unlink(tmp_path)


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

        print(f"Подключение к Weaviate: {url}")

        # Подключение к Weaviate
        try:
            self.client = weaviate.connect_to_local(
                host="localhost",
                port=8080,
                grpc_port=50051
            )
            print("Успешно подключено к Weaviate")
        except Exception as e:
            print(f"Ошибка подключения к Weaviate: {e}")
            print("Убедитесь, что Weaviate запущен: docker-compose up -d")
            raise

        self.class_name = class_name

        print(f"Загрузка embedding модели: {embedding_model}")

        # Специальная обработка для Qwen моделей
        if "Qwen" in embedding_model:
            print(f"  Используется Qwen модель, загружаем с trust_remote_code=True")
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
                print(f"Класс {self.class_name} уже существует")
                self.collection = self.client.collections.get(self.class_name)
                return

            # Создаем новый класс
            print(f"Создание класса {self.class_name}...")

            self.collection = self.client.collections.create(
                name=self.class_name,
                # Используем простую схему без векторайзера
                properties=[
                    Property(name="chunk_id", data_type=DataType.TEXT),
                    Property(name="web_id", data_type=DataType.INT),
                    Property(name="title", data_type=DataType.TEXT),
                    Property(name="text", data_type=DataType.TEXT),
                    # Дополнительные метаданные из websites.csv
                    Property(name="url", data_type=DataType.TEXT),
                    Property(name="kind", data_type=DataType.TEXT),
                    # Информация о чанке
                    Property(name="chunk_index", data_type=DataType.INT),  # номер чанка в документе
                ]
            )
            print(f"Класс {self.class_name} создан успешно")

        except Exception as e:
            print(f"Ошибка при создании схемы: {e}")
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
        print(f"Генерация эмбеддингов для {len(texts)} текстов...")

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        print(f"Создано эмбеддингов: {embeddings.shape}")
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

        print(f"Индексация {len(chunks_df)} документов в Weaviate...")
        print(f"Обработка по {embedding_chunk_size} документов за раз для экономии памяти GPU")

        total_docs = len(chunks_df)
        num_chunks = (total_docs + embedding_chunk_size - 1) // embedding_chunk_size

        # Обрабатываем документы итеративно по частям
        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * embedding_chunk_size
            end_idx = min(start_idx + embedding_chunk_size, total_docs)

            print(f"\n[Батч {chunk_idx + 1}/{num_chunks}] Обработка документов {start_idx}-{end_idx}...")

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
                print(f"  GPU память очищена")

        print(f"\nИндексация завершена. Всего документов: {total_docs}")

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
        query_embedding = self.model.encode([query], normalize_embeddings=True)[0]

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
        query_embedding = self.model.encode([query], normalize_embeddings=True)[0]

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
            })

        return scores, results

    def delete_all(self):
        """Удаление всех документов из коллекции"""
        try:
            self.client.collections.delete(self.class_name)
            print(f"Коллекция {self.class_name} удалена")
            self._create_schema()
        except Exception as e:
            print(f"Ошибка при удалении: {e}")

    def close(self):
        """Закрытие соединения с Weaviate"""
        if hasattr(self, 'client'):
            self.client.close()
            print("Соединение с Weaviate закрыто")

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
        print(f"Построение BM25 индекса для {len(texts)} текстов...")

        self.tokenized_corpus = [self.tokenize(text) for text in tqdm(texts)]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

        print("BM25 индекс построен")

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
        print(f"BM25 индекс сохранен: {filepath}")

    def load_index(self, filepath: str):
        """Загрузка BM25 индекса"""
        filepath = str(filepath)
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.bm25 = data['bm25']
            self.tokenized_corpus = data['tokenized_corpus']
        print(f"BM25 индекс загружен: {filepath}")


def build_indexes(chunks_df: pd.DataFrame,
                 save_dir: str = None) -> Tuple[EmbeddingIndexer, BM25Indexer]:
    """
    Построение обоих индексов (векторного и BM25)

    Args:
        chunks_df: DataFrame с чанками
        save_dir: директория для сохранения индексов

    Returns:
        (embedding_indexer, bm25_indexer)
    """
    if save_dir is None:
        save_dir = MODELS_DIR

    # 1. Векторный индекс
    print("\n" + "="*80)
    print("ПОСТРОЕНИЕ ВЕКТОРНОГО ИНДЕКСА")
    print("="*80)

    embedding_indexer = EmbeddingIndexer()

    # Генерация эмбеддингов
    texts = chunks_df['text'].tolist()
    embeddings = embedding_indexer.create_embeddings(texts)

    # Построение FAISS индекса
    embedding_indexer.build_faiss_index(embeddings)

    # Сохранение метаданных чанков
    embedding_indexer.chunk_metadata = chunks_df

    # Сохранение индекса
    if save_dir:
        save_dir = str(save_dir)  # Конвертируем Path в строку
        os.makedirs(save_dir, exist_ok=True)  # Создаем директорию

        embedding_indexer.save_index(os.path.join(save_dir, "faiss.index"))

        # Сохранение эмбеддингов и метаданных
        np.save(os.path.join(save_dir, "embeddings.npy"), embeddings)
        chunks_df.to_pickle(os.path.join(save_dir, "chunks_metadata.pkl"))

    # 2. BM25 индекс
    print("\n" + "="*80)
    print("ПОСТРОЕНИЕ BM25 ИНДЕКСА")
    print("="*80)

    bm25_indexer = BM25Indexer()
    bm25_indexer.build_index(texts)

    # Сохранение BM25
    if save_dir:
        # save_dir уже строка после конвертации выше
        bm25_indexer.save_index(os.path.join(save_dir, "bm25.pkl"))

    print("\n" + "="*80)
    print("ИНДЕКСЫ ПОСТРОЕНЫ")
    print("="*80)

    return embedding_indexer, bm25_indexer


if __name__ == "__main__":
    # Тестирование
    test_chunks = pd.DataFrame([
        {'chunk_id': '1_0', 'web_id': 1, 'text': 'Альфа-Банк предлагает кредиты'},
        {'chunk_id': '1_1', 'web_id': 1, 'text': 'Кэшбэк на покупки до 10%'},
        {'chunk_id': '2_0', 'web_id': 2, 'text': 'Оплата коммунальных услуг без комиссии'},
    ])

    emb_idx, bm25_idx = build_indexes(test_chunks, save_dir=None)

    # Тест поиска
    query = "кэшбэк за покупки"
    query_emb = emb_idx.model.encode([query])[0]

    print(f"\nПоиск по запросу: '{query}'")
    print("\nВекторный поиск:")
    scores, indices = emb_idx.search(query_emb, k=2)
    for i, (score, idx) in enumerate(zip(scores, indices)):
        print(f"{i+1}. Score: {score:.3f} - {test_chunks.iloc[idx]['text']}")

    print("\nBM25 поиск:")
    scores, indices = bm25_idx.search(query, k=2)
    for i, (score, idx) in enumerate(zip(scores, indices)):
        print(f"{i+1}. Score: {score:.3f} - {test_chunks.iloc[idx]['text']}")
