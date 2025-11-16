"""
Потоковая обработка документов для построения базы знаний

Вместо загрузки всех документов в память (batch processing):
    documents_df (все) → llm_clean (все) → chunk (все) → index (все)

Используем потоковую обработку (streaming):
    для каждого документа: load → clean → chunk → накопить → index батч

Преимущества:
- Меньше памяти (не держим весь DataFrame)
- Быстрее (индексируем по мере обработки для Weaviate)
- Прогресс виден сразу
"""
import pandas as pd
import json
from pathlib import Path
from typing import List, Dict, Optional, Union
import gc
from tqdm import tqdm

from src.preprocessing import TextPreprocessor
from src.chunking import DocumentChunker
from src.logger import get_logger, log_timing
from src.config import CSV_CHUNKSIZE, CSV_COUNT_CHUNKSIZE

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class StreamingDocumentProcessor:
    """
    Потоковая обработка документов: load → clean → chunk → accumulate

    Уменьшает использование памяти обрабатывая документы по одному
    вместо загрузки всего датасета в память.
    """

    def __init__(self,
                 llm_clean: bool = False,
                 min_usefulness: float = 0.3,
                 chunk_batch_size: int = 500,
                 csv_chunksize: int = None):
        """
        Args:
            llm_clean: использовать LLM для очистки документов
            min_usefulness: минимальный порог полезности (0.0-1.0)
            chunk_batch_size: сколько чанков накапливать перед индексацией батча
            csv_chunksize: сколько строк CSV читать за раз (если None - из config.CSV_CHUNKSIZE)
        """
        self.llm_clean = llm_clean
        self.min_usefulness = min_usefulness
        self.chunk_batch_size = chunk_batch_size
        self.csv_chunksize = csv_chunksize if csv_chunksize is not None else CSV_CHUNKSIZE

        self.logger = get_logger(__name__)

        # Инициализация компонентов
        self.preprocessor = TextPreprocessor()
        self.chunker = DocumentChunker()

        # LLM cleaner (загружаем только если нужен)
        self.llm_cleaner = None
        self.llm_clean_failed = False  # флаг что LLM очистка запрошена, но не загрузилась
        if llm_clean:
            try:
                from src.llm_preprocessing import LLMDocumentCleaner
                self.llm_cleaner = LLMDocumentCleaner(verbose=True)
                self.llm_cleaner.load_model()
                self.logger.info("✓ LLM Document Cleaner загружен")
            except Exception as e:
                self.logger.warning(f"Не удалось загрузить LLM cleaner: {e}")
                self.logger.warning("Продолжаем без LLM очистки")
                self.llm_clean_failed = True
                # НЕ меняем self.llm_clean на False, чтобы показать правильное сообщение

    def process_document(self, doc_row: pd.Series) -> List[Dict]:
        """
        Обработка одного документа: preprocess → llm_clean → chunk

        Args:
            doc_row: строка из CSV (pandas Series)

        Returns:
            list of chunk dicts
        """
        # 1. Предобработка
        processed_text = self.preprocessor.preprocess_document(
            text=doc_row.get('text', ''),
            title=doc_row.get('title', ''),
            apply_lemmatization=False
        )

        if not processed_text or len(processed_text.strip()) < 10:
            # Слишком короткий текст - пропускаем
            return []

        # 2. LLM очистка (если включена)
        entities = ''
        topics = ''

        if self.llm_clean and self.llm_cleaner:
            try:
                cleaned_result = self.llm_cleaner.clean_document(processed_text)

                # Фильтруем по полезности
                usefulness = cleaned_result.get('usefulness_score', 1.0)
                is_useful = cleaned_result.get('is_useful', True)

                if is_useful and usefulness >= self.min_usefulness:
                    # Используем очищенный текст
                    processed_text = cleaned_result.get('clean_text', processed_text)

                    # Собираем метаданные
                    products = cleaned_result.get('products', [])
                    actions = cleaned_result.get('actions', [])
                    conditions = cleaned_result.get('conditions', [])

                    # Комбинируем entities
                    all_entities = products + actions + conditions
                    if all_entities:
                        entities = json.dumps(all_entities, ensure_ascii=False)

                    # Темы
                    topics_list = cleaned_result.get('topics', [])
                    if topics_list:
                        topics = json.dumps(topics_list, ensure_ascii=False)
                else:
                    # Документ бесполезен - пропускаем
                    self.logger.debug(f"Пропущен документ web_id={doc_row.get('web_id')} (usefulness={usefulness:.2f})")
                    return []

            except Exception as e:
                # Ошибка LLM - используем исходный текст
                self.logger.debug(f"Ошибка LLM для web_id={doc_row.get('web_id')}: {e}")
                pass

        # 3. Чанкинг
        chunks = self.chunker.chunk_by_words(
            text=processed_text,
            web_id=int(doc_row.get('web_id', 0)),
            title=str(doc_row.get('title', '')),
            url=str(doc_row.get('url', '')),
            kind=str(doc_row.get('kind', '')),
            entities=entities,
            topics=topics
        )

        return chunks

    def process_csv_streaming(self,
                             csv_path: str,
                             indexer = None,
                             for_weaviate: bool = False) -> Optional[pd.DataFrame]:
        """
        Потоковая обработка CSV (режим Weaviate):
        - Читает по csv_chunksize документов за раз
        - Обрабатывает каждый: preprocess → llm_clean → chunk
        - Накапливает чанки в батчи
        - Индексирует батчи сразу в Weaviate и очищает память

        Args:
            csv_path: путь к websites.csv
            indexer: WeaviateIndexer (для streaming индексации) или None
            for_weaviate: True если используем Weaviate (индексируем сразу)

        Returns:
            DataFrame со всеми чанками (для FAISS) или None (для Weaviate)
        """
        self.logger.info("="*80)
        self.logger.info("ПОТОКОВАЯ ОБРАБОТКА ДОКУМЕНТОВ")
        self.logger.info("="*80)
        self.logger.info(f"Режим: Weaviate (streaming index)")
        self.logger.info(f"LLM очистка: {'ВКЛ' if self.llm_clean and not self.llm_clean_failed else 'ВЫКЛ'}")
        if self.llm_clean:
            if self.llm_clean_failed:
                # LLM очистка запрошена, но модель не загрузилась
                from src.config import MODELS_DIR, LLM_MODEL_FILE
                model_path = MODELS_DIR / LLM_MODEL_FILE
                self.logger.warning(f"⚠ LLM очистка запрошена, но модель не найдена!")
                self.logger.warning(f"   Ожидаемый путь: {model_path}")
                self.logger.warning(f"   Логи не будут писаться. Скачайте модель: python scripts/download_models.py")
                print(f"  ⚠ LLM очистка запрошена, но модель не найдена!")
                print(f"     Ожидаемый путь: {model_path}")
                print(f"     Логи не будут писаться. Скачайте модель: python scripts/download_models.py")
            elif self.llm_cleaner:
                self.logger.info(f"Порог полезности: {self.min_usefulness}")
                self.logger.info(f"✓ LLM Document Cleaner готов (логи: outputs/llm_cleaning.log)")
                print(f"  ✓ LLM Document Cleaner готов (логи: outputs/llm_cleaning.log)")
            else:
                self.logger.warning("⚠ LLM Document Cleaner не загружен - логи не будут писаться!")
                print(f"  ⚠ LLM Document Cleaner не загружен - логи не будут писаться!")
        else:
            self.logger.info("ℹ LLM очистка выключена - файл llm_cleaning.log будет пустым")
            print(f"  ℹ LLM очистка ВЫКЛЮЧЕНА (запустите с --llm-clean для логирования)")
        self.logger.info(f"Размер батча чанков: {self.chunk_batch_size}")
        self.logger.info(f"Размер чанка CSV: {self.csv_chunksize} документов")
        self.logger.info("="*80)

        # Подсчитываем общее количество документов для прогресс-бара
        print("\n" + "="*80)
        print("📊 Подсчет общего количества документов...")
        print("="*80)
        try:
            # Быстрый подсчет: читаем только первую колонку большими батчами
            # Используем большой chunksize (из config) т.к. мы только считаем строки,
            # не обрабатываем данные - это намного быстрее чем обработка (где chunksize=5-10)
            total_docs = 0
            for chunk in pd.read_csv(csv_path, chunksize=CSV_COUNT_CHUNKSIZE, usecols=[0]):
                total_docs += len(chunk)
            print(f"✓ Всего документов в CSV: {total_docs:,}")
        except Exception as e:
            # Если не получилось - используем динамический прогресс
            print(f"⚠ Не удалось подсчитать документы: {e}")
            print("   Используется динамический прогресс-бар")
            total_docs = None
        print("="*80 + "\n")

        all_chunks = []
        chunk_batch = []

        total_docs_processed = 0
        total_docs_filtered = 0
        total_chunks_created = 0
        batches_indexed = 0

        # Читаем CSV по частям (streaming)
        csv_reader = pd.read_csv(csv_path, chunksize=self.csv_chunksize)

        # Создаем tqdm прогресс-бар
        desc = "🚀 Построение базы знаний"
        if self.llm_clean:
            desc += " (с LLM очисткой)"
        pbar = tqdm(
            total=total_docs,
            desc=desc,
            unit="док",
            ncols=100,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
        )

        try:
            for csv_chunk_idx, doc_chunk_df in enumerate(csv_reader):
                self.logger.info(f"\n[Батч CSV {csv_chunk_idx + 1}] Обработка {len(doc_chunk_df)} документов...")

                for idx, doc_row in doc_chunk_df.iterrows():
                    # Обработка одного документа
                    doc_chunks = self.process_document(doc_row)

                    if doc_chunks:
                        chunk_batch.extend(doc_chunks)
                        total_chunks_created += len(doc_chunks)
                    else:
                        total_docs_filtered += 1

                    total_docs_processed += 1
                    
                    # Обновляем прогресс-бар
                    pbar.update(1)
                    pbar.set_postfix({
                        'чанков': total_chunks_created,
                        'отфильтр.': total_docs_filtered,
                        'батчей': batches_indexed
                    })

                    # Если накопили достаточно чанков для батча
                    if len(chunk_batch) >= self.chunk_batch_size:
                        # Конвертируем в DataFrame
                        batch_df = pd.DataFrame(chunk_batch)

                        if for_weaviate and indexer is not None:
                            # Weaviate: индексируем сразу
                            self.logger.info(f"  → Индексация батча {batches_indexed + 1}: {len(batch_df)} чанков в Weaviate...")

                            with log_timing(self.logger, f"Индексация батча {batches_indexed + 1}"):
                                indexer.index_documents(batch_df, show_progress=False)

                            batches_indexed += 1

                            # Для Weaviate: сохраняем метаданные чанков (без эмбеддингов)
                            all_chunks.extend(chunk_batch)

                            # Очищаем батч из памяти
                            chunk_batch = []
                            del batch_df

                            # Чистим GPU память
                            gc.collect()
                            if TORCH_AVAILABLE and torch.cuda.is_available():
                                torch.cuda.empty_cache()
                        else:
                            raise RuntimeError("Ожидался режим Weaviate с активным indexer")

            # Прогресс уже обновляется через tqdm
            self.logger.info(
                f"  Прогресс: {total_docs_processed} документов | "
                f"Чанков создано: {total_chunks_created} | "
                f"Отфильтровано: {total_docs_filtered}"
            )
        
        finally:
            # Закрываем прогресс-бар
            pbar.close()

        # Обработка остатка
        if chunk_batch:
            batch_df = pd.DataFrame(chunk_batch)

            if for_weaviate and indexer is not None:
                self.logger.info(f"  → Индексация финального батча: {len(batch_df)} чанков...")
                indexer.index_documents(batch_df, show_progress=False)
                batches_indexed += 1

                # Сохраняем метаданные
                all_chunks.extend(chunk_batch)
                del batch_df
            else:
                raise RuntimeError("Ожидался режим Weaviate с активным indexer")

        # Итоговая статистика
        print("\n" + "="*80)
        print("📈 СТАТИСТИКА ОБРАБОТКИ")
        print("="*80)
        print(f"Документов обработано: {total_docs_processed:,}")
        print(f"Документов отфильтровано: {total_docs_filtered:,} ({total_docs_filtered/max(total_docs_processed,1)*100:.1f}%)")
        print(f"Чанков создано: {total_chunks_created:,}")
        print(f"Среднее чанков/документ: {total_chunks_created/max(total_docs_processed-total_docs_filtered,1):.1f}")

        if for_weaviate:
            print(f"Батчей проиндексировано в Weaviate: {batches_indexed}")

        print("="*80 + "\n")
        
        # Дублируем в лог
        self.logger.info("\n" + "="*80)
        self.logger.info("СТАТИСТИКА ОБРАБОТКИ")
        self.logger.info("="*80)
        self.logger.info(f"Документов обработано: {total_docs_processed}")
        self.logger.info(f"Документов отфильтровано: {total_docs_filtered} ({total_docs_filtered/max(total_docs_processed,1)*100:.1f}%)")
        self.logger.info(f"Чанков создано: {total_chunks_created}")
        self.logger.info(f"Среднее чанков/документ: {total_chunks_created/max(total_docs_processed-total_docs_filtered,1):.1f}")

        if for_weaviate:
            self.logger.info(f"Батчей проиндексировано в Weaviate: {batches_indexed}")

        self.logger.info("="*80)

        # Конвертируем в DataFrame
        if all_chunks:
            chunks_df = pd.DataFrame(all_chunks)
            self.logger.info(f"DataFrame создан: {len(chunks_df)} строк")
            return chunks_df
        else:
            self.logger.warning("Ни одного чанка не создано!")
            return pd.DataFrame()


def build_knowledge_base_streaming(csv_path: str,
                                   indexer = None,
                                   for_weaviate: bool = False,
                                   llm_clean: bool = False,
                                   min_usefulness: float = 0.3,
                                   chunk_batch_size: int = 500,
                                   csv_chunksize: int = None) -> Optional[pd.DataFrame]:
    """
    Удобная функция для построения базы знаний потоковым методом

    Args:
        csv_path: путь к websites.csv
        indexer: WeaviateIndexer или None
        for_weaviate: True если используем Weaviate
        llm_clean: использовать LLM очистку
        min_usefulness: минимальный порог полезности
        chunk_batch_size: размер батча для индексации
        csv_chunksize: сколько документов читать за раз (если None - из config.CSV_CHUNKSIZE)

    Returns:
        DataFrame с чанками (для FAISS) или None (для Weaviate)
    """
    processor = StreamingDocumentProcessor(
        llm_clean=llm_clean,
        min_usefulness=min_usefulness,
        chunk_batch_size=chunk_batch_size,
        csv_chunksize=csv_chunksize
    )

    chunks_df = processor.process_csv_streaming(
        csv_path=csv_path,
        indexer=indexer,
        for_weaviate=for_weaviate
    )

    return chunks_df


if __name__ == "__main__":
    # Простой тест
    print("StreamingDocumentProcessor")
    print("Используйте через main_pipeline.py build")
