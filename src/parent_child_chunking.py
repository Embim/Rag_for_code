"""
Parent-Child Chunking для улучшения точности и полноты контекста

Концепция:
- Child chunks (маленькие, 100 слов) - для точного поиска
- Parent chunks (большие, 300 слов) - для полного контекста
- При поиске используем child, но возвращаем parent для LLM

Преимущества:
- Точность поиска (малые чанки лучше матчатся)
- Полнота контекста (большие чанки содержат больше информации)
- Лучшее качество ответов
"""
import pandas as pd
from typing import List, Dict, Tuple
from razdel import sentenize

from src.config import MIN_CHUNK_SIZE


class ParentChildChunker:
    """Создание Parent-Child структуры чанков"""

    def __init__(self,
                 child_size: int = 100,  # слов
                 parent_size: int = 300,  # слов
                 overlap: int = 20):     # слов
        """
        Args:
            child_size: размер child чанка в словах
            parent_size: размер parent чанка в словах
            overlap: перекрытие между чанками
        """
        self.child_size = child_size
        self.parent_size = parent_size
        self.overlap = overlap

        # Валидация
        if child_size >= parent_size:
            raise ValueError(f"child_size ({child_size}) должен быть < parent_size ({parent_size})")

        print(f"[ParentChildChunker] Инициализация:")
        print(f"  Child: {child_size} слов (для поиска)")
        print(f"  Parent: {parent_size} слов (для контекста)")
        print(f"  Overlap: {overlap} слов")

    def create_parent_chunk(self, text: str, web_id: int, parent_idx: int,
                           title: str = "", url: str = "", kind: str = "") -> Dict:
        """
        Создание одного parent chunk

        Args:
            text: текст чанка
            web_id: ID документа
            parent_idx: индекс parent чанка
            title: заголовок
            url: URL
            kind: тип документа

        Returns:
            словарь с parent chunk
        """
        title_prefix = f"[{title}] " if title else ""
        full_text = title_prefix + text

        return {
            'chunk_id': f"{web_id}_p{parent_idx}",
            'parent_id': f"{web_id}_p{parent_idx}",  # parent сам себе parent
            'web_id': web_id,
            'title': title,
            'text': full_text,
            'original_text': text,
            'chunk_type': 'parent',
            'chunk_index': parent_idx,
            'word_count': len(text.split()),
            'url': url,
            'kind': kind,
        }

    def create_child_chunks(self, parent_text: str, web_id: int, parent_idx: int,
                           title: str = "", url: str = "", kind: str = "") -> List[Dict]:
        """
        Создание child chunks из parent chunk

        Args:
            parent_text: текст parent чанка
            web_id: ID документа
            parent_idx: индекс parent чанка
            title: заголовок
            url: URL
            kind: тип

        Returns:
            список child chunks
        """
        words = parent_text.split()
        children = []
        child_idx = 0

        title_prefix = f"[{title}] " if title else ""
        parent_id = f"{web_id}_p{parent_idx}"

        # Разбиваем parent на child chunks скользящим окном
        for i in range(0, len(words), self.child_size - self.overlap):
            chunk_words = words[i:i + self.child_size]

            # Пропускаем слишком маленькие
            if len(chunk_words) < MIN_CHUNK_SIZE and i + self.child_size < len(words):
                continue

            chunk_text = ' '.join(chunk_words)
            full_text = title_prefix + chunk_text

            children.append({
                'chunk_id': f"{web_id}_p{parent_idx}_c{child_idx}",
                'parent_id': parent_id,  # ссылка на parent
                'web_id': web_id,
                'title': title,
                'text': full_text,
                'original_text': chunk_text,
                'chunk_type': 'child',
                'chunk_index': child_idx,
                'parent_index': parent_idx,
                'word_count': len(chunk_words),
                'url': url,
                'kind': kind,
            })

            child_idx += 1

        return children

    def chunk_document(self, text: str, web_id: int,
                      title: str = "", url: str = "", kind: str = "") -> Tuple[List[Dict], List[Dict]]:
        """
        Создание parent и child chunks для одного документа

        Args:
            text: текст документа
            web_id: ID документа
            title: заголовок
            url: URL
            kind: тип

        Returns:
            (parent_chunks, child_chunks)
        """
        if not text:
            return [], []

        words = text.split()
        parents = []
        all_children = []
        parent_idx = 0

        # Создаем parent chunks
        for i in range(0, len(words), self.parent_size - self.overlap):
            parent_words = words[i:i + self.parent_size]

            if len(parent_words) < MIN_CHUNK_SIZE and i + self.parent_size < len(words):
                continue

            parent_text = ' '.join(parent_words)

            # Создаем parent
            parent_chunk = self.create_parent_chunk(
                parent_text, web_id, parent_idx, title, url, kind
            )
            parents.append(parent_chunk)

            # Создаем child chunks из этого parent
            children = self.create_child_chunks(
                parent_text, web_id, parent_idx, title, url, kind
            )
            all_children.extend(children)

            parent_idx += 1

        return parents, all_children

    def chunk_documents_dataframe(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Создание parent-child chunks для всех документов

        Args:
            df: DataFrame с колонками: web_id, title, processed_text, url, kind

        Returns:
            (parent_df, child_df)
        """
        all_parents = []
        all_children = []

        print(f"\n[ParentChildChunker] Обработка {len(df)} документов...")

        for idx, row in df.iterrows():
            parents, children = self.chunk_document(
                row['processed_text'],
                row['web_id'],
                row.get('title', ''),
                row.get('url', ''),
                row.get('kind', '')
            )

            all_parents.extend(parents)
            all_children.extend(children)

            if (idx + 1) % 100 == 0:
                print(f"  Обработано {idx + 1}/{len(df)} документов...")

        parents_df = pd.DataFrame(all_parents)
        children_df = pd.DataFrame(all_children)

        print(f"\n[ParentChildChunker] Результаты:")
        print(f"  Parent chunks: {len(parents_df)} (средний размер: {parents_df['word_count'].mean():.0f} слов)")
        print(f"  Child chunks:  {len(children_df)} (средний размер: {children_df['word_count'].mean():.0f} слов)")
        print(f"  Соотношение child/parent: {len(children_df) / len(parents_df):.1f}")

        return parents_df, children_df


class ParentChildRetriever:
    """
    Retriever с поддержкой Parent-Child chunks

    Стратегия:
    1. Поиск по child chunks (точный)
    2. Получение parent chunks для найденных child
    3. Возврат parent chunks (полный контекст)
    """

    def __init__(self, parents_df: pd.DataFrame, children_df: pd.DataFrame):
        """
        Args:
            parents_df: DataFrame с parent chunks
            children_df: DataFrame с child chunks
        """
        self.parents_df = parents_df
        self.children_df = children_df

        # Создаем индекс parent_id -> parent_chunk для быстрого доступа
        self.parent_index = {
            row['parent_id']: row.to_dict()
            for _, row in parents_df.iterrows()
        }

        print(f"[ParentChildRetriever] Инициализация:")
        print(f"  Parents: {len(self.parents_df)}")
        print(f"  Children: {len(self.children_df)}")

    def get_parents_for_children(self, child_results: pd.DataFrame) -> pd.DataFrame:
        """
        Получить parent chunks для найденных child chunks

        Args:
            child_results: DataFrame с результатами поиска по child chunks
                          (должен содержать колонку 'parent_id')

        Returns:
            DataFrame с parent chunks
        """
        if len(child_results) == 0:
            return pd.DataFrame()

        # Получаем уникальные parent_id
        parent_ids = child_results['parent_id'].unique()

        # Получаем parent chunks
        parent_chunks = []
        for parent_id in parent_ids:
            if parent_id in self.parent_index:
                parent = self.parent_index[parent_id].copy()

                # Копируем метрики от лучшего child chunk
                best_child = child_results[
                    child_results['parent_id'] == parent_id
                ].iloc[0]

                # Переносим scores
                if 'retrieval_score' in best_child:
                    parent['retrieval_score'] = best_child['retrieval_score']
                if 'rerank_score' in best_child:
                    parent['rerank_score'] = best_child['rerank_score']
                if 'final_score' in best_child:
                    parent['final_score'] = best_child['final_score']

                parent_chunks.append(parent)

        parents_df = pd.DataFrame(parent_chunks)

        # Сортируем по score
        if 'rerank_score' in parents_df.columns:
            parents_df = parents_df.sort_values('rerank_score', ascending=False)
        elif 'retrieval_score' in parents_df.columns:
            parents_df = parents_df.sort_values('retrieval_score', ascending=False)

        return parents_df.reset_index(drop=True)


def create_parent_child_chunks(documents_df: pd.DataFrame,
                               child_size: int = 100,
                               parent_size: int = 300,
                               overlap: int = 20) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Удобная функция для создания parent-child chunks

    Args:
        documents_df: DataFrame с документами
        child_size: размер child чанка
        parent_size: размер parent чанка
        overlap: перекрытие

    Returns:
        (parents_df, children_df)
    """
    chunker = ParentChildChunker(
        child_size=child_size,
        parent_size=parent_size,
        overlap=overlap
    )

    return chunker.chunk_documents_dataframe(documents_df)


if __name__ == "__main__":
    # Тест
    print("="*80)
    print("ТЕСТ PARENT-CHILD CHUNKING")
    print("="*80)

    # Тестовый документ
    test_doc = {
        'web_id': 1,
        'title': 'Тестовый документ',
        'processed_text': ' '.join(['Слово'] * 500),  # 500 слов
        'url': 'http://test.com',
        'kind': 'test'
    }

    df = pd.DataFrame([test_doc])

    # Создаем chunks
    parents_df, children_df = create_parent_child_chunks(
        df,
        child_size=100,
        parent_size=300,
        overlap=20
    )

    print(f"\n✅ Результаты:")
    print(f"   Parent chunks: {len(parents_df)}")
    print(f"   Child chunks: {len(children_df)}")
    print(f"\nПервый parent chunk:")
    print(f"   ID: {parents_df.iloc[0]['chunk_id']}")
    print(f"   Слов: {parents_df.iloc[0]['word_count']}")
    print(f"\nПервый child chunk:")
    print(f"   ID: {children_df.iloc[0]['chunk_id']}")
    print(f"   Parent ID: {children_df.iloc[0]['parent_id']}")
    print(f"   Слов: {children_df.iloc[0]['word_count']}")
