"""
Очистка и обогащение документов через LLM
"""
import pandas as pd
from llama_cpp import Llama
from pathlib import Path
import json
from tqdm import tqdm

from src.config import (
    LLM_MODEL_FILE,
    LLM_CONTEXT_SIZE,
    LLM_GPU_LAYERS,
    MODELS_DIR
)


class DocumentCleaner:
    """Очистка документов от мусора и извлечение сущностей"""

    def __init__(self, model_path: str = None):
        """Инициализация LLM для очистки"""
        if model_path is None:
            model_path = str(MODELS_DIR / LLM_MODEL_FILE)

        print(f"Загрузка LLM для очистки документов: {model_path}")

        self.llm = Llama(
            model_path=model_path,
            n_ctx=LLM_CONTEXT_SIZE,
            n_gpu_layers=LLM_GPU_LAYERS,
            n_batch=512,
            n_threads=8,
            verbose=False
        )

        print("  LLM загружена успешно")

    def clean_document(self, text: str) -> dict:
        """
        Очистка одного документа через LLM

        Args:
            text: исходный текст документа

        Returns:
            dict с полями:
                - clean_text: очищенный текст
                - entities: список сущностей
                - topics: список тем
        """
        # Ограничиваем длину текста
        text_truncated = text[:3000]

        prompt = f"""<|im_start|>system
Ты - эксперт по обработке банковских документов. Твоя задача:
1. Убрать мусор: навигацию, футеры, меню, повторы, рекламу
2. Оставить только полезный контент
3. Выделить ключевые сущности (продукты, услуги, термины)
4. Определить основные темы<|im_end|>
<|im_start|>user
Обработай документ:

{text_truncated}

Верни JSON:
{{
  "clean_text": "очищенный текст без мусора",
  "entities": ["Альфа-Карта", "кэшбэк", "ЖКХ"],
  "topics": ["кредитные карты", "оплата услуг"]
}}<|im_end|>
<|im_start|>assistant
"""

        try:
            response = self.llm(
                prompt,
                max_tokens=2048,
                temperature=0.1,
                stop=["<|im_end|>"],
                echo=False
            )

            result_text = response['choices'][0]['text'].strip()

            # Пытаемся распарсить JSON
            try:
                result = json.loads(result_text)
            except:
                # Если не удалось - вернуть как есть
                result = {
                    "clean_text": text_truncated,
                    "entities": [],
                    "topics": []
                }

            return result

        except Exception as e:
            print(f"  Ошибка очистки: {e}")
            return {
                "clean_text": text_truncated,
                "entities": [],
                "topics": []
            }

    def clean_documents_df(self, docs_df: pd.DataFrame,
                          text_column: str = 'text') -> pd.DataFrame:
        """
        Очистка всех документов в DataFrame

        Args:
            docs_df: DataFrame с документами
            text_column: название колонки с текстом

        Returns:
            DataFrame с дополнительными колонками:
                - clean_text
                - entities
                - topics
        """
        print(f"\nОчистка {len(docs_df)} документов...")

        results = []

        for idx, row in tqdm(docs_df.iterrows(), total=len(docs_df)):
            text = row[text_column]
            cleaned = self.clean_document(text)

            results.append({
                **row.to_dict(),
                'clean_text': cleaned['clean_text'],
                'entities': json.dumps(cleaned['entities'], ensure_ascii=False),
                'topics': json.dumps(cleaned['topics'], ensure_ascii=False)
            })

        cleaned_df = pd.DataFrame(results)
        print(f"Очистка завершена!")

        return cleaned_df


def main():
    """Тестовый запуск очистки документов"""
    from src.config import CHUNKS_CSV

    # Загружаем чанки
    chunks_df = pd.read_csv(CHUNKS_CSV)
    print(f"Загружено {len(chunks_df)} чанков")

    # Берем первые 10 для теста
    test_df = chunks_df.head(10)

    # Очищаем
    cleaner = DocumentCleaner()
    cleaned_df = cleaner.clean_documents_df(test_df)

    # Сохраняем
    output_path = "outputs/cleaned_chunks_test.csv"
    cleaned_df.to_csv(output_path, index=False)
    print(f"\nРезультат сохранен: {output_path}")

    # Показываем пример
    print("\n" + "="*80)
    print("ПРИМЕР ОЧИСТКИ:")
    print("="*80)
    print("\nИсходный текст:")
    print(test_df.iloc[0]['text'][:500])
    print("\nОчищенный текст:")
    print(cleaned_df.iloc[0]['clean_text'][:500])
    print("\nСущности:")
    print(cleaned_df.iloc[0]['entities'])
    print("\nТемы:")
    print(cleaned_df.iloc[0]['topics'])


if __name__ == "__main__":
    main()
