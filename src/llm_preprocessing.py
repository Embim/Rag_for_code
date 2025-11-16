"""
LLM-based предобработка документов

Интегрирована в main_pipeline.py через флаг --llm-clean

Что делает:
1. Удаляет мусор (навигация, футеры, реклама) через LLM
2. Извлекает сущности (продукты, услуги, термины)
3. Определяет темы документов
4. Оценивает полезность
5. Сохраняет метаданные для улучшения поиска
"""
import pandas as pd
from llama_cpp import Llama
from tqdm import tqdm
import json
import re
from pathlib import Path
from typing import Dict, Optional

from src.config import (
    LLM_MODEL_FILE,
    LLM_CONTEXT_SIZE,
    LLM_GPU_LAYERS,
    MODELS_DIR
)


class LLMDocumentCleaner:
    """
    LLM-based очистка и обогащение документов

    Использует Qwen3-32B (или другую LLM) для:
    - Удаления мусора из веб-документов
    - Извлечения ключевой информации
    - Добавления метаданных для улучшения поиска
    """

    def __init__(self, model_path: Optional[str] = None, verbose: bool = True):
        """
        Args:
            model_path: путь к GGUF модели (если None - использует из config)
            verbose: выводить прогресс
        """
        if model_path is None:
            model_path = str(MODELS_DIR / LLM_MODEL_FILE)

        self.model_path = model_path
        self.verbose = verbose
        self.llm = None

        if verbose:
            print(f"\n{'='*80}")
            print(f"📥 Инициализация LLM Document Cleaner")
            print(f"   Модель: {Path(model_path).name}")
            print(f"{'='*80}\n")

    def load_model(self):
        """Загрузка LLM модели"""
        if self.llm is not None:
            return  # уже загружена

        model_path = Path(self.model_path)

        if not model_path.exists():
            raise FileNotFoundError(
                f"LLM модель не найдена: {model_path}\n"
                f"Скачайте модель: python download_models.py"
            )

        if self.verbose:
            print(f"⏳ Загрузка модели...")

        self.llm = Llama(
            model_path=str(model_path),
            n_ctx=LLM_CONTEXT_SIZE,
            n_gpu_layers=LLM_GPU_LAYERS,
            n_batch=512,
            n_threads=8,
            use_mlock=True,
            verbose=False
        )

        if self.verbose:
            print(f"✅ Модель загружена!")

    def clean_document(self, text: str) -> Dict:
        """
        Очистка одного документа через LLM

        Args:
            text: исходный текст документа

        Returns:
            dict с полями:
                - clean_text: очищенный текст
                - products: список продуктов
                - actions: список действий
                - conditions: список условий
                - topics: список тем
                - usefulness_score: оценка полезности (0-1)
                - is_useful: bool
        """
        # Убеждаемся что модель загружена
        if self.llm is None:
            self.load_model()

        # Ограничиваем длину для контекста
        text_truncated = text[:4000]

        prompt = f"""<|im_start|>system
Ты - эксперт по анализу банковских документов Альфа-Банка. Твоя задача:

1. УДАЛИТЬ весь мусор:
   - Навигацию (меню, ссылки "Поделиться", "Назад", хлебные крошки)
   - Футеры (© 2001-2025, адреса офисов, лицензии)
   - Рекламные блоки ("Откройте карту сегодня!")
   - Cookie-баннеры и уведомления
   - Контактные данные (если они не часть основной информации)
   - Списки продуктов без деталей

2. ИЗВЛЕЧЬ ключевую информацию:
   - Продукты/услуги: Альфа-Карта, А-Клуб, ипотека, вклады и т.д.
   - Действия: оплата ЖКХ, перевод, открытие счета, получение справки
   - Условия: комиссии (0%, 1.5%), лимиты (100000), требования (от 18 лет)

3. ОПРЕДЕЛИТЬ темы (максимум 3):
   кредитные_карты, дебетовые_карты, переводы, жкх, кэшбэк,
   счета_реквизиты, комиссии, лимиты, безопасность,
   мобильное_приложение, альфа_онлайн, ипотека, кредиты,
   вклады, инвестиции, страхование

4. ОЦЕНИТЬ полезность:
   - 0.0-0.3: мусор (только навигация/реклама/общие фразы)
   - 0.4-0.6: частично полезно (есть информация, но неполная)
   - 0.7-1.0: очень полезный документ (конкретная инструкция/условия)
<|im_end|>
<|im_start|>user
Документ:
{text_truncated}

Верни ТОЛЬКО JSON в формате:
{{
  "clean_text": "очищенный текст без мусора, только суть",
  "products": ["Альфа-Карта", "А-Клуб"],
  "actions": ["оплата ЖКХ", "получение кэшбэка"],
  "conditions": ["комиссия 0%", "лимит 15000"],
  "topics": ["кредитные_карты", "кэшбэк"],
  "usefulness_score": 0.8,
  "is_useful": true
}}
<|im_end|>
<|im_start|>assistant
"""

        try:
            response = self.llm(
                prompt,
                max_tokens=2048,
                temperature=0.1,
                stop=["<|im_end|>"],
            )

            response_text = response['choices'][0]['text']

            # Извлекаем JSON из ответа
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group(0))
                return result
            else:
                # Fallback если JSON не найден
                return self._fallback_result(text_truncated)

        except Exception as e:
            if self.verbose:
                print(f"  ⚠️  Ошибка обработки: {e}")
            return self._fallback_result(text_truncated)

    def _fallback_result(self, text: str) -> Dict:
        """Fallback результат если LLM не работает"""
        return {
            "clean_text": text,
            "products": [],
            "actions": [],
            "conditions": [],
            "topics": [],
            "usefulness_score": 0.5,
            "is_useful": True
        }

    def process_documents(self, documents_df: pd.DataFrame,
                         text_column: str = 'text') -> pd.DataFrame:
        """
        Обработка DataFrame с документами

        Args:
            documents_df: DataFrame с документами
            text_column: название колонки с текстом

        Returns:
            DataFrame с добавленными колонками:
                - clean_text
                - products (JSON list)
                - actions (JSON list)
                - conditions (JSON list)
                - topics (JSON list)
                - usefulness_score (float)
                - is_useful (bool)
        """
        # Убеждаемся что модель загружена
        if self.llm is None:
            self.load_model()

        results = []

        if self.verbose:
            print(f"\n🚀 Обработка {len(documents_df)} документов через LLM...")
            iterator = tqdm(documents_df.iterrows(), total=len(documents_df), desc="LLM Cleaning")
        else:
            iterator = documents_df.iterrows()

        for idx, row in iterator:
            text = row[text_column]

            # Очищаем через LLM
            cleaned = self.clean_document(text)

            # Создаем новую строку с оригинальными + новыми данными
            result_row = {
                **row.to_dict(),
                'clean_text': cleaned.get('clean_text', text),
                'products': json.dumps(cleaned.get('products', []), ensure_ascii=False),
                'actions': json.dumps(cleaned.get('actions', []), ensure_ascii=False),
                'conditions': json.dumps(cleaned.get('conditions', []), ensure_ascii=False),
                'topics': json.dumps(cleaned.get('topics', []), ensure_ascii=False),
                'usefulness_score': cleaned.get('usefulness_score', 0.5),
                'is_useful': cleaned.get('is_useful', True)
            }

            results.append(result_row)

        result_df = pd.DataFrame(results)

        if self.verbose:
            # Статистика
            useful_count = result_df['is_useful'].sum()
            avg_score = result_df['usefulness_score'].mean()

            print(f"\n📈 Статистика LLM обработки:")
            print(f"   Полезных: {useful_count}/{len(result_df)} ({useful_count/len(result_df)*100:.1f}%)")
            print(f"   Средняя оценка: {avg_score:.2f}")

            # Топ темы
            all_topics = []
            for topics_json in result_df['topics']:
                try:
                    topics = json.loads(topics_json)
                    all_topics.extend(topics)
                except:
                    pass

            if all_topics:
                from collections import Counter
                topic_counts = Counter(all_topics)

                print(f"\n📊 Топ-5 тем:")
                for topic, count in topic_counts.most_common(5):
                    print(f"   {topic}: {count}")

        return result_df

    def filter_by_usefulness(self, documents_df: pd.DataFrame,
                            min_score: float = 0.3) -> pd.DataFrame:
        """
        Фильтрация документов по полезности

        Args:
            documents_df: DataFrame с документами (после process_documents)
            min_score: минимальный порог usefulness_score

        Returns:
            Отфильтрованный DataFrame
        """
        before_count = len(documents_df)

        filtered_df = documents_df[
            (documents_df['is_useful'] == True) &
            (documents_df['usefulness_score'] >= min_score)
        ].copy()

        after_count = len(filtered_df)
        removed_count = before_count - after_count

        if self.verbose:
            print(f"\n🗑️  Фильтрация по полезности (min_score={min_score}):")
            print(f"   Было: {before_count}")
            print(f"   Осталось: {after_count}")
            print(f"   Удалено: {removed_count} ({removed_count/before_count*100:.1f}%)")

        return filtered_df


def apply_llm_cleaning(documents_df: pd.DataFrame,
                      min_usefulness: float = 0.3,
                      verbose: bool = True) -> pd.DataFrame:
    """
    Удобная функция для применения LLM очистки

    Args:
        documents_df: DataFrame с документами
        min_usefulness: минимальный порог полезности для фильтрации
        verbose: выводить прогресс

    Returns:
        Очищенный и обогащенный DataFrame
    """
    cleaner = LLMDocumentCleaner(verbose=verbose)

    # Обработка
    cleaned_df = cleaner.process_documents(documents_df)

    # Фильтрация (опционально)
    if min_usefulness > 0:
        cleaned_df = cleaner.filter_by_usefulness(cleaned_df, min_score=min_usefulness)

    return cleaned_df


if __name__ == "__main__":
    # Простой тест
    print("="*80)
    print("ТЕСТ LLM DOCUMENT CLEANER")
    print("="*80)

    # Тестовый документ
    test_doc = """
    Главная / Карты / Альфа-Карта

    Альфа-Карта - дебетовая карта с кэшбэком

    Получайте кэшбэк 2% на все покупки и до 10% в категориях на выбор.
    Бесплатное обслуживание при сумме покупок от 10000 рублей в месяц.

    Оформить онлайн

    © 2001-2025 Альфа-Банк. Лицензия ЦБ РФ №1326
    """

    test_df = pd.DataFrame([{'text': test_doc, 'web_id': 1}])

    cleaner = LLMDocumentCleaner(verbose=True)
    result_df = cleaner.process_documents(test_df)

    print("\n" + "="*80)
    print("РЕЗУЛЬТАТ:")
    print("="*80)
    print(f"\nОчищенный текст:\n{result_df.iloc[0]['clean_text']}")
    print(f"\nПродукты: {result_df.iloc[0]['products']}")
    print(f"Темы: {result_df.iloc[0]['topics']}")
    print(f"Полезность: {result_df.iloc[0]['usefulness_score']}")
