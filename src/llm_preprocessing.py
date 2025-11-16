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
import logging
from logging.handlers import RotatingFileHandler

from src.config import (
    LLM_MODEL_FILE,
    LLM_CONTEXT_SIZE,
    LLM_GPU_LAYERS,
    MODELS_DIR,
    OUTPUTS_DIR,
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

        # Отдельный логгер для хранения результатов работы LLM
        # (чтобы можно было анализировать, что именно вернула модель)
        self.llm_logger = logging.getLogger("llm_cleaning")
        self._init_llm_logger()

        if verbose:
            print(f"\n{'='*80}")
            print(f"📥 Инициализация LLM Document Cleaner")
            print(f"   Модель: {Path(model_path).name}")
            print(f"{'='*80}\n")

    def _init_llm_logger(self):
        """
        Инициализация отдельного лог-файла для результатов LLM очистки.

        Формат: одна строка = один JSON с результатом clean_document.
        """
        try:
            OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        except Exception:
            # Если не удалось создать директорию — тихо выходим, чтобы не ломать пайплайн
            return

        log_path = OUTPUTS_DIR / "llm_cleaning.log"

        # Проверяем, не добавлен ли уже хендлер на этот файл
        handler_exists = any(
            isinstance(h, RotatingFileHandler) and getattr(h, "baseFilename", None) == str(log_path)
            for h in self.llm_logger.handlers
        )
        if handler_exists:
            return

        handler = RotatingFileHandler(
            str(log_path),
            maxBytes=25 * 1024 * 1024,
            backupCount=3,
            encoding="utf-8",
        )
        # Логируем в компактном формате: только сообщение (JSON)
        formatter = logging.Formatter("%(message)s")
        handler.setFormatter(formatter)

        # INFO достаточно, т.к. каждая запись — один результат LLM
        handler.setLevel(logging.INFO)
        self.llm_logger.setLevel(logging.INFO)
        self.llm_logger.addHandler(handler)
        # Не дублируем в root, оставляем только отдельный файл
        self.llm_logger.propagate = False
        
        # Отладочный вывод (только в verbose режиме)
        if self.verbose:
            print(f"  📝 LLM лог-файл: {log_path}")
            print(f"     Хендлеров: {len(self.llm_logger.handlers)}")

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

        prompt = f"""Ты — аккуратный и строгий эксперт по анализу банковских документов Альфа-Банка. 
Твоя задача — очистить текст документа и структурировать его содержимое. 
НЕ придумывай и НЕ добавляй новую информацию, которой нет в исходном документе.

ТЕБЕ ДАН ИСХОДНЫЙ ТЕКСТ (ДОКУМЕНТ):
{text_truncated}

ВЫПОЛНИ СЛЕДУЮЩЕЕ:

1. ОЧИСТКА ДОКУМЕНТА
Удалить из текста всё, что не несёт смысловой нагрузки для клиента:
- навигацию (меню, хлебные крошки, «Назад», «Поделиться», «Вверх», пункты сайта),
- футеры (© 2001–2025, юридические адреса офисов, лицензии, копирайты),
- рекламные лозунги и баннеры («Откройте карту сегодня!», «Узнайте больше», «Оставьте заявку»),
- cookie-баннеры и уведомления,
- технические блоки («нажмите здесь», «перейдите по ссылке», «смотрите также»),
- контактные данные, если они сами по себе не являются частью конкретной инструкции или условия продукта,
- пустые/общие списки продуктов без конкретных условий или деталей.

СОХРАНИ:
- описания продуктов/услуг и условий,
- конкретные шаги, инструкции, требования,
- важные числовые параметры (комиссии, лимиты, сроки, возрастные ограничения),
- формулировки, которые реально помогут ответить на вопросы клиента.

Не перефразируй текст без необходимости — по возможности сохраняй исходную формулировку, только убирай мусор и дубли.

2. ИЗВЛЕЧЕНИЕ КЛЮЧЕВОЙ ИНФОРМАЦИИ
Определи, какая полезная информация содержится в документе:
- Продукты/услуги (например: Альфа-Карта, А-Клуб, ипотека, вклад, кредит, дебетовая карта и т.п.).
- Действия (например: оплата ЖКХ, перевод, открытие счёта, оформление карты, получение выписки).
- Условия (комиссии, проценты, лимиты, требования к клиенту, сроки, валюты, минимальные суммы и т.д.).

3. ОПРЕДЕЛЕНИЕ ТЕМ (МАКСИМУМ 3)
Выбери не более трёх тем, которые лучше всего описывают очищенный документ, только из этого списка:

кредитные_карты, дебетовые_карты, переводы, жкх, кэшбэк,
счета_реквизиты, комиссии, лимиты, безопасность,
мобильное_приложение, альфа_онлайн, ипотека, кредиты,
вклады, инвестиции, страхование

4. ОЦЕНКА ПОЛЕЗНОСТИ ДОКУМЕНТА
Оцени, насколько документ полезен для ответа на реальные вопросы клиента банка:

- 0.0–0.3 — почти бесполезный (в основном навигация, реклама, общие маркетинговые фразы).
- 0.4–0.6 — частично полезный (есть важная информация, но она обрывочная или недостаточно конкретная).
- 0.7–1.0 — очень полезный (конкретные условия, инструкции, тарифы, шаги).

Не ставь оценку > 0.6, если в тексте нет конкретики (чисел, чётких условий, понятных шагов).

ФОРМАТ ОТВЕТА
Верни результат СТРОГО в формате JSON без дополнительного текста:

{{
  "clean_text": "очищенный текст документа, без мусора",
  "key_info_summary": "краткое текстовое резюме того, о чём этот документ (1–3 предложения)",
  "topics": ["тема_1", "тема_2"],
  "usefulness_score": 0.0,
  "reasoning": "кратко объясни, почему поставлена именно такая оценка полезности (1–2 предложения)"
}}"""

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
                raw_result = json.loads(json_match.group(0))
                # Нормализация полей под downstream:
                # заполняем отсутствующие поля пустыми структурами
                raw_result.setdefault("clean_text", text_truncated)
                raw_result.setdefault("topics", [])
                raw_result.setdefault("usefulness_score", 0.5)
                # совместимость: эти поля могут использоваться downstream (entities сборка)
                raw_result.setdefault("products", [])
                raw_result.setdefault("actions", [])
                raw_result.setdefault("conditions", [])
                # derive is_useful по прежней логике (порог ~0.3)
                raw_result["is_useful"] = bool(raw_result.get("usefulness_score", 0.5) >= 0.3)

                # Логируем результат в отдельный лог-файл (без лишнего шума)
                self._log_llm_result(raw_result, original_text=text_truncated)

                return raw_result
            else:
                # Fallback если JSON не найден
                fallback = self._fallback_result(text_truncated)
                self._log_llm_result(fallback, original_text=text_truncated, reason="json_parse_failed")
                return fallback

        except Exception as e:
            if self.verbose:
                print(f"  ⚠️  Ошибка обработки: {e}")
            fallback = self._fallback_result(text_truncated)
            self._log_llm_result(fallback, original_text=text_truncated, reason=str(e))
            return fallback

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

    def _log_llm_result(self, result: Dict, original_text: str, reason: Optional[str] = None) -> None:
        """
        Логирование результата LLM очистки в отдельный JSON-лог.

        Мы логируем:
        - краткие метаданные,
        - усечённый original_text и clean_text (чтобы лог не разрастался бесконечно).
        """
        if not self.llm_logger.handlers:
            # Логгер не инициализирован (например, не удалось создать файл)
            # Попробуем переинициализировать
            try:
                self._init_llm_logger()
            except Exception:
                pass
            
            # Если все еще нет хендлеров - выходим
            if not self.llm_logger.handlers:
                return

        try:
            log_record = {
                "reason": reason,
                "usefulness_score": result.get("usefulness_score"),
                "is_useful": result.get("is_useful"),
                "topics": result.get("topics", []),
                "products": result.get("products", []),
                "actions": result.get("actions", []),
                "conditions": result.get("conditions", []),
                # web_id может добавляться на следующих этапах — здесь обычно None,
                # но поле оставляем для единообразия если в будущем туда будут писать
                "web_id": result.get("web_id"),
                # Превью текстов (усекаем, чтобы файл был разумного размера)
                "original_text_preview": original_text[:1000],
                "clean_text_preview": str(result.get("clean_text", ""))[:1000],
            }
            self.llm_logger.info(json.dumps(log_record, ensure_ascii=False))
            
            # Принудительно сбрасываем буферы всех хендлеров
            for handler in self.llm_logger.handlers:
                handler.flush()
        except Exception as e:
            # Логирование не должно ломать основной пайплайн
            # Но можем вывести предупреждение в verbose режиме
            if self.verbose:
                print(f"  ⚠️  Ошибка логирования LLM результата: {e}")
            pass

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
