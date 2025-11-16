"""
Предочистка документов через OpenRouter (Sherlock-Think)
Используется ОДИН РАЗ оффлайн для улучшения качества документов
"""
import os
import requests
import json
import pandas as pd
from tqdm import tqdm
import time
from typing import Dict

from src.config import CHUNKS_CSV


class OpenRouterCleaner:
    """Очистка документов через OpenRouter API"""

    def __init__(self, api_key: str = None, model: str = "openrouter/sherlock-think-alpha"):
        """
        Инициализация

        Args:
            api_key: OpenRouter API key (или через env OPENROUTER_API_KEY)
            model: модель для использования
        """
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenRouter API key не найден!\n"
                "Установите: export OPENROUTER_API_KEY='your-key'\n"
                "Или получите бесплатно: https://openrouter.ai/keys"
            )

        self.model = model
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"

        print(f"OpenRouter Cleaner инициализирован: {model}")

    def clean_document(self, text: str, max_retries: int = 3) -> Dict:
        """
        Очистка одного документа через Sherlock-Think

        Args:
            text: исходный текст
            max_retries: количество попыток при ошибке

        Returns:
            dict с очищенными данными
        """
        # Ограничиваем длину (API limits)
        text_truncated = text[:4000]

        prompt = f"""Ты - эксперт по анализу банковских документов. Проанализируй этот документ и:

1. Удали весь мусор:
   - Навигационные элементы (меню, ссылки "Поделиться", "Назад")
   - Футеры (копирайты, юридические данные, адреса банка)
   - Рекламные блоки
   - Cookie-баннеры
   - Повторяющиеся фразы
   - Нерелевантные разделы (контакты, соцсети)

2. Извлеки ключевую информацию:
   - О каких ПРОДУКТАХ/УСЛУГАХ говорится? (Альфа-Карта, А-Клуб, ипотека и т.д.)
   - Какие ДЕЙСТВИЯ описаны? (оплата, перевод, открытие счета)
   - Какие УСЛОВИЯ/ОГРАНИЧЕНИЯ? (комиссии, лимиты, требования)

3. Определи ТЕМЫ документа (максимум 3):
   - Категории: кредитные карты, дебетовые карты, переводы, ЖКХ, кэшбэк,
                 счета и реквизиты, комиссии, лимиты, безопасность,
                 мобильное приложение, онлайн-банк

Исходный документ:
{text_truncated}

Верни СТРОГО JSON в формате:
{{
  "clean_text": "очищенный текст без мусора (только полезная информация)",
  "products": ["Альфа-Карта", "А-Клуб"],
  "actions": ["оплата ЖКХ", "перевод денег"],
  "conditions": ["комиссия 0%", "лимит 100000"],
  "topics": ["кредитные карты", "кэшбэк"],
  "is_useful": true,
  "usefulness_score": 0.8,
  "reason": "Документ содержит полезную информацию о кэшбэке по картам"
}}

Если документ полностью мусор (только навигация/реклама), установи is_useful=false."""

        for attempt in range(max_retries):
            try:
                response = requests.post(
                    self.base_url,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.model,
                        "messages": [
                            {
                                "role": "user",
                                "content": prompt
                            }
                        ],
                        "max_tokens": 2048,
                        "temperature": 0.1,
                    },
                    timeout=60
                )

                if response.status_code == 200:
                    result = response.json()
                    content = result['choices'][0]['message']['content']

                    # Парсим JSON из ответа
                    try:
                        # Ищем JSON в ответе
                        json_start = content.find('{')
                        json_end = content.rfind('}') + 1
                        if json_start != -1 and json_end > json_start:
                            json_str = content[json_start:json_end]
                            parsed = json.loads(json_str)
                            return parsed
                        else:
                            raise ValueError("JSON не найден в ответе")

                    except (json.JSONDecodeError, ValueError) as e:
                        print(f"  Ошибка парсинга JSON: {e}")
                        print(f"  Ответ: {content[:200]}")
                        # Возвращаем минимальный результат
                        return {
                            "clean_text": text_truncated,
                            "products": [],
                            "actions": [],
                            "conditions": [],
                            "topics": [],
                            "is_useful": True,
                            "usefulness_score": 0.5,
                            "reason": "Ошибка обработки"
                        }

                elif response.status_code == 429:  # Rate limit
                    wait_time = 5 * (attempt + 1)
                    print(f"  Rate limit, ждем {wait_time}с...")
                    time.sleep(wait_time)
                    continue

                else:
                    print(f"  Ошибка API: {response.status_code}")
                    print(f"  {response.text[:200]}")
                    time.sleep(2)
                    continue

            except Exception as e:
                print(f"  Ошибка запроса (попытка {attempt + 1}/{max_retries}): {e}")
                time.sleep(2)
                continue

        # Если все попытки провалились
        return {
            "clean_text": text_truncated,
            "products": [],
            "actions": [],
            "conditions": [],
            "topics": [],
            "is_useful": True,
            "usefulness_score": 0.5,
            "reason": "Ошибка обработки"
        }

    def clean_documents_df(self, docs_df: pd.DataFrame,
                          text_column: str = 'text',
                          save_every: int = 10) -> pd.DataFrame:
        """
        Очистка документов с прогресс-баром и checkpoint'ами

        Args:
            docs_df: DataFrame с документами
            text_column: колонка с текстом
            save_every: сохранять checkpoint каждые N документов

        Returns:
            DataFrame с очищенными данными
        """
        print(f"\nОчистка {len(docs_df)} документов через OpenRouter...")
        print(f"Модель: {self.model}")
        print(f"Checkpoint каждые {save_every} документов")

        results = []
        checkpoint_file = "outputs/openrouter_clean_checkpoint.csv"

        # Загружаем checkpoint если есть
        if os.path.exists(checkpoint_file):
            checkpoint_df = pd.read_csv(checkpoint_file)
            results = checkpoint_df.to_dict('records')
            start_idx = len(results)
            print(f"Найден checkpoint, продолжаем с {start_idx}")
        else:
            start_idx = 0

        for idx in tqdm(range(start_idx, len(docs_df)), desc="Очистка"):
            row = docs_df.iloc[idx]
            text = row[text_column]

            # Очищаем через API
            cleaned = self.clean_document(text)

            # Добавляем к результатам
            result_row = {
                **row.to_dict(),
                'clean_text': cleaned.get('clean_text', text),
                'products': json.dumps(cleaned.get('products', []), ensure_ascii=False),
                'actions': json.dumps(cleaned.get('actions', []), ensure_ascii=False),
                'conditions': json.dumps(cleaned.get('conditions', []), ensure_ascii=False),
                'topics': json.dumps(cleaned.get('topics', []), ensure_ascii=False),
                'is_useful': cleaned.get('is_useful', True),
                'usefulness_score': cleaned.get('usefulness_score', 0.5),
                'clean_reason': cleaned.get('reason', '')
            }
            results.append(result_row)

            # Сохраняем checkpoint
            if (idx + 1) % save_every == 0:
                temp_df = pd.DataFrame(results)
                temp_df.to_csv(checkpoint_file, index=False)
                print(f"\n  Checkpoint сохранен: {len(results)} документов")

            # Rate limiting
            time.sleep(1)  # 1 сек между запросами для free tier

        # Финальное сохранение
        cleaned_df = pd.DataFrame(results)
        cleaned_df.to_csv(checkpoint_file, index=False)
        print(f"\nОчистка завершена! Результаты: {checkpoint_file}")

        return cleaned_df


def main():
    """Запуск очистки документов"""
    import sys

    # Проверяем API key
    if not os.environ.get("OPENROUTER_API_KEY"):
        print("\n❌ OPENROUTER_API_KEY не установлен!")
        print("\nКак получить бесплатный ключ:")
        print("1. Перейди на https://openrouter.ai/keys")
        print("2. Создай бесплатный аккаунт")
        print("3. Получи API key")
        print("4. Установи: export OPENROUTER_API_KEY='your-key'")
        sys.exit(1)

    # Загружаем чанки
    chunks_df = pd.read_csv(CHUNKS_CSV)
    print(f"Загружено {len(chunks_df)} чанков")

    # Спрашиваем подтверждение
    print(f"\n⚠️  ВНИМАНИЕ:")
    print(f"   Будет сделано ~{len(chunks_df)} API запросов к OpenRouter")
    print(f"   Free tier: 20 запросов/мин = ~{len(chunks_df)/20:.0f} минут")
    print(f"   С паузами ~{len(chunks_df)/10:.0f} минут")

    confirm = input("\nПродолжить? (yes/no): ")
    if confirm.lower() != 'yes':
        print("Отменено")
        sys.exit(0)

    # Для теста - только первые 50
    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        print("\n🧪 Тестовый режим: только первые 50 документов")
        chunks_df = chunks_df.head(50)

    # Очищаем
    cleaner = OpenRouterCleaner()
    cleaned_df = cleaner.clean_documents_df(chunks_df)

    # Финальное сохранение
    output_path = "data/chunks_openrouter_cleaned.csv"
    cleaned_df.to_csv(output_path, index=False)
    print(f"\n✅ Готово! Результат: {output_path}")

    # Статистика
    useful_count = cleaned_df['is_useful'].sum()
    avg_score = cleaned_df['usefulness_score'].mean()
    print(f"\n📊 Статистика:")
    print(f"   Полезных документов: {useful_count}/{len(cleaned_df)} ({useful_count/len(cleaned_df)*100:.1f}%)")
    print(f"   Средняя полезность: {avg_score:.2f}")


if __name__ == "__main__":
    main()
