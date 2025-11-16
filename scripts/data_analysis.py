"""
Анализ структуры данных для RAG пайплайна
"""
import pandas as pd
import numpy as np

def analyze_data():
    print("=" * 80)
    print("АНАЛИЗ ДАННЫХ RAG ПАЙПЛАЙНА")
    print("=" * 80)

    # 1. Анализ websites.csv
    print("\n1. WEBSITES.CSV (База документов)")
    print("-" * 80)
    df_websites = pd.read_csv('websites.csv')
    print(f"Количество документов: {len(df_websites)}")
    print(f"Колонки: {list(df_websites.columns)}")
    print(f"\nПримеры web_id: {df_websites['web_id'].head(10).tolist()}")
    print(f"Типы документов (kind): {df_websites['kind'].value_counts().to_dict()}")

    # Анализ размеров текстов
    df_websites['text_len'] = df_websites['text'].fillna('').str.len()
    print(f"\nСтатистика длины текстов:")
    print(f"  Минимум: {df_websites['text_len'].min()} символов")
    print(f"  Максимум: {df_websites['text_len'].max()} символов")
    print(f"  Среднее: {df_websites['text_len'].mean():.0f} символов")
    print(f"  Медиана: {df_websites['text_len'].median():.0f} символов")

    # Примеры документов
    print(f"\nПример документа (web_id={df_websites.iloc[0]['web_id']}):")
    print(f"  URL: {df_websites.iloc[0]['url']}")
    print(f"  Title: {df_websites.iloc[0]['title'][:100]}...")
    print(f"  Text (first 200 chars): {df_websites.iloc[0]['text'][:200]}...")

    # 2. Анализ questions_clean.csv
    print("\n" + "=" * 80)
    print("2. QUESTIONS_CLEAN.CSV (Вопросы)")
    print("-" * 80)
    df_questions = pd.read_csv('questions_clean.csv')
    print(f"Количество вопросов: {len(df_questions)}")
    print(f"Колонки: {list(df_questions.columns)}")

    # Анализ длины вопросов
    df_questions['query_len'] = df_questions['query'].str.len()
    print(f"\nСтатистика длины вопросов:")
    print(f"  Минимум: {df_questions['query_len'].min()} символов")
    print(f"  Максимум: {df_questions['query_len'].max()} символов")
    print(f"  Среднее: {df_questions['query_len'].mean():.1f} символов")
    print(f"  Медиана: {df_questions['query_len'].median():.0f} символов")

    print(f"\nПримеры вопросов:")
    for i in range(min(5, len(df_questions))):
        print(f"  q_id={df_questions.iloc[i]['q_id']}: {df_questions.iloc[i]['query']}")

    # 3. Анализ examples_for_participants.csv
    print("\n" + "=" * 80)
    print("3. EXAMPLES_FOR_PARTICIPANTS.CSV (Эталонные примеры)")
    print("-" * 80)
    df_examples = pd.read_csv('examples_for_participants.csv')
    print(f"Количество примеров: {len(df_examples)}")
    print(f"Колонки: {list(df_examples.columns)}")

    print(f"\nПример 1:")
    print(f"  Query: {df_examples.iloc[0]['query']}")
    print(f"  Chunk 1 (first 150 chars): {str(df_examples.iloc[0]['chunk_1'])[:150]}...")
    print(f"  Perfect answer (first 150 chars): {str(df_examples.iloc[0]['perfect_answer'])[:150]}...")

    # Подсчет количества непустых чанков для каждого примера
    chunk_cols = ['chunk_1', 'chunk_2', 'chunk_3', 'chunk_4', 'chunk_5']
    for idx, row in df_examples.iterrows():
        non_empty_chunks = sum(1 for col in chunk_cols if pd.notna(row[col]) and str(row[col]).strip())
        if idx < 3:
            print(f"\nПример {idx+1}: {non_empty_chunks} релевантных чанков")

    print("\n" + "=" * 80)
    print("ИТОГОВАЯ СТАТИСТИКА")
    print("=" * 80)
    print(f"Документов в базе: {len(df_websites)}")
    print(f"Вопросов для обработки: {len(df_questions)}")
    print(f"Эталонных примеров: {len(df_examples)}")
    print(f"Общий объем текстов: {df_websites['text_len'].sum() / 1024**2:.2f} МБ")
    print("\nАнализ завершен!")

if __name__ == "__main__":
    analyze_data()
