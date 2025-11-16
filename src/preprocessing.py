"""
Предобработка текстов документов и вопросов
"""
import re
import pandas as pd
from typing import List, Dict
from natasha import (
    Segmenter,
    MorphVocab,
    NewsEmbedding,
    NewsMorphTagger,
    Doc
)
import pymorphy2

from src.config import SYNONYMS


class TextPreprocessor:
    """Класс для предобработки текстов"""

    def __init__(self):
        # Natasha компоненты для продвинутого NLP
        self.segmenter = Segmenter()
        self.morph_vocab = MorphVocab()

        # Pymorphy2 для лемматизации (быстрее для простых случаев)
        self.morph = pymorphy2.MorphAnalyzer()

        # Словарь синонимов
        self.synonyms = SYNONYMS

    def clean_text(self, text: str) -> str:
        """
        Очистка текста от лишних символов и HTML

        Args:
            text: исходный текст

        Returns:
            очищенный текст
        """
        if pd.isna(text) or not text:
            return ""

        text = str(text)

        # Убираем HTML-подобные артефакты
        text = re.sub(r'<[^>]+>', '', text)

        # Убираем множественные пробелы и переносы
        text = re.sub(r'\s+', ' ', text)

        # Убираем спецсимволы, оставляя буквы, цифры и базовую пунктуацию
        text = re.sub(r'[^\w\s\.\,\!\?\-\:\;\(\)\"\'№]', ' ', text)

        # Убираем множественные пробелы снова
        text = re.sub(r'\s+', ' ', text)

        return text.strip()

    def normalize_text(self, text: str, lowercase: bool = True) -> str:
        """
        Нормализация текста с применением синонимов

        Args:
            text: текст для нормализации
            lowercase: приводить к нижнему регистру

        Returns:
            нормализованный текст
        """
        if not text:
            return ""

        # Очистка
        text = self.clean_text(text)

        # Lowercase
        if lowercase:
            text = text.lower()

        # Применяем словарь синонимов
        for original, replacement in self.synonyms.items():
            text = re.sub(r'\b' + re.escape(original) + r'\b',
                         replacement, text, flags=re.IGNORECASE)

        return text

    def lemmatize_text(self, text: str) -> str:
        """
        Лемматизация текста (приведение слов к начальной форме)

        Args:
            text: текст для лемматизации

        Returns:
            лемматизированный текст
        """
        if not text:
            return ""

        words = text.split()
        lemmatized_words = []

        for word in words:
            # Пропускаем пунктуацию и короткие слова
            if len(word) < 2 or not any(c.isalpha() for c in word):
                lemmatized_words.append(word)
                continue

            # Лемматизация с pymorphy2
            parsed = self.morph.parse(word)[0]
            lemmatized_words.append(parsed.normal_form)

        return ' '.join(lemmatized_words)

    def preprocess_document(self, text: str, title: str = "",
                          apply_lemmatization: bool = False) -> str:
        """
        Полная предобработка документа

        Args:
            text: текст документа
            title: заголовок документа
            apply_lemmatization: применять ли лемматизацию

        Returns:
            предобработанный текст
        """
        # Объединяем заголовок и текст
        full_text = f"{title}. {text}" if title else text

        # Нормализация
        processed = self.normalize_text(full_text)

        # Лемматизация (опционально, может замедлить)
        if apply_lemmatization:
            processed = self.lemmatize_text(processed)

        return processed

    def preprocess_query(self, query: str,
                        apply_lemmatization: bool = False) -> str:
        """
        Предобработка пользовательского вопроса

        Args:
            query: вопрос пользователя
            apply_lemmatization: применять ли лемматизацию

        Returns:
            предобработанный вопрос
        """
        # Базовая нормализация
        processed = self.normalize_text(query)

        # Исправление частых опечаток
        processed = self.fix_common_typos(processed)

        # Лемматизация
        if apply_lemmatization:
            processed = self.lemmatize_text(processed)

        return processed

    def fix_common_typos(self, text: str) -> str:
        """
        Исправление частых опечаток

        Args:
            text: текст с возможными опечатками

        Returns:
            исправленный текст
        """
        typo_dict = {
            'вібрать': 'выбрать',
            'віднять': 'отнять',
            'приходять': 'приходят',
            'не приходить': 'не приходит',
        }

        for typo, correct in typo_dict.items():
            text = text.replace(typo, correct)

        return text


def load_and_preprocess_documents(csv_path: str,
                                  apply_lemmatization: bool = False) -> pd.DataFrame:
    """
    Загрузка и предобработка документов из CSV

    Args:
        csv_path: путь к файлу websites.csv
        apply_lemmatization: применять ли лемматизацию

    Returns:
        DataFrame с предобработанными текстами
    """
    print(f"Загрузка документов из {csv_path}...")
    df = pd.read_csv(csv_path)

    preprocessor = TextPreprocessor()

    print("Предобработка текстов...")
    df['processed_text'] = df.apply(
        lambda row: preprocessor.preprocess_document(
            row['text'],
            row['title'],
            apply_lemmatization=apply_lemmatization
        ),
        axis=1
    )

    print(f"Обработано {len(df)} документов")
    return df


def load_and_preprocess_questions(csv_path: str,
                                  apply_lemmatization: bool = False) -> pd.DataFrame:
    """
    Загрузка и предобработка вопросов из CSV

    Args:
        csv_path: путь к файлу questions_clean.csv
        apply_lemmatization: применять ли лемматизацию

    Returns:
        DataFrame с предобработанными вопросами
    """
    print(f"Загрузка вопросов из {csv_path}...")
    df = pd.read_csv(csv_path)

    preprocessor = TextPreprocessor()

    print("Предобработка вопросов...")
    df['processed_query'] = df['query'].apply(
        lambda q: preprocessor.preprocess_query(q, apply_lemmatization)
    )

    print(f"Обработано {len(df)} вопросов")
    return df


if __name__ == "__main__":
    # Тестирование предобработки
    preprocessor = TextPreprocessor()

    test_text = "Кешбек за оплату услуг ЖКХ не начисляется!"
    print(f"Исходный текст: {test_text}")
    print(f"После нормализации: {preprocessor.normalize_text(test_text)}")
    print(f"После лемматизации: {preprocessor.lemmatize_text(preprocessor.normalize_text(test_text))}")
