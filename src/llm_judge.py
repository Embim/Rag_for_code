"""
LLM as Judge для оценки полноты контекста (Coverage Judge)
Использует Qwen3-4B-Instruct-2507-GGUF для оценки, достаточно ли информации
в найденных фрагментах для ответа на вопрос
"""
import json
from typing import List, Dict, Tuple
import pandas as pd
from pathlib import Path
import os

# Фикс для CUDA путей перед импортом llama_cpp
if os.name == 'nt':  # Windows
    cuda_path = os.environ.get('CUDA_PATH', '')

    # Если CUDA_PATH установлен но не существует - пытаемся исправить
    if cuda_path and not os.path.exists(cuda_path):
        cuda_base = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
        if os.path.exists(cuda_base):
            # Находим все установленные версии CUDA
            versions = sorted([d for d in os.listdir(cuda_base) if d.startswith('v')])
            if versions:
                # Берем первую найденную версию
                correct_path = os.path.join(cuda_base, versions[0])
                os.environ['CUDA_PATH'] = correct_path
            else:
                # CUDA папка есть, но версий нет - удаляем CUDA_PATH
                del os.environ['CUDA_PATH']
        else:
            # CUDA вообще не установлена - удаляем переменную
            del os.environ['CUDA_PATH']

try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except Exception as e:
    LLAMA_CPP_AVAILABLE = False
    print(f"Warning: llama-cpp-python не загружен: {e}")
    print("LLM Judge будет недоступен.")

from src.config import (
    LLM_JUDGE_MODEL,
    LLM_JUDGE_FILE,
    LLM_JUDGE_CONTEXT_SIZE,
    LLM_JUDGE_TEMPERATURE,
    LLM_JUDGE_MAX_TOKENS,
    LLM_GPU_LAYERS,
    MODELS_DIR
)


class SubquestionAnalyzer:
    """Анализ вопроса и разбиение на подвопросы"""

    def __init__(self, llm_model=None):
        """
        Args:
            llm_model: модель LLM (опционально, если None - используем эвристики)
        """
        self.llm_model = llm_model

    def analyze_question(self, question: str) -> List[str]:
        """
        Разбивает вопрос на подвопросы (subquestions)

        Args:
            question: исходный вопрос

        Returns:
            список подвопросов
        """
        if self.llm_model:
            return self._analyze_with_llm(question)
        else:
            return self._analyze_heuristic(question)

    def _analyze_with_llm(self, question: str) -> List[str]:
        """Разбиение с помощью LLM"""
        prompt = f"""Проанализируй следующий вопрос пользователя банка и разбей его на отдельные подвопросы (аспекты), на которые нужно ответить.

Вопрос: {question}

Выведи список подвопросов в формате JSON:
{{"subquestions": ["подвопрос 1", "подвопрос 2", ...]}}

Ответ (JSON):"""

        response = self.llm_model.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=512
        )

        try:
            content = response['choices'][0]['message']['content']
            # Извлекаем JSON из ответа
            start_idx = content.find('{')
            end_idx = content.rfind('}') + 1
            if start_idx != -1 and end_idx > start_idx:
                json_str = content[start_idx:end_idx]
                data = json.loads(json_str)
                return data.get('subquestions', [question])
        except Exception as e:
            print(f"Ошибка парсинга подвопросов: {e}")
            return [question]

        return [question]

    def _analyze_heuristic(self, question: str) -> List[str]:
        """
        Простая эвристическая разбивка на основе ключевых слов
        """
        # Для большинства простых вопросов возвращаем сам вопрос
        # Для сложных можно добавить логику разбиения
        subquestions = [question]

        # Если вопрос содержит "и", "а также", "или" - пытаемся разбить
        if " и " in question.lower() or " а также " in question.lower():
            # Простая эвристика: разделяем по союзам
            parts = question.replace(" а также ", " и ").split(" и ")
            if len(parts) > 1:
                subquestions = [p.strip() for p in parts if p.strip()]

        return subquestions


class CoverageJudge:
    """LLM Judge для оценки полноты контекста"""

    def __init__(self, model_path: str = None):
        """
        Args:
            model_path: путь к GGUF файлу модели
        """
        if not LLAMA_CPP_AVAILABLE:
            raise ImportError(
                "llama-cpp-python не установлен. "
                "Установите: pip install llama-cpp-python"
            )

        # Определяем путь к модели
        if model_path is None:
            model_path = MODELS_DIR / LLM_JUDGE_FILE

        model_path = Path(model_path)

        if not model_path.exists():
            raise FileNotFoundError(
                f"Модель не найдена: {model_path}\n"
                f"Скачайте модель с HuggingFace:\n"
                f"huggingface-cli download {LLM_JUDGE_MODEL} {LLM_JUDGE_FILE} "
                f"--local-dir {MODELS_DIR}"
            )

        print(f"Загрузка LLM Judge модели: {model_path}")

        # Загружаем модель
        self.llm = Llama(
            model_path=str(model_path),
            n_ctx=LLM_JUDGE_CONTEXT_SIZE,
            n_gpu_layers=LLM_GPU_LAYERS,  # используем настройку из config
            verbose=False
        )

        self.analyzer = SubquestionAnalyzer(llm_model=self.llm)

        print(f"LLM Judge загружен успешно (GPU layers: {LLM_GPU_LAYERS})")

    def evaluate_coverage(
        self,
        question: str,
        subquestions: List[str],
        chunks: pd.DataFrame,
        top_k: int = 10
    ) -> Dict:
        """
        Оценивает полноту контекста для ответа на вопрос

        Args:
            question: исходный вопрос
            subquestions: список подвопросов
            chunks: DataFrame с найденными чанками (text, rerank_score, web_id)
            top_k: количество топовых чанков для анализа

        Returns:
            словарь с оценкой полноты:
            {
                'coverage_score': float (0-1),
                'subquestion_coverage': {
                    'sub_1': 'full|partial|none',
                    'sub_2': 'full|partial|none',
                    ...
                },
                'missing_aspects': List[str],
                'recommendation': str
            }
        """
        # Берем топ-K чанков
        top_chunks = chunks.head(top_k)

        # Формируем контекст из чанков
        context = "\n\n---\n\n".join([
            f"[Фрагмент {i+1}]\n{row['text']}"
            for i, (_, row) in enumerate(top_chunks.iterrows())
        ])

        # Формируем промпт для оценки
        prompt = self._build_coverage_prompt(question, subquestions, context)

        # Получаем оценку от LLM
        response = self.llm.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            temperature=LLM_JUDGE_TEMPERATURE,
            max_tokens=LLM_JUDGE_MAX_TOKENS
        )

        # Парсим ответ
        result = self._parse_coverage_response(
            response['choices'][0]['message']['content'],
            subquestions
        )

        return result

    def _build_coverage_prompt(
        self,
        question: str,
        subquestions: List[str],
        context: str
    ) -> str:
        """Формирует промпт для оценки полноты контекста"""

        subquestions_text = "\n".join([
            f"{i+1}. {sq}" for i, sq in enumerate(subquestions)
        ])

        prompt = f"""Ты - эксперт по оценке полноты информации для ответа на вопросы клиентов банка.

ВОПРОС КЛИЕНТА:
{question}

ПОДВОПРОСЫ (АСПЕКТЫ):
{subquestions_text}

НАЙДЕННЫЕ ФРАГМЕНТЫ ДОКУМЕНТОВ:
{context}

ЗАДАЧА:
Оцени для каждого подвопроса, достаточно ли информации в найденных фрагментах для полного ответа.

Для каждого подвопроса укажи:
- "full" - есть полный, явный ответ
- "partial" - есть частичная информация, но ответ неполный
- "none" - информации нет или она не релевантна

Выведи оценку в формате JSON:
{{
    "subquestion_coverage": {{
        "1": "full|partial|none",
        "2": "full|partial|none",
        ...
    }},
    "missing_aspects": ["список отсутствующих аспектов, если есть"],
    "recommendation": "нужно ли искать дополнительную информацию (да/нет)"
}}

Ответ (JSON):"""

        return prompt

    def _parse_coverage_response(
        self,
        response_text: str,
        subquestions: List[str]
    ) -> Dict:
        """
        Парсит ответ LLM и вычисляет coverage_score
        """
        try:
            # Извлекаем JSON из ответа
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1

            if start_idx == -1 or end_idx <= start_idx:
                raise ValueError("JSON не найден в ответе")

            json_str = response_text[start_idx:end_idx]
            data = json.loads(json_str)

            # Вычисляем coverage_score
            coverage_map = data.get('subquestion_coverage', {})

            full_count = sum(1 for v in coverage_map.values() if v == 'full')
            partial_count = sum(1 for v in coverage_map.values() if v == 'partial')
            total = len(subquestions)

            if total == 0:
                coverage_score = 1.0
            else:
                coverage_score = (full_count + 0.5 * partial_count) / total

            return {
                'coverage_score': coverage_score,
                'subquestion_coverage': coverage_map,
                'missing_aspects': data.get('missing_aspects', []),
                'recommendation': data.get('recommendation', 'нет'),
                'full_count': full_count,
                'partial_count': partial_count,
                'none_count': total - full_count - partial_count
            }

        except Exception as e:
            print(f"Ошибка парсинга ответа LLM Judge: {e}")
            print(f"Ответ: {response_text}")

            # Возвращаем дефолтную оценку
            return {
                'coverage_score': 0.5,
                'subquestion_coverage': {},
                'missing_aspects': [],
                'recommendation': 'нет',
                'error': str(e)
            }

    def analyze_and_evaluate(
        self,
        question: str,
        chunks: pd.DataFrame,
        top_k: int = 10
    ) -> Tuple[List[str], Dict]:
        """
        Комбинированный метод: анализирует вопрос и оценивает покрытие

        Args:
            question: вопрос
            chunks: найденные чанки
            top_k: количество топовых чанков

        Returns:
            (subquestions, coverage_result)
        """
        # 1. Анализируем вопрос
        subquestions = self.analyzer.analyze_question(question)

        # 2. Оцениваем покрытие
        coverage = self.evaluate_coverage(question, subquestions, chunks, top_k)

        return subquestions, coverage

    def __del__(self):
        """Освобождаем ресурсы"""
        if hasattr(self, 'llm'):
            del self.llm


# Singleton для избежания множественной загрузки модели
_judge_instance = None


def get_coverage_judge(model_path: str = None) -> CoverageJudge:
    """
    Получить singleton экземпляр CoverageJudge

    Args:
        model_path: путь к модели (опционально)

    Returns:
        CoverageJudge instance
    """
    global _judge_instance

    if _judge_instance is None:
        _judge_instance = CoverageJudge(model_path)

    return _judge_instance


if __name__ == "__main__":
    # Тестирование
    print("Тестирование LLM Judge")

    # Проверка доступности модели
    model_path = MODELS_DIR / LLM_JUDGE_FILE

    if not model_path.exists():
        print(f"\n[ВНИМАНИЕ] Модель не найдена: {model_path}")
        print(f"\nДля скачивания модели выполните:")
        print(f"  huggingface-cli download {LLM_JUDGE_MODEL} {LLM_JUDGE_FILE} --local-dir {MODELS_DIR}")
        print(f"\nИли используйте:")
        print(f"  pip install huggingface_hub")
        print(f"  from huggingface_hub import hf_hub_download")
        print(f"  hf_hub_download(repo_id='{LLM_JUDGE_MODEL}', filename='{LLM_JUDGE_FILE}', local_dir='{MODELS_DIR}')")
    else:
        print(f"Модель найдена: {model_path}")

        try:
            # Создаем judge
            judge = get_coverage_judge()

            # Тестовый вопрос
            test_question = "Почему не начисляется кэшбэк за оплату коммунальных услуг?"

            # Тестовые чанки
            test_chunks = pd.DataFrame([
                {
                    'text': 'Кэшбэк начисляется на покупки в магазинах, ресторанах и онлайн-сервисах.',
                    'rerank_score': 0.9,
                    'web_id': 1
                },
                {
                    'text': 'На оплату коммунальных услуг и налогов кэшбэк не распространяется согласно условиям программы.',
                    'rerank_score': 0.95,
                    'web_id': 2
                },
                {
                    'text': 'Подробные условия программы лояльности можно посмотреть в мобильном приложении.',
                    'rerank_score': 0.7,
                    'web_id': 3
                }
            ])

            # Анализируем
            print(f"\nВопрос: {test_question}")
            subquestions, coverage = judge.analyze_and_evaluate(test_question, test_chunks)

            print(f"\nПодвопросы: {subquestions}")
            print(f"\nОценка полноты: {coverage['coverage_score']:.2f}")
            print(f"Покрытие по подвопросам: {coverage['subquestion_coverage']}")
            print(f"Отсутствующие аспекты: {coverage['missing_aspects']}")
            print(f"Рекомендация: {coverage['recommendation']}")

            print("\n✓ Тест завершен успешно!")

        except Exception as e:
            print(f"\n❌ Ошибка: {e}")
