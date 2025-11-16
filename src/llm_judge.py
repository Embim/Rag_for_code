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
    LLM_MODE,
    LLM_API_MODEL,
    LLM_API_ROUTING,
    OPENROUTER_API_KEY,
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

    def __init__(self, model_path: str = None, use_api: bool = None):
        """
        Args:
            model_path: путь к GGUF файлу модели (для локального режима)
            use_api: использовать ли API (если None - определяется из LLM_MODE)
        """
        # Определяем режим работы
        if use_api is None:
            use_api = (LLM_MODE == "api")
        
        self.use_api = use_api
        
        if use_api:
            # API режим (OpenRouter)
            print(f"Инициализация LLM Judge (API режим, модель: {LLM_API_MODEL})")
            try:
                from openai import OpenAI
                base_url = "https://openrouter.ai/api/v1"
                
                # OpenRouter требует API ключ даже для бесплатных моделей
                if not OPENROUTER_API_KEY:
                    raise ValueError(
                        "OPENROUTER_API_KEY не установлен!\n"
                        "Получите бесплатный ключ на https://openrouter.ai/keys\n"
                        "Установите: export OPENROUTER_API_KEY=sk-or-v1-..."
                    )
                
                default_headers = {
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "HTTP-Referer": "https://github.com/your-repo",
                    "X-Title": "AlfaBank RAG Pipeline"
                }
                
                # Добавляем провайдера для роутинга (если указан)
                if LLM_API_ROUTING:
                    default_headers["X-OpenRouter-Provider"] = LLM_API_ROUTING
                
                self.client = OpenAI(
                    base_url=base_url,
                    api_key=OPENROUTER_API_KEY,
                    timeout=60,
                    default_headers=default_headers
                )
                self.model_name = LLM_API_MODEL
                self.llm = None  # для совместимости с SubquestionAnalyzer
                self.analyzer = SubquestionAnalyzer(llm_model=None)  # используем эвристики для API
                print(f"LLM Judge (API) инициализирован")
            except ImportError:
                raise ImportError("Установите openai: pip install openai")
        else:
            # Локальный режим
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
        if self.use_api:
            # API режим
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=LLM_JUDGE_TEMPERATURE,
                max_tokens=LLM_JUDGE_MAX_TOKENS
            )
            content = response.choices[0].message.content
        else:
            # Локальный режим
            response = self.llm.create_chat_completion(
                messages=[{"role": "user", "content": prompt}],
                temperature=LLM_JUDGE_TEMPERATURE,
                max_tokens=LLM_JUDGE_MAX_TOKENS
            )
            content = response['choices'][0]['message']['content']

        # Парсим ответ
        result = self._parse_coverage_response(
            content,
            subquestions
        )

        return result

    def _build_coverage_prompt(
        self,
        question: str,
        subquestions: List[str],
        context: str
    ) -> str:
        """Формирует промпт для оценки полноты контекста (строгий JSON-вывод)"""

        subquestions_text = "\n".join([
            f"{i+1}. {sq}" for i, sq in enumerate(subquestions)
        ])

        prompt = f"""Ты — строгий и беспристрастный эксперт по оценке полноты информации в контексте, используемом для ответа на вопросы клиентов банка. 
Твоя задача — НЕ генерировать ответ пользователю, а только оценивать качество найденной информации.

ТЕБЕ ДАНО:
1) Основной вопрос клиента.
2) Список под-вопросов (аспектов), которые должны быть покрыты контекстом.
3) Набор найденных фрагментов документов (контекст).

ВОПРОС КЛИЕНТА:
{question}

ПОДВОПРОСЫ (АСПЕКТЫ):
{subquestions_text}

НАЙДЕННЫЕ ФРАГМЕНТЫ ДОКУМЕНТОВ:
{context}

ЗАДАЧА:
Для КАЖДОГО подвопроса оцени, содержит ли контекст достаточно информации, чтобы корректно ответить.

ОПРЕДЕЛЕНИЯ КАТЕГОРИЙ:
- "full"  — контекст содержит полный, прямой, однозначный ответ. 
             Фрагменты явно закрывают аспект без необходимости поиска дополнительных данных.
- "partial" — есть релевантная информация, но она НЕ покрывает аспект полностью, 
              содержит лишь часть ответа или требует дополнительных уточнений.
- "none" — в контексте НЕТ информации по аспекту, 
           или данные нерелевантны / явно недостаточны для осмысленного ответа.

ADDATIVE SCORING (числовая шкала):
Для каждого подвопроса дополнительно поставь числовой балл:
- "full"    → score = 2
- "partial" → score = 1
- "none"    → score = 0

ИТОГО:
- Общий итоговый балл = сумма score по всем подвопросам.
- Максимальный балл = 2 × (количество подвопросов).
- Нормализованный балл = Общий балл / Максимальный балл (от 0 до 1, округли до 3 знаков).

ТРЕБОВАНИЯ:
- Не придумывай факты, НЕ пытайся дополнить или угадать отсутствующую информацию.
- Оценивай строго только то, что есть в найденных фрагментах.
- Будь объективен и следуй приведённым определениям.
- Выведи ТОЛЬКО JSON без какого-либо текста до или после.

ФОРМАТ ОТВЕТА (СТРОГО JSON):

{{
  "subquestion_coverage": {{
    "1": "full | partial | none",
    "2": "full | partial | none",
    ...
  }},
  "subquestion_scores": {{
    "1": 0 | 1 | 2,
    "2": 0 | 1 | 2,
    ...
  }},
  "coverage_summary": {{
    "full": <количество аспектов с меткой "full">,
    "partial": <количество аспектов с меткой "partial">,
    "none": <количество аспектов с меткой "none">
  }},
  "additive_score": {{
    "total_score": <целое число, сумма по всем аспектам>,
    "max_score": <целое число, максимум возможных баллов>,
    "normalized_score": <число от 0 до 1 с точностью до 3 знаков>
  }},
  "missing_aspects": [
    "краткие формулировки тех аспектов, где coverage = 'none'"
  ],
  "recommendation": "yes" | "no"
}}

ПРАВИЛО ДЛЯ recommendation:
- "yes" — если есть хотя бы один аспект с coverage = "none".
- "no"  — если coverage для всех аспектов "full" или "partial".

Ответ (ТОЛЬКО JSON):"""

        return prompt

    def _parse_coverage_response(
        self,
        response_text: str,
        subquestions: List[str]
    ) -> Dict:
        """
        Парсит ответ LLM и вычисляет coverage_score (поддержка нового JSON-формата)
        """
        try:
            # Извлекаем JSON из ответа
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1

            if start_idx == -1 or end_idx <= start_idx:
                raise ValueError("JSON не найден в ответе")

            json_str = response_text[start_idx:end_idx]
            data = json.loads(json_str)

            # Основные поля нового формата
            coverage_map = data.get('subquestion_coverage', {}) or {}
            subquestion_scores = data.get('subquestion_scores', {}) or {}
            coverage_summary = data.get('coverage_summary', {}) or {}
            additive_score = data.get('additive_score', {}) or {}

            # Подсчеты при отсутствии coverage_summary
            if not coverage_summary and coverage_map:
                full_count = sum(1 for v in coverage_map.values() if str(v).strip().lower() == 'full')
                partial_count = sum(1 for v in coverage_map.values() if str(v).strip().lower() == 'partial')
                none_count = sum(1 for v in coverage_map.values() if str(v).strip().lower() == 'none')
                coverage_summary = {
                    'full': full_count,
                    'partial': partial_count,
                    'none': none_count
                }
            else:
                # безопасные значения по умолчанию
                full_count = int(coverage_summary.get('full', 0))
                partial_count = int(coverage_summary.get('partial', 0))
                none_count = int(coverage_summary.get('none', 0))

            # Нормализованный скор (coverage_score)
            total_subqs = len(subquestions) if subquestions else (full_count + partial_count + none_count)
            normalized_score = None
            if additive_score and 'normalized_score' in additive_score:
                try:
                    normalized_score = float(additive_score['normalized_score'])
                except Exception:
                    normalized_score = None

            if normalized_score is None:
                # Fallback на старую метрику: (full + 0.5 * partial) / total
                if total_subqs == 0:
                    coverage_score = 1.0
                else:
                    coverage_score = (full_count + 0.5 * partial_count) / max(1, total_subqs)
            else:
                coverage_score = normalized_score

            # Missing aspects (если нет — соберем из coverage_map)
            missing_aspects = data.get('missing_aspects', None)
            if missing_aspects is None and coverage_map:
                # Возьмем индексы с none
                missing_aspects = []
                for idx, v in coverage_map.items():
                    if str(v).strip().lower() == 'none':
                        # попробуем взять текст подвопроса
                        try:
                            i = int(idx) - 1
                            if 0 <= i < len(subquestions):
                                missing_aspects.append(subquestions[i])
                            else:
                                missing_aspects.append(str(idx))
                        except Exception:
                            missing_aspects.append(str(idx))

            return {
                'coverage_score': coverage_score,
                'subquestion_coverage': coverage_map,
                'subquestion_scores': subquestion_scores,
                'coverage_summary': coverage_summary,
                'additive_score': additive_score,
                'missing_aspects': missing_aspects if missing_aspects is not None else [],
                'recommendation': data.get('recommendation', 'no'),
                'full_count': full_count,
                'partial_count': partial_count,
                'none_count': none_count
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


def get_coverage_judge(model_path: str = None, use_api: bool = None) -> CoverageJudge:
    """
    Получить singleton экземпляр CoverageJudge

    Args:
        model_path: путь к модели (опционально, для локального режима)
        use_api: использовать ли API (если None - определяется из LLM_MODE)

    Returns:
        CoverageJudge instance
    """
    global _judge_instance

    if _judge_instance is None:
        _judge_instance = CoverageJudge(model_path=model_path, use_api=use_api)

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
