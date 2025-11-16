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
import hashlib
from pathlib import Path
from typing import Dict, Optional
import logging
from logging.handlers import RotatingFileHandler
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from src.config import (
    LLM_MODEL_FILE,
    LLM_CONTEXT_SIZE,
    LLM_GPU_LAYERS,
    LLM_MAX_TOKENS,
    LLM_N_BATCH,
    LLM_N_THREADS,
    LLM_MODE,
    LLM_API_MODEL,
    LLM_API_MAX_WORKERS,
    LLM_API_TIMEOUT,
    LLM_API_RETRIES,
    LLM_API_ROUTING,
    LLM_PARALLEL_WORKERS,
    OPENROUTER_API_KEY,
    MODELS_DIR,
    OUTPUTS_DIR,
)


class LLMDocumentCleanerAPI:
    """
    API-based очистка документов через OpenRouter
    
    OpenRouter предоставляет единый API для доступа к 400+ моделям:
    - OpenAI (GPT-4, GPT-3.5)
    - Anthropic (Claude)
    - Google (Gemini)
    - Meta (Llama)
    - DeepSeek (R1T2 Chimera - бесплатно!)
    - И многие другие
    
    Преимущества:
    - Параллельные запросы (ускорение в 10-20 раз)
    - Не занимает VRAM
    - Быстрее локальной модели
    - Бесплатные модели доступны (DeepSeek R1T2 Chimera)
    
    Рекомендуется: DeepSeek R1T2 Chimera (бесплатно, быстрая, хорошее качество)
    """
    
    def __init__(self, verbose: bool = True):
        """
        Args:
            verbose: выводить прогресс
        """
        self.verbose = verbose
        self.model = LLM_API_MODEL
        self.max_workers = LLM_API_MAX_WORKERS
        self.timeout = LLM_API_TIMEOUT
        self.retries = LLM_API_RETRIES
        
        # Кэш для похожих документов
        self._cache = {}
        self._cache_max_size = 100
        
        # Инициализация API клиента
        self.client = None
        self._init_api_client()
        
        # Логгер
        self.llm_logger = logging.getLogger("llm_cleaning")
        self._init_llm_logger()
        
        if verbose:
            print(f"\n{'='*80}")
            print(f"📡 Инициализация LLM Document Cleaner (OpenRouter API)")
            print(f"   Модель: {self.model}")
            print(f"   Параллельных запросов: {self.max_workers}")
            print(f"{'='*80}\n")
    
    def _init_api_client(self):
        """Инициализация OpenRouter API клиента"""
        # OpenRouter использует OpenAI-совместимый API
        # OpenRouter требует API ключ даже для бесплатных моделей
        try:
            from openai import OpenAI
            # OpenRouter endpoint
            base_url = "https://openrouter.ai/api/v1"
            
            # OpenRouter требует API ключ даже для бесплатных моделей
            if not OPENROUTER_API_KEY:
                raise ValueError(
                    "OPENROUTER_API_KEY не установлен!\n"
                    "Получите бесплатный ключ на https://openrouter.ai/keys\n"
                    "Установите: export OPENROUTER_API_KEY=sk-or-v1-..."
                )
            
            # Заголовки для OpenRouter
            default_headers = {
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "HTTP-Referer": "https://github.com/your-repo",  # можно изменить
                "X-Title": "AlfaBank RAG Pipeline"
            }
            
            # Добавляем провайдера для роутинга (если указан)
            if LLM_API_ROUTING:
                default_headers["X-OpenRouter-Provider"] = LLM_API_ROUTING
            
            self.client = OpenAI(
                base_url=base_url,
                api_key=OPENROUTER_API_KEY,
                timeout=self.timeout,
                default_headers=default_headers
            )
        except ImportError:
            raise ImportError("Установите openai: pip install openai")
    
    def _init_llm_logger(self):
        """Инициализация логгера для LLM результатов"""
        if not self.llm_logger.handlers:
            log_file = OUTPUTS_DIR / "llm_cleaning.log"
            log_file.parent.mkdir(parents=True, exist_ok=True)
            
            handler = RotatingFileHandler(
                log_file,
                maxBytes=10 * 1024 * 1024,  # 10 MB
                backupCount=5,
                encoding='utf-8'
            )
            handler.setFormatter(logging.Formatter(
                '%(asctime)s | %(levelname)s | %(name)s | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            ))
            self.llm_logger.addHandler(handler)
            self.llm_logger.setLevel(logging.INFO)
    
    def _extract_final_answer(self, text: str) -> str:
        """
        Извлечение финального ответа из reasoning моделей
        
        Reasoning модели (sherlock-think-alpha, deepseek-r1 и т.д.) возвращают
        reasoning процесс в тегах <think>, <think> и т.д.
        Нужно извлечь только финальный ответ.
        """
        if not text or not isinstance(text, str):
            return text
        
        original_text = text
        
        # Удаляем reasoning теги и их содержимое (разные варианты)
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<reasoning>.*?</reasoning>', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<think>.*', '', text, flags=re.DOTALL | re.IGNORECASE)  # незакрытый тег
        
        # Удаляем строки, начинающиеся с reasoning-маркеров
        lines = text.split('\n')
        cleaned_lines = []
        skip_reasoning = True
        reasoning_markers = [
            'хорошо', 'давайте', 'сначала', 'нужно', 'возможно', 'итак', 'теперь',
            'well', 'let', 'first', 'need', 'maybe', 'so', 'now', 'then'
        ]
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Пропускаем reasoning строки
            if skip_reasoning:
                line_lower = line.lower()
                # Проверяем начало строки
                if any(line_lower.startswith(marker) for marker in reasoning_markers):
                    continue
                # Проверяем если строка содержит только reasoning-текст
                if len(line) < 30 and any(marker in line_lower for marker in reasoning_markers):
                    continue
                # Если нашли что-то похожее на финальный ответ, начинаем собирать
                if len(line) > 15 and not line.startswith('<') and not any(marker in line_lower[:20] for marker in reasoning_markers):
                    skip_reasoning = False
            
            if not skip_reasoning:
                # Пропускаем строки, которые явно являются reasoning
                line_lower = line.lower()
                if len(line) < 30 and any(marker in line_lower for marker in reasoning_markers):
                    continue
                cleaned_lines.append(line)
        
        result = ' '.join(cleaned_lines).strip()
        
        # Если ничего не осталось или результат слишком короткий, пробуем другой подход
        if not result or len(result) < 10:
            # Пробуем взять последнюю значимую строку
            lines = original_text.split('\n')
            for line in reversed(lines):
                line = line.strip()
                if line and len(line) > 15:
                    line_lower = line.lower()
                    if not any(line_lower.startswith(marker) for marker in reasoning_markers):
                        result = line
                        break
        
        # Если все еще ничего, возвращаем оригинал (на случай если это не reasoning модель)
        if not result or len(result) < 5:
            result = original_text.strip()
        
        return result

    def _call_api(self, prompt: str) -> tuple[str, str]:
        """Вызов OpenRouter API
        
        Returns:
            tuple: (raw_response, cleaned_response)
        """
        # OpenRouter использует OpenAI-совместимый API
        # Подготавливаем параметры запроса
        request_params = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": LLM_MAX_TOKENS,
        }
        
        # Добавляем провайдера через extra_headers если указан
        if LLM_API_ROUTING:
            request_params["extra_headers"] = {"X-OpenRouter-Provider": LLM_API_ROUTING}
        
        response = self.client.chat.completions.create(**request_params)
        raw_response = response.choices[0].message.content
        
        # Извлекаем финальный ответ из reasoning моделей
        cleaned_response = self._extract_final_answer(raw_response)
        
        return raw_response, cleaned_response
    
    def _preprocess_text_before_llm(self, text: str) -> str:
        """Агрессивная предобработка текста (та же логика что и в локальной версии)"""
        if not text:
            return ""
        
        patterns_to_remove = [
            r'(?i)(главная|назад|вверх|поделиться|следите за нами|подписаться)',
            r'(?i)(меню|навигация|breadcrumb|хлебные крошки)',
            r'©\s*\d{4}[-\s]*\d{4}.*?',
            r'(?i)(все права защищены|лицензия|лицензия цб рф)',
            r'(?i)(юридический адрес|офис|контакты).*?(?=\n\n|\Z)',
            r'(?i)(откройте карту сегодня|узнайте больше|оставьте заявку|оформить онлайн)',
            r'(?i)(скачать приложение|app store|google play)',
            r'(?i)(подпишитесь|рассылка|новости)',
            r'(?i)(cookie|cookies|использование cookie)',
            r'(?i)(согласие на обработку|политика конфиденциальности)',
            r'(?i)(нажмите здесь|перейдите по ссылке|смотрите также|читайте также)',
            r'(?i)(подробнее|детали|узнать больше)',
            r'[-=]{3,}',
            r'_{3,}',
            r'\n{3,}',
        ]
        
        for pattern in patterns_to_remove:
            text = re.sub(pattern, ' ', text, flags=re.MULTILINE)
        
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def clean_document(self, text: str) -> Dict:
        """Очистка одного документа через API"""
        # Предобработка
        text_preprocessed = self._preprocess_text_before_llm(text)
        
        if len(text_preprocessed.strip()) < 100:
            return self._fallback_result(text_preprocessed)
        
        # Проверка кэша
        text_hash = hashlib.md5(text_preprocessed[:2000].encode('utf-8')).hexdigest()
        if text_hash in self._cache:
            return self._cache[text_hash].copy()
        
        # Сокращенный промпт
        text_truncated = text_preprocessed[:2500]
        prompt = f"""Очисти банковский документ и верни JSON:

ДОКУМЕНТ:
{text_truncated}

ЗАДАЧИ:
1. Удали: навигацию, футеры, рекламу, cookie-баннеры, технические блоки
2. Сохрани: описания продуктов, инструкции, числовые параметры (комиссии, лимиты, сроки)
3. Темы (макс 3): кредитные_карты, дебетовые_карты, переводы, жкх, кэшбэк, счета_реквизиты, комиссии, лимиты, безопасность, мобильное_приложение, альфа_онлайн, ипотека, кредиты, вклады, инвестиции, страхование
4. Полезность: 0.0-0.3 (мусор), 0.4-0.6 (частично), 0.7-1.0 (конкретика)

JSON:
{{
  "clean_text": "очищенный текст",
  "topics": ["тема_1", "тема_2"],
  "usefulness_score": 0.0
}}"""
        
        # Вызов API с повторными попытками
        raw_json_response = None
        for attempt in range(self.retries):
            try:
                raw_response, response_text = self._call_api(prompt)
                raw_json_response = raw_response  # Сохраняем сырой ответ для логирования
                
                # Удаляем markdown код-блоки (```json ... ```)
                # Модели часто возвращают JSON в markdown формате
                # Стратегия: ищем содержимое между ```json и ```
                json_block_match = re.search(r'```(?:json)?\s*\n(.*?)\n```', response_text, re.DOTALL)
                if json_block_match:
                    # Извлекаем содержимое из markdown блока
                    response_text = json_block_match.group(1).strip()
                elif response_text.strip().startswith('```'):
                    # Fallback: если формат немного другой, просто удаляем маркеры
                    response_text = re.sub(r'^```(?:json)?\s*\n?', '', response_text, flags=re.MULTILINE)
                    response_text = re.sub(r'\n?```\s*$', '', response_text, flags=re.MULTILINE)
                    response_text = response_text.strip()
                
                # Парсинг JSON (улучшенная логика с обработкой reasoning)
                raw_result = None
                
                # Стратегия 1: ищем первый валидный JSON объект, начиная с первой {
                first_brace = response_text.find('{')
                if first_brace != -1:
                    brace_count = 0
                    last_brace = -1
                    for i in range(first_brace, len(response_text)):
                        if response_text[i] == '{':
                            brace_count += 1
                        elif response_text[i] == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                last_brace = i
                                break
                    
                    if last_brace != -1:
                        try:
                            json_str = response_text[first_brace:last_brace + 1]
                            raw_result = json.loads(json_str)
                        except json.JSONDecodeError:
                            pass
                
                # Стратегия 2: если не получилось, пробуем найти JSON между первыми { и последними }
                if raw_result is None:
                    first_brace = response_text.find('{')
                    last_brace = response_text.rfind('}')
                    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
                        try:
                            json_str = response_text[first_brace:last_brace + 1]
                            raw_result = json.loads(json_str)
                        except json.JSONDecodeError:
                            pass
                
                # Стратегия 3: пробуем парсить весь ответ как JSON (на случай если там только JSON)
                if raw_result is None:
                    try:
                        raw_result = json.loads(response_text.strip())
                    except json.JSONDecodeError:
                        pass
                
                # Стратегия 4: пробуем найти JSON после reasoning маркеров
                if raw_result is None:
                    # Ищем JSON после строк типа "JSON:", "Ответ:", "Результат:"
                    json_markers = ['json:', 'ответ:', 'результат:', 'result:', 'output:']
                    for marker in json_markers:
                        marker_pos = response_text.lower().find(marker)
                        if marker_pos != -1:
                            # Ищем { после маркера
                            brace_pos = response_text.find('{', marker_pos)
                            if brace_pos != -1:
                                try:
                                    # Пробуем извлечь JSON от этой позиции
                                    brace_count = 0
                                    last_brace = -1
                                    for i in range(brace_pos, len(response_text)):
                                        if response_text[i] == '{':
                                            brace_count += 1
                                        elif response_text[i] == '}':
                                            brace_count -= 1
                                            if brace_count == 0:
                                                last_brace = i
                                                break
                                    if last_brace != -1:
                                        json_str = response_text[brace_pos:last_brace + 1]
                                        raw_result = json.loads(json_str)
                                        break
                                except json.JSONDecodeError:
                                    continue
                
                if raw_result:
                    raw_result.setdefault("clean_text", text_truncated)
                    raw_result.setdefault("topics", [])
                    raw_result.setdefault("usefulness_score", 0.5)
                    raw_result.setdefault("products", [])
                    raw_result.setdefault("actions", [])
                    raw_result.setdefault("conditions", [])
                    raw_result["is_useful"] = bool(raw_result.get("usefulness_score", 0.5) >= 0.3)
                    
                    # Кэширование
                    if len(self._cache) >= self._cache_max_size:
                        oldest_key = next(iter(self._cache))
                        del self._cache[oldest_key]
                    self._cache[text_hash] = raw_result.copy()
                    
                    # Логируем результат с сырым JSON ответом
                    self._log_llm_result(raw_result, original_text=text_truncated, raw_json_response=raw_json_response)
                    
                    return raw_result
                else:
                    # Fallback если JSON не найден
                    fallback = self._fallback_result(text_truncated)
                    self._log_llm_result(fallback, original_text=text_truncated, reason="json_parse_failed", raw_json_response=raw_json_response)
                    return fallback
            
            except Exception as e:
                if attempt < self.retries - 1:
                    if self.verbose:
                        print(f"  ⚠️  Ошибка API (попытка {attempt + 1}/{self.retries}): {e}")
                    continue
                else:
                    if self.verbose:
                        print(f"  ⚠️  Ошибка API после {self.retries} попыток: {e}")
                    fallback = self._fallback_result(text_truncated)
                    self._log_llm_result(fallback, original_text=text_truncated, reason=str(e), raw_json_response=raw_json_response)
                    return fallback
        
        # Если все попытки исчерпаны
        fallback = self._fallback_result(text_truncated)
        self._log_llm_result(fallback, original_text=text_truncated, reason="all_retries_exhausted", raw_json_response=raw_json_response)
        return fallback
    
    def _fallback_result(self, text: str) -> Dict:
        """Fallback результат если API не сработал"""
        return {
            "clean_text": text,
            "topics": [],
            "usefulness_score": 0.5,
            "products": [],
            "actions": [],
            "conditions": [],
            "is_useful": True
        }
    
    def _log_llm_result(self, result: Dict, original_text: str, reason: Optional[str] = None, raw_json_response: Optional[str] = None) -> None:
        """
        Логирование результата LLM очистки в отдельный JSON-лог.
        
        Args:
            result: результат clean_document
            original_text: оригинальный текст документа
            reason: причина логирования (например, "json_parse_failed")
            raw_json_response: сырой JSON ответ от API (если доступен)
        """
        # Проверяем, что логгер инициализирован
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
            
            # Добавляем сырой JSON ответ от API (если доступен)
            if raw_json_response is not None:
                # Ограничиваем длину сырого ответа (первые 2000 символов)
                log_record["raw_json_response"] = raw_json_response[:2000]
            
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


class LLMDocumentCleaner:
    """
    LLM-based очистка и обогащение документов

    Использует Qwen3-32B (или другую LLM) для:
    - Удаления мусора из веб-документов
    - Извлечения ключевой информации
    - Добавления метаданных для улучшения поиска
    """

    def __init__(self, model_path: Optional[str] = None, verbose: bool = True, n_workers: Optional[int] = None):
        """
        Args:
            model_path: путь к GGUF модели (если None - использует из config)
            verbose: выводить прогресс
            n_workers: количество параллельных воркеров (если None - использует LLM_PARALLEL_WORKERS из config)
        """
        if model_path is None:
            model_path = str(MODELS_DIR / LLM_MODEL_FILE)

        self.model_path = model_path
        self.verbose = verbose
        self.llm = None

        # Количество параллельных воркеров
        self.n_workers = n_workers if n_workers is not None else LLM_PARALLEL_WORKERS

        # Простой кэш для похожих документов (по хэшу текста)
        # Кэшируем только последние 100 результатов для экономии памяти
        self._cache = {}
        self._cache_max_size = 100
        self._cache_lock = threading.Lock()  # Lock для thread-safe доступа к кэшу

        # Отдельный логгер для хранения результатов работы LLM
        # (чтобы можно было анализировать, что именно вернула модель)
        self.llm_logger = logging.getLogger("llm_cleaning")
        self._init_llm_logger()

        if verbose:
            print(f"\n{'='*80}")
            print(f"📥 Инициализация LLM Document Cleaner")
            print(f"   Модель: {Path(model_path).name}")
            print(f"   Параллельных воркеров: {self.n_workers}")
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
            n_batch=LLM_N_BATCH,  # из config (1024 для ускорения)
            n_threads=LLM_N_THREADS,  # из config (16 для ускорения)
            use_mlock=True,
            verbose=False
        )

        if self.verbose:
            print(f"✅ Модель загружена!")

    def _preprocess_text_before_llm(self, text: str) -> str:
        """
        Агрессивная предобработка текста перед LLM для ускорения.
        Удаляет очевидный мусор через regex, чтобы сократить контекст для LLM.
        
        Args:
            text: исходный текст
            
        Returns:
            предобработанный текст
        """
        if not text:
            return ""
        
        # Паттерны для удаления очевидного мусора
        patterns_to_remove = [
            # Навигация и меню
            r'(?i)(главная|назад|вверх|поделиться|следите за нами|подписаться)',
            r'(?i)(меню|навигация|breadcrumb|хлебные крошки)',
            
            # Футеры и копирайты
            r'©\s*\d{4}[-\s]*\d{4}.*?',
            r'(?i)(все права защищены|лицензия|лицензия цб рф)',
            r'(?i)(юридический адрес|офис|контакты).*?(?=\n\n|\Z)',
            
            # Реклама и призывы к действию
            r'(?i)(откройте карту сегодня|узнайте больше|оставьте заявку|оформить онлайн)',
            r'(?i)(скачать приложение|app store|google play)',
            r'(?i)(подпишитесь|рассылка|новости)',
            
            # Cookie и баннеры
            r'(?i)(cookie|cookies|использование cookie)',
            r'(?i)(согласие на обработку|политика конфиденциальности)',
            
            # Технические блоки
            r'(?i)(нажмите здесь|перейдите по ссылке|смотрите также|читайте также)',
            r'(?i)(подробнее|детали|узнать больше)',
            
            # Повторяющиеся разделители
            r'[-=]{3,}',
            r'_{3,}',
            
            # Множественные переносы строк
            r'\n{3,}',
        ]
        
        # Удаляем паттерны
        for pattern in patterns_to_remove:
            text = re.sub(pattern, ' ', text, flags=re.MULTILINE)
        
        # Удаляем HTML теги (если остались)
        text = re.sub(r'<[^>]+>', '', text)
        
        # Нормализуем пробелы
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text

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

        # Агрессивная предобработка перед LLM (удаляем очевидный мусор)
        text_preprocessed = self._preprocess_text_before_llm(text)
        
        # Пропускаем очень короткие документы (после предобработки) - экономим время LLM
        if len(text_preprocessed.strip()) < 100:
            fallback = self._fallback_result(text_preprocessed)
            self._log_llm_result(fallback, original_text=text, reason="too_short_after_preprocessing")
            return fallback
        
        # Проверяем кэш (по хэшу предобработанного текста) - thread-safe
        text_hash = hashlib.md5(text_preprocessed[:2000].encode('utf-8')).hexdigest()
        with self._cache_lock:
            if text_hash in self._cache:
                cached_result = self._cache[text_hash].copy()
                self._log_llm_result(cached_result, original_text=text, reason="cached")
                return cached_result
        
        # Ограничиваем длину для контекста (уменьшено для ускорения)
        text_truncated = text_preprocessed[:2500]  # было 3000, уменьшено т.к. уже предобработано

        # Сокращенный промпт для ускорения (убраны избыточные пояснения)
        prompt = f"""Очисти банковский документ и верни JSON:

ДОКУМЕНТ:
{text_truncated}

ЗАДАЧИ:
1. Удали: навигацию, футеры, рекламу, cookie-баннеры, технические блоки
2. Сохрани: описания продуктов, инструкции, числовые параметры (комиссии, лимиты, сроки)
3. Темы (макс 3): кредитные_карты, дебетовые_карты, переводы, жкх, кэшбэк, счета_реквизиты, комиссии, лимиты, безопасность, мобильное_приложение, альфа_онлайн, ипотека, кредиты, вклады, инвестиции, страхование
4. Полезность: 0.0-0.3 (мусор), 0.4-0.6 (частично), 0.7-1.0 (конкретика)

JSON:
{{
  "clean_text": "очищенный текст",
  "topics": ["тема_1", "тема_2"],
  "usefulness_score": 0.0
}}"""

        try:
            response = self.llm(
                prompt,
                max_tokens=LLM_MAX_TOKENS,  # из config.py (1024 для ускорения)
                temperature=0.1,
                stop=["<|im_end|>"],
                top_p=0.9,  # nucleus sampling для ускорения
                top_k=40,  # ограничение словаря для ускорения
            )

            response_text = response['choices'][0]['text']

            # Извлекаем JSON из ответа (более устойчивый парсинг)
            # Пробуем несколько стратегий:
            raw_result = None
            
            # Стратегия 1: ищем первый валидный JSON объект, начиная с первой {
            # Используем жадный поиск первого JSON объекта
            first_brace = response_text.find('{')
            if first_brace != -1:
                # Пробуем найти закрывающую скобку, начиная с позиции первой {
                # Идем от конца к началу, чтобы найти правильную закрывающую скобку
                brace_count = 0
                last_brace = -1
                for i in range(first_brace, len(response_text)):
                    if response_text[i] == '{':
                        brace_count += 1
                    elif response_text[i] == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            last_brace = i
                            break
                
                if last_brace != -1:
                    try:
                        json_str = response_text[first_brace:last_brace + 1]
                        raw_result = json.loads(json_str)
                    except json.JSONDecodeError:
                        pass
            
            # Стратегия 2: если не получилось, пробуем найти JSON между первыми { и последними }
            if raw_result is None:
                first_brace = response_text.find('{')
                last_brace = response_text.rfind('}')
                if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
                    try:
                        json_str = response_text[first_brace:last_brace + 1]
                        raw_result = json.loads(json_str)
                    except json.JSONDecodeError:
                        pass
            
            # Стратегия 3: пробуем парсить весь ответ как JSON (на случай если там только JSON)
            if raw_result is None:
                try:
                    raw_result = json.loads(response_text.strip())
                except json.JSONDecodeError:
                    pass
            
            if raw_result:
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

                # Сохраняем в кэш (ограничиваем размер) - thread-safe
                with self._cache_lock:
                    if len(self._cache) >= self._cache_max_size:
                        # Удаляем самый старый элемент (FIFO)
                        oldest_key = next(iter(self._cache))
                        del self._cache[oldest_key]
                    self._cache[text_hash] = raw_result.copy()

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

    def _log_llm_result(self, result: Dict, original_text: str, reason: Optional[str] = None, raw_json_response: Optional[str] = None) -> None:
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
            
            # Добавляем сырой JSON ответ от API (если доступен)
            if raw_json_response is not None:
                # Ограничиваем длину сырого ответа (первые 2000 символов)
                log_record["raw_json_response"] = raw_json_response[:2000]
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
