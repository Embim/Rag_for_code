"""
Конфигурация для RAG пайплайна
"""
import os
from pathlib import Path
import torch

# Пути к данным
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# Логирование
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
LOG_FILE = OUTPUTS_DIR / os.environ.get("LOG_FILE", "pipeline.log")

# Входные файлы
WEBSITES_CSV = PROJECT_ROOT / "websites_sample.csv"
QUESTIONS_CSV = PROJECT_ROOT / "questions_clean.csv"
EXAMPLES_CSV = PROJECT_ROOT / "examples_for_participants.csv"

# Параметры чанкинга
CHUNK_SIZE = 200  # слов
CHUNK_OVERLAP = 50  # слов
MIN_CHUNK_SIZE = 50  # минимальный размер чанка

# Параметры Parent-Child Chunking (улучшенный чанкинг)
ENABLE_PARENT_CHILD_CHUNKING = os.environ.get("ENABLE_PARENT_CHILD_CHUNKING", "false").lower() == "true"
CHILD_CHUNK_SIZE = 100  # маленькие чанки для точного поиска
PARENT_CHUNK_SIZE = 300  # большие чанки для полного контекста
PARENT_CHILD_OVERLAP = 20  # перекрытие между чанками

# Параметры чтения CSV (streaming)
CSV_CHUNKSIZE = int(os.environ.get("CSV_CHUNKSIZE", "10"))  # сколько строк CSV читать за раз при обработке (маленький для LLM)
CSV_COUNT_CHUNKSIZE = int(os.environ.get("CSV_COUNT_CHUNKSIZE", "50000"))  # chunksize для быстрого подсчета документов

# Параметры embedding модели
# A100 80GB оптимизация: используем BGE-M3 - лучшую multilingual модель
EMBEDDING_MODEL = "BAAI/bge-m3"  #  (1024 dim, hybrid retrieval)
# Альтернативы:
# EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"  # Baseline (512 dim, быстрая)
# EMBEDDING_MODEL = "intfloat/multilingual-e5-large"  # Хорошая (768 dim)
# EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"  # Универсальная

# Настройка PyTorch для уменьшения фрагментации GPU памяти
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Автоопределение устройства (GPU если доступно, иначе CPU)
if torch.cuda.is_available():
    EMBEDDING_DEVICE = "cuda"
    # Оптимизация для A100 80GB - максимальные batch size
    EMBEDDING_BATCH_SIZE = 128  # A100 80GB может обрабатывать большие батчи
    EMBEDDING_CHUNK_SIZE = 100  # Увеличено для A100 (больше VRAM = меньше итераций)
    print(f"[GPU] Используется: {torch.cuda.get_device_name(0)}")
    print(f"      CUDA версия: {torch.version.cuda}")
    print(f"      Batch size: {EMBEDDING_BATCH_SIZE}")
    print(f"      Chunk size: {EMBEDDING_CHUNK_SIZE} документов за раз")
else:
    EMBEDDING_DEVICE = "cpu"
    EMBEDDING_BATCH_SIZE = 16  # Меньший batch size для CPU (было 32)
    EMBEDDING_CHUNK_SIZE = 1000  # Больше для CPU т.к. больше RAM
    print("[CPU] GPU не доступен, используется CPU")
    print(f"      Batch size: {EMBEDDING_BATCH_SIZE}")
    print(f"      Chunk size: {EMBEDDING_CHUNK_SIZE} документов за раз")

# Можно принудительно установить CPU через переменную окружения
if os.environ.get("FORCE_CPU", "").lower() == "true":
    EMBEDDING_DEVICE = "cpu"
    EMBEDDING_BATCH_SIZE = 16
    EMBEDDING_CHUNK_SIZE = 1000
    print("[CPU] Принудительно установлен CPU (FORCE_CPU=true)")

# Параметры Weaviate
WEAVIATE_URL = os.environ.get("WEAVIATE_URL", "http://localhost:8080")
WEAVIATE_CLASS_NAME = "DocumentChunk"
USE_WEAVIATE = os.environ.get("USE_WEAVIATE", "true").lower() == "true"  # по умолчанию используем Weaviate

# Параметры поиска
TOP_K_DENSE = 25  # топ-K для векторного поиска (уменьшено для скорости)
TOP_K_BM25 = 25   # топ-K для BM25 (уменьшено для скорости)
TOP_K_RERANK = 20 # топ-K после rerank для финального выбора
HYBRID_ALPHA = 0.5  # вес для dense (1-alpha для BM25)

# Параметры Reciprocal Rank Fusion
ENABLE_RRF = os.environ.get("ENABLE_RRF", "true").lower() == "true"  # ✅ ВКЛЮЧЕНО по умолчанию
RRF_K = 60  # константа для RRF (обычно 60, можно 20-100)

# Параметры Context Window
ENABLE_CONTEXT_WINDOW = os.environ.get("ENABLE_CONTEXT_WINDOW", "true").lower() == "true"  # ✅ ВКЛЮЧЕНО
CONTEXT_WINDOW_SIZE = 3  # ±1 чанк (можно 2-3 для большего контекста)
CONTEXT_MERGE_MODE = "merged"  # "separate" или "merged" (объединять тексты или нет)

# Параметры Multi-hop Reasoning
# Глобальный переключатель продвинутых функций (простая и понятная явная настройка)
ENABLE_ADVANCED_FEATURES = True

# Явные флаги (по умолчанию включены для простоты)
ENABLE_MULTI_HOP = True
MAX_HOPS = 3  # максимальное количество итераций (обычно 2-3)

# Параметры Query Reformulation
ENABLE_QUERY_REFORMULATION = True
QUERY_REFORMULATION_METHOD = "all"  # "simple", "expanded", "multi", "all"
QUERY_REFORMULATION_CACHE = True  # кэширование переформулированных запросов

# Параметры Reranker
# Типы: "llm" (Qwen3-32B, медленно но качественно),
#       "cross_encoder" (быстро и качественно),
#       "transformer" (Qwen3-Reranker, НЕ РАБОТАЕТ - не обучена),
#       "none" (без reranking)
RERANKER_TYPE = os.environ.get("RERANKER_TYPE", "cross_encoder")  # по умолчанию cross_encoder

# Параметры для разных reranker'ов
USE_TRANSFORMER_RERANKER = False  # False = LLM/CrossEncoder reranker
RERANKER_MODEL_PATH = MODELS_DIR / "Qwen3-Reranker-0.6B"
RERANKER_BATCH_SIZE = 1  # Batch size для reranker
RERANKER_MAX_LENGTH = 512  # Максимальная длина текста для reranker

# Cross-Encoder параметры
CROSS_ENCODER_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"  # multilingual (русский!)
# Альтернативы:
# "cross-encoder/ms-marco-MiniLM-L-12-v2" - быстрая, хорошее качество (English)
# "cross-encoder/ms-marco-MiniLM-L-6-v2" - очень быстрая (English)

# Параметры LLM (используется И для предочистки И для reranking)
# ЕДИНАЯ МОДЕЛЬ: Qwen3-32B для всех задач
# A100 80GB оптимизация: используем 8-bit для максимального качества
# Варианты:
# LLM_MODEL_FILE = "Qwen3-4B-Instruct-2507-UD-Q4_K_XL.gguf"  # 4B baseline (VRAM: 2-3 GB, качество среднее)
# LLM_MODEL_FILE = "Qwen3-32B-2507-Q4_K_M.gguf"  # 32B 4-bit (VRAM: 16-18 GB, качество хорошее)
# LLM_MODEL_FILE = "Qwen3-32B-2507-Q8_0.gguf"  # 32B 8-bit с датой (VRAM: 32-34 GB)
LLM_MODEL_FILE = "Qwen3-32B-Q8_0.gguf"  # 32B 8-bit (VRAM: 32-34 GB) - фактическое имя файла в models/

LLM_CONTEXT_SIZE = 8192  # Qwen3 поддерживает до 32k, но 8k достаточно
LLM_TEMPERATURE = 0.1  # низкая температура для более детерминированных оценок
LLM_MAX_TOKENS = 2048  # для предочистки нужно больше (было 512)
LLM_GPU_LAYERS = -1  # -1 = все слои на GPU, 0 = только CPU

# Финальные результаты
TOP_N_DOCUMENTS = 5  # количество документов в финальном ответе

# Устаревшие параметры (для обратной совместимости)
RERANKER_MODEL = None  # Устарело
LLM_JUDGE_MODEL = "Qwen3-32B-2507-Q8_0.gguf"  # Для подсказок при ошибках
LLM_JUDGE_FILE = LLM_MODEL_FILE
LLM_JUDGE_CONTEXT_SIZE = LLM_CONTEXT_SIZE
LLM_JUDGE_TEMPERATURE = LLM_TEMPERATURE
LLM_JUDGE_MAX_TOKENS = LLM_MAX_TOKENS

# Параметры агентного RAG
ENABLE_AGENT_RAG = os.environ.get("ENABLE_AGENT_RAG", "false").lower() == "true"
COVERAGE_THRESHOLD = 0.7  # порог полноты контекста для остановки итераций
MAX_AGENT_ITERATIONS = 3  # максимальное количество итераций поиска

# Параметры Grid Search оптимизации
GRID_SEARCH_SAMPLE_SIZE = 50  # размер выборки для оптимизации гиперпараметров (рекомендуется 50)
GRID_SEARCH_USE_LLM = os.environ.get("GRID_SEARCH_USE_LLM", "true").lower() == "true"  # использовать LLM для оценки (точнее, но медленнее ~1-2 часа)
GRID_SEARCH_MODE = "quick"  # режим: "quick" (54 комбинации) или "full" (1225 комбинаций)

# Параметры Query Expansion (расширение запроса)
ENABLE_QUERY_EXPANSION = os.environ.get("ENABLE_QUERY_EXPANSION", "true").lower() == "true"  # ✅ ВКЛЮЧЕНО
QUERY_EXPANSION_METHOD = "hybrid"  # "synonyms", "llm", "hybrid" (synonyms - быстро и эффективно)

# Параметры Metadata Filtering (фильтрация по метаданным)
ENABLE_METADATA_FILTER = os.environ.get("ENABLE_METADATA_FILTER", "true").lower() == "true"  # по умолчанию ВКЛ
METADATA_BOOST_SCORE = 1.3  # коэффициент усиления для документов с совпадающими метаданными

# Параметры Usefulness Filtering (фильтрация по качеству)
MIN_USEFULNESS_SCORE = 0.3  # минимальный порог полезности (0.0-1.0)
ENABLE_USEFULNESS_FILTER = os.environ.get("ENABLE_USEFULNESS_FILTER", "true").lower() == "true"

# Параметры Dynamic TOP_K (адаптивный выбор количества результатов)
ENABLE_DYNAMIC_TOP_K = os.environ.get("ENABLE_DYNAMIC_TOP_K", "true").lower() == "true"  # по умолчанию ВКЛ

# Словарь синонимов для нормализации
SYNONYMS = {
    "кешбек": "кэшбэк",
    "кешбэк": "кэшбэк",
    "кэшбек": "кэшбэк",
    "жкх": "коммунальные услуги",
    "смс": "sms",
    "sms": "смс",
    "бик": "БИК",
}

# Создание директорий если их нет
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
