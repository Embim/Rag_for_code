"""
Автоматическое скачивание необходимых моделей для Code RAG.

Модели автоматически скачиваются при первом использовании,
но этот скрипт позволяет скачать их заранее.

Запуск: python scripts/download_models.py
"""
import os
import sys
import subprocess
from pathlib import Path
from dotenv import load_dotenv

# Загружаем .env
load_dotenv()

# Директория для моделей
MODELS_DIR = Path(__file__).parent.parent / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Список моделей для скачивания
MODELS = {
    "embedding": {
        "repo": os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3"),
        "description": "Embedding модель (векторизация кода)",
        "size": "~2 GB",
        "required": True,  # Обязательная
        "local": True  # Локальная модель
    },
    "reranker": {
        "repo": os.getenv("RERANKER_MODEL", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"),
        "description": "Reranker модель (перерангирование результатов)",
        "size": "~500 MB",
        "required": True,  # Обязательная
        "local": True  # Локальная модель
    },
    "llm_local": {
        "repo": "unsloth/Qwen3-32B-GGUF",
        "files": ["Qwen3-32B-IQ4_NL.gguf"],
        "description": "Локальная LLM (опционально, если не используете OpenRouter API)",
        "size": "~20 GB",
        "required": False,  # Опциональная
        "local": True  # Локальная модель
    }
}

def check_huggingface_cli():
    """Проверка установки huggingface-cli"""
    try:
        # Пробуем через python -m (работает на всех платформах)
        subprocess.run(
            [sys.executable, "-m", "huggingface_hub.commands.huggingface_cli", "--help"],
            check=True,
            capture_output=True
        )
        print("✅ huggingface-cli установлен")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ huggingface-cli не найден!")
        print("\nУстанови:")
        print("  pip install huggingface_hub")
        return False

def download_model(model_name, config):
    """Скачивание одной модели"""
    print(f"\n{'='*80}")
    print(f"📥 Скачивание: {model_name}")
    print(f"   Репозиторий: {config['repo']}")
    print(f"   Описание: {config['description']}")
    print(f"   Размер: {config['size']}")
    print(f"{'='*80}\n")

    # Базовая команда для всех платформ
    base_cmd = [sys.executable, "-m", "huggingface_hub.commands.huggingface_cli", "download"]

    # Для embedding моделей - скачиваем весь репо
    if model_name == "embedding":
        cmd = base_cmd + [
            config["repo"],
            "--local-dir", str(MODELS_DIR / "bge-m3")
        ]

    # Для GGUF файлов - только нужные файлы
    elif "files" in config:
        for file in config["files"]:
            cmd = base_cmd + [
                config["repo"],
                file,
                "--local-dir", str(MODELS_DIR)
            ]
            print(f"Команда: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)
        return True

    # Для HuggingFace моделей - весь репо
    else:
        model_dir = MODELS_DIR / config["repo"].split("/")[-1]
        cmd = base_cmd + [
            config["repo"],
            "--local-dir", str(model_dir)
        ]

    print(f"Команда: {' '.join(cmd)}")

    try:
        subprocess.run(cmd, check=True)
        print(f"✅ {model_name} скачан!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Ошибка при скачивании {model_name}: {e}")
        return False

def main():
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                  СКАЧИВАНИЕ МОДЕЛЕЙ ДЛЯ CODE RAG                             ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)

    # Проверка HuggingFace CLI
    if not check_huggingface_cli():
        return

    # Показываем список моделей
    print("\n📦 Доступные модели:\n")
    required_models = {k: v for k, v in MODELS.items() if v.get('required', True)}
    optional_models = {k: v for k, v in MODELS.items() if not v.get('required', True)}

    print("ОБЯЗАТЕЛЬНЫЕ (необходимы для работы):")
    total_required_size = 0
    for name, config in required_models.items():
        status = "✅ Обязательно"
        print(f"  • {name}: {config['description']}")
        print(f"    Размер: {config['size']}, Модель: {config['repo']}")
        # Примерный расчет
        if "GB" in config['size']:
            total_required_size += float(config['size'].split('~')[1].split(' ')[0])
        elif "MB" in config['size']:
            total_required_size += float(config['size'].split('~')[1].split(' ')[0]) / 1024

    print(f"\nОПЦИОНАЛЬНЫЕ (только если используете локальный режим LLM):")
    for name, config in optional_models.items():
        print(f"  • {name}: {config['description']}")
        print(f"    Размер: {config['size']}, Модель: {config['repo']}")

    # Выбор режима
    print(f"\n📊 Требуемое место на диске:")
    print(f"   Обязательные модели:  ~{total_required_size:.1f} GB")
    print(f"   + Опциональные:       ~20 GB (если нужна локальная LLM)")
    print("   " + "-" * 50)

    print("\nЧто скачивать?")
    print("1. Только обязательные (рекомендуется, если используете OpenRouter API)")
    print("2. Все модели (если планируете использовать локальный режим)")
    print("3. Выбрать вручную")

    choice = input("\nВаш выбор (1/2/3): ").strip()

    models_to_download = {}
    if choice == "1":
        models_to_download = required_models
    elif choice == "2":
        models_to_download = MODELS
    elif choice == "3":
        print("\nВыберите модели для скачивания (y/n):")
        for name, config in MODELS.items():
            response = input(f"  Скачать {name}? (y/n): ").strip().lower()
            if response == 'y':
                models_to_download[name] = config
    else:
        print("❌ Неверный выбор. Отменено.")
        return

    if not models_to_download:
        print("❌ Не выбрано ни одной модели. Отменено.")
        return

    confirm = input(f"\n⚠️  Скачать {len(models_to_download)} модель(и)? (yes/no): ")
    if confirm.lower() != 'yes':
        print("Отменено.")
        return

    # Скачиваем модели
    success = []
    failed = []

    for model_name, config in models_to_download.items():
        print(f"\n[{len(success)+len(failed)+1}/{len(models_to_download)}] {model_name}")

        if download_model(model_name, config):
            success.append(model_name)
        else:
            failed.append(model_name)

    # Итоги
    print("\n" + "="*80)
    print("📊 ИТОГИ СКАЧИВАНИЯ:")
    print(f"   ✅ Успешно: {len(success)}/{len(MODELS)}")
    if success:
        print(f"      {', '.join(success)}")

    if failed:
        print(f"   ❌ Ошибки: {len(failed)}/{len(MODELS)}")
        print(f"      {', '.join(failed)}")

    print("="*80)

    # Следующие шаги
    if success:
        print(f"\n✅ Скачано {len(success)} модель(и)!")
        print("\n📋 Следующие шаги:")
        print("1. Убедитесь что .env настроен:")
        print("   cp .env.example .env")
        print("   nano .env  # добавьте OPENROUTER_API_KEY и другие переменные")
        print("\n2. Запустите базы данных:")
        print("   docker-compose up -d")
        print("\n3. Проиндексируйте репозиторий:")
        print("   python -m src.core.graph.build_and_index /path/to/repo")
        print("\n4. Запустите Telegram бота или API:")
        print("   python -m src.interfaces.telegram_bot.bot")
        print("   # или")
        print("   uvicorn src.api.main:app --reload")
        print("\n📚 Документация:")
        print("   docs/QUICKSTART.md - быстрый старт")
        print("   docs/MODEL_CONFIGURATION.md - настройка моделей")

    if failed:
        print("\n⚠️  Некоторые модели не скачались. Проверь ошибки выше.")
        print("Можно повторить командой: python scripts/download_models.py")
        print("\nℹ️  Примечание: Модели также скачаются автоматически при первом использовании.")

if __name__ == "__main__":
    main()
