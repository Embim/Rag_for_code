"""
Автоматическое скачивание всех необходимых моделей для A100 80GB
Запуск: python download_models.py
"""
import os
import subprocess
from pathlib import Path

# Директория для моделей
MODELS_DIR = Path(__file__).parent / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Список моделей для скачивания
# ЕДИНАЯ МОДЕЛЬ: Qwen3-32B для всех задач (и очистка и reranking)
MODELS = {
    "embedding": {
        "repo": "BAAI/bge-m3",
        "description": "BGE-M3 embedding модель (лучшая multilingual)",
        "size": "~2 GB"
    },
    "llm": {
        "repo": "bartowski/Qwen3-32B-2507-GGUF",
        "files": ["Qwen3-32B-2507-Q8_0.gguf"],
        "description": "Qwen3-32B 8-bit (ЕДИНАЯ для очистки И reranking)",
        "size": "~32 GB"
    }
}

def check_huggingface_cli():
    """Проверка установки huggingface-cli"""
    try:
        subprocess.run(["huggingface-cli", "--version"], check=True, capture_output=True)
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

    # Для embedding моделей - скачиваем весь репо
    if model_name == "embedding":
        cmd = [
            "huggingface-cli", "download",
            config["repo"],
            "--local-dir", str(MODELS_DIR / "bge-m3")
        ]

    # Для GGUF файлов - только нужные файлы
    elif "files" in config:
        for file in config["files"]:
            cmd = [
                "huggingface-cli", "download",
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
        cmd = [
            "huggingface-cli", "download",
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
║                   СКАЧИВАНИЕ МОДЕЛЕЙ ДЛЯ A100 80GB                           ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)

    # Проверка HuggingFace CLI
    if not check_huggingface_cli():
        return

    # Расчет общего размера
    print("\n📊 Требуемое место на диске:")
    print("   BGE-M3:            ~2 GB")
    print("   Qwen3-32B (8-bit): ~32 GB (ЕДИНАЯ модель для всех задач)")
    print("   " + "-" * 50)
    print("   ИТОГО:             ~34 GB")

    confirm = input("\n⚠️  Продолжить скачивание? (yes/no): ")
    if confirm.lower() != 'yes':
        print("Отменено.")
        return

    # Скачиваем модели
    success = []
    failed = []

    for model_name, config in MODELS.items():
        print(f"\n[{len(success)+len(failed)+1}/{len(MODELS)}] {model_name}")

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
    if len(success) == len(MODELS):
        print("\n✅ ВСЕ МОДЕЛИ СКАЧАНЫ!")
        print("\n📋 Следующие шаги:")
        print("1. Запустить полный пайплайн автоматически:")
        print("   bash full_pipeline.sh")
        print("\nИли вручную по шагам:")
        print("1. Настроить Weaviate:")
        print("   docker-compose up -d")
        print("\n2. Создать чанки:")
        print("   python main_pipeline.py chunk")
        print("\n3. Запустить предочистку через Qwen3-32B:")
        print("   python scripts/preprocess_documents_qwen25.py")
        print("\n4. Сгенерировать embeddings:")
        print("   python main_pipeline.py build --input data/processed/chunks_cleaned.csv")
        print("\n5. Запустить inference:")
        print("   python main_pipeline.py search")
    else:
        print("\n⚠️  Некоторые модели не скачались. Проверь ошибки выше.")
        print("Можно повторить командой: python download_models.py")

if __name__ == "__main__":
    main()
