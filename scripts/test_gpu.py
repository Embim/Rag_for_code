# -*- coding: utf-8 -*-
"""
Скрипт для проверки использования GPU
"""
import torch
import time
import numpy as np
from sentence_transformers import SentenceTransformer


def print_gpu_info():
    """Вывод информации о GPU"""
    print("=" * 80)
    print("ИНФОРМАЦИЯ О GPU")
    print("=" * 80)

    print(f"\nPyTorch версия: {torch.__version__}")
    print(f"CUDA доступна: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA версия (PyTorch): {torch.version.cuda}")
        print(f"cuDNN версия: {torch.backends.cudnn.version()}")
        print(f"\nКоличество GPU: {torch.cuda.device_count()}")

        for i in range(torch.cuda.device_count()):
            print(f"\nGPU {i}:")
            print(f"  Название: {torch.cuda.get_device_name(i)}")
            print(f"  Память (общая): {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")

            if torch.cuda.is_initialized():
                print(f"  Память (выделено): {torch.cuda.memory_allocated(i) / 1024**3:.2f} GB")
                print(f"  Память (зарезервировано): {torch.cuda.memory_reserved(i) / 1024**3:.2f} GB")
    else:
        print("\n[!] CUDA не доступна!")
        print("Установите PyTorch с CUDA:")
        print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")


def test_embedding_speed():
    """Тест скорости генерации эмбеддингов на CPU vs GPU"""
    print("\n" + "=" * 80)
    print("ТЕСТ СКОРОСТИ ЭМБЕДДИНГОВ")
    print("=" * 80)

    model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    # Тестовые тексты
    texts = [
        "Как получить кэшбэк за покупки?",
        "Где посмотреть номер счета?",
        "Оплата коммунальных услуг без комиссии",
        "БИК банка для перевода",
        "Альфа-Банк предлагает выгодные кредиты"
    ] * 20  # 100 текстов

    print(f"\nМодель: {model_name}")
    print(f"Количество текстов: {len(texts)}")

    # Тест на CPU
    print("\n" + "-" * 80)
    print("Тест на CPU:")
    print("-" * 80)

    model_cpu = SentenceTransformer(model_name, device='cpu')

    start = time.time()
    embeddings_cpu = model_cpu.encode(texts, batch_size=32, show_progress_bar=False)
    cpu_time = time.time() - start

    print(f"Время: {cpu_time:.2f} сек")
    print(f"Скорость: {len(texts) / cpu_time:.1f} текстов/сек")
    print(f"Размер эмбеддингов: {embeddings_cpu.shape}")

    del model_cpu
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Тест на GPU
    if torch.cuda.is_available():
        print("\n" + "-" * 80)
        print("Тест на GPU:")
        print("-" * 80)

        model_gpu = SentenceTransformer(model_name, device='cuda')

        # Прогрев GPU
        _ = model_gpu.encode(texts[:10], batch_size=32, show_progress_bar=False)

        start = time.time()
        embeddings_gpu = model_gpu.encode(texts, batch_size=128, show_progress_bar=False)
        gpu_time = time.time() - start

        print(f"Время: {gpu_time:.2f} сек")
        print(f"Скорость: {len(texts) / gpu_time:.1f} текстов/сек")
        print(f"Размер эмбеддингов: {embeddings_gpu.shape}")

        # Сравнение
        print("\n" + "-" * 80)
        print("СРАВНЕНИЕ:")
        print("-" * 80)
        speedup = cpu_time / gpu_time
        print(f"Ускорение на GPU: {speedup:.2f}x")

        if speedup > 1.5:
            print("[OK] GPU работает корректно и дает значительное ускорение!")
        elif speedup > 1.0:
            print("[!] GPU работает, но ускорение небольшое.")
        else:
            print("[X] GPU медленнее CPU! Проверьте конфигурацию.")

        # Проверка результатов
        diff = np.abs(embeddings_cpu - embeddings_gpu).max()
        print(f"\nМакс. разница между CPU и GPU: {diff:.6f}")

        if diff < 0.001:
            print("[OK] Результаты идентичны!")
        else:
            print("[!] Есть небольшие различия (это нормально)")

        # Использование памяти GPU
        print("\n" + "-" * 80)
        print("ИСПОЛЬЗОВАНИЕ ПАМЯТИ GPU:")
        print("-" * 80)
        print(f"Выделено: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
        print(f"Зарезервировано: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
        print(f"Максимум: {torch.cuda.max_memory_allocated(0) / 1024**3:.2f} GB")

        del model_gpu
        torch.cuda.empty_cache()
    else:
        print("\n[!] GPU не доступен, тест пропущен")


def test_config():
    """Проверка конфигурации проекта"""
    print("\n" + "=" * 80)
    print("ПРОВЕРКА КОНФИГУРАЦИИ ПРОЕКТА")
    print("=" * 80)

    try:
        from src.config import EMBEDDING_DEVICE, EMBEDDING_BATCH_SIZE, EMBEDDING_MODEL

        print(f"\nУстройство: {EMBEDDING_DEVICE}")
        print(f"Batch size: {EMBEDDING_BATCH_SIZE}")
        print(f"Модель: {EMBEDDING_MODEL}")

        if EMBEDDING_DEVICE == "cuda" and torch.cuda.is_available():
            print("\n[OK] Конфигурация правильная - будет использоваться GPU!")
        elif EMBEDDING_DEVICE == "cpu":
            print("\n[!] Конфигурация установлена на CPU")
            if torch.cuda.is_available():
                print("    GPU доступен, но не используется")
        else:
            print("\n[X] Несоответствие конфигурации!")

    except ImportError as e:
        print(f"\n[X] Ошибка импорта: {e}")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("ПРОВЕРКА ИСПОЛЬЗОВАНИЯ GPU")
    print("="*80)

    try:
        print_gpu_info()
        test_config()

        print("\n")
        response = input("Запустить тест скорости? (y/n): ")
        if response.lower() == 'y':
            test_embedding_speed()

        print("\n" + "="*80)
        print("[OK] ПРОВЕРКА ЗАВЕРШЕНА")
        print("="*80)

    except KeyboardInterrupt:
        print("\n\n[!] Тест прерван")
    except Exception as e:
        print(f"\n\n[X] Ошибка: {e}")
        import traceback
        traceback.print_exc()
