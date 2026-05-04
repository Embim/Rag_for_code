"""
Единая настройка логирования для всего проекта.
"""
import logging
import logging.handlers
from pathlib import Path
from typing import Optional
import time
from contextlib import contextmanager

from .config import OUTPUTS_DIR


def setup_logging(level: str = "INFO",
                  log_file: Optional[Path] = None,
                  enable_console: bool = True) -> None:
    """
    Инициализация root-логгера с выводом в файл и консоль.
    Не создает дубликатов хендлеров при повторных вызовах.
    """
    root = logging.getLogger()

    # Всегда выставляем уровень (мог быть переопределён сторонними либами)
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Форматтеры
    console_fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S"
    )
    file_fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(filename)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Файл логов
    try:
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    if log_file is None:
        log_file = OUTPUTS_DIR / "pipeline.log"

    # Добавляем файловый хендлер, только если ещё нет хендлера на этот файл
    file_handler_exists = any(
        isinstance(h, logging.handlers.RotatingFileHandler) and getattr(h, 'baseFilename', None) == str(log_file)
        for h in root.handlers
    )
    if not file_handler_exists:
        file_handler = logging.handlers.RotatingFileHandler(
            str(log_file), maxBytes=25 * 1024 * 1024, backupCount=3, encoding="utf-8"
        )
        file_handler.setLevel(getattr(logging, level.upper(), logging.INFO))
        file_handler.setFormatter(file_fmt)
        root.addHandler(file_handler)

    # Консоль
    if enable_console:
        # Не дублируем консольный хендлер
        console_exists = any(isinstance(h, logging.StreamHandler) for h in root.handlers)
        if not console_exists:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(getattr(logging, level.upper(), logging.INFO))
            console_handler.setFormatter(console_fmt)
            root.addHandler(console_handler)


def get_logger(name: str) -> logging.Logger:
    """
    Возвращает именованный логгер.
    """
    return logging.getLogger(name)


@contextmanager
def log_timing(logger: logging.Logger, message: str):
    """
    Контекст-менеджер для измерения времени выполнения блока кода.
    """
    start = time.time()
    logger.info(f"{message} - старт")
    try:
        yield
    finally:
        took = time.time() - start
        logger.info(f"{message} - завершено за {took:.2f}s")


