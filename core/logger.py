"""
Модуль конфигурации расширенного логирования (Enterprise Logging).
Обеспечивает вывод логов как в консоль (для отладки), так и в ротируемые файлы
(для мониторинга внешними системами и разбора инцидентов).
"""

import os
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

def setup_system_logger(log_dir: str = "logs") -> logging.Logger:
    """
    Инициализирует и настраивает корневой логгер приложения.
    
    Args:
        log_dir (str): Директория для хранения файлов логов.
        
    Returns:
        logging.Logger: Настроенный инстанс логгера.
    """
    # Создаем директорию для логов, если ее нет
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    log_file_path = os.path.join(log_dir, "media_indexer.log")
    error_file_path = os.path.join(log_dir, "error.log")

    # Форматирование логов по стандартам (Время - Модуль - Уровень - Сообщение)
    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)-8s] [%(name)s:%(lineno)d] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # 1. Обработчик для консоли (вывод в терминал)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # 2. Обработчик для общего файла логов (с ротацией: макс 5 МБ, 3 бэкапа)
    file_handler = RotatingFileHandler(
        log_file_path, maxBytes=5*1024*1024, backupCount=3, encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    # 3. Обработчик только для ошибок (Critical/Error)
    error_handler = RotatingFileHandler(
        error_file_path, maxBytes=2*1024*1024, backupCount=2, encoding='utf-8'
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)

    # Настройка корневого логгера (Root Logger)
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    
    # Очистка старых обработчиков (чтобы логи не дублировались)
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(error_handler)

    logging.info("Система логирования успешно инициализирована.")
    
    return root_logger

# Инициализируем при импорте модуля
system_logger = setup_system_logger()