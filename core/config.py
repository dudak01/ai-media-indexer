"""
Централизованный менеджер конфигурации.
Отвечает за пути к файлам, базе данных и весам нейросетей.
"""

import os
from pathlib import Path
from dataclasses import dataclass

# Определяем корень проекта динамически
BASE_DIR = Path(__file__).resolve().parent.parent

@dataclass
class SystemConfig:
    """Конфигурационный DTO-объект."""
    # Базовые пути
    base_dir: Path = BASE_DIR
    data_dir: Path = BASE_DIR / "data"
    raw_data_dir: Path = BASE_DIR / "data" / "raw"
    test_dir: Path = BASE_DIR / "data" / "test_folder"
    
    # База данных
    db_path: Path = BASE_DIR / "db" / "diploma_system.db"
    
    # ML модели и веса
    weights_dir: Path = BASE_DIR / "ml" / "weights"
    ner_model_path: Path = BASE_DIR / "ml" / "weights" / "ner_model.pt"
    
    # Настройки логирования
    log_level: str = "INFO"
    
    def __post_init__(self):
        """Гарантирует существование критически важных директорий."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.weights_dir.mkdir(parents=True, exist_ok=True)
        self.test_dir.mkdir(parents=True, exist_ok=True)

# Глобальный экземпляр конфигурации
settings = SystemConfig()