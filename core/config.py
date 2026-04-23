"""
Централизованный менеджер конфигурации.
Отвечает за пути к файлам, базе данных и весам нейросетей.
"""

import os
from pathlib import Path
from dataclasses import dataclass

# Определяем корень проекта динамически
BASE_DIR = Path(__file__).resolve().parent.parent

# =================================================================
# DEVOPS НАСТРОЙКА: ПЕРЕНОСИМ КЭШ ИИ НА ЛОКАЛЬНЫЙ ДИСК ПРОЕКТА
# Это спасет диск C: от переполнения и сделает проект портативным.
# =================================================================
HF_CACHE_DIR = BASE_DIR / "ml" / "cache"
HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ['HF_HOME'] = str(HF_CACHE_DIR)
os.environ['TRANSFORMERS_CACHE'] = str(HF_CACHE_DIR)

@dataclass
class SystemConfig:
    """Конфигурационный DTO-объект."""
    base_dir: Path = BASE_DIR
    data_dir: Path = BASE_DIR / "data"
    raw_data_dir: Path = BASE_DIR / "data" / "raw"
    test_dir: Path = BASE_DIR / "data" / "test_folder"
    
    db_path: Path = BASE_DIR / "db" / "diploma_system.db"
    
    weights_dir: Path = BASE_DIR / "ml" / "weights"
    ner_model_path: Path = BASE_DIR / "ml" / "weights" / "ner_model.pt"
    
    log_level: str = "INFO"
    
    def __post_init__(self):
        """Гарантирует существование критически важных директорий при запуске."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.weights_dir.mkdir(parents=True, exist_ok=True)
        self.test_dir.mkdir(parents=True, exist_ok=True)

settings = SystemConfig()