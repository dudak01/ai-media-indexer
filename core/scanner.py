"""
Модуль сканирования файловой системы.
Отвечает за рекурсивный обход директорий и классификацию найденных медиафайлов.
"""

import logging
from pathlib import Path
from typing import Dict, List, Set
from collections import defaultdict
from core.models import MediaFile

logger = logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DirectoryScanner:
    """
    Класс для интеллектуального сканирования директорий.
    Автоматически фильтрует системные файлы и распределяет медиа по категориям.
    """
    
    # Конфигурация поддерживаемых форматов вынесена на уровень класса
    SUPPORTED_FORMATS: Dict[str, Set[str]] = {
        'video': {'.mp4', '.mkv', '.avi', '.mov', '.flv', '.wmv', '.webm', '.m4v'},
        'audio': {'.mp3', '.flac', '.wav', '.m4a', '.aac', '.ogg', '.wma', '.opus'},
        'image': {'.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp'}
    }
    
    def __init__(self, root_path: str):
        """
        Инициализация сканера.
        
        Args:
            root_path (str): Абсолютный или относительный путь к сканируемой папке.
        """
        self.root_path = Path(root_path)
        self.media_inventory: Dict[str, List[MediaFile]] = defaultdict(list)
        
    def validate_path(self) -> bool:
        """Проверка доступности директории."""
        if not self.root_path.exists():
            logging.error(f"Директория не найдена: {self.root_path}")
            return False
        if not self.root_path.is_dir():
            logging.error(f"Указанный путь не является директорией: {self.root_path}")
            return False
        return True
        
    def scan(self) -> Dict[str, List[MediaFile]]:
        """
        Рекурсивный обход и сбор информации о файлах.
        
        Returns:
            Dict[str, List[MediaFile]]: Словарь, где ключи - типы медиа, 
            а значения - списки объектов MediaFile.
        """
        logging.info(f"Запуск глубокого сканирования директории: {self.root_path}")
        self.media_inventory.clear()
        
        if not self.validate_path():
            return {}

        total_files = 0
        
        for media_type, extensions in self.SUPPORTED_FORMATS.items():
            for ext in extensions:
                # rglob обеспечивает рекурсивный поиск во всех подпапках
                for file_path in self.root_path.rglob(f'*{ext}'):
                    # Игнорируем скрытые и системные файлы
                    if file_path.is_file() and not file_path.name.startswith('.'):
                        try:
                            size_mb = round(file_path.stat().st_size / (1024 * 1024), 2)
                            media_obj = MediaFile(
                                full_path=str(file_path),
                                name=file_path.name,
                                relative_path=str(file_path.relative_to(self.root_path)),
                                size_mb=size_mb,
                                extension=file_path.suffix.lower(),
                                media_type=media_type
                            )
                            self.media_inventory[media_type].append(media_obj)
                            total_files += 1
                        except OSError as e:
                            logging.warning(f"Ошибка доступа к файлу {file_path.name}: {e}")
                            
        logging.info(f"Сканирование завершено. Успешно проиндексировано файлов: {total_files}")
        return dict(self.media_inventory)