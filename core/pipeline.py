"""
Центральный контроллер (Оркестратор) системы.
Связывает вместе Scanner, Extractor, EnrichmentService и нашу нейросеть NER.
"""

import os
import sqlite3
import logging
from typing import List, Dict, Any
from core.models import MediaFile, MediaMetadata
from core.scanner import DirectoryScanner
from core.extractor import TechnicalMetadataExtractor
from core.enrichment import EnrichmentService

# Устанавливаем пути
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, 'db', 'diploma_system.db')

logger = logging.getLogger(__name__)

class MediaPipeline:
    """
    Главный класс обработки. Управляет жизненным циклом данных:
    Сканирование -> Извлечение технических данных -> Семантическое обогащение -> Сохранение.
    """
    
    def __init__(self):
        logger.info("Инициализация Интеллектуального Ядра v2.0...")
        self.extractor = TechnicalMetadataExtractor()
        self.enricher = EnrichmentService()
        self._init_db_connection()

    def _init_db_connection(self):
        """Проверка доступа к базе данных SQLite."""
        if not os.path.exists(DB_PATH):
            logger.error(f"БД не найдена по пути: {DB_PATH}. Запустите python db/init_db.py")
        self.conn = sqlite3.connect(DB_PATH, check_same_thread=False)

    def process_directory(self, target_dir: str) -> None:
        """
        Полный цикл обработки папки с медиафайлами.
        
        Args:
            target_dir (str): Путь к директории для анализа.
        """
        scanner = DirectoryScanner(target_dir)
        inventory = scanner.scan()
        
        # Разворачиваем словарь в плоский список файлов для обработки
        files_to_process: List[MediaFile] = []
        for media_list in inventory.values():
            files_to_process.extend(media_list)
            
        if not files_to_process:
            logger.info("Медиафайлы для обработки не найдены.")
            return

        logger.info(f"Начало обработки {len(files_to_process)} файлов...")
        
        for media in files_to_process:
            self._process_single_file(media)
            
        logger.info("Пайплайн завершил работу.")

    def _process_single_file(self, media: MediaFile) -> None:
        """Обработка одного конкретного файла: Экстракция -> Обогащение -> БД."""
        logger.info(f"Анализ: {media.name}")
        
        # 1. Извлечение технических метаданных (FFmpeg / Pillow)
        tech_meta: MediaMetadata = self.extractor.extract(media)
        
        # Если файл битый, логируем и пропускаем обогащение
        if tech_meta.status == 'error':
            logger.warning(f"Пропуск обогащения из-за технической ошибки: {media.name}")
            enrich_meta = {'extracted_title': media.name} 
        else:
            # 2. Семантическое обогащение (IMDb, Shazam, OCR)
            enrich_meta: Dict[str, Any] = self.enricher.enrich(media.full_path, media.media_type)

        # 3. Сохранение в базу
        self._save_to_db(media, tech_meta, enrich_meta)

    def _save_to_db(self, media: MediaFile, tech: MediaMetadata, enrich: Dict[str, Any]) -> None:
        """Запись собранной информации в SQLite."""
        cursor = self.conn.cursor()
        title = enrich.get('extracted_title', media.name)
        
        try:
            # Запись базовой информации в scanned_files
            cursor.execute('''
                INSERT OR IGNORE INTO scanned_files (file_name, file_path, size_mb)
                VALUES (?, ?, ?)
            ''', (title, media.full_path, media.size_mb))
            self.conn.commit()
            
            # В реальном проекте здесь будет логика распределения по таблицам video/audio/image
            # на основе media.media_type. Пока просто подтверждаем успешную обработку.
            
        except sqlite3.Error as e:
            logger.error(f"Ошибка записи в БД для {media.name}: {e}")

    def __del__(self):
        """Безопасное закрытие соединения с БД при уничтожении объекта."""
        if hasattr(self, 'conn'):
            self.conn.close()

if __name__ == "__main__":
    # Локальный тест пайплайна
    test_dir = os.path.join(BASE_DIR, 'data', 'test_folder')
    os.makedirs(test_dir, exist_ok=True)
    
    # Создаем фейковый файл для теста
    with open(os.path.join(test_dir, 'Inception.2010.1080p.mkv'), 'w') as f:
        f.write('dummy')
        
    pipeline = MediaPipeline()
    pipeline.process_directory(test_dir)