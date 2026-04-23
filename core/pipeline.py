"""
Центральный контроллер (Оркестратор) системы.
Связывает вместе Scanner, Extractor, EnrichmentService, NERPredictor и FAISS.
"""

import sqlite3
import logging
from typing import List, Dict, Any

from core.config import settings
from core.models import MediaFile, MediaMetadata
from core.scanner import DirectoryScanner
from core.extractor import TechnicalMetadataExtractor
from core.enrichment import EnrichmentService
from core.ner_predictor import NERPredictor
from core.vector_db import SemanticSearchEngine
from core.exceptions import DatabaseConnectionError

logger = logging.getLogger(__name__)

class MediaPipeline:
    def __init__(self):
        logger.info("Инициализация Интеллектуального Ядра v4.0 (С поддержкой FAISS)...")
        self.extractor = TechnicalMetadataExtractor()
        self.enricher = EnrichmentService()
        self.ner = NERPredictor()
        self.vector_db = SemanticSearchEngine()  # Инициализация Векторной БД
        self._init_db_connection()

    def _init_db_connection(self):
        if not settings.db_path.exists():
            logger.error(f"БД не найдена по пути: {settings.db_path}")
        self.conn = sqlite3.connect(str(settings.db_path), check_same_thread=False)

    def process_directory(self, target_dir: str) -> None:
        scanner = DirectoryScanner(target_dir)
        inventory = scanner.scan()
        
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
        logger.info(f"Анализ: {media.name}")
        
        tech_meta: MediaMetadata = self.extractor.extract(media)
        
        if tech_meta.status == 'error':
            logger.warning(f"Файл не читается. Пропуск глубокого анализа: {media.name}")
            enrich_meta = {'extracted_title': media.name} 
        else:
            enrich_meta: Dict[str, Any] = self.enricher.enrich(media.full_path, media.media_type)
            
        # Инференс NER
        ner_extracted = self.ner.extract_entities(media.name)
        enrich_meta['ner_analysis'] = ner_extracted
        logger.info(f"NER извлек: {ner_extracted}")

        # =======================================================
        # ИНТЕГРАЦИЯ FAISS: Создаем семантический слепок файла
        # =======================================================
        semantic_parts = [
            ner_extracted.get('title', ''),
            ner_extracted.get('year', ''),
            ner_extracted.get('artist', ''),
            enrich_meta.get('ocr_text', ''),
        ]
        # Если IMDb нашел сюжет - добавляем его для поиска!
        if 'imdb' in enrich_meta:
            semantic_parts.append(enrich_meta['imdb'].get('plot', ''))
            
        semantic_text = " ".join(filter(bool, semantic_parts))
        
        # Записываем вектор в FAISS
        self.vector_db.add_to_index(semantic_text, media.full_path)

        # Сохранение в реляционную БД (SQLite)
        self._save_to_db(media, tech_meta, enrich_meta)

    def _save_to_db(self, media: MediaFile, tech: MediaMetadata, enrich: Dict[str, Any]) -> None:
        cursor = self.conn.cursor()
        title = enrich.get('extracted_title', media.name)
        try:
            cursor.execute('''
                INSERT OR IGNORE INTO scanned_files (file_name, file_path, size_mb)
                VALUES (?, ?, ?)
            ''', (title, media.full_path, media.size_mb))
            self.conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Ошибка записи в БД: {e}")

    def __del__(self):
        if hasattr(self, 'conn'):
            self.conn.close()

if __name__ == "__main__":
    # Тестовый запуск
    settings.test_dir.mkdir(parents=True, exist_ok=True)
    
    # Фейковые файлы
    (settings.test_dir / 'The.Dark.Knight.2008.1080p.mkv').write_text('dummy')
    (settings.test_dir / 'Interstellar.2014.WEB-DL.mkv').write_text('dummy')
        
    pipeline = MediaPipeline()
    pipeline.process_directory(str(settings.test_dir))
    
    # ТЕСТ СЕМАНТИЧЕСКОГО ПОИСКА
    print("\n" + "="*50)
    print(" ДЕМОНСТРАЦИЯ ИНТЕЛЛЕКТУАЛЬНОГО ВЕКТОРНОГО ПОИСКА")
    print("="*50)
    
    # Ищем не по названию, а по смысловому описанию!
    query = "фильм про космос черную дыру и гравитацию"
    print(f"ЗАПРОС ПОЛЬЗОВАТЕЛЯ: '{query}'\n")
    
    results = pipeline.vector_db.search(query, top_k=1)
    for res in results:
        print(f"🤖 ИИ нашел файл: {res['file_path']}")
        print(f"📊 Уверенность (Relevance Score): {res['relevance_score']}")
    print("="*50 + "\n")