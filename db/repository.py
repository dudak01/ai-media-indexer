"""
Модуль доступа к данным (Data Access Layer) с использованием паттерна Repository.
Обеспечивает безопасное взаимодействие интеллектуального ядра с базой данных SQLite.
Реализует транзакционную запись результатов ML-инференса и обогащения.
"""

import sqlite3
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path

logger = logging.getLogger(__name__)

class MediaRepository:
    """
    Класс-репозиторий для централизованного управления данными системы.
    Изолирует SQL-запросы от бизнес-логики (ML-моделей).
    """

    def __init__(self, db_path: str = "db/diploma_system.db"):
        self.db_path = Path(db_path)
        self._ensure_db_exists()

    def _ensure_db_exists(self):
        """Проверка доступности файла базы данных и инициализация базовых таблиц."""
        if not self.db_path.parent.exists():
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            
        # --- ДОБАВЛЕНО ТОЛЬКО ЭТО: Авто-создание таблиц для краш-теста ---
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS scanned_files (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_path TEXT UNIQUE, file_name TEXT, file_extension TEXT, 
                    file_size_mb REAL, discovered_at TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ner_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, file_id INTEGER,
                    original_text TEXT, extracted_title TEXT, extracted_year TEXT, 
                    extracted_quality TEXT, confidence_score REAL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS media_metadata (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, file_id INTEGER,
                    duration_seconds REAL, bit_rate INTEGER, width INTEGER, 
                    height INTEGER, video_codec TEXT, audio_codec TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ml_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, operation_type TEXT,
                    model_name TEXT, status TEXT, execution_time_ms REAL, details TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS error_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, message TEXT
                )
            """)
        # -----------------------------------------------------------------

        if not self.db_path.exists():
            logger.warning(f"База данных не найдена по пути {self.db_path}. Требуется запуск init_db.py")

    def _get_connection(self) -> sqlite3.Connection:
        """Создает и возвращает подключение к БД с включенными внешними ключами."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = 1")
        return conn

    def save_scanned_file(self, file_path: str, file_name: str, file_type: str, size_mb: float) -> Optional[int]:
        """
        Регистрирует новый найденный медиафайл в системе.
        Возвращает ID вставленной записи.
        """
        query = """
            INSERT INTO scanned_files (file_path, file_name, file_extension, file_size_mb, discovered_at)
            VALUES (?, ?, ?, ?, ?)
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query, (
                    file_path,
                    file_name,
                    file_type,
                    size_mb,
                    datetime.now().isoformat()
                ))
                conn.commit()
                return cursor.lastrowid
        except sqlite3.IntegrityError:
            logger.debug(f"Файл {file_name} уже существует в базе (IntegrityError).")
            return self.get_file_id_by_path(file_path)
        except Exception as e:
            logger.error(f"Ошибка БД при сохранении файла {file_name}: {e}")
            return None

    def get_file_id_by_path(self, file_path: str) -> Optional[int]:
        """Получает внутренний ID файла по его абсолютному пути."""
        query = "SELECT id FROM scanned_files WHERE file_path = ?"
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query, (file_path,))
                row = cursor.fetchone()
                return row['id'] if row else None
        except Exception as e:
            logger.error(f"Ошибка чтения ID для {file_path}: {e}")
            return None

    def save_technical_metadata(self, file_id: int, metadata: Dict[str, Any]) -> bool:
        """Сохраняет технические параметры (битрейт, кодеки, разрешение) от FFmpeg."""
        query = """
            INSERT INTO media_metadata (
                file_id, duration_seconds, bit_rate, width, height, video_codec, audio_codec
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        try:
            with self._get_connection() as conn:
                conn.execute(query, (
                    file_id,
                    metadata.get('duration_seconds', 0.0),
                    metadata.get('bit_rate', 0),
                    metadata.get('width', 0),
                    metadata.get('height', 0),
                    metadata.get('video_codec'),
                    metadata.get('audio_codec')
                ))
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Ошибка сохранения метаданных для ID {file_id}: {e}")
            return False

    def save_ner_result(self, file_id: int, original_text: str, entities: Dict[str, str], confidence: float) -> bool:
        """
        Сохраняет результаты работы нейросети DistilBERT (извлеченные сущности).
        """
        query = """
            INSERT INTO ner_results (
                file_id, original_text, extracted_title, extracted_year, extracted_quality, confidence_score
            ) VALUES (?, ?, ?, ?, ?, ?)
        """
        try:
            with self._get_connection() as conn:
                conn.execute(query, (
                    file_id,
                    original_text,
                    entities.get('title'),
                    entities.get('year'),
                    entities.get('quality'),
                    confidence
                ))
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Ошибка сохранения NER-результатов для ID {file_id}: {e}")
            return False

    def log_ml_operation(self, operation_type: str, model_name: str, status: str, execution_time_ms: float, details: str = ""):
        """Логирует работу ML-моделей в БД для последующей аналитики и метрик."""
        query = """
            INSERT INTO ml_logs (operation_type, model_name, status, execution_time_ms, details)
            VALUES (?, ?, ?, ?, ?)
        """
        try:
            with self._get_connection() as conn:
                conn.execute(query, (operation_type, model_name, status, execution_time_ms, details))
                conn.commit()
        except Exception as e:
            logger.error(f"Не удалось записать ML-лог: {e}")

    def get_statistics(self) -> Dict[str, int]:
        """Возвращает сводную статистику системы для передачи во внешние системы (Ada)."""
        stats = {}
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                stats['total_files'] = cursor.execute("SELECT count(*) FROM scanned_files").fetchone()[0]
                stats['ner_processed'] = cursor.execute("SELECT count(*) FROM ner_results").fetchone()[0]
                stats['errors'] = cursor.execute("SELECT count(*) FROM error_logs").fetchone()[0]
            return stats
        except Exception as e:
            logger.error(f"Ошибка сбора статистики: {e}")
            return {"total_files": 0, "ner_processed": 0, "errors": 0}