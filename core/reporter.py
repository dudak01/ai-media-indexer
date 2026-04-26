"""
Модуль генерации аналитической отчетности (Analytics & Reporting Engine).
Выполняет агрегацию данных из базы SQLite, рассчитывает метрики качества 
работы ИИ-моделей и экспортирует результаты в различных форматах (CSV, JSON, TXT).
Сгенерированные отчеты используются для аудита системы и прикрепления к ВКР.
"""

import os
import csv
import json
import logging
import sqlite3
from datetime import datetime
from typing import Dict, List, Any
from pathlib import Path

# Фикс путей для корректного импорта из корня
try:
    from db.repository import MediaRepository
except ImportError:
    import sys
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if BASE_DIR not in sys.path:
        sys.path.insert(0, BASE_DIR)
    from db.repository import MediaRepository

logger = logging.getLogger(__name__)

class ReportGenerator:
    """
    Генератор сводных отчетов. Работает напрямую с DAL (Data Access Layer),
    собирая статистику по проиндексированным файлам и результатам NER.
    """

    def __init__(self, export_dir: str = "reports"):
        """
        Инициализация генератора.
        
        Args:
            export_dir (str): Директория для сохранения готовых отчетов.
        """
        self.export_dir = Path(export_dir)
        self.export_dir.mkdir(parents=True, exist_ok=True)
        self.repo = MediaRepository()

    def _fetch_raw_data(self) -> List[sqlite3.Row]:
        """
        Выполняет сложный SQL JOIN для объединения базовых данных о файлах
        с результатами работы нейросети (NER).
        """
        query = """
            SELECT 
                sf.file_name, 
                sf.file_extension, 
                sf.file_size_mb,
                nr.extracted_title,
                nr.extracted_year,
                nr.extracted_quality,
                nr.confidence_score
            FROM scanned_files sf
            LEFT JOIN ner_results nr ON sf.id = nr.file_id
            ORDER BY sf.discovered_at DESC
        """
        try:
            with self.repo._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query)
                return cursor.fetchall()
        except Exception as e:
            logger.error(f"Ошибка выгрузки данных для отчета: {e}")
            return []

    def generate_json_report(self) -> str:
        """Генерирует полную выгрузку системы в формате JSON."""
        data = self._fetch_raw_data()
        if not data:
            logger.warning("Нет данных для генерации JSON отчета.")
            return ""

        report_data = []
        for row in data:
            report_data.append({
                "original_file": row["file_name"],
                "format": row["file_extension"],
                "size_mb": row["file_size_mb"],
                "ai_predictions": {
                    "title": row["extracted_title"],
                    "year": row["extracted_year"],
                    "quality": row["extracted_quality"],
                    "confidence": row["confidence_score"]
                }
            })

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = self.export_dir / f"system_dump_{timestamp}.json"
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, ensure_ascii=False, indent=4)
            
        logger.info(f"JSON отчет успешно сохранен: {filepath}")
        return str(filepath)

    def generate_csv_analytics(self) -> str:
        """Генерирует плоский CSV файл для построения графиков (например, в Excel)."""
        data = self._fetch_raw_data()
        if not data:
            return ""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = self.export_dir / f"ai_analytics_{timestamp}.csv"

        headers = [
            "File Name", "Extension", "Size (MB)", 
            "AI Title", "AI Year", "AI Quality", "Confidence Score"
        ]

        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, delimiter=';')
            writer.writerow(headers)
            
            for row in data:
                writer.writerow([
                    row["file_name"],
                    row["file_extension"],
                    row["file_size_mb"],
                    row["extracted_title"] or "N/A",
                    row["extracted_year"] or "N/A",
                    row["extracted_quality"] or "N/A",
                    row["confidence_score"] or 0.0
                ])

        logger.info(f"CSV аналитика успешно сохранена: {filepath}")
        return str(filepath)

    def print_summary(self):
        """Выводит краткую текстовую сводку в консоль."""
        data = self._fetch_raw_data()
        total_files = len(data)
        if total_files == 0:
            print("База данных пуста.")
            return

        processed_by_ai = sum(1 for r in data if r["confidence_score"] is not None)
        total_size = sum(r["file_size_mb"] for r in data if r["file_size_mb"])
        
        avg_confidence = 0.0
        if processed_by_ai > 0:
            avg_confidence = sum(r["confidence_score"] for r in data if r["confidence_score"]) / processed_by_ai

        print("\n" + "="*50)
        print(" АНАЛИТИЧЕСКАЯ СВОДКА СИСТЕМЫ ИНДЕКСАЦИИ ".center(50))
        print("="*50)
        print(f" Всего файлов в базе:   {total_files}")
        print(f" Общий объем данных:    {total_size:.2f} МБ")
        print(f" Обработано ИИ (NER):   {processed_by_ai} файлов")
        print(f" Средняя уверенность:   {avg_confidence:.1f}%")
        print("="*50 + "\n")

if __name__ == "__main__":
    # Быстрый тест модуля
    reporter = ReportGenerator()
    reporter.print_summary()
    reporter.generate_csv_analytics()