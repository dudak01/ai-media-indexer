import os
import sqlite3
import logging
import ffmpeg
import torch
from transformers import DistilBertTokenizerFast, DistilBertForTokenClassification

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# Железобетонные пути относительно расположения этого скрипта
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, 'db', 'diploma_system.db')
MODEL_PATH = os.path.join(BASE_DIR, 'ml', 'weights', 'ner_model.pt')

class MediaPipeline:
    def __init__(self):
        logging.info("Инициализация Интеллектуального Ядра...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load_model()
        
    def load_model(self):
        logging.info("Поднятие NER-модели в память...")
        # 9 - это количество наших тегов из train.py
        self.tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
        self.model = DistilBertForTokenClassification.from_pretrained(
            'distilbert-base-uncased', num_labels=9 
        )
        if os.path.exists(MODEL_PATH):
            self.model.load_state_dict(torch.load(MODEL_PATH, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
            logging.info("Успех: Модель загружена на видеокарту!")
        else:
            logging.error(f"Веса модели не найдены по пути {MODEL_PATH}")

    def get_db_connection(self):
        return sqlite3.connect(DB_PATH)

    def scan_directory(self, target_dir):
        logging.info(f"Сканирование директории: {target_dir}")
        found_files = []
        valid_exts = {'.mp4', '.mkv', '.avi', '.mp3', '.flac', '.jpg', '.png'}
        
        for root, _, files in os.walk(target_dir):
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in valid_exts:
                    found_files.append((file, os.path.join(root, file), ext))
        
        logging.info(f"Найдено файлов: {len(found_files)}")
        return found_files

    def extract_metadata(self, file_path):
        """Извлечение технических данных через FFmpeg"""
        try:
            probe = ffmpeg.probe(file_path)
            video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
            if video_stream:
                return {
                    'codec': video_stream.get('codec_name', 'unknown'),
                    'resolution': f"{video_stream.get('width', 0)}x{video_stream.get('height', 0)}",
                    'duration': float(probe['format'].get('duration', 0))
                }
        except Exception:
            pass # Если файл битый или это пустышка - игнорим ошибку FFmpeg
        return {'codec': 'unknown', 'resolution': 'unknown', 'duration': 0.0}

    def process_and_save(self, target_dir):
        files = self.scan_directory(target_dir)
        if not files:
            return

        conn = self.get_db_connection()
        cursor = conn.cursor()

        for file_name, file_path, ext in files:
            # Черновая очистка имени (в будущем тут будет предикт нейросети)
            clean_title = file_name.replace(ext, '').replace('.', ' ').replace('_', ' ')
            
            # Извлекаем метаданные FFmpeg
            meta = self.extract_metadata(file_path)
            size_mb = os.path.getsize(file_path) / (1024 * 1024)

            # Сохраняем в таблицу scanned_files
            try:
                cursor.execute('''
                    INSERT OR IGNORE INTO scanned_files (file_name, file_path, size_mb)
                    VALUES (?, ?, ?)
                ''', (clean_title, file_path, size_mb))
                logging.info(f"В БД сохранен файл: {clean_title} | Кодек: {meta['codec']}")
            except sqlite3.OperationalError as e:
                logging.error(f"Ошибка БД: {e}")

        conn.commit()
        conn.close()
        logging.info("Пайплайн завершил работу. Данные обогащены и сохранены.")

if __name__ == "__main__":
    # Тестовый прогон: создадим фейковый файл для проверки
    os.makedirs(os.path.join(BASE_DIR, 'data', 'test_folder'), exist_ok=True)
    with open(os.path.join(BASE_DIR, 'data', 'test_folder', 'The.Matrix.1999.1080p.mkv'), 'w') as f:
        f.write('dummy')
        
    pipeline = MediaPipeline()
    pipeline.process_and_save(os.path.join(BASE_DIR, 'data', 'test_folder'))