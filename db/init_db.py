import sqlite3
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_database():
    os.makedirs('db', exist_ok=True)
    db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'diploma_system.db')

    
    logging.info(f"Инициализация базы данных: {db_path}")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 1. Пользователи
    cursor.execute('''CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT, username VARCHAR(50) NOT NULL UNIQUE, role VARCHAR(20), created_at DATETIME DEFAULT CURRENT_TIMESTAMP)''')

    # 2. Настройки
    cursor.execute('''CREATE TABLE IF NOT EXISTS app_settings (
        id INTEGER PRIMARY KEY AUTOINCREMENT, setting_key VARCHAR(50), setting_value VARCHAR(255))''')

    # 3. Директории
    cursor.execute('''CREATE TABLE IF NOT EXISTS directories (
        id INTEGER PRIMARY KEY AUTOINCREMENT, path TEXT UNIQUE, last_scanned DATETIME)''')

    # 4. Сырые файлы
    cursor.execute('''CREATE TABLE IF NOT EXISTS scanned_files (
        id INTEGER PRIMARY KEY AUTOINCREMENT, file_name TEXT, file_path TEXT UNIQUE, size_mb REAL)''')

    # 5. Метаданные медиа
    cursor.execute('''CREATE TABLE IF NOT EXISTS media_metadata (
        id INTEGER PRIMARY KEY AUTOINCREMENT, file_id INTEGER, duration_sec REAL, video_codec VARCHAR(50), audio_codec VARCHAR(50))''')

    # 6. Результаты ИИ (NER)
    cursor.execute('''CREATE TABLE IF NOT EXISTS ner_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT, file_id INTEGER, extracted_title VARCHAR(255), extracted_year VARCHAR(4), quality VARCHAR(50))''')

    # 7. Кэш IMDb
    cursor.execute('''CREATE TABLE IF NOT EXISTS imdb_cache (
        id INTEGER PRIMARY KEY AUTOINCREMENT, query_title VARCHAR(255), official_title VARCHAR(255), release_year INTEGER)''')

    # 8. Кэш Shazam
    cursor.execute('''CREATE TABLE IF NOT EXISTS shazam_cache (
        id INTEGER PRIMARY KEY AUTOINCREMENT, file_id INTEGER, artist VARCHAR(255), track_title VARCHAR(255))''')

    # 9. Логи машинного обучения
    cursor.execute('''CREATE TABLE IF NOT EXISTS ml_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT, epoch INTEGER, loss REAL, accuracy REAL, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')

    # 10. Логи ошибок
    cursor.execute('''CREATE TABLE IF NOT EXISTS error_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT, module_name VARCHAR(50), error_message TEXT, occurred_at DATETIME DEFAULT CURRENT_TIMESTAMP)''')

    # Создаем админа по умолчанию
    cursor.execute('''INSERT OR IGNORE INTO users (username, role) VALUES ('admin', 'superuser')''')

    conn.commit()
    conn.close()
    logging.info("Успех: 10 таблиц создано.")

if __name__ == "__main__":
    create_database()