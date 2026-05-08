import os
import sqlite3
from pathlib import Path

print("="*50)
print(" ДИАГНОСТИКА ПРОЕКТА ДЛЯ ИИ ".center(50))
print("="*50)

# 1. Читаем реальную файловую структуру
print("\n[1] ФАЙЛОВАЯ СТРУКТУРА (без кэша и виртуального окружения):")
for root, dirs, files in os.walk('.'):
    # Игнорируем мусор
    if any(ignore in root for ignore in ['.venv', '.git', '__pycache__', 'ml\\cache']):
        continue
    
    level = root.replace('.', '').count(os.sep)
    indent = ' ' * 4 * level
    print(f"{indent}📁 {os.path.basename(root) or 'diploma_final'}/")
    subindent = ' ' * 4 * (level + 1)
    for f in files:
        if f.endswith('.pyc') or f.endswith('.csv'): continue # Скрываем тяжелые данные
        print(f"{subindent}📄 {f}")

# 2. Проверяем веса обученной модели
print("\n[2] СОСТОЯНИЕ ML МОДЕЛИ:")
model_path = Path('ml/weights/model.pt')
if model_path.exists():
    size_mb = model_path.stat().st_size / (1024 * 1024)
    print(f" ✅ Файл весов найден: {model_path}")
    print(f" 📊 Размер файла: {size_mb:.2f} МБ")
else:
    print(f" ❌ ФАЙЛ ВЕСОВ НЕ НАЙДЕН по пути {model_path}!")

# 3. Проверяем базу данных
print("\n[3] СОСТОЯНИЕ БАЗЫ ДАННЫХ:")
db_path = Path('db/diploma_system.db')
if db_path.exists():
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [t[0] for t in cursor.fetchall() if t[0] != 'sqlite_sequence']
        
        print(f" ✅ БД найдена. Подключение успешно.")
        print(f" 🗄 Найдено таблиц: {len(tables)}")
        
        for table in tables:
            cursor.execute(f"SELECT count(*) FROM {table};")
            count = cursor.fetchone()[0]
            print(f"    - Таблица '{table}': {count} записей")
        conn.close()
    except Exception as e:
        print(f" ❌ Ошибка чтения БД: {e}")
else:
    print(f" ❌ БАЗА ДАННЫХ НЕ НАЙДЕНА по пути {db_path}!")

print("\n" + "="*50)