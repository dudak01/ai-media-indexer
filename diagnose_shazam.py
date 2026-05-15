"""
Изолированная диагностика Shazam.
Проверяет работоспособность распознавания на одном mp3-файле.

Запуск:
    python diagnose_shazam.py
"""
import asyncio
import sys
import os
from pathlib import Path

# Путь к тестовому файлу — поправь если у тебя другой
TEST_FILE = r"C:\Users\Никита Феденко\Desktop\test_media\Хабиб - Ягода Малинка.mp3"


def main():
    print("=" * 60)
    print(" ДИАГНОСТИКА SHAZAM ".center(60))
    print("=" * 60)

    # [1] Проверка установки
    print("\n[1] Проверка установленных библиотек...")
    try:
        import shazamio
        print(f"   ✅ shazamio установлен (версия: {shazamio.__version__ if hasattr(shazamio, '__version__') else 'неизвестна'})")
    except ImportError as e:
        print(f"   ❌ shazamio НЕ установлен: {e}")
        return

    try:
        from shazamio import Shazam
        print(f"   ✅ Shazam импортируется")
    except ImportError as e:
        print(f"   ❌ Не могу импортировать Shazam: {e}")
        return

    # [2] Проверка существования файла
    print(f"\n[2] Проверка файла: {TEST_FILE}")
    if not os.path.exists(TEST_FILE):
        print(f"   ❌ Файл не найден!")
        print(f"   Поправь TEST_FILE в начале скрипта на путь к любому mp3.")
        return
    size_mb = os.path.getsize(TEST_FILE) / (1024 * 1024)
    print(f"   ✅ Файл существует, размер: {size_mb:.2f} МБ")

    # [3] Async-распознавание
    print(f"\n[3] Запуск распознавания (это может занять до 30 секунд)...")
    try:
        shazam = Shazam()
        result = asyncio.run(shazam.recognize(TEST_FILE))
        print(f"   ✅ Shazam вернул результат")

        # [4] Анализ структуры ответа
        print(f"\n[4] Структура ответа:")
        print(f"   Тип: {type(result).__name__}")
        print(f"   Ключи верхнего уровня: {list(result.keys())}")

        if 'track' in result:
            track = result['track']
            print(f"\n   === TRACK ===")
            print(f"   Title:    {track.get('title')}")
            print(f"   Subtitle: {track.get('subtitle')}")
            print(f"   Genres:   {track.get('genres')}")
            print(f"   Все ключи track: {list(track.keys())}")
        else:
            print(f"   ⚠ Ключа 'track' нет — Shazam не узнал композицию")
            print(f"   Полный ответ (первые 500 символов):")
            print(f"   {str(result)[:500]}")

    except Exception as e:
        print(f"   ❌ ОШИБКА: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
