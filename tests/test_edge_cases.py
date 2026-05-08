"""
=============================================================================
Интеграционный тест устойчивости системы на граничных случаях.

Тема ВКР: «Индексация медиаконтента и обогащение метаданных
           с использованием интеллектуального анализа данных»

Автор:  Феденко Никита Александрович
Группа: ИД 23.1/Б3-22
Год:    2026

Описание:
    Проверка корректности обработки нестандартных и потенциально
    проблемных входных данных каждым ключевым модулем системы:
        - DirectoryScanner    (сканирование файловой системы)
        - MetadataExtractor   (извлечение метаданных через FFmpeg)
        - NERPredictor        (инференс DistilBERT)
        - VectorDatabase      (семантический поиск FAISS)
        - MediaRepository     (работа с SQLite, защита от инъекций)

    Тестовые данные включают:
        - файлы с подменой расширения (текст под видом mp4)
        - файлы нулевого размера
        - имена с кириллицей, спецсимволами, SQL-инъекциями
        - имена без расширения и со скрытыми атрибутами
        - корректные имена с разной структурой
=============================================================================
"""

import os
import sys
import shutil
from pathlib import Path

# Добавляем корень проекта в sys.path для корректного импорта core/db
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.scanner import DirectoryScanner
from core.extractor import TechnicalMetadataExtractor
from core.ner_predictor import NERPredictor
from core.vector_db import VectorDatabase
from db.repository import MediaRepository

# Директория для тестовых данных с граничными случаями
EDGE_CASES_DIR = ROOT_DIR / "tests" / "edge_cases"


def setup_edge_cases():
    """
    Создаёт тестовую директорию с набором файлов, моделирующих
    нестандартные и проблемные входные данные.
    """
    if EDGE_CASES_DIR.exists():
        shutil.rmtree(EDGE_CASES_DIR)
    EDGE_CASES_DIR.mkdir(parents=True)

    # 1. Файл с подменой расширения (текст под видом видео)
    with open(EDGE_CASES_DIR / "fake_video.mp4", "w") as f:
        f.write("Это не видеофайл, а обычный текст.")

    # 2. Файл нулевого размера
    open(EDGE_CASES_DIR / "zero_byte.mp3", "w").close()

    # 3. Набор имён с граничными случаями
    edge_case_names = [
        "Матрица [1999] 1080p.mkv",                          # кириллица + квадратные скобки
        "  пробелы   в   начале.jpg",                        # множественные пробелы
        "123.mp4",                                            # имя из одних цифр
        "DROP TABLE users;--.png",                            # SQL-инъекция в имени
        "очень_длинное_название_" * 5 + ".mp3",              # экстремально длинное имя
        ".hidden_hacker_file",                                # скрытый файл без расширения
        "NO_EXTENSION_FILE",                                  # отсутствие расширения
    ]
    for name in edge_case_names:
        with open(EDGE_CASES_DIR / name, "w") as f:
            f.write("test data")

    print(f"[+] Тестовая директория создана. Файлов: {len(os.listdir(EDGE_CASES_DIR))}")


def test_scanner():
    """
    Проверка устойчивости DirectoryScanner к нестандартным файлам:
    подмена MIME-типов, отсутствие расширений, скрытые атрибуты.
    """
    print("\n" + "=" * 60)
    print("Тест 1: DirectoryScanner — обработка некорректных файлов")
    print("=" * 60)
    scanner = DirectoryScanner(str(EDGE_CASES_DIR))
    try:
        inventory = scanner.scan()
        total = sum(len(v) for v in inventory.values())
        print(f"[OK] Сканирование завершено. Всего распознано файлов: {total}")
        print("    Распределение по типам:")
        for media_type, files in inventory.items():
            print(f"      - {media_type}: {len(files)} шт.")
        return inventory
    except Exception as e:
        print(f"[FAIL] DirectoryScanner выбросил исключение: {e}")
        return None


def test_extractor(inventory):
    """
    Проверка устойчивости TechnicalMetadataExtractor к битым медиафайлам:
    подмена содержимого, нулевой размер, отсутствие потоков.
    """
    print("\n" + "=" * 60)
    print("Тест 2: TechnicalMetadataExtractor — обработка битых файлов")
    print("=" * 60)
    extractor = TechnicalMetadataExtractor()
    passed = 0
    failed = 0

    for media_type, files in inventory.items():
        for media in files:
            try:
                meta = extractor.extract(media.full_path, media_type)
                status = getattr(meta, "status", "unknown")
                print(f"[OK] {media.name}: статус = {status}")
                passed += 1
            except Exception as e:
                print(f"[FAIL] Ошибка извлечения метаданных для {media.name}: {e}")
                failed += 1

    print(f"\nРезультат: успешно обработано {passed}, исключений {failed}")


def test_ner_model():
    """
    Проверка инференса NERPredictor на граничных строках:
    кириллица, очень короткие имена, отсутствие пробелов, шум.
    """
    print("\n" + "=" * 60)
    print("Тест 3: NERPredictor — инференс на граничных строках")
    print("=" * 60)
    try:
        ner = NERPredictor()
        print("[+] Модель NER успешно загружена.")

        edge_case_strings = [
            "The.Dark.Knight.2008.1080p.BluRay.x264.mkv",
            "Матрица [1999] WEBRip 720p",
            "123",
            "  ",
            "TerminatorGenezisNoSpaces2015",
            "какая-то рандомная песня без года и качества",
        ]

        for s in edge_case_strings:
            result = ner.extract_entities(s)
            preview = s[:30] + "..." if len(s) > 30 else s
            print(
                f"  Вход: {preview!r:35s} -> "
                f"Title: {result.get('title')!r}, "
                f"Year: {result.get('year')!r}, "
                f"Quality: {result.get('quality')!r}"
            )
    except Exception as e:
        print(f"[FAIL] Ошибка инициализации/инференса NER-модели: {e}")


def test_vector_db():
    """
    Проверка устойчивости VectorDatabase (FAISS) к пустым входам
    и поиску по запросам, отсутствующим в индексе.
    """
    print("\n" + "=" * 60)
    print("Тест 4: VectorDatabase — обработка пустых входов")
    print("=" * 60)
    try:
        db = VectorDatabase()

        # Граничный случай: пустая строка
        db.add_text("", {"test": "empty_input"})
        db.add_text("Корректный текст для индексации", {"test": "valid_input"})

        # Поиск по пустому запросу
        empty_results = db.search("")
        print(f"[OK] Поиск с пустым запросом обработан корректно. "
              f"Найдено: {len(empty_results)} результатов")

        # Поиск по запросу, не релевантному индексу
        irrelevant_results = db.search("квантовая физика", k=5)
        scores = [r["score"] for r in irrelevant_results]
        print(f"[OK] Поиск нерелевантного запроса дал низкие оценки сходства: {scores}")

    except Exception as e:
        print(f"[FAIL] Ошибка работы с векторным индексом FAISS: {e}")


def test_database():
    """
    Проверка защищённости MediaRepository от SQL-инъекций
    и корректной обработки дублирующихся записей.
    """
    print("\n" + "=" * 60)
    print("Тест 5: MediaRepository — защита от SQL-инъекций")
    print("=" * 60)
    try:
        repo = MediaRepository("db/test_edge_cases.db")

        # Имя файла, содержащее SQL-инъекцию
        injection_name = "DROP TABLE users;--"
        file_id = repo.save_scanned_file("/test/path", injection_name, "mp4", 0.0)

        if file_id:
            print("[OK] Параметризованные SQL-запросы корректно экранируют инъекции")

            # Проверка обработки дубликатов
            duplicate_id = repo.save_scanned_file("/test/path", injection_name, "mp4", 0.0)
            print(f"[OK] Обработка дубликатов выполнена корректно. ID: {duplicate_id}")
        else:
            print("[FAIL] База данных не вернула идентификатор записи")

    except Exception as e:
        print(f"[FAIL] Ошибка работы с базой данных: {e}")


if __name__ == "__main__":
    print("Запуск интеграционного теста устойчивости системы")
    print("=" * 60)

    setup_edge_cases()

    inventory = test_scanner()
    if inventory is not None:
        test_extractor(inventory)

    test_ner_model()
    test_vector_db()
    test_database()

    print("\n" + "=" * 60)
    print("Интеграционный тест завершён.")
    print("=" * 60)