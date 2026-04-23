"""
Набор Unit-тестов для проверки работы Ядра (Core) системы.
Использует Pytest для изоляции и мок-тестирования файловой системы.
"""

import os
import pytest
from pathlib import Path
from core.models import MediaFile
from core.scanner import DirectoryScanner
from core.extractor import TechnicalMetadataExtractor

@pytest.fixture
def test_env(tmp_path):
    """
    Фикстура (Fixture): создает изолированную файловую систему
    специально для этого теста, чтобы не мусорить на реальном диске.
    """
    test_dir = tmp_path / "test_media"
    test_dir.mkdir()
    
    # Создаем фейковые файлы разных форматов
    (test_dir / "movie.mp4").write_text("fake video data")
    (test_dir / "song.mp3").write_text("fake audio data")
    (test_dir / "photo.jpg").write_text("fake image data")
    
    # Мусорные файлы, которые сканер должен проигнорировать
    (test_dir / ".hidden_file").write_text("should be ignored")
    (test_dir / "document.txt").write_text("should be ignored")
    
    return test_dir

def test_directory_scanner(test_env):
    """Проверка работы сканера: должен находить только медиа и игнорировать мусор."""
    scanner = DirectoryScanner(str(test_env))
    inventory = scanner.scan()
    
    # Проверяем, что сканер нашел все 3 категории
    assert 'video' in inventory, "Сканер не нашел категорию video"
    assert 'audio' in inventory, "Сканер не нашел категорию audio"
    assert 'image' in inventory, "Сканер не нашел категорию image"
    
    # Проверяем конкретные файлы
    assert len(inventory['video']) == 1
    assert inventory['video'][0].name == "movie.mp4"
    assert inventory['video'][0].extension == ".mp4"
    
    # Текстовые и скрытые файлы должны быть проигнорированы!
    total_files = sum(len(files) for files in inventory.values())
    assert total_files == 3, f"Сканер нашел {total_files} файлов вместо 3 (захватил мусор)"

def test_metadata_extractor_error_handling():
    """
    Проверка отказоустойчивости Экстрактора.
    Если подсунуть ему битый файл, он не должен крашить программу, 
    а должен мягко перехватить ошибку и вернуть status='error'.
    """
    extractor = TechnicalMetadataExtractor()
    
    fake_media = MediaFile(
        full_path="non_existent_file.mp4",
        name="dummy.mp4",
        relative_path="dummy.mp4",
        size_mb=0.1,
        extension=".mp4",
        media_type="video"
    )
    
    metadata = extractor.extract(fake_media)
    
    # Система должна мягко обработать отсутствие файла (или его битый формат)
    assert metadata.status == 'error', "Экстрактор не выставил статус error для битого файла"
    assert metadata.error_msg is not None, "Экстрактор не записал причину ошибки"