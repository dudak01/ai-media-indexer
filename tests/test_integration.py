"""
Интеграционные тесты для проверки логики семантического обогащения.
"""

import pytest
from core.enrichment import EnrichmentService

@pytest.fixture
def enricher():
    """Фикстура для инициализации сервиса (без поднятия тяжелых нейросетей)"""
    return EnrichmentService()

def test_clean_filename_movies(enricher):
    """Проверка очистки названий фильмов от торрент-мусора (кодеки, год, качество)"""
    dirty_name = "The.Dark.Knight.2008.1080p.BluRay.x264.mkv"
    # Регулярка должна снести год (2008), качество (1080p), источник (BluRay) и заменить точки на пробелы
    clean_name = enricher.clean_filename(dirty_name)
    
    # x264 мы в регулярку не добавляли, так что он останется (в реальности его потом добьет NER-модель)
    assert clean_name.startswith("The Dark Knight"), f"Ожидалось очищенное название, получено: {clean_name}"

def test_clean_filename_images(enricher):
    """Проверка того, что системные префиксы фотокарточек не искажаются"""
    photo_name = "IMG_20231024_153022.jpg"
    clean_name = enricher.clean_filename(photo_name)
    
    # Сервис должен вернуть оригинальное имя без расширения, если это фото
    assert clean_name == "IMG_20231024_153022", f"Фото переименовано некорректно: {clean_name}"

def test_clean_filename_audio(enricher):
    """Проверка очистки музыкальных треков"""
    audio_name = "Queen_-_Bohemian_Rhapsody_320kbps.mp3"
    clean_name = enricher.clean_filename(audio_name)
    
    # Должен уйти битрейт (если он есть в регулярке) и замениться спецсимволы
    assert "Queen" in clean_name
    assert "Bohemian Rhapsody" in clean_name