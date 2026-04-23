"""
Модуль семантического обогащения метаданных.
Использует внешние API (IMDb, Shazam) и локальные нейросети (EasyOCR) 
для извлечения смысловой информации из медиафайлов.
"""

import re
import logging
import asyncio
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
from PIL import Image

# Импортируем тяжелые библиотеки внутри try/except, чтобы система не падала при их сбое
try:
    from imdb import Cinemagoer
except ImportError:
    Cinemagoer = None

try:
    from shazamio import Shazam
except ImportError:
    Shazam = None

try:
    import easyocr
except ImportError:
    easyocr = None

logger = logging.getLogger(__name__)

class EnrichmentService:
    """
    Сервис интеллектуального анализа контента.
    Автоматически определяет тип файла и применяет соответствующий алгоритм обогащения.
    """
    
    def __init__(self):
        """Инициализация модулей машинного обучения и API."""
        logger.info("Загрузка моделей обогащения...")
        
        self.ia = Cinemagoer() if Cinemagoer else None
        if not self.ia:
            logger.warning("Cinemagoer не установлен, обогащение видео недоступно.")
            
        self.shazam = Shazam() if Shazam else None
        if not self.shazam:
            logger.warning("Shazamio не установлен, распознавание аудио недоступно.")
            
        self.reader = None
        if easyocr:
            try:
                # Используем gpu=True для ускорения на твоей RTX 2060
                self.reader = easyocr.Reader(['ru', 'en'], gpu=True)
                logger.info("Модель EasyOCR успешно загружена в VRAM.")
            except Exception as e:
                logger.error(f"Ошибка загрузки EasyOCR: {e}")

    def clean_filename(self, filename: str) -> str:
        """
        Очистка имени файла от технического мусора с помощью регулярных выражений.
        """
        name = Path(filename).stem
        if name.upper().startswith(("IMG", "DSC", "СНИМОК")):
            return name
            
        name = re.sub(r'[._\-]', ' ', name)
        name = re.sub(r'\b(1080p|720p|4K|WEB-DL|BluRay|mp3|flac|avi|mkv|jpg|png|jpeg)\b', '', name, flags=re.IGNORECASE)
        name = re.sub(r'\d{4}', '', name) # Удаляем год (он часто мешает поиску)
        return name.strip()

    def enrich_image(self, file_path: str) -> Dict[str, Any]:
        """Распознавание текста на изображении (OCR)."""
        result = {}
        if not self.reader:
            return result
            
        try:
            with Image.open(file_path) as img:
                img_array = np.array(img)
                text_results = self.reader.readtext(img_array, detail=0)
                extracted_text = " ".join(text_results)
                
                if extracted_text.strip():
                    result['ocr_text'] = extracted_text.strip()
                    logger.info(f"OCR нашел текст: {extracted_text[:30]}...")
        except Exception as e:
            logger.warning(f"Ошибка OCR для {file_path}: {e}")
            
        return result

    def enrich_video(self, clean_title: str) -> Dict[str, Any]:
        """Поиск информации о фильме/сериале в базе IMDb."""
        result = {}
        if not self.ia:
            return result
            
        try:
            search_results = self.ia.search_movie(clean_title)
            if search_results:
                movie = search_results[0]
                self.ia.update(movie)
                result['imdb'] = {
                    'title': movie.get('title'),
                    'year': movie.get('year'),
                    'plot': movie.get('plot outline', ''),
                    'genres': movie.get('genres', [])
                }
        except Exception as e:
            logger.warning(f"Ошибка поиска IMDb для '{clean_title}': {e}")
            
        return result

    async def _recognize_audio_async(self, file_path: str) -> Dict[str, Any]:
        """Асинхронная функция отправки запроса в Shazam API."""
        if not self.shazam:
            return {}
        try:
            out = await self.shazam.recognize(file_path)
            if 'track' in out:
                track = out['track']
                return {
                    'title': track.get('title', 'Unknown'),
                    'artist': track.get('subtitle', 'Unknown'),
                    'genre': track.get('genres', {}).get('primary', 'Unknown')
                }
        except Exception as e:
            logger.warning(f"Ошибка Shazam для {file_path}: {e}")
        return {}

    def enrich_audio(self, file_path: str) -> Dict[str, Any]:
        """Синхронная обертка для вызова Shazam."""
        # Создаем новый цикл событий (event loop) для безопасного запуска в потоках
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self._recognize_audio_async(file_path))
        finally:
            loop.close()

    def enrich(self, file_path: str, media_type: str) -> Dict[str, Any]:
        """
        Главный метод-маршрутизатор.
        Определяет нужный алгоритм на основе типа медиафайла.
        """
        clean_title = self.clean_filename(Path(file_path).name)
        enrichment_data = {'extracted_title': clean_title}
        
        if media_type == 'image':
            enrichment_data.update(self.enrich_image(file_path))
        elif media_type == 'video':
            enrichment_data.update(self.enrich_video(clean_title))
        elif media_type == 'audio':
            shazam_data = self.enrich_audio(file_path)
            if shazam_data:
                enrichment_data['shazam'] = shazam_data
                enrichment_data['extracted_title'] = f"{shazam_data.get('artist')} - {shazam_data.get('title')}"
                
        return enrichment_data