"""
Модуль семантического обогащения метаданных.
"""

import logging
import asyncio
from typing import Dict, Any

logger = logging.getLogger(__name__)

class EnrichmentService:
    def __init__(self):
        logger.info("Инициализация EnrichmentService...")

        try:
            from imdb import Cinemagoer
            self.ia = Cinemagoer()
        except ImportError:
            self.ia = None
            logger.warning("Библиотека Cinemagoer (IMDb) не установлена.")

        try:
            import easyocr
            self.reader = easyocr.Reader(['ru', 'en'])
        except ImportError:
            self.reader = None
            logger.warning("Библиотека EasyOCR не установлена.")

        try:
            from shazamio import Shazam
            self.shazam = Shazam()
        except ImportError:
            self.shazam = None
            logger.warning("Библиотека Shazamio не установлена.")

        self.local_cache = {
            "inter stellar": "Фильм про космос, черную дыру, гравитацию и спасение человечества.",
            "the dark knight": "Бэтмен противостоит Джокеру в Готэме. Криминальный триллер.",
            "the matrix": "Хакер Нео узнает, что наш мир - это матрица. Киберпанк и ИИ.",
            "terminator": "Робот-убийца из будущего охотится на человека. Научная фантастика.",
            "inception": "Команда проникает в сны людей чтобы украсть секреты. Триллер Нолана.",
            "interstellar": "Фильм про космос, черную дыру, гравитацию и спасение человечества.",
            "incredibles": "Семья супергероев скрывает свои способности. Мультфильм Pixar.",
        }

    def enrich(self, file_path: str, media_type: str) -> Dict[str, Any]:
        if media_type == 'video':
            return {}
        elif media_type == 'audio':
            return self._run_async(self.enrich_audio(file_path))
        elif media_type == 'image':
            return self.enrich_image(file_path)
        return {}

    def enrich_video(self, title: str) -> Dict[str, Any]:
        if self.ia:
            try:
                movies = self.ia.search_movie(title)
                if movies:
                    best_match = movies[0]
                    self.ia.update(best_match, info=['plot'])
                    plot = best_match.get('plot', [''])[0]
                    return {'imdb': {'plot': plot}}
            except Exception as e:
                logger.warning("IMDb недоступен. Активация Fallback-кэша...")

        clean_title_lower = title.lower().strip()
        for key, plot in self.local_cache.items():
            if key in clean_title_lower or clean_title_lower in key:
                logger.info(f"Сюжет из резервного кэша для: '{title}'")
                return {'imdb': {'plot': plot}}

        return {}

    def enrich_image(self, file_path: str) -> Dict[str, Any]:
        """Распознавание текста на изображении через EasyOCR."""
        if not self.reader:
            return {}
        try:
            # Читаем файл через numpy чтобы обойти проблему кириллицы в пути (Windows)
            import numpy as np
            img_array = np.fromfile(file_path, dtype=np.uint8)
            import cv2
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if img is None:
                logger.warning(f"Не удалось декодировать изображение: {file_path}")
                return {}
            results = self.reader.readtext(img, detail=0)
            text = " ".join(results)
            return {'ocr_text': text} if text else {}
        except Exception as e:
            logger.warning(f"Ошибка EasyOCR при обработке {file_path}: {e}")
            return {}

    async def enrich_audio(self, file_path: str) -> Dict[str, Any]:
        if not self.shazam:
            return {}
        try:
            out = await self.shazam.recognize(file_path)
            if 'track' in out:
                return {
                    'shazam_title': out['track'].get('title'),
                    'shazam_subtitle': out['track'].get('subtitle'),
                    'shazam_genre': out['track'].get('genres', {}).get('primary')
                }
        except Exception as e:
            logger.warning(f"Ошибка Shazam при обработке {file_path}: {e}")
        return {}

    def _run_async(self, coro) -> Dict[str, Any]:
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(coro)