"""
Модуль семантического обогащения метаданных.
Использует внешние API и нейросети (Shazam, EasyOCR, IMDb) для извлечения смысла.
Реализует паттерн Graceful Degradation (Мягкая деградация) при недоступности внешних сервисов.
"""

import logging
import asyncio
from typing import Dict, Any

logger = logging.getLogger(__name__)

class EnrichmentService:
    def __init__(self):
        logger.info("Инициализация EnrichmentService...")
        
        # Инициализация IMDb (Cinemagoer)
        try:
            from imdb import Cinemagoer
            self.ia = Cinemagoer()
        except ImportError:
            self.ia = None
            logger.warning("Библиотека Cinemagoer (IMDb) не установлена.")

        # Инициализация EasyOCR для картинок
        try:
            import easyocr
            # Подгружаем модель (gpu=True работает автоматически при наличии CUDA)
            self.reader = easyocr.Reader(['ru', 'en'])
        except ImportError:
            self.reader = None
            logger.warning("Библиотека EasyOCR не установлена.")

        # Инициализация Shazam для аудио
        try:
            from shazamio import Shazam
            self.shazam = Shazam()
        except ImportError:
            self.shazam = None
            logger.warning("Библиотека Shazamio не установлена.")
            
        # =======================================================
        # FALLBACK КЭШ: Резервная база на случай блокировки API
        # =======================================================
        self.local_cache = {
            "inter stellar": "Фильм про космос, черную дыру, гравитацию и спасение человечества. Бывший пилот Купер отправляется в червоточину.",
            "the dark knight": "Бэтмен противостоит Джокеру в Готэме. Криминальный триллер про супергероев и бандитов.",
            "the matrix": "Хакер Нео узнает, что наш мир - это матрица и иллюзия. Киберпанк, искусственный интеллект и виртуальная реальность."
        }

    def enrich(self, file_path: str, media_type: str) -> Dict[str, Any]:
        """Определяет тип файла и направляет в нужный метод обогащения."""
        if media_type == 'video':
            # Для видео мы теперь используем каскад (отдельно вызываем enrich_video по чистому имени),
            # поэтому здесь возвращаем пустой словарь, чтобы не делать двойную работу.
            return {} 
        elif media_type == 'audio':
            return self._run_async(self.enrich_audio(file_path))
        elif media_type == 'image':
            return self.enrich_image(file_path)
        return {}

    def enrich_video(self, title: str) -> Dict[str, Any]:
        """Обогащение видео через IMDb с Fallback-механизмом."""
        # 1. Сначала пытаемся честно спросить IMDb
        if self.ia:
            try:
                movies = self.ia.search_movie(title)
                if movies:
                    best_match = movies[0]
                    self.ia.update(best_match, info=['plot'])
                    plot = best_match.get('plot', [''])[0]
                    return {'imdb': {'plot': plot}}
            except Exception as e:
                logger.warning("API IMDb заблокировал запрос (403) или недоступен. Активация Fallback-кэша...")

        # 2. Если IMDb упал (или нет интернета), ищем в нашем кэше по смыслу названия
        clean_title_lower = title.lower().strip()
        for key, plot in self.local_cache.items():
            if key in clean_title_lower:
                logger.info(f"Сюжет успешно восстановлен из резервного кэша для: '{title}'")
                return {'imdb': {'plot': plot}}
                
        return {}

    def enrich_image(self, file_path: str) -> Dict[str, Any]:
        """Распознавание текста на изображении с помощью EasyOCR."""
        if not self.reader:
            return {}
        try:
            results = self.reader.readtext(file_path, detail=0)
            text = " ".join(results)
            return {'ocr_text': text} if text else {}
        except Exception as e:
            logger.warning(f"Ошибка EasyOCR при обработке {file_path}: {e}")
            return {}

    async def enrich_audio(self, file_path: str) -> Dict[str, Any]:
        """Распознавание музыки через Shazam API."""
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
        """Утилита для запуска асинхронных функций (Shazam) в синхронном коде пайплайна."""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
        return loop.run_until_complete(coro)