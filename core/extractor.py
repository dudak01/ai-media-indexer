"""
Модуль извлечения технических метаданных.
Отвечает за работу с бинарными файлами через FFmpeg (видео/аудио) и Pillow (изображения).
"""

import logging
import ffmpeg
from PIL import Image
from typing import Optional
from core.models import MediaFile, MediaMetadata

# Подключаем встроенный FFmpeg (как было в твоем оригинальном коде)
try:
    import static_ffmpeg
    static_ffmpeg.add_paths()
except ImportError:
    logging.warning("Модуль static_ffmpeg не найден. Убедитесь, что FFmpeg установлен в системе.")

logger = logging.getLogger(__name__)

class TechnicalMetadataExtractor:
    """
    Анализатор медиафайлов. Читает заголовки файлов и извлекает кодеки, 
    разрешение, битрейт и длительность, не загружая файл целиком в оперативную память.
    """

    def extract(self, media: MediaFile) -> MediaMetadata:
        """
        Единая точка входа для извлечения данных.
        
        Args:
            media (MediaFile): Объект файла, полученный от сканера.
            
        Returns:
            MediaMetadata: Заполненный DTO с технической информацией.
        """
        metadata = MediaMetadata(file_type=media.media_type, file_path=media.full_path)
        
        if media.media_type == 'image':
            return self._extract_image_data(media.full_path, metadata)
            
        return self._extract_av_data(media.full_path, media.media_type, metadata)

    def _extract_image_data(self, file_path: str, metadata: MediaMetadata) -> MediaMetadata:
        """Приватный метод обработки изображений через Pillow."""
        try:
            with Image.open(file_path) as img:
                metadata.width, metadata.height = img.size
                metadata.image_format = img.format
        except Exception as e:
            logger.error(f"Ошибка чтения изображения {file_path}: {e}")
            metadata.status = 'error'
            metadata.error_msg = str(e)
        return metadata

    def _extract_av_data(self, file_path: str, media_type: str, metadata: MediaMetadata) -> MediaMetadata:
        """Приватный метод обработки видео и аудио через FFmpeg Probe."""
        try:
            probe = ffmpeg.probe(file_path)
            fmt = probe.get('format', {})
            
            metadata.duration_seconds = float(fmt.get('duration', 0))
            metadata.size_bytes = int(fmt.get('size', 0))
            metadata.bit_rate = int(fmt.get('bit_rate', 0))
            
            if media_type == 'video':
                video_stream = next((s for s in probe['streams'] if s['codec_type'] == 'video'), None)
                if video_stream:
                    metadata.video_codec = video_stream.get('codec_name')
                    metadata.width = int(video_stream.get('width', 0))
                    metadata.height = int(video_stream.get('height', 0))
                    
            elif media_type == 'audio':
                audio_stream = next((s for s in probe['streams'] if s['codec_type'] == 'audio'), None)
                if audio_stream:
                    metadata.audio_codec = audio_stream.get('codec_name')
                    metadata.sample_rate = int(audio_stream.get('sample_rate', 0))
                    
        except ffmpeg.Error as e:
            logger.error(f"FFmpeg ошибка при обработке {file_path}: {e.stderr.decode('utf8') if e.stderr else str(e)}")
            metadata.status = 'error'
            metadata.error_msg = "FFmpeg probe failed"
        except Exception as e:
            logger.error(f"Критическая ошибка метаданных {file_path}: {e}")
            metadata.status = 'error'
            metadata.error_msg = str(e)
            
        return metadata