"""
Модуль моделей данных (Data Models).
Содержит DTO (Data Transfer Objects) для строгой типизации передаваемых данных
между слоями архитектуры приложения.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime

@dataclass
class MediaFile:
    """
    Базовая структура найденного медиафайла на жестком диске.
    
    Attributes:
        full_path (str): Абсолютный путь к файлу.
        name (str): Имя файла с расширением.
        relative_path (str): Путь относительно корневой директории сканирования.
        size_mb (float): Размер файла в мегабайтах.
        extension (str): Расширение файла (в нижнем регистре).
        media_type (str): Тип контента ('video', 'audio', 'image').
    """
    full_path: str
    name: str
    relative_path: str
    size_mb: float
    extension: str
    media_type: str

@dataclass
class MediaMetadata:
    """
    Структура для хранения технических метаданных, извлеченных FFmpeg или Pillow.
    """
    file_type: str
    file_path: str
    status: str = "success"
    error_msg: Optional[str] = None
    
    # Общие
    size_bytes: int = 0
    
    # Видео / Аудио
    duration_seconds: float = 0.0
    bit_rate: int = 0
    
    # Видео / Изображения
    width: int = 0
    height: int = 0
    
    # Специфичные
    video_codec: Optional[str] = None
    audio_codec: Optional[str] = None
    sample_rate: int = 0
    image_format: Optional[str] = None

    def to_dict(self) -> dict:
        """Сериализация в словарь для сохранения в БД."""
        return {k: v for k, v in self.__dict__.items() if v is not None}