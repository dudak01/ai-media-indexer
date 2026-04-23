"""
Генератор мультимодального датасета для обучения NER-модели.
Версия 2.0 (С поддержкой аугментации Data Corruption)
"""

import csv
import random
import os
from pathlib import Path

# Базовые словари
TITLES = ["The Matrix", "Inception", "The Dark Knight", "Interstellar", "Pulp Fiction", "Avatar", "Fight Club", "Forrest Gump"]
YEARS = ["1999", "2010", "2008", "2014", "1994", "2009", "1999", "1994"]
QUALITIES = ["1080p", "720p", "4K", "2160p", "WEB-DL", "BluRay", "HDRip", "HDTV"]
ARTISTS = ["Hans Zimmer", "Queen", "The Beatles", "Daft Punk", "Nirvana", "Pink Floyd", "David Bowie", "Eminem"]
TRACKS = ["Bohemian Rhapsody", "Time", "Smells Like Teen Spirit", "Let It Be", "Lose Yourself", "Heroes", "Starboy", "Money"]

def corrupt_text(text: str) -> str:
    """Аугментация: заменяет пробелы на типичные торрент-разделители."""
    if random.random() > 0.3: # С вероятностью 70% "пачкаем" строку
        separator = random.choice(['.', '_', '-'])
        return text.replace(' ', separator)
    return text

def generate_video_sample():
    title = random.choice(TITLES)
    year = random.choice(YEARS)
    quality = random.choice(QUALITIES)
    
    # Собираем чистую строку для тегов
    clean_parts = title.split() + [year, quality]
    tags = ["B-TITLE"] + ["I-TITLE"] * (len(title.split()) - 1) + ["B-YEAR", "B-QUALITY"]
    
    # Генерируем грязную строку для обучения
    raw_string = f"{title} {year} {quality}"
    dirty_string = corrupt_text(raw_string)
    
    return dirty_string, tags

def generate_audio_sample():
    artist = random.choice(ARTISTS)
    track = random.choice(TRACKS)
    quality = random.choice(["320kbps", "FLAC", "Lossless"])
    
    clean_parts = artist.split() + track.split() + [quality]
    tags = (["B-ARTIST"] + ["I-ARTIST"] * (len(artist.split()) - 1) + 
            ["B-TITLE"] + ["I-TITLE"] * (len(track.split()) - 1) + 
            ["B-QUALITY"])
    
    raw_string = f"{artist} {track} {quality}"
    dirty_string = corrupt_text(raw_string)
    
    return dirty_string, tags

def main():
    BASE_DIR = Path(__file__).resolve().parent.parent
    OUTPUT_FILE = BASE_DIR / 'data' / 'raw' / 'synthetic_media_names.csv'
    
    # Убедимся, что папка существует
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Генерация аугментированного датасета: {OUTPUT_FILE}")
    
    with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["text", "tags"])
        
        for _ in range(7000):
            text, tags = generate_video_sample()
            writer.writerow([text, ",".join(tags)])
            
        for _ in range(3000):
            text, tags = generate_audio_sample()
            writer.writerow([text, ",".join(tags)])
            
    print("Успех: Сгенерировано 10 000 мультимодальных 'грязных' строк.")

if __name__ == "__main__":
    main()