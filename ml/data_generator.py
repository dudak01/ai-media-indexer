import pandas as pd
import random
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

MOVIES = ["The Matrix", "Inception", "Interstellar", "Avatar", "The Dark Knight", "Pulp Fiction", "Forrest Gump", "Fight Club", "Gladiator", "The Godfather"]
ARTISTS = ["Queen", "Eminem", "Daft Punk", "The Beatles", "Nirvana", "Rihanna", "Hans Zimmer", "Linkin Park", "Metallica", "Adele"]
SONGS = ["Bohemian Rhapsody", "Lose Yourself", "Get Lucky", "Yesterday", "Smells Like Teen Spirit", "Umbrella", "Time", "Numb", "Nothing Else Matters", "Rolling in the Deep"]
PHOTO_EVENTS = ["New York Trip", "Wedding", "Birthday Party", "Summer Vacation", "Graduation", "Conference", "Family Dinner", "Camping"]

VIDEO_TAGS = ["1080p", "720p", "4K", "BDRip", "WEB-DL", "CAMRip", "HDRip"]
VIDEO_CODECS = ["x264", "H.265", "HEVC", "XviD", "AAC"]
AUDIO_TAGS = ["320kbps", "FLAC", "192kbps", "V0", "Remastered", "Lossless"]
IMAGE_PREFIXES = ["IMG", "DSC", "Photo", "Screenshot", "PXL"]

def generate_video():
    title = random.choice(MOVIES)
    year = str(random.randint(1980, 2024))
    quality = random.choice(VIDEO_TAGS)
    codec = random.choice(VIDEO_CODECS)
    ext = random.choice([".mkv", ".mp4", ".avi"])
    
    sep = random.choice([".", "_", " "])
    raw_name = f"{title.replace(' ', sep)}{sep}{year}{sep}{quality}{sep}{codec}{ext}"
    
    return {"raw_filename": raw_name, "type": "video", "title": title, "year": year, "quality": quality, "artist": None}

def generate_audio():
    artist = random.choice(ARTISTS)
    song = random.choice(SONGS)
    tag = random.choice(AUDIO_TAGS)
    track_num = str(random.randint(1, 15)).zfill(2)
    ext = random.choice([".mp3", ".flac", ".wav", ".m4a"])
    
    sep = random.choice([" - ", "_", ". "])
    raw_name = f"{track_num}{sep}{artist}{sep}{song}{sep}[{tag}]{ext}"
    
    return {"raw_filename": raw_name, "type": "audio", "title": song, "year": None, "quality": tag, "artist": artist}

def generate_image():
    prefix = random.choice(IMAGE_PREFIXES)
    event = random.choice(PHOTO_EVENTS)
    year = str(random.randint(2010, 2024))
    ext = random.choice([".jpg", ".png", ".jpeg"])
    
    sep = random.choice(["_", "-"] )
    raw_name = f"{prefix}{sep}{year}{sep}{event.replace(' ', sep)}{ext}"
    
    return {"raw_filename": raw_name, "type": "image", "title": event, "year": year, "quality": None, "artist": None}

def build_dataset(num_samples=10000):
    logging.info(f"Запуск генерации {num_samples} строк синтетических данных...")
    data = []
    for _ in range(num_samples):
        media_type = random.choices(['video', 'audio', 'image'], weights=[0.5, 0.3, 0.2])[0]
        if media_type == 'video':
            data.append(generate_video())
        elif media_type == 'audio':
            data.append(generate_audio())
        else:
            data.append(generate_image())
            
    df = pd.DataFrame(data)
    
    os.makedirs('data/raw', exist_ok=True)
    file_path = 'data/raw/synthetic_media_names.csv'
    
    df.to_csv(file_path, index=False, encoding='utf-8')
    logging.info(f"Успех: Датасет сохранен в {file_path}")
    
    print("\n--- ПРИМЕРЫ СГЕНЕРИРОВАННЫХ ФАЙЛОВ ---")
    print(df[['raw_filename', 'title', 'type']].head(5))

if __name__ == "__main__":
    build_dataset(10000)