"""
Скрипт проверки инференса (предсказаний) дообученной модели NER.
Демонстрирует способность ИИ вытаскивать метаданные из файлового мусора.
"""
import time
from core.ner_predictor import NERPredictor

def run_ai_test():
    print("="*60)
    print(" ЗАГРУЗКА ИНТЕЛЛЕКТУАЛЬНОГО ЯДРА (DistilBERT) ".center(60))
    print("="*60)
    
    start_time = time.time()
    # Эта строка поднимет модель на 253 МБ с жесткого диска в оперативку/видеопамять
    ner = NERPredictor() 
    print(f"✅ Модель успешно загружена за {time.time() - start_time:.2f} сек.\n")

    # Это самый грязный торрент-мусор, на котором обычный код ломается
    test_cases = [
        "The.Dark.Knight.2008.1080p.BluRay.x264.mkv",  # Классика с точками
        "Inception_2010_WEB-DL_720p_YIFY.mp4",         # С нижними подчеркиваниями
        "Pink_Floyd_-_Time_320kbps_Lossless.mp3",      # Музыкальный трек
        "Interstellar 2014 2160p 4K HDRip.avi"         # Много мусора в конце
    ]

    print("="*60)
    print(" РЕЗУЛЬТАТЫ РАСПОЗНАВАНИЯ СУЩНОСТЕЙ ".center(60))
    print("="*60)

    for filename in test_cases:
        print(f"\n[СЫРОЙ ФАЙЛ] : {filename}")
        
        # Прогон через нейросеть
        t0 = time.time()
        result = ner.extract_entities(filename)
        t_ms = (time.time() - t0) * 1000
        
        # Вывод чистых предсказаний
        print(f"┣ 🎬 Название : {result.get('title') or '---'}")
        print(f"┣ 📅 Год      : {result.get('year') or '---'}")
        print(f"┣ 📺 Качество : {result.get('quality') or '---'}")
        print(f"┣ 🎵 Исполнит.: {result.get('artist') or '---'}")
        print(f"┗ ⏱ Скорость : {t_ms:.1f} мс")

if __name__ == "__main__":
    run_ai_test()