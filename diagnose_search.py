"""
Диагностика: показывает что лежит в text-индексе VectorDatabase
после индексации тестовой папки.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from core.scanner import DirectoryScanner
from core.extractor import TechnicalMetadataExtractor
from core.enrichment import EnrichmentService
from core.ner_predictor import NERPredictor
from core.vector_db import VectorDatabase

# Импорт хелперов и логики из app.py
sys.path.insert(0, str(ROOT / "ui"))
from app import is_screenshot_filename, has_cyrillic

TEST_DIR = r"C:\Users\Никита Феденко\Desktop\test_media"

scanner = DirectoryScanner(TEST_DIR, compute_hashes=False)
extractor = TechnicalMetadataExtractor()
enricher = EnrichmentService()
ner = NERPredictor()
vector_db = VectorDatabase()

inventory = scanner.scan()

for ftype, files in inventory.items():
    for media in files:
        ner_result = ner.extract_entities(media.name)
        title = ner_result.get('title') or ''
        year = ner_result.get('year') or '---'
        quality = ner_result.get('quality') or '---'
        artist = ner_result.get('artist') or ''
        enriched = enricher.enrich(media.full_path, ftype)
        stem = Path(media.name).stem

        payload = {'real_file_name': media.name, 'ftype': ftype}

        if ftype == 'image':
            vector_db.add_image(media.full_path, payload)
            if is_screenshot_filename(stem):
                type_tokens = ['скриншот снимок экрана screenshot скрин']
            else:
                type_tokens = ['фотография фото изображение photo image']
            parts = type_tokens + [stem]
            ocr_text = enriched.get('ocr_text', '')
            if ocr_text:
                parts.append(ocr_text)
            search_text = " ".join(filter(bool, parts))
            print(f"\n=== IMAGE: {media.name} ===")
            print(f"  Тип: {'скриншот' if is_screenshot_filename(stem) else 'фото'}")
            print(f"  Полный text для индекса: {search_text[:300]}")
            vector_db.add_text(search_text, payload)
        elif ftype == 'audio':
            parts = ['музыка аудио песня трек']
            if artist: parts.append(f"исполнитель {artist}")
            if title: parts.append(f"название {title}")
            if quality and quality != '---': parts.append(quality)
            if has_cyrillic(stem): parts.append('русская русский на русском')
            parts.append(stem)
            search_text = " ".join(filter(bool, parts))
            print(f"\n=== AUDIO: {media.name} ===")
            print(f"  Полный text для индекса: {search_text}")
            vector_db.add_text(search_text, payload)
        elif ftype == 'video':
            parts = ['видео фильм кино']
            if title: parts.append(f"название {title}")
            if year != '---': parts.append(f"год {year}")
            if quality != '---': parts.append(quality)
            if has_cyrillic(stem): parts.append('русское кино русский фильм')
            parts.append(stem)
            search_text = " ".join(filter(bool, parts))
            print(f"\n=== VIDEO: {media.name} ===")
            print(f"  Полный text для индекса: {search_text}")
            vector_db.add_text(search_text, payload)

print("\n\n=== ТЕСТ ПОИСКА ===")
for q in ['скриншот', 'Хабиб', 'снимок экрана', 'фотография']:
    print(f"\nЗапрос: '{q}'")
    results = vector_db.search(q, k=10)
    for r in results:
        name = r['payload'].get('real_file_name', '?')
        print(f"  {r['score']:5.1f}%  {r['type']}  {name}")