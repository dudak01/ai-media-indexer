"""
Модуль векторного поиска на базе FAISS и мультимодальных нейросетей.
Реализует семантический поиск по тексту (MiniLM) и по смыслу изображений (CLIP).
"""

import logging
import numpy as np
from pathlib import Path
from PIL import Image
from typing import Dict, List, Any

import faiss
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

class VectorDatabase:
    """Мультимодальная векторная база данных."""
    
    def __init__(self):
        logger.info("Инициализация текстовой модели семантического поиска...")
        self.text_model = SentenceTransformer('distiluse-base-multilingual-cased-v2')
        
        logger.info("Инициализация визуальной модели (CLIP)...")
        self.clip_text = SentenceTransformer('clip-ViT-B-32-multilingual-v1')
        self.clip_vision = SentenceTransformer('clip-ViT-B-32')
        
        self.text_index = faiss.IndexFlatIP(512)
        self.image_index = faiss.IndexFlatIP(512)
        
        self.text_docs = []
        self.image_docs = []
        logger.info("Векторные модели (FAISS + CLIP) успешно загружены.")

    def add_text(self, text: str, payload: Dict):
        if not text.strip(): return
        emb = self.text_model.encode(text, normalize_embeddings=True).astype('float32')
        self.text_index.add(np.array([emb]))
        self.text_docs.append({'text': text, 'payload': payload})

    def add_image(self, image_path: str, payload: Dict):
        try:
            img = Image.open(image_path)
            emb = self.clip_vision.encode(img, normalize_embeddings=True).astype('float32')
            self.image_index.add(np.array([emb]))
            self.image_docs.append({'text': f"[ВИЗУАЛ] {Path(image_path).name}", 'payload': payload})
        except Exception as e:
            logger.warning(f"Ошибка CLIP при чтении {image_path}: {e}")

    def search(self, query: str, k: int = 3) -> List[Dict]:
        """Ищет файлы по смыслу запроса (сразу по тексту и по картинкам)."""
        results = []
        
        # 1. Текстовый поиск
        if self.text_docs:
            q_text_emb = self.text_model.encode(query, normalize_embeddings=True).astype('float32')
            D_t, I_t = self.text_index.search(np.array([q_text_emb]), k)
            for i, similarity in zip(I_t[0], D_t[0]):
                if i < len(self.text_docs) and i != -1:
                    score = round(float(similarity) * 100, 1)
                    if score > 20: 
                        res = self.text_docs[i].copy()
                        res['score'] = score
                        res['type'] = '🎬 [МЕДИА/ТЕКСТ]'
                        results.append(res)
                        
        # 2. Визуальный поиск (поиск по картинкам!)
        if self.image_docs:
            q_img_emb = self.clip_text.encode(query, normalize_embeddings=True).astype('float32')
            D_i, I_i = self.image_index.search(np.array([q_img_emb]), k)
            for i, similarity in zip(I_i[0], D_i[0]):
                if i < len(self.image_docs) and i != -1:
                    raw_sim = float(similarity)
                    if raw_sim > 0.22:
                        scaled_score = ((raw_sim - 0.22) / 0.12) * 60 + 40
                        res = self.image_docs[i].copy()
                        res['score'] = min(99.9, round(scaled_score, 1))
                        res['type'] = '📸 [ФОТО]'
                        results.append(res)
                        
        # Сортируем все результаты по уверенности ИИ
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:k]