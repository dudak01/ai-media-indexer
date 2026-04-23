"""
Модуль векторного поиска на базе FAISS (Facebook AI Similarity Search).
Преобразует текстовые метаданные в плотные векторы (эмбеддинги) и обеспечивает
мгновенный семантический поиск по базе медиафайлов.
"""

import logging
import numpy as np
from typing import List, Dict, Any

try:
    import faiss
    from sentence_transformers import SentenceTransformer
except ImportError:
    faiss = None
    SentenceTransformer = None

from core.exceptions import VectorDBError

logger = logging.getLogger(__name__)

class SemanticSearchEngine:
    """Движок семантического поиска для интеллектуального анализа данных."""
    
    def __init__(self, model_name: str = 'paraphrase-multilingual-MiniLM-L12-v2'):
        """Инициализация энкодера и FAISS-индекса (с поддержкой русского языка)."""
        if not faiss or not SentenceTransformer:
            raise VectorDBError("Критическая ошибка: faiss или sentence-transformers не установлены.")
            
        logger.info(f"Загрузка модели эмбеддингов: {model_name}...")
        try:
            self.encoder = SentenceTransformer(model_name)
            self.dimension = self.encoder.get_sentence_embedding_dimension()
            
            # Используем FlatL2 индекс для точного поиска (Евклидово расстояние)
            self.index = faiss.IndexFlatL2(self.dimension)
            self.metadata_store: List[str] = [] 
            
            logger.info(f"Векторная БД успешно инициализирована (Размерность: {self.dimension}).")
        except Exception as e:
            raise VectorDBError(f"Сбой при загрузке модели: {e}")

    def add_to_index(self, semantic_text: str, file_path: str) -> None:
        """Превращает текст в математический вектор и добавляет в FAISS."""
        if not semantic_text.strip():
            return
            
        vector = self.encoder.encode([semantic_text])
        self.index.add(np.array(vector, dtype=np.float32))
        self.metadata_store.append(file_path)
        logger.debug(f"Вектор для {file_path} добавлен в индекс.")

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Ищет самые похожие файлы по смыслу запроса."""
        if self.index.ntotal == 0:
            logger.warning("Индекс FAISS пуст. Поиск невозможен.")
            return []
            
        query_vector = self.encoder.encode([query])
        distances, indices = self.index.search(np.array(query_vector, dtype=np.float32), top_k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx != -1 and idx < len(self.metadata_store):
                results.append({
                    "file_path": self.metadata_store[idx],
                    "l2_distance": float(dist),
                    "relevance_score": round(1 / (1 + float(dist)), 4)
                })
                
        return results