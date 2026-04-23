"""
Модуль инференса (предсказания) NER-модели.
Отвечает за загрузку весов PyTorch, токенизацию сырого текста,
прогон через трансформер и восстановление (Subword Alignment) извлеченных сущностей.
"""

import os
import logging
import torch
from typing import Dict, List, Tuple
from transformers import DistilBertTokenizerFast, DistilBertForTokenClassification

logger = logging.getLogger(__name__)

# Те же теги, на которых мы обучали модель в ml/train.py
LABELS = ["O", "B-TITLE", "I-TITLE", "B-YEAR", "I-YEAR", "B-QUALITY", "I-QUALITY", "B-ARTIST", "I-ARTIST"]
ID2LABEL = {i: label for i, label in enumerate(LABELS)}

# Железобетонный путь к весам
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_WEIGHTS_PATH = os.path.join(BASE_DIR, 'ml', 'weights', 'ner_model.pt')

class NERPredictor:
    """
    Класс-обертка для предсказания именованных сущностей (Named Entity Recognition).
    """
    
    def __init__(self):
        """Инициализация токенизатора и загрузка модели на доступное устройство (GPU/CPU)."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Инициализация NERPredictor на устройстве: {self.device}")
        
        try:
            self.tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
            self.model = DistilBertForTokenClassification.from_pretrained(
                'distilbert-base-uncased', 
                num_labels=len(LABELS)
            )
            self._load_weights()
        except Exception as e:
            logger.error(f"Критическая ошибка при инициализации NER-модели: {e}")
            self.model = None

    def _load_weights(self):
        """Загрузка локально обученных весов."""
        if os.path.exists(MODEL_WEIGHTS_PATH):
            state_dict = torch.load(MODEL_WEIGHTS_PATH, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()  # Перевод модели в режим предсказания (отключение Dropout)
            logger.info("Успех: Локальные веса NER-модели загружены.")
        else:
            logger.warning(f"Веса модели не найдены по пути {MODEL_WEIGHTS_PATH}. Будет использована базовая модель (предикты будут мусорными).")
            self.model.to(self.device)

    def extract_entities(self, text: str) -> Dict[str, str]:
        """
        Главный метод: принимает сырую строку и возвращает найденные сущности.
        
        Args:
            text (str): Грязное имя файла (например, "The.Matrix.1999.1080p").
            
        Returns:
            Dict[str, str]: Словарь с извлеченными данными (title, year, quality, artist).
        """
        if not self.model or not text.strip():
            return {}

        # 1. Токенизация с сохранением маппинга символов
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            return_offsets_mapping=True
        )
        
        offset_mapping = inputs.pop("offset_mapping").squeeze().tolist()
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # 2. Прогон через модель
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=2).squeeze().tolist()

        # 3. Восстановление слов из подсловных токенов (Subword Alignment)
        extracted_data = {"title": [], "year": [], "quality": [], "artist": []}
        
        for idx, pred_idx in enumerate(predictions):
            # Пропускаем специальные токены [CLS], [SEP] и паддинги
            if offset_mapping[idx] == [0, 0]:
                continue
                
            label = ID2LABEL[pred_idx]
            if label == "O":
                continue
                
            # Получаем реальный кусок текста по смещениям
            start, end = offset_mapping[idx]
            token_text = text[start:end]
            
            # Распределяем по категориям в зависимости от тега
            if "TITLE" in label:
                extracted_data["title"].append(token_text)
            elif "YEAR" in label:
                extracted_data["year"].append(token_text)
            elif "QUALITY" in label:
                extracted_data["quality"].append(token_text)
            elif "ARTIST" in label:
                extracted_data["artist"].append(token_text)

        # 4. Финальная сборка строк
        # (Так как мы разбивали по саб-токенам, просто склеиваем их обратно)
        return {
            "title": self._clean_assembled_text(extracted_data["title"]),
            "year": self._clean_assembled_text(extracted_data["year"]),
            "quality": self._clean_assembled_text(extracted_data["quality"]),
            "artist": self._clean_assembled_text(extracted_data["artist"]),
        }

    def _clean_assembled_text(self, tokens: List[str]) -> str:
        """Вспомогательный метод для склейки кусков текста и удаления артефактов."""
        if not tokens:
            return ""
        # Склеиваем токены. В реальности DistilBERT разделяет подслова символами '##',
        # но так как мы берем куски прямо из оригинального текста через offset_mapping, 
        # нам нужно просто правильно склеить их с пробелами.
        assembled = " ".join(tokens)
        return assembled.strip()