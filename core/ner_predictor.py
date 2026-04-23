"""
Модуль инференса (предсказания) NER-модели.
Отвечает за загрузку весов PyTorch, токенизацию сырого текста,
прогон через трансформер и восстановление извлеченных сущностей.
"""

import os
import re
import logging
import torch
from typing import Dict, List
from transformers import DistilBertTokenizerFast, DistilBertForTokenClassification

logger = logging.getLogger(__name__)

LABELS = ["O", "B-TITLE", "I-TITLE", "B-YEAR", "I-YEAR", "B-QUALITY", "I-QUALITY", "B-ARTIST", "I-ARTIST"]
ID2LABEL = {i: label for i, label in enumerate(LABELS)}

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_WEIGHTS_PATH = os.path.join(BASE_DIR, 'ml', 'weights', 'ner_model.pt')

class NERPredictor:
    def __init__(self):
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
        if os.path.exists(MODEL_WEIGHTS_PATH):
            state_dict = torch.load(MODEL_WEIGHTS_PATH, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()
            logger.info("Успех: Локальные веса NER-модели загружены.")
        else:
            logger.warning("Веса модели не найдены. Будет использована базовая модель.")
            self.model.to(self.device)

    def extract_entities(self, text: str) -> Dict[str, str]:
        if not self.model or not text.strip():
            return {}

        # =================================================================
        # ЭТАП ПРЕПРОЦЕССИНГА (DATA CLEANING)
        # =================================================================
        # 1. Удаляем расширение файла (например, .mkv, .mp3, .1080p.mp4)
        clean_text = re.sub(r'\.[a-zA-Z0-9]{2,4}$', '', text)
        
        # 2. Заменяем технические разделители на пробелы для токенизатора
        clean_text = clean_text.replace('.', ' ').replace('_', ' ').replace('-', ' ')
        
        # Схлопываем двойные пробелы в один
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()
        
        inputs = self.tokenizer(
            clean_text, 
            return_tensors="pt", 
            truncation=True, 
            return_offsets_mapping=True
        )
        
        offset_mapping = inputs.pop("offset_mapping").squeeze().tolist()
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=2).squeeze().tolist()

        extracted_data = {"title": [], "year": [], "quality": [], "artist": []}
        
        # Проходим по предсказаниям и вытаскиваем слова из ОЧИЩЕННОГО текста
        for idx, pred_idx in enumerate(predictions):
            if offset_mapping[idx] == [0, 0]:
                continue
                
            label = ID2LABEL[pred_idx]
            if label == "O":
                continue
                
            start, end = offset_mapping[idx]
            token_text = clean_text[start:end]
            
            if "TITLE" in label:
                extracted_data["title"].append(token_text)
            elif "YEAR" in label:
                extracted_data["year"].append(token_text)
            elif "QUALITY" in label:
                extracted_data["quality"].append(token_text)
            elif "ARTIST" in label:
                extracted_data["artist"].append(token_text)

        return {
            "title": self._clean_assembled_text(extracted_data["title"]),
            "year": self._clean_assembled_text(extracted_data["year"]),
            "quality": self._clean_assembled_text(extracted_data["quality"]),
            "artist": self._clean_assembled_text(extracted_data["artist"]),
        }

    def _clean_assembled_text(self, tokens: List[str]) -> str:
        if not tokens:
            return ""
        # DistilBERT часто разбивает числа (1080 -> 108, 0). 
        # Убираем пробелы внутри слов, которые склеиваются.
        assembled = " ".join(tokens)
        assembled = assembled.replace(" ##", "").replace("##", "")
        return assembled.strip()