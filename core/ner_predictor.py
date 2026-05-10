"""
=============================================================================
ФАЙЛ МОДЕЛИ — Инференс DistilBERT NER

Тема ВКР: «Индексация медиаконтента и обогащение метаданных
           с использованием интеллектуального анализа данных»

Автор:  Феденко Никита Александрович
Группа: Группа: ИД 23.1/Б3-22
Год:    2026

Описание:
    Модуль загружает веса предобученной DistilBERT-модели (model.pt)
    и применяет её для извлечения именованных сущностей из имён
    медиафайлов. Распознаваемые сущности:
        - TITLE   (название произведения)
        - YEAR    (год выпуска)
        - QUALITY (качество видео/аудио)
        - ARTIST  (исполнитель / автор)
=============================================================================
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
MODEL_WEIGHTS_PATH = os.path.join(BASE_DIR, 'ml', 'weights', 'model.pt')

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

        # Препроцессинг: удаляем расширение, скобки и разделители
        clean_text = re.sub(r'\.[a-zA-Z0-9]{2,4}$', '', text)
        # Убираем все виды скобок (круглые, квадратные, фигурные)
        clean_text = re.sub(r'[\(\)\[\]\{\}]', ' ', clean_text)
        # Заменяем разделители имён файлов на пробелы
        clean_text = clean_text.replace('.', ' ').replace('_', ' ').replace('-', ' ')
        # Схлопываем последовательные пробелы
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

        assembled = " ".join(tokens)
        assembled = assembled.replace(" ##", "").replace("##", "")

        # Числа: "108 0" -> "1080", повторяем дважды
        assembled = re.sub(r'(\d)\s+(\d)', r'\1\2', assembled)
        assembled = re.sub(r'(\d)\s+(\d)', r'\1\2', assembled)

        # Разрешения: "1080 p" -> "1080p"
        assembled = re.sub(r'(\d)\s+p\b', r'\1p', assembled)

        # 4K/2K: "4 K" -> "4K"
        assembled = re.sub(r'(\d)\s+K\b', r'\1K', assembled)

        # BluRay
        assembled = re.sub(r'\bBlu\s*R\s*ay\b', 'BluRay', assembled, flags=re.IGNORECASE)

        # WEB-DL
        assembled = re.sub(r'\bWEB\s+DL\b', 'WEB-DL', assembled, flags=re.IGNORECASE)

        # HDRip, HDTV
        assembled = re.sub(r'\bHD\s+(Rip|TV)\b', r'HD\1', assembled, flags=re.IGNORECASE)

        # Lossless
        assembled = re.sub(r'\bLoss\s+less\b', 'Lossless', assembled, flags=re.IGNORECASE)

        # kbps
        assembled = re.sub(r'\bk\s+b\s+ps\b', 'kbps', assembled, flags=re.IGNORECASE)

        # x264, x265, h264, h265
        assembled = re.sub(r'\b(x|h)\s*26\s*(\d)\b', r'\g<1>26\2', assembled, flags=re.IGNORECASE)

        # Склейка только латинских разбитых слов типа "Inter stellar" -> "Interstellar"
        # Условие: первая часть латиница 2-6 букв, вторая часть строчная латиница
        # НЕ трогаем кириллицу и числа
        assembled = re.sub(r'\b([A-Za-z]{2,6})\s+([a-z]{2,})\b', r'\1\2', assembled)

        return assembled.strip()