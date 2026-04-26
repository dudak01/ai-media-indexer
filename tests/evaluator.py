"""
Модуль сравнительного анализа и оценки качества моделей (Model Evaluator).
Реализует автоматизированное тестирование Baseline-алгоритма (Регулярные выражения)
и интеллектуальной модели (NER DistilBERT) на синтетическом наборе данных.

Автор: Белкина Анна Сергеевна
Тема ВКР: Автоматизация индексации медиаконтента с использованием ИИ
"""

import os
import re
import time
import pandas as pd
from typing import List, Dict
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import warnings

# Подавляем предупреждения трансформеров для чистого вывода в консоль
warnings.filterwarnings('ignore')

from core.ner_predictor import NERPredictor

class ModelEvaluator:
    """
    Класс для расчета ML-метрик и проведения A/B тестирования алгоритмов.
    Сравнивает эвристический подход (Regex) с нейросетевым (NER).
    """
    
    def __init__(self, dataset_path: str, sample_size: int = 100):
        self.dataset_path = dataset_path
        self.sample_size = sample_size
        print("[1/4] Инициализация интеллектуального ядра...")
        self.ner_model = NERPredictor()
        self.dataset = self._load_dataset()

    def _load_dataset(self) -> pd.DataFrame:
        """Загрузка тестовой выборки из CSV файла."""
        print(f"[2/4] Загрузка датасета из {self.dataset_path}...")
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Датасет не найден: {self.dataset_path}")
        
        df = pd.read_csv(self.dataset_path)
        # Берем случайную подвыборку для ускорения тестов (фиксированный random_state для воспроизводимости)
        return df.sample(n=min(self.sample_size, len(df)), random_state=42)

    def _baseline_regex_predict(self, text: str) -> Dict[str, str]:
        """
        Базовый алгоритм (Baseline). 
        Использует жесткие регулярные выражения для поиска года.
        """
        result = {"year": "", "quality": ""}
        
        # Поиск года (4 цифры от 1900 до 2099)
        year_match = re.search(r'\b(19\d{2}|20\d{2})\b', text)
        if year_match:
            result["year"] = year_match.group(1)
            
        return result

    def _extract_ground_truth(self, text: str, tags_str: str) -> Dict[str, str]:
        """Извлекает истинные значения (Ground Truth) из разметки BIO-тегов датасета."""
        words = text.split()
        tags = tags_str.split(',')
        
        truth = {"year": ""}
        for word, tag in zip(words, tags):
            if "YEAR" in tag:
                truth["year"] += word + " "
                
        return {k: v.strip() for k, v in truth.items()}

    def evaluate(self):
        """Главный метод запуска бенчмарка и расчета метрик."""
        print(f"[3/4] Запуск бенчмарка на {len(self.dataset)} строках...")
        
        y_true_year, y_pred_regex_year, y_pred_ner_year = [], [], []
        
        start_time_regex = 0.0
        start_time_ner = 0.0 
        
        # Прогоняем данные через обе модели
        for _, row in self.dataset.iterrows():
            text = str(row['text'])
            tags = str(row['tags'])
            
            # 1. Ground Truth
            truth = self._extract_ground_truth(text, tags)
            y_true_year.append(truth.get('year', ''))
            
            # 2. Baseline (Regex)
            t0 = time.time()
            regex_pred = self._baseline_regex_predict(text)
            start_time_regex += (time.time() - t0)
            y_pred_regex_year.append(regex_pred.get('year', ''))
            
            # 3. AI Model (NER)
            t1 = time.time()
            ner_pred = self.ner_model.extract_entities(text)
            start_time_ner += (time.time() - t1)
            y_pred_ner_year.append(ner_pred.get('year', ''))

        print("[4/4] Расчет метрик Precision, Recall, F1-Score...")
        self._print_report("Извлечение ГОДА (YEAR)", y_true_year, y_pred_regex_year, y_pred_ner_year)
        
        print("\nСводка по производительности (Скорость инференса):")
        print(f"┣ Baseline (Regex) : {start_time_regex:.4f} сек.")
        print(f"┗ AI Model (NER)   : {start_time_ner:.4f} сек.")

    def _print_report(self, task_name: str, y_true: List[str], y_pred_regex: List[str], y_pred_ner: List[str]):
        """Форматированный вывод результатов в консоль."""
        
        # Переводим в бинарный формат (успех/провал) для расчета метрик scikit-learn
        true_binary = [1] * len(y_true)
        
        # Очищаем предсказания NER от пробелов (так как токенизатор бьет 2008 на 20 08)
        regex_binary = [1 if p == t and t != "" else 0 for p, t in zip(y_pred_regex, y_true)]
        ner_binary = [1 if str(p).replace(" ", "") == str(t).replace(" ", "") and t != "" else 0 for p, t in zip(y_pred_ner, y_true)]
        
        regex_acc = accuracy_score(true_binary, regex_binary)
        ner_acc = accuracy_score(true_binary, ner_binary)
        
        print(f"\n{'='*60}")
        print(f" РЕЗУЛЬТАТЫ: {task_name}".center(60))
        print(f"{'='*60}")
        print(f"{'Метрика':<15} | {'Baseline (Regex)':<20} | {'AI Model (NER)':<20}")
        print(f"{'-'*60}")
        print(f"{'Accuracy':<15} | {regex_acc:.4f}               | {ner_acc:.4f}")
        
        # Расчет детальных метрик
        p_reg, r_reg, f1_reg, _ = precision_recall_fscore_support(true_binary, regex_binary, average='binary', zero_division=0)
        p_ner, r_ner, f1_ner, _ = precision_recall_fscore_support(true_binary, ner_binary, average='binary', zero_division=0)
        
        print(f"{'Precision':<15} | {p_reg:.4f}               | {p_ner:.4f}")
        print(f"{'Recall':<15} | {r_reg:.4f}               | {r_ner:.4f}")
        print(f"{'F1-Score':<15} | {f1_reg:.4f}               | {f1_ner:.4f}")
        print(f"{'-'*60}")

if __name__ == "__main__":
    # Указываем путь к нашему синтетическому датасету
    dataset_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'raw', 'synthetic_media_names.csv')
    
    evaluator = ModelEvaluator(dataset_path=dataset_file, sample_size=100)
    evaluator.evaluate()