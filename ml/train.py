import pandas as pd
import torch
import sqlite3
import logging
import os
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizerFast, DistilBertForTokenClassification
from torch.optim import AdamW
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# Настраиваем устройство (CUDA для твоей RTX 2060)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Вычислительное устройство: {DEVICE}")

# Теги, которые мы будем предсказывать (BIO формат)
LABELS = ["O", "B-TITLE", "I-TITLE", "B-YEAR", "I-YEAR", "B-QUALITY", "I-QUALITY", "B-ARTIST", "I-ARTIST"]
LABEL_MAP = {label: i for i, label in enumerate(LABELS)}

class MediaDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_len=64):
        self.data = pd.read_csv(csv_file).dropna(subset=['raw_filename'])
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        text = str(row['raw_filename'])
        
        # Токенизация
        encoding = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors='pt',
            return_offsets_mapping=True
        )
        
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        
        # Для упрощения старта (MVP) мы пока будем учить модель как классификатор последовательностей, 
        # но архитектуру закладываем под Token Classification.
        # Создаем фиктивные лейблы для заглушки (в реальном дипломе тут будет алгоритм выравнивания).
        labels = torch.zeros(self.max_len, dtype=torch.long)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

def log_to_db(epoch, loss, accuracy):
    try:
        db_path = os.path.abspath('db/diploma_system.db')
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO ml_logs (epoch, loss, accuracy) VALUES (?, ?, ?)",
            (epoch, float(loss), float(accuracy))
        )
        conn.commit()
        conn.close()
    except Exception as e:
        logging.error(f"Ошибка записи в БД: {e}")

def train_model():
    # 1. Загрузка токенизатора и модели
    logging.info("Загрузка предобученной модели DistilBERT...")
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    model = DistilBertForTokenClassification.from_pretrained(
        'distilbert-base-uncased', 
        num_labels=len(LABELS)
    )
    model.to(DEVICE)

   # 2. Подготовка данных (БОЕВОЙ РЕЖИМ)
    dataset = MediaDataset('data/raw/synthetic_media_names.csv', tokenizer)
    logging.info(f"Загружено строк для обучения: {len(dataset)}")
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    # 3. Настройка обучения
    optimizer = AdamW(model.parameters(), lr=5e-5)
    epochs = 3

    logging.info("Старт процесса обучения...")
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        correct_preds = 0
        total_preds = 0
        
        loop = tqdm(dataloader, leave=True)
        for batch in loop:
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)
            
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Подсчет фейковой accuracy для логов (для проверки пайплайна)
            predictions = torch.argmax(outputs.logits, dim=2)
            correct_preds += (predictions == labels).sum().item()
            total_preds += labels.numel()
            
            loop.set_description(f"Epoch {epoch+1}/{epochs}")
            loop.set_postfix(loss=loss.item())
            
        avg_loss = total_loss / len(dataloader)
        accuracy = correct_preds / total_preds
        
        logging.info(f"Эпоха {epoch+1} завершена | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.4f}")
        log_to_db(epoch+1, avg_loss, accuracy)

    # 4. Сохранение весов
    os.makedirs('ml/weights', exist_ok=True)
    save_path = 'ml/weights/ner_model.pt'
    torch.save(model.state_dict(), save_path)
    logging.info(f"Успех! Модель сохранена в {save_path}")

if __name__ == "__main__":
    train_model()