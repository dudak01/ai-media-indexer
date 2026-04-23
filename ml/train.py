"""
Скрипт дообучения (Fine-Tuning) модели DistilBERT для задачи NER.
Обучается на аугментированном датасете для распознавания сущностей в грязных именах файлов.
"""

import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizerFast, DistilBertForTokenClassification
from torch.optim import AdamW
from tqdm import tqdm

# Те же теги, что и в предикторе
LABELS = ["O", "B-TITLE", "I-TITLE", "B-YEAR", "I-YEAR", "B-QUALITY", "I-QUALITY", "B-ARTIST", "I-ARTIST"]
LABEL2ID = {label: i for i, label in enumerate(LABELS)}

class MediaDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_len=64):
        # Читаем наш новый датасет (колонки 'text' и 'tags')
        self.data = pd.read_csv(csv_file).dropna(subset=['text', 'tags'])
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        text = str(row['text'])
        tags = str(row['tags']).split(',')

        # Токенизируем слова
        encoding = self.tokenizer(
            text.split(),
            is_split_into_words=True,
            return_offsets_mapping=False,
            padding='max_length',
            truncation=True,
            max_length=self.max_len
        )

        word_ids = encoding.word_ids()
        label_ids = []
        
        # Выравниваем теги под токены (Subword alignment)
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100) # Игнорируем спец-токены при расчете Loss
            else:
                tag = tags[word_idx] if word_idx < len(tags) else "O"
                label_ids.append(LABEL2ID.get(tag, 0))

        item = {key: torch.tensor(val) for key, val in encoding.items()}
        item['labels'] = torch.tensor(label_ids)
        return item

def calculate_accuracy(preds, labels):
    """Считает Accuracy, игнорируя токены с меткой -100 (паддинги)"""
    active_preds = preds[labels != -100]
    active_labels = labels[labels != -100]
    if len(active_labels) == 0:
        return 0.0
    return (active_preds == active_labels).float().mean().item()

def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Запуск обучения на: {device}")
    
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    model = DistilBertForTokenClassification.from_pretrained(
        'distilbert-base-uncased', 
        num_labels=len(LABELS)
    ).to(device)

    dataset = MediaDataset('data/raw/synthetic_media_names.csv', tokenizer)
    # Используем размер батча 16, чтобы не перегрузить твою RTX 2060
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    optimizer = AdamW(model.parameters(), lr=5e-5)
    
    model.train()
    epochs = 3
    
    for epoch in range(epochs):
        total_loss = 0
        total_acc = 0
        
        loop = tqdm(loader, leave=True)
        for batch in loop:
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Подсчет метрик для отчета
            preds = torch.argmax(outputs.logits, dim=2)
            acc = calculate_accuracy(preds, labels)
            total_acc += acc
            
            loop.set_description(f"Эпоха {epoch+1}/{epochs}")
            loop.set_postfix(loss=loss.item(), accuracy=f"{acc*100:.2f}%")
            
        avg_acc = total_acc / len(loader)
        print(f"Эпоха {epoch+1} завершена. Средняя точность (Accuracy): {avg_acc*100:.2f}%")
        
    # Сохраняем веса согласно требованиям ВКР (файл с расширением .pt)
    os.makedirs('ml/weights', exist_ok=True)
    save_path = 'ml/weights/ner_model.pt'
    torch.save(model.state_dict(), save_path)
    print(f"Обучение завершено. Веса сохранены в {save_path}")

if __name__ == "__main__":
    train_model()