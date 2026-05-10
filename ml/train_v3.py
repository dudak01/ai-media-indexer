"""
=============================================================================
ФАЙЛ МОДЕЛИ — Обучение (fine-tuning) DistilBERT NER (версия v3)

Тема ВКР: «Индексация медиаконтента и обогащение метаданных
           с использованием интеллектуального анализа данных»

Автор:  Феденко Никита Александрович
Группа: ИД 23.1/Б3-22
Год:    2026

Описание:
    Версия v2 файла обучения NER. Отличия от train.py (v1):
        - базовая модель заменена с distilbert-base-uncased
          на distilbert-base-multilingual-cased для поддержки
          кириллических текстов;
        - результаты сохраняются в model3.pt и metrics_v3.json
          (требование методички №11 о нумерации множественных
          моделей);
        - графики сохраняются с суффиксом _v2.

    Все остальные параметры обучения (датасет, train/test split,
    гиперпараметры, метрики) идентичны v1 — это обеспечивает
    корректное сравнение моделей в соответствии с методологическим
    требованием №15 (улучшение модели + сравнение).
=============================================================================
"""

import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import DistilBertTokenizerFast, DistilBertForTokenClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)
from tqdm import tqdm


# =========================================================================
# КОНФИГУРАЦИЯ
# =========================================================================

LABELS = ["O", "B-TITLE", "I-TITLE", "B-YEAR", "I-YEAR",
          "B-QUALITY", "I-QUALITY", "B-ARTIST", "I-ARTIST"]
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = {i: label for i, label in enumerate(LABELS)}

# Гиперпараметры обучения (идентичны v1 для корректного сравнения)
RANDOM_STATE = 42
TEST_SIZE = 0.2
MAX_SEQ_LEN = 64
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 5e-5

# Базовая модель — мультиязычная версия DistilBERT
BASE_MODEL_NAME = "distilbert-base-multilingual-cased"

# Пути проекта (с суффиксом v2)
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "raw" / "synthetic_media_names.csv"
WEIGHTS_DIR = BASE_DIR / "ml" / "weights"
PLOTS_DIR = BASE_DIR / "ml" / "plots"
METRICS_PATH = BASE_DIR / "ml" / "metrics_v3.json"
MODEL_SAVE_PATH = WEIGHTS_DIR / "model3.pt"


# =========================================================================
# DATASET
# =========================================================================

class MediaDataset(Dataset):
    """
    PyTorch-датасет для задачи NER на именах медиафайлов.
    Преобразует пары (text, tags) в тензоры с выравниванием
    меток на уровне subword-токенов.
    """

    def __init__(self, dataframe, tokenizer, max_len=MAX_SEQ_LEN):
        self.data = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        text = str(row["text"])
        tags = str(row["tags"]).split(",")

        encoding = self.tokenizer(
            text.split(),
            is_split_into_words=True,
            return_offsets_mapping=False,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
        )

        word_ids = encoding.word_ids()
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            else:
                tag = tags[word_idx] if word_idx < len(tags) else "O"
                label_ids.append(LABEL2ID.get(tag, 0))

        item = {key: torch.tensor(val) for key, val in encoding.items()}
        item["labels"] = torch.tensor(label_ids)
        return item


# =========================================================================
# ОЦЕНКА КАЧЕСТВА МОДЕЛИ
# =========================================================================

def evaluate_model(model, dataloader, device):
    """
    Оценка модели на тестовой выборке.
    Возвращает Accuracy, Precision, Recall, F1 (weighted и macro)
    и матрицу ошибок. Расчёт ведётся только по активным токенам
    (без padding'а и спецтокенов CLS/SEP).
    """
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += outputs.loss.item()
            num_batches += 1

            preds = torch.argmax(outputs.logits, dim=2)

            mask = labels != -100
            active_preds = preds[mask].cpu().numpy()
            active_labels = labels[mask].cpu().numpy()

            all_preds.extend(active_preds.tolist())
            all_labels.extend(active_labels.tolist())

    accuracy = accuracy_score(all_labels, all_preds)
    p_w, r_w, f1_w, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="weighted", zero_division=0
    )
    p_m, r_m, f1_m, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="macro", zero_division=0
    )
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(LABELS))))

    return {
        "loss": total_loss / max(num_batches, 1),
        "accuracy": float(accuracy),
        "precision_weighted": float(p_w),
        "recall_weighted": float(r_w),
        "f1_weighted": float(f1_w),
        "precision_macro": float(p_m),
        "recall_macro": float(r_m),
        "f1_macro": float(f1_m),
        "confusion_matrix": cm.tolist(),
    }


# =========================================================================
# ВИЗУАЛИЗАЦИЯ
# =========================================================================

def plot_training_history(history, save_path):
    """График loss и accuracy по эпохам (train vs test)."""
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, history["train_loss"], "o-", label="Train Loss")
    ax1.plot(epochs, history["test_loss"], "s-", label="Test Loss")
    ax1.set_xlabel("Эпоха")
    ax1.set_ylabel("Loss")
    ax1.set_title("Динамика функции потерь (v3)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history["train_accuracy"], "o-", label="Train Accuracy")
    ax2.plot(epochs, history["test_accuracy"], "s-", label="Test Accuracy")
    ax2.set_xlabel("Эпоха")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Динамика точности (v3)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  График обучения сохранён: {save_path}")


def plot_confusion_matrix(cm, labels, save_path):
    """Матрица ошибок на тестовой выборке."""
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, cmap="Blues", aspect="auto")

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Предсказанный класс")
    ax.set_ylabel("Истинный класс")
    ax.set_title("Confusion Matrix (тестовая выборка, v2)")

    max_val = cm.max() if cm.max() > 0 else 1
    for i in range(len(labels)):
        for j in range(len(labels)):
            value = cm[i][j]
            color = "white" if value > max_val / 2 else "black"
            ax.text(j, i, str(value), ha="center", va="center",
                    color=color, fontsize=8)

    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Confusion matrix сохранена: {save_path}")


# =========================================================================
# ОСНОВНОЙ ЦИКЛ ОБУЧЕНИЯ
# =========================================================================

def train_model():
    """Полный пайплайн обучения v2 с поэпоховой оценкой на тесте."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Устройство: {device}")
    print(f"[*] Random state: {RANDOM_STATE}")
    print(f"[*] Базовая модель: {BASE_MODEL_NAME}")

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # ----- 1. ЗАГРУЗКА И РАЗДЕЛЕНИЕ ДАТАСЕТА -----
    # Применяется group-based split по уникальным значениям колонки 'text'
    # для исключения утечки данных (см. Запись №1 в RESEARCH_LOG).
    print(f"\n[1] Загрузка датасета: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH).dropna(subset=["text", "tags"]).reset_index(drop=True)
    print(f"    Всего примеров в датасете:           {len(df)}")
    print(f"    Уникальных текстов (без дубликатов): {df['text'].nunique()}")

    unique_texts = df["text"].drop_duplicates().reset_index(drop=True)
    train_texts, test_texts = train_test_split(
        unique_texts,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        shuffle=True,
    )

    train_df = df[df["text"].isin(train_texts)].reset_index(drop=True)
    test_df = df[df["text"].isin(test_texts)].reset_index(drop=True)

    overlap = set(train_df["text"]).intersection(set(test_df["text"]))
    assert len(overlap) == 0, f"Утечка данных: {len(overlap)} текстов в обеих выборках"

    print(f"    Train: {len(train_df)} строк ({train_df['text'].nunique()} уник. текстов)")
    print(f"    Test:  {len(test_df)} строк ({test_df['text'].nunique()} уник. текстов)")
    print(f"    Пересечение train/test: 0 (group split корректен)")

    # ----- 2. ИНИЦИАЛИЗАЦИЯ МОДЕЛИ -----
    print(f"\n[2] Инициализация DistilBERT ({BASE_MODEL_NAME})")
    tokenizer = DistilBertTokenizerFast.from_pretrained(BASE_MODEL_NAME)
    model = DistilBertForTokenClassification.from_pretrained(
        BASE_MODEL_NAME,
        num_labels=len(LABELS),
    ).to(device)

    train_dataset = MediaDataset(train_df, tokenizer)
    test_dataset = MediaDataset(test_df, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    # ----- 3. ЦИКЛ ОБУЧЕНИЯ -----
    history = {
        "train_loss": [], "train_accuracy": [],
        "test_loss": [], "test_accuracy": [],
        "test_precision_weighted": [], "test_recall_weighted": [], "test_f1_weighted": [],
    }

    print(f"\n[3] Старт обучения ({EPOCHS} эпох, batch_size={BATCH_SIZE}, lr={LEARNING_RATE})")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        loop = tqdm(train_loader, desc=f"Эпоха {epoch}/{EPOCHS}", leave=False)

        for batch in loop:
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            preds = torch.argmax(outputs.logits, dim=2)
            mask = labels != -100
            epoch_correct += (preds[mask] == labels[mask]).sum().item()
            epoch_total += mask.sum().item()

            loop.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = epoch_loss / len(train_loader)
        train_acc = epoch_correct / max(epoch_total, 1)

        test_metrics = evaluate_model(model, test_loader, device)

        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_acc)
        history["test_loss"].append(test_metrics["loss"])
        history["test_accuracy"].append(test_metrics["accuracy"])
        history["test_precision_weighted"].append(test_metrics["precision_weighted"])
        history["test_recall_weighted"].append(test_metrics["recall_weighted"])
        history["test_f1_weighted"].append(test_metrics["f1_weighted"])

        print(
            f"  Эпоха {epoch}: "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f} | "
            f"test_loss={test_metrics['loss']:.4f}, "
            f"test_acc={test_metrics['accuracy']:.4f}, "
            f"test_f1={test_metrics['f1_weighted']:.4f}"
        )

    # ----- 4. ФИНАЛЬНАЯ ОЦЕНКА -----
    print(f"\n[4] Финальная оценка на тестовой выборке")
    final_metrics = evaluate_model(model, test_loader, device)

    print(f"  Accuracy:             {final_metrics['accuracy']:.4f}")
    print(f"  Precision (weighted): {final_metrics['precision_weighted']:.4f}")
    print(f"  Recall    (weighted): {final_metrics['recall_weighted']:.4f}")
    print(f"  F1-score  (weighted): {final_metrics['f1_weighted']:.4f}")
    print(f"  Precision (macro):    {final_metrics['precision_macro']:.4f}")
    print(f"  Recall    (macro):    {final_metrics['recall_macro']:.4f}")
    print(f"  F1-score  (macro):    {final_metrics['f1_macro']:.4f}")

    if final_metrics["accuracy"] >= 0.70:
        print(f"\n  Accuracy {final_metrics['accuracy']:.4f} >= 0.70 — требование методички выполнено")
    else:
        print(f"\n  Accuracy {final_metrics['accuracy']:.4f} < 0.70 — требуется оптимизация")

    # ----- 5. СОХРАНЕНИЕ РЕЗУЛЬТАТОВ -----
    print(f"\n[5] Сохранение результатов")

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"  Веса модели:    {MODEL_SAVE_PATH}")

    output = {
        "model_version": "v3_multilingual_extended_data",
        "base_model": BASE_MODEL_NAME,
        "hyperparameters": {
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "max_seq_len": MAX_SEQ_LEN,
            "train_size": len(train_df),
            "test_size": len(test_df),
            "test_split_ratio": TEST_SIZE,
            "random_state": RANDOM_STATE,
        },
        "training_history": history,
        "final_metrics": final_metrics,
        "labels": LABELS,
    }

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"  Метрики (JSON): {METRICS_PATH}")

    plot_training_history(history, PLOTS_DIR / "training_history_v3.png")
    plot_confusion_matrix(
        np.array(final_metrics["confusion_matrix"]),
        LABELS,
        PLOTS_DIR / "confusion_matrix_v3.png",
    )

    print(f"\n[OK] Обучение v3 завершено")


if __name__ == "__main__":
    train_model()