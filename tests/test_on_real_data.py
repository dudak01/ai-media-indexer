"""
=============================================================================
Оценка NER-модели на реалистичном тестовом наборе.

Тема ВКР: «Индексация медиаконтента и обогащение метаданных
           с использованием интеллектуального анализа данных»

Автор:  Феденко Никита Александрович
Группа: ИД 23.1/Б3-22
Год:    2026

Описание:
    Тестовый набор представляет собой 52 реальных имени медиафайлов
    в форматах, типичных для пользовательских коллекций (раздачи
    P2P-сетей, локальные библиотеки, сохранения с различных
    источников). В отличие от синтетического теста train_test_split,
    этот набор содержит:
        - названия, не встречавшиеся в обучающем датасете;
        - реальные кириллические названия;
        - различные шаблоны именования с разделителями и расширениями;
        - граничные случаи (короткие, нестандартные имена).

    Скрипт поддерживает оценку обеих версий модели:
        python tests/test_on_real_data.py             (по умолчанию v1)
        python tests/test_on_real_data.py --model v1  (явно v1)
        python tests/test_on_real_data.py --model v2  (мультиязычная)

    Это позволяет сравнить v1 и v2 на одном и том же реалистичном
    наборе и зафиксировать улучшение в соответствии с
    методологическим требованием №15 методички.
=============================================================================
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)
from transformers import DistilBertTokenizerFast, DistilBertForTokenClassification

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

LABELS = ["O", "B-TITLE", "I-TITLE", "B-YEAR", "I-YEAR",
          "B-QUALITY", "I-QUALITY", "B-ARTIST", "I-ARTIST"]
ID2LABEL = {i: label for i, label in enumerate(LABELS)}


# =========================================================================
# КОНФИГУРАЦИЯ ВЕРСИЙ МОДЕЛИ
# =========================================================================

MODEL_VERSIONS = {
    "v1": {
        "base_model": "distilbert-base-uncased",
        "weights_path": ROOT_DIR / "ml" / "weights" / "model.pt",
        "output_path": ROOT_DIR / "ml" / "metrics_v1_real_world.json",
        "label": "v1_baseline_en",
    },
    "v2": {
        "base_model": "distilbert-base-multilingual-cased",
        "weights_path": ROOT_DIR / "ml" / "weights" / "model2.pt",
        "output_path": ROOT_DIR / "ml" / "metrics_v2_real_world.json",
        "label": "v2_multilingual_cased",
    },
    "v3": {
        "base_model": "distilbert-base-multilingual-cased",
        "weights_path": ROOT_DIR / "ml" / "weights" / "model3.pt",
        "output_path": ROOT_DIR / "ml" / "metrics_v3_real_world.json",
        "label": "v3_multilingual_extended_data",
    },
}


# =========================================================================
# РЕАЛИСТИЧНЫЙ TEST SET
# =========================================================================

REAL_WORLD_TEST_SET = [
    # --- АНГЛОЯЗЫЧНЫЕ ФИЛЬМЫ ---
    ("The.Wolf.of.Wall.Street.2013.1080p.BluRay.x264-AMIABLE.mkv",
     ["B-TITLE", "I-TITLE", "I-TITLE", "I-TITLE", "I-TITLE",
      "B-YEAR", "B-QUALITY", "I-QUALITY", "I-QUALITY", "O"]),

    ("Trainspotting.1996.1080p.BluRay.x264-CtrlHD.mkv",
     ["B-TITLE", "B-YEAR", "B-QUALITY", "I-QUALITY", "I-QUALITY", "O"]),

    ("Eternal.Sunshine.of.the.Spotless.Mind.2004.720p.BluRay-YTS.mkv",
     ["B-TITLE", "I-TITLE", "I-TITLE", "I-TITLE", "I-TITLE", "I-TITLE",
      "B-YEAR", "B-QUALITY", "I-QUALITY", "O"]),

    ("Lost.in.Translation.2003.BDRip.XviD.mkv",
     ["B-TITLE", "I-TITLE", "I-TITLE", "B-YEAR", "B-QUALITY", "I-QUALITY"]),

    ("There.Will.Be.Blood.2007.1080p.WEB-DL.mp4",
     ["B-TITLE", "I-TITLE", "I-TITLE", "I-TITLE",
      "B-YEAR", "B-QUALITY", "I-QUALITY", "I-QUALITY"]),

    ("No.Country.for.Old.Men.2007.2160p.UHD.BluRay.HEVC.mkv",
     ["B-TITLE", "I-TITLE", "I-TITLE", "I-TITLE", "I-TITLE",
      "B-YEAR", "B-QUALITY", "I-QUALITY", "I-QUALITY", "I-QUALITY"]),

    ("Drive.2011.1080p.BluRay.DTS.x264-EbP.mkv",
     ["B-TITLE", "B-YEAR", "B-QUALITY", "I-QUALITY", "I-QUALITY", "I-QUALITY", "O"]),

    ("Whiplash 2014 1080p BluRay YTS.mp4",
     ["B-TITLE", "B-YEAR", "B-QUALITY", "I-QUALITY", "O"]),

    ("Sicario_2015_BDRip_1080p.avi",
     ["B-TITLE", "B-YEAR", "B-QUALITY", "I-QUALITY"]),

    ("The Grand Budapest Hotel.2014.4K.HEVC.mkv",
     ["B-TITLE", "I-TITLE", "I-TITLE", "I-TITLE",
      "B-YEAR", "B-QUALITY", "I-QUALITY"]),

    ("Shutter Island 2010 720p HDRip.mp4",
     ["B-TITLE", "I-TITLE", "B-YEAR", "B-QUALITY", "I-QUALITY"]),

    ("Gone.Girl.2014.WEB-DL.1080p.x265.mkv",
     ["B-TITLE", "I-TITLE", "B-YEAR",
      "B-QUALITY", "I-QUALITY", "I-QUALITY", "I-QUALITY"]),

    ("Inception.2010.1080p.BluRay.x264.mkv",
     ["B-TITLE", "B-YEAR", "B-QUALITY", "I-QUALITY", "I-QUALITY"]),

    ("The.Matrix.1999.4K.UHD.BluRay.HEVC.mkv",
     ["B-TITLE", "I-TITLE", "B-YEAR",
      "B-QUALITY", "I-QUALITY", "I-QUALITY", "I-QUALITY"]),

    ("Interstellar (2014).mkv",
     ["B-TITLE", "B-YEAR"]),

    # --- РУССКИЕ ФИЛЬМЫ КИРИЛЛИЦЕЙ ---
    ("Брат.1997.DVDRip.avi",
     ["B-TITLE", "B-YEAR", "B-QUALITY"]),

    ("Москва.слезам.не.верит.1980.HDTV.mkv",
     ["B-TITLE", "I-TITLE", "I-TITLE", "I-TITLE", "B-YEAR", "B-QUALITY"]),

    ("Иван.Васильевич.меняет.профессию.1973.BDRip.1080p.mkv",
     ["B-TITLE", "I-TITLE", "I-TITLE", "I-TITLE",
      "B-YEAR", "B-QUALITY", "I-QUALITY"]),

    ("Бриллиантовая рука 1968 720p.mp4",
     ["B-TITLE", "I-TITLE", "B-YEAR", "B-QUALITY"]),

    ("Чебурашка.2023.WEB-DL.1080p.mkv",
     ["B-TITLE", "B-YEAR", "B-QUALITY", "I-QUALITY", "I-QUALITY"]),

    ("Левиафан-2014-BluRay-1080p.mkv",
     ["B-TITLE", "B-YEAR", "B-QUALITY", "I-QUALITY"]),

    ("Высоцкий_Спасибо_что_живой_2011_HDRip.avi",
     ["B-TITLE", "I-TITLE", "I-TITLE", "I-TITLE", "B-YEAR", "B-QUALITY"]),

    ("Ирония судьбы.1975.DVDRip.avi",
     ["B-TITLE", "I-TITLE", "B-YEAR", "B-QUALITY"]),

    ("Слово пацана 2023 1080p.mkv",
     ["B-TITLE", "I-TITLE", "B-YEAR", "B-QUALITY"]),

    ("Текст 2019 BDRip.mkv",
     ["B-TITLE", "B-YEAR", "B-QUALITY"]),

    # --- РУССКИЕ ФИЛЬМЫ ТРАНСЛИТОМ ---
    ("Brat.1997.DVDRip.avi",
     ["B-TITLE", "B-YEAR", "B-QUALITY"]),

    ("Moskva.Slezam.Ne.Verit.1980.HDTV.mkv",
     ["B-TITLE", "I-TITLE", "I-TITLE", "I-TITLE", "B-YEAR", "B-QUALITY"]),

    ("Levafan.2014.BluRay.1080p.mkv",
     ["B-TITLE", "B-YEAR", "B-QUALITY", "I-QUALITY"]),

    ("Stalingrad-2013-1080p-BDRip.mkv",
     ["B-TITLE", "B-YEAR", "B-QUALITY", "I-QUALITY"]),

    ("Ironia_Sudby_1975_DVDRip.avi",
     ["B-TITLE", "I-TITLE", "B-YEAR", "B-QUALITY"]),

    # --- АНГЛОЯЗЫЧНАЯ МУЗЫКА ---
    ("Bjork - Hyperballad - 320kbps.mp3",
     ["B-ARTIST", "B-TITLE", "B-QUALITY"]),

    ("Tool - Lateralus - FLAC.flac",
     ["B-ARTIST", "B-TITLE", "B-QUALITY"]),

    ("Arctic_Monkeys_-_Do_I_Wanna_Know_-_Lossless.flac",
     ["B-ARTIST", "I-ARTIST", "B-TITLE", "I-TITLE", "I-TITLE", "I-TITLE", "B-QUALITY"]),

    ("Tame Impala - The Less I Know The Better.mp3",
     ["B-ARTIST", "I-ARTIST", "B-TITLE", "I-TITLE", "I-TITLE",
      "I-TITLE", "I-TITLE", "I-TITLE"]),

    ("Rage.Against.the.Machine - Killing.in.the.Name - 320.mp3",
     ["B-ARTIST", "I-ARTIST", "I-ARTIST", "I-ARTIST",
      "B-TITLE", "I-TITLE", "I-TITLE", "I-TITLE", "B-QUALITY"]),

    ("Queen - Bohemian Rhapsody.flac",
     ["B-ARTIST", "B-TITLE", "I-TITLE"]),

    ("Pink Floyd - Time - FLAC.flac",
     ["B-ARTIST", "I-ARTIST", "B-TITLE", "B-QUALITY"]),

    ("Eminem - Lose Yourself - 320kbps.mp3",
     ["B-ARTIST", "B-TITLE", "I-TITLE", "B-QUALITY"]),

    # --- РУССКАЯ МУЗЫКА КИРИЛЛИЦЕЙ ---
    ("Кино - Группа крови.flac",
     ["B-ARTIST", "B-TITLE", "I-TITLE"]),

    ("ДДТ - Что такое осень.mp3",
     ["B-ARTIST", "B-TITLE", "I-TITLE", "I-TITLE"]),

    ("Земфира - Хочешь - Lossless.flac",
     ["B-ARTIST", "B-TITLE", "B-QUALITY"]),

    ("Сектор Газа - Лирика - 320kbps.mp3",
     ["B-ARTIST", "I-ARTIST", "B-TITLE", "B-QUALITY"]),

    ("Хабиб - Ягода Малинка.mp3",
     ["B-ARTIST", "B-TITLE", "I-TITLE"]),

    # --- СЕРИАЛЫ ---
    ("Breaking.Bad.S03E07.1080p.BluRay.x264-DEMAND.mkv",
     ["B-TITLE", "I-TITLE", "O",
      "B-QUALITY", "I-QUALITY", "I-QUALITY", "O"]),

    ("Stranger.Things.S04E01.WEB-DL.1080p.mkv",
     ["B-TITLE", "I-TITLE", "O",
      "B-QUALITY", "I-QUALITY", "I-QUALITY"]),

    ("Game of Thrones S01E09 1080p.mkv",
     ["B-TITLE", "I-TITLE", "I-TITLE", "O", "B-QUALITY"]),

    ("The Witcher S02E03 4K HDR.mkv",
     ["B-TITLE", "I-TITLE", "O", "B-QUALITY", "O"]),

    # --- EDGE CASES ---
    ("IMG_20240515_142233.jpg",
     ["O", "O", "O"]),

    ("video_001.mp4",
     ["O", "O"]),

    ("DSC02145.JPG",
     ["O"]),

    ("untitled.mp3",
     ["O"]),

    ("2024_отпуск.mp4",
     ["B-YEAR", "O"]),
]


# =========================================================================
# ПРЕПРОЦЕССИНГ
# =========================================================================

def preprocess_filename(text: str) -> str:
    """
    Применяет ту же последовательность преобразований, что и
    NERPredictor.extract_entities() в core/ner_predictor.py:
        1. Удаление расширения файла (последняя точка + 2-4 символа)
        2. Замена разделителей '.', '_', '-' на пробелы
        3. Схлопывание последовательных пробелов
    """
    clean_text = re.sub(r"\.[a-zA-Z0-9]{2,4}$", "", text)
    clean_text = clean_text.replace(".", " ").replace("_", " ").replace("-", " ")
    clean_text = re.sub(r"\s+", " ", clean_text).strip()
    return clean_text


# =========================================================================
# ЗАГРУЗКА МОДЕЛИ ВЫБРАННОЙ ВЕРСИИ
# =========================================================================

def load_model(version: str):
    """
    Загружает базовую DistilBERT-модель указанной версии (v1 или v2)
    и подгружает дообученные веса. Возвращает (tokenizer, model, device).
    """
    config = MODEL_VERSIONS[version]
    base_model = config["base_model"]
    weights_path = config["weights_path"]

    if not weights_path.exists():
        raise FileNotFoundError(
            f"Файл весов не найден: {weights_path}\n"
            f"Сначала обучите модель {version} соответствующим train-скриптом."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = DistilBertTokenizerFast.from_pretrained(base_model)
    model = DistilBertForTokenClassification.from_pretrained(
        base_model,
        num_labels=len(LABELS),
    ).to(device)

    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    return tokenizer, model, device


# =========================================================================
# ИНФЕРЕНС НА УРОВНЕ СЛОВ
# =========================================================================

def predict_word_tags(tokenizer, model, device, text: str):
    """
    Применяет модель к строке и возвращает список IOB-меток
    на уровне слов. Для каждого слова берётся метка ПЕРВОГО subtoken'a.
    """
    words = text.split()
    if not words:
        return []

    encoding = tokenizer(
        words,
        is_split_into_words=True,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=64,
        return_offsets_mapping=False,
    )
    word_ids = encoding.word_ids()
    inputs = {k: v.to(device) for k, v in encoding.items()}

    with torch.no_grad():
        logits = model(**inputs).logits
        preds = torch.argmax(logits, dim=2).squeeze().tolist()

    word_to_pred = {}
    for tok_idx, word_idx in enumerate(word_ids):
        if word_idx is not None and word_idx not in word_to_pred:
            word_to_pred[word_idx] = preds[tok_idx]

    return [ID2LABEL[word_to_pred.get(i, 0)] for i in range(len(words))]


# =========================================================================
# ВАЛИДАЦИЯ TEST SET
# =========================================================================

def validate_test_set():
    """Проверяет, что для каждого примера длина words после preprocess
    совпадает с длиной разметки."""
    errors = []
    for filename, tags in REAL_WORLD_TEST_SET:
        clean = preprocess_filename(filename)
        words = clean.split()
        if len(words) != len(tags):
            errors.append(
                f"  '{filename}'\n"
                f"    после preprocess: {words} (длина {len(words)})\n"
                f"    разметка: длина {len(tags)}"
            )
    if errors:
        print("Ошибки в test set:")
        for e in errors:
            print(e)
        return False
    return True


# =========================================================================
# ОСНОВНАЯ ОЦЕНКА
# =========================================================================

def evaluate_on_real_data(version: str):
    config = MODEL_VERSIONS[version]

    print("=" * 70)
    print(f"Оценка NER-модели {version} ({config['label']}) на реалистичном тесте")
    print("=" * 70)

    if not validate_test_set():
        print("\n[FAIL] Test set содержит ошибки разметки. Прерываю.")
        return

    print(f"\n[1] Загрузка модели {version}...")
    print(f"    Базовая модель: {config['base_model']}")
    print(f"    Веса:           {config['weights_path']}")

    try:
        tokenizer, model, device = load_model(version)
    except FileNotFoundError as e:
        print(f"\n[FAIL] {e}")
        return

    print(f"    Устройство:     {device}")
    print(f"    Загружено: {len(REAL_WORLD_TEST_SET)} реалистичных примеров")

    all_true_label_ids = []
    all_pred_label_ids = []

    exact_match_count = 0
    error_log = []

    print(f"\n[2] Прогон инференса...")
    for filename, true_tags in REAL_WORLD_TEST_SET:
        clean_text = preprocess_filename(filename)
        pred_tags = predict_word_tags(tokenizer, model, device, clean_text)

        true_ids = [LABELS.index(t) if t in LABELS else 0 for t in true_tags]
        pred_ids = [LABELS.index(t) if t in LABELS else 0 for t in pred_tags]

        all_true_label_ids.extend(true_ids)
        all_pred_label_ids.extend(pred_ids)

        if true_tags == pred_tags:
            exact_match_count += 1
        else:
            error_log.append({
                "filename": filename,
                "preprocessed": clean_text,
                "true": true_tags,
                "pred": pred_tags,
            })

    accuracy = accuracy_score(all_true_label_ids, all_pred_label_ids)
    p_w, r_w, f1_w, _ = precision_recall_fscore_support(
        all_true_label_ids, all_pred_label_ids,
        average="weighted", zero_division=0,
    )
    p_m, r_m, f1_m, _ = precision_recall_fscore_support(
        all_true_label_ids, all_pred_label_ids,
        average="macro", zero_division=0,
    )
    cm = confusion_matrix(
        all_true_label_ids, all_pred_label_ids,
        labels=list(range(len(LABELS))),
    )

    exact_match_ratio = exact_match_count / len(REAL_WORLD_TEST_SET)

    print(f"\n[3] Результаты")
    print("-" * 70)
    print(f"Версия модели:           {config['label']}")
    print(f"Всего примеров:          {len(REAL_WORLD_TEST_SET)}")
    print(f"Точные совпадения:       {exact_match_count} / {len(REAL_WORLD_TEST_SET)} "
          f"({exact_match_ratio:.2%})")
    print()
    print(f"Token-level Accuracy:    {accuracy:.4f}")
    print(f"Precision (weighted):    {p_w:.4f}")
    print(f"Recall    (weighted):    {r_w:.4f}")
    print(f"F1-score  (weighted):    {f1_w:.4f}")
    print(f"Precision (macro):       {p_m:.4f}")
    print(f"Recall    (macro):       {r_m:.4f}")
    print(f"F1-score  (macro):       {f1_m:.4f}")

    if error_log:
        print(f"\n[4] Примеры ошибок (показаны первые 10):")
        print("-" * 70)
        for err in error_log[:10]:
            print(f"  Файл:             {err['filename']}")
            print(f"  После preprocess: {err['preprocessed']}")
            print(f"  Эталон:           {err['true']}")
            print(f"  Предсказание:     {err['pred']}")
            print()

    output = {
        "model_version": config["label"],
        "base_model": config["base_model"],
        "test_set_type": "real_world",
        "test_set_size": len(REAL_WORLD_TEST_SET),
        "exact_match_count": exact_match_count,
        "exact_match_ratio": float(exact_match_ratio),
        "token_level_metrics": {
            "accuracy": float(accuracy),
            "precision_weighted": float(p_w),
            "recall_weighted": float(r_w),
            "f1_weighted": float(f1_w),
            "precision_macro": float(p_m),
            "recall_macro": float(r_m),
            "f1_macro": float(f1_m),
        },
        "confusion_matrix": cm.tolist(),
        "labels": LABELS,
        "errors": error_log,
    }

    output_path = config["output_path"]
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n[5] Результаты сохранены: {output_path}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Оценка NER-модели на реалистичном тестовом наборе"
    )
    parser.add_argument(
        "--model",
        choices=["v1", "v2", "v3"],
        default="v1",
        help="Версия модели для оценки (v1 — baseline; v2 — multilingual; "
             "v3 — multilingual + extended data). По умолчанию: v1.",
    )
    args = parser.parse_args()
    evaluate_on_real_data(args.model)


if __name__ == "__main__":
    main()