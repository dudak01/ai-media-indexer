"""
=============================================================================
Сравнительный анализ моделей v1 и v2 на синтетическом и реалистичном
тестовых наборах. Генерирует таблицу метрик и столбчатую диаграмму.

Тема ВКР: «Индексация медиаконтента и обогащение метаданных
           с использованием интеллектуального анализа данных»

Автор:  Феденко Никита Александрович
Группа: ИД 23.1/Б3-22
Год:    2026
=============================================================================
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
ML_DIR = ROOT_DIR / "ml"
PLOTS_DIR = ML_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def load_metrics():
    """Загружает все 4 JSON-файла с метриками."""
    files = {
        "v1_synthetic": ML_DIR / "metrics_v1.json",
        "v2_synthetic": ML_DIR / "metrics_v2.json",
        "v1_real": ML_DIR / "metrics_v1_real_world.json",
        "v2_real": ML_DIR / "metrics_v2_real_world.json",
    }

    data = {}
    for key, path in files.items():
        if not path.exists():
            print(f"[!] Файл не найден: {path}")
            return None
        with open(path, "r", encoding="utf-8") as f:
            data[key] = json.load(f)
    return data


def extract_metrics(data):
    """Извлекает 4 ключевых метрики из каждого JSON в единый формат."""
    result = {}
    for key in ["v1_synthetic", "v2_synthetic"]:
        m = data[key]["final_metrics"]
        result[key] = {
            "accuracy": m["accuracy"],
            "precision_weighted": m["precision_weighted"],
            "recall_weighted": m["recall_weighted"],
            "f1_weighted": m["f1_weighted"],
            "f1_macro": m["f1_macro"],
        }
    for key in ["v1_real", "v2_real"]:
        m = data[key]["token_level_metrics"]
        result[key] = {
            "accuracy": m["accuracy"],
            "precision_weighted": m["precision_weighted"],
            "recall_weighted": m["recall_weighted"],
            "f1_weighted": m["f1_weighted"],
            "f1_macro": m["f1_macro"],
        }
    return result


def print_comparison_table(metrics):
    """Печатает сравнительную таблицу в консоль."""
    print("\n" + "=" * 80)
    print(" СРАВНЕНИЕ МОДЕЛЕЙ v1 vs v2 ".center(80, "="))
    print("=" * 80)

    metric_names = [
        ("accuracy",            "Accuracy"),
        ("precision_weighted",  "Precision (weighted)"),
        ("recall_weighted",     "Recall (weighted)"),
        ("f1_weighted",         "F1-score (weighted)"),
        ("f1_macro",            "F1-score (macro)"),
    ]

    for test_type, suffix in [("synthetic", "Synthetic test"),
                              ("real", "Real-world test")]:
        v1 = metrics[f"v1_{test_type}"]
        v2 = metrics[f"v2_{test_type}"]
        print(f"\n--- {suffix} ---")
        print(f"{'Метрика':<25} | {'v1':>10} | {'v2':>10} | {'Δ':>8}")
        print("-" * 60)
        for key, label in metric_names:
            delta = v2[key] - v1[key]
            sign = "+" if delta >= 0 else ""
            print(f"{label:<25} | {v1[key]:>10.4f} | {v2[key]:>10.4f} | "
                  f"{sign}{delta*100:>6.2f}%")

    print("\n" + "=" * 80)


def plot_comparison(metrics, save_path):
    """Строит сгруппированную столбчатую диаграмму сравнения."""
    metric_keys = ["accuracy", "precision_weighted", "recall_weighted",
                   "f1_weighted", "f1_macro"]
    metric_labels = ["Accuracy", "Precision\n(weighted)", "Recall\n(weighted)",
                     "F1-score\n(weighted)", "F1-score\n(macro)"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax, test_type, title in [
        (axes[0], "synthetic", "Synthetic test (in-distribution)"),
        (axes[1], "real", "Real-world test (out-of-distribution)"),
    ]:
        v1_vals = [metrics[f"v1_{test_type}"][k] for k in metric_keys]
        v2_vals = [metrics[f"v2_{test_type}"][k] for k in metric_keys]

        x = np.arange(len(metric_keys))
        width = 0.35

        bars1 = ax.bar(x - width/2, v1_vals, width,
                       label="v1 (distilbert-base-uncased)",
                       color="#4F81BD")
        bars2 = ax.bar(x + width/2, v2_vals, width,
                       label="v2 (multilingual-cased)",
                       color="#C0504D")

        # Подписи значений над столбцами
        for bars in [bars1, bars2]:
            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, h + 0.005,
                        f"{h:.3f}", ha="center", va="bottom", fontsize=8)

        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(metric_labels, fontsize=9)
        ax.set_ylim(0, 1.08)
        ax.set_ylabel("Значение метрики")
        ax.legend(loc="lower right")
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Сравнение моделей v1 (English baseline) и "
                 "v2 (Multilingual)", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nГрафик сохранён: {save_path}")


def main():
    data = load_metrics()
    if data is None:
        print("[FAIL] Не удалось загрузить все метрики. "
              "Запустите train.py, train_v2.py и оба прогона "
              "test_on_real_data.py перед сравнением.")
        return

    metrics = extract_metrics(data)
    print_comparison_table(metrics)
    plot_comparison(metrics, PLOTS_DIR / "comparison_v1_vs_v2.png")


if __name__ == "__main__":
    main()