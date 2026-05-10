"""
=============================================================================
Сравнительный анализ моделей v1, v2 и v3 на синтетическом и реалистичном
тестовых наборах. Генерирует таблицу метрик и столбчатую диаграмму.

Тема ВКР: «Индексация медиаконтента и обогащение метаданных
           с использованием интеллектуального анализа данных»

Автор:  Феденко Никита Александрович
Группа: ИД 23.1/Б3-22
Год:    2026

Описание:
    Скрипт строит итоговое сравнение трёх итераций модели:
        v1 — baseline (distilbert-base-uncased)
        v2 — multilingual (distilbert-base-multilingual-cased)
        v3 — multilingual + расширенный обучающий датасет

    Для каждой модели сравниваются метрики на двух тестах:
        - синтетический (in-distribution)
        - реалистичный (out-of-distribution, 52 примера вручную)
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

VERSIONS = ["v1", "v2", "v3"]
VERSION_LABELS = {
    "v1": "v1\n(baseline)",
    "v2": "v2\n(multilingual)",
    "v3": "v3\n(multi + ext.data)",
}
VERSION_COLORS = {
    "v1": "#4F81BD",  # синий
    "v2": "#9BBB59",  # зелёный
    "v3": "#C0504D",  # красный
}


def load_metrics():
    """Загружает все 6 JSON-файлов с метриками."""
    files = {}
    for v in VERSIONS:
        files[f"{v}_synthetic"] = ML_DIR / f"metrics_{v}.json"
        files[f"{v}_real"] = ML_DIR / f"metrics_{v}_real_world.json"

    data = {}
    missing = []
    for key, path in files.items():
        if not path.exists():
            missing.append(str(path))
            continue
        with open(path, "r", encoding="utf-8") as f:
            data[key] = json.load(f)

    if missing:
        print("[!] Не найдены файлы:")
        for m in missing:
            print(f"    {m}")
        return None
    return data


def extract_metrics(data):
    """Извлекает 5 ключевых метрик из каждого JSON в единый формат."""
    result = {}
    for v in VERSIONS:
        # Synthetic
        m = data[f"{v}_synthetic"]["final_metrics"]
        result[f"{v}_synthetic"] = {
            "accuracy": m["accuracy"],
            "precision_weighted": m["precision_weighted"],
            "recall_weighted": m["recall_weighted"],
            "f1_weighted": m["f1_weighted"],
            "f1_macro": m["f1_macro"],
            "exact_match": None,  # для synthetic не определён
        }
        # Real-world
        m = data[f"{v}_real"]["token_level_metrics"]
        result[f"{v}_real"] = {
            "accuracy": m["accuracy"],
            "precision_weighted": m["precision_weighted"],
            "recall_weighted": m["recall_weighted"],
            "f1_weighted": m["f1_weighted"],
            "f1_macro": m["f1_macro"],
            "exact_match": data[f"{v}_real"]["exact_match_ratio"],
        }
    return result


def print_comparison_table(metrics):
    """Печатает сравнительные таблицы в консоль."""
    metric_names = [
        ("accuracy",            "Accuracy"),
        ("precision_weighted",  "Precision (weighted)"),
        ("recall_weighted",     "Recall (weighted)"),
        ("f1_weighted",         "F1-score (weighted)"),
        ("f1_macro",            "F1-score (macro)"),
    ]

    print("\n" + "=" * 80)
    print(" СРАВНЕНИЕ ТРЁХ МОДЕЛЕЙ ".center(80, "="))
    print("=" * 80)

    for test_type, test_label in [
        ("synthetic", "СИНТЕТИЧЕСКИЙ ТЕСТ (in-distribution)"),
        ("real",      "РЕАЛИСТИЧНЫЙ ТЕСТ (out-of-distribution, 52 примера)"),
    ]:
        print(f"\n--- {test_label} ---")
        header = f"{'Метрика':<25}"
        for v in VERSIONS:
            header += f" | {v:>10}"
        print(header)
        print("-" * len(header))

        for key, label in metric_names:
            row = f"{label:<25}"
            for v in VERSIONS:
                row += f" | {metrics[f'{v}_{test_type}'][key]:>10.4f}"
            print(row)

        # Exact match только для real
        if test_type == "real":
            row = f"{'Exact match':<25}"
            for v in VERSIONS:
                em = metrics[f"{v}_real"]["exact_match"]
                row += f" | {em:>9.2%}"
            print(row)

    # Дельты для real-world
    print("\n--- ПРИРОСТЫ НА РЕАЛИСТИЧНОМ ТЕСТЕ ---")
    print(f"{'Метрика':<25} | {'v1→v2':>10} | {'v2→v3':>10} | {'v1→v3':>10}")
    print("-" * 70)
    for key, label in metric_names:
        v1 = metrics["v1_real"][key]
        v2 = metrics["v2_real"][key]
        v3 = metrics["v3_real"][key]
        d12 = (v2 - v1) * 100
        d23 = (v3 - v2) * 100
        d13 = (v3 - v1) * 100
        print(f"{label:<25} | {d12:>+9.2f}% | {d23:>+9.2f}% | {d13:>+9.2f}%")

    print("\n" + "=" * 80)


def plot_comparison(metrics, save_path):
    """Строит сгруппированную столбчатую диаграмму сравнения трёх моделей."""
    metric_keys = ["accuracy", "precision_weighted", "recall_weighted",
                   "f1_weighted", "f1_macro"]
    metric_labels = ["Accuracy", "Precision\n(weighted)", "Recall\n(weighted)",
                     "F1-score\n(weighted)", "F1-score\n(macro)"]

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    for ax, test_type, title in [
        (axes[0], "synthetic", "Synthetic test (in-distribution)"),
        (axes[1], "real", "Real-world test (out-of-distribution)"),
    ]:
        x = np.arange(len(metric_keys))
        width = 0.27

        for i, v in enumerate(VERSIONS):
            vals = [metrics[f"{v}_{test_type}"][k] for k in metric_keys]
            offset = (i - 1) * width
            bars = ax.bar(x + offset, vals, width,
                          label=VERSION_LABELS[v].replace("\n", " "),
                          color=VERSION_COLORS[v])
            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, h + 0.005,
                        f"{h:.3f}", ha="center", va="bottom", fontsize=7,
                        rotation=0)

        ax.set_title(title, fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(metric_labels, fontsize=9)
        ax.set_ylim(0, 1.08)
        ax.set_ylabel("Значение метрики")
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Сравнение моделей v1, v2 и v3 (DistilBERT NER)",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nГрафик сохранён: {save_path}")


def plot_realworld_only(metrics, save_path):
    """Отдельный график только для real-world (главный для ВКР)."""
    metric_keys = ["accuracy", "f1_weighted", "f1_macro"]
    metric_labels = ["Token\nAccuracy", "F1-score\n(weighted)", "F1-score\n(macro)"]

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(metric_keys))
    width = 0.27

    for i, v in enumerate(VERSIONS):
        vals = [metrics[f"{v}_real"][k] for k in metric_keys]
        offset = (i - 1) * width
        bars = ax.bar(x + offset, vals, width,
                      label=VERSION_LABELS[v].replace("\n", " "),
                      color=VERSION_COLORS[v])
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.003,
                    f"{h:.4f}", ha="center", va="bottom", fontsize=9)

    ax.set_title("Сравнение моделей на реалистичном тестовом наборе",
                 fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=10)
    ax.set_ylim(0.80, 0.96)  # обрезаем для лучшей видимости разницы
    ax.set_ylabel("Значение метрики")
    ax.legend(loc="lower right")
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"График real-world сохранён: {save_path}")


def main():
    data = load_metrics()
    if data is None:
        print("[FAIL] Не удалось загрузить все метрики.")
        return

    metrics = extract_metrics(data)
    print_comparison_table(metrics)
    plot_comparison(metrics, PLOTS_DIR / "comparison_v1_v2_v3.png")
    plot_realworld_only(metrics, PLOTS_DIR / "comparison_realworld_v1_v2_v3.png")


if __name__ == "__main__":
    main()