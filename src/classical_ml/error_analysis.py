# Назначение файла: F1.8: анализ сохранённых предсказаний без повторного обучения.
# Формирует агрегированные метрики, матрицу ошибок и список ID без текстов Q&A.
# Примечание: комментарии не влияют на работу программы.

"""F1.8: text-free analysis of frozen F1.7 predictions.

This module never loads a trained joblib model, refits TF-IDF, or copies Q&A
contexts into public-facing report objects. Case IDs are local-only artifacts.
"""
from __future__ import annotations

from collections import Counter
from statistics import median
from typing import Any

import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


# Ошибка входных прогнозов, аудита или метрик анализа ошибок.
class ErrorAnalysisError(ValueError):
    """Invalid or inconsistent F1.7 prediction/audit inputs."""


# Сверяет предсказания, подсчитывает ошибки по классам и их пары.
def analyze_errors(test_rows: list[dict], predictions: dict, f17_report: dict) -> tuple[dict, list[dict]]:
    """Compare frozen test predictions with the same ordered F1.6 test split.

    The returned report is aggregate-only. The error-case list includes only
    IDs, true/predicted labels and text-length counts; keep it local.
    """
    if not isinstance(test_rows, list) or not test_rows:
        raise ErrorAnalysisError("nonempty F1.6 test rows required")
    if not isinstance(predictions, dict) or not isinstance(predictions.get("test"), list):
        raise ErrorAnalysisError("missing F1.7 test predictions")
    if not isinstance(f17_report, dict) or not isinstance(f17_report.get("test"), dict):
        raise ErrorAnalysisError("missing F1.7 test metrics")

    expected = f17_report["test"]
    labels = expected.get("labels")
    if (not isinstance(labels, list) or len(labels) < 2
            or any(not isinstance(x, str) or not x.strip() for x in labels)
            or len(set(labels)) != len(labels)):
        raise ErrorAnalysisError("invalid F1.7 labels")
    test_predictions = predictions["test"]
    if len(test_rows) != len(test_predictions):
        raise ErrorAnalysisError("test rows/predictions count mismatch")
    if type(expected.get("n")) is not int or expected["n"] != len(test_rows):
        raise ErrorAnalysisError("F1.7 report sample size mismatch")
    label_set = set(labels)
    seen: set[str] = set()
    y_true: list[str] = []
    y_pred: list[str] = []
    error_cases: list[dict] = []
    right_lengths: list[int] = []
    wrong_lengths: list[int] = []
    for idx, (row, pred) in enumerate(zip(test_rows, test_predictions)):
        if not isinstance(row, dict) or not isinstance(pred, dict):
            raise ErrorAnalysisError(f"invalid test/prediction row {idx}")
        if set(pred) != {"id", "true", "predicted"}:
            raise ErrorAnalysisError("prediction row must contain only id/true/predicted")
        uid = row.get("id")
        truth = row.get("domain")
        predicted = pred.get("predicted")
        context = row.get("context")
        if (not isinstance(uid, str) or not uid.strip() or uid in seen
                or row.get("split") != "test" or not isinstance(context, str)
                or not context.strip() or truth not in label_set):
            raise ErrorAnalysisError("invalid or duplicate F1.6 test row")
        if pred.get("id") != uid or pred.get("true") != truth or predicted not in label_set:
            raise ErrorAnalysisError("F1.7 predictions do not match the same ordered test split")
        seen.add(uid)
        y_true.append(truth)
        y_pred.append(predicted)
        words = len(context.split())
        if truth == predicted:
            right_lengths.append(words)
        else:
            wrong_lengths.append(words)
            error_cases.append({"id": uid, "true": truth,
                                "predicted": predicted, "question_words": words})

    matrix = confusion_matrix(y_true, y_pred, labels=labels).tolist()
    if matrix != expected.get("confusion_matrix"):
        raise ErrorAnalysisError("F1.7 confusion matrix differs from frozen predictions")
    accuracy = sum(a == b for a, b in zip(y_true, y_pred)) / len(y_true)
    if (not isinstance(expected.get("accuracy"), (int, float))
            or abs(accuracy - expected["accuracy"]) > 1e-10):
        raise ErrorAnalysisError("F1.7 accuracy differs from frozen predictions")
    prec, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    if (not isinstance(expected.get("macro_f1"), (int, float))
            or abs(float(np.mean(f1)) - expected["macro_f1"]) > 1e-10):
        raise ErrorAnalysisError("F1.7 macro F1 differs from frozen predictions")

    per_class: dict[str, dict] = {}
    for i, label in enumerate(labels):
        fp = sum(matrix[j][i] for j in range(len(labels)) if j != i)
        fn = sum(matrix[i][j] for j in range(len(labels)) if j != i)
        per_class[label] = {
            "support": int(support[i]), "precision": float(prec[i]),
            "recall": float(recall[i]), "f1": float(f1[i]),
            "false_positives": int(fp), "false_negatives": int(fn),
        }
    pair_counts = Counter((r["true"], r["predicted"]) for r in error_cases)
    confusions = [
        {"true": truth, "predicted": predicted, "count": count}
        for (truth, predicted), count in sorted(
            pair_counts.items(), key=lambda item: (-item[1], item[0][0], item[0][1])
        )
    ]

    # Сводит длины ошибочных вопросов к агрегированной статистике без текста.
    def lengths_stats(values: list[int]) -> dict:
        return {"n": len(values),
                "median_words": float(median(values)) if values else None,
                "mean_words": float(np.mean(values)) if values else None}

    n = len(test_rows)
    summary = {
        "stage": "F1.8", "data_split": "test", "frozen_model": "F1.7",
        "n": n, "correct": n - len(error_cases),
        "errors": len(error_cases), "error_rate": len(error_cases) / n,
        "accuracy": accuracy, "macro_f1": float(np.mean(f1)),
        "labels": labels, "confusion_matrix": matrix,
        "per_class": per_class, "confusion_pairs": confusions,
        "question_lengths": {"correct": lengths_stats(right_lengths),
                             "incorrect": lengths_stats(wrong_lengths)},
        "limitations": [
            "Labels describe Stack Exchange source domains, not answer correctness.",
            "A confusion pair alone does not establish why the model erred.",
            "Question-length differences are associations, not causal explanations.",
            "Only the frozen test predictions are inspected; no model retraining.",
        ],
    }
    return summary, error_cases


# Строит матрицу ошибок на рисунке без текстов исходных вопросов.
def draw_confusion_matrix(report: dict, destination: Any) -> None:
    """Draw aggregate confusion counts only; no raw Q&A text or IDs."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    labels = report["labels"]
    matrix = report["confusion_matrix"]
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(im, ax=ax, label="Count")
    ax.set_xticks(range(len(labels)), labels=labels, rotation=35, ha="right")
    ax.set_yticks(range(len(labels)), labels=labels)
    ax.set_xlabel("Predicted domain")
    ax.set_ylabel("True domain")
    ax.set_title("Classical baseline — held-out test confusion matrix")
    for i, row in enumerate(matrix):
        for j, count in enumerate(row):
            ax.text(j, i, str(count), ha="center", va="center",
                    color="white" if count > max(map(max, matrix)) / 2 else "black")
    fig.tight_layout()
    try:
        fig.savefig(destination, dpi=150)
    finally:
        plt.close(fig)
