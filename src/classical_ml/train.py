"""F1.7 baseline: fit TF-IDF and classifier on train exclusively.

The validation/test sets are never passed to .fit().  Pure functions here
are useful for offline synthetic-data tests, without proprietary Q&A data.
"""
from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.pipeline import Pipeline


class ClassicalError(ValueError):
    """A split contract, configuration, or model-training error."""


def validate_splits(datasets: dict[str, list[dict]]) -> dict[str, Any]:
    """Validate disjoint split IDs/groups and labels without returning texts."""
    if not isinstance(datasets, dict) or set(datasets) != {"train", "val", "test"}:
        raise ClassicalError("required splits: train, val, test")
    all_ids: set[str] = set()
    group_sets: dict[str, set[Any]] = {}
    counts: dict[str, int] = {}
    train_labels: set[str] = set()
    for name in ("train", "val", "test"):
        rows = datasets[name]
        if not isinstance(rows, list) or not rows:
            raise ClassicalError(f"split {name} must be a nonempty list")
        groups: set[Any] = set()
        local_ids: set[str] = set()
        for i, row in enumerate(rows):
            if not isinstance(row, dict) or any(
                not isinstance(row.get(k), str) or not row[k].strip()
                for k in ("id", "context", "domain")
            ):
                raise ClassicalError(f"invalid text/id/label in {name}[{i}]")
            if row.get("split") != name:
                raise ClassicalError(f"split marker mismatch in {name}[{i}]")
            group_id = row.get("group_id")
            if type(group_id) not in (str, int) or str(group_id) == "":
                raise ClassicalError(f"missing group_id in {name}[{i}]")
            if row["id"] in local_ids or row["id"] in all_ids:
                raise ClassicalError("duplicate record IDs across splits")
            local_ids.add(row["id"])
            groups.add(group_id)
            if name == "train":
                train_labels.add(row["domain"])
        all_ids.update(local_ids)
        group_sets[name] = groups
        counts[name] = len(rows)
    if len(train_labels) < 2:
        raise ClassicalError("train must have at least two classes")
    for name in ("val", "test"):
        unknown = {row["domain"] for row in datasets[name]} - train_labels
        if unknown:
            raise ClassicalError(f"unseen classes in {name}")
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        if group_sets[a] & group_sets[b]:
            raise ClassicalError(f"group leakage between {a} and {b}")
    return {
        "split_counts": counts,
        "train_classes": sorted(train_labels),
        "train_class_counts": dict(sorted(Counter(r["domain"] for r in datasets["train"]).items())),
        "group_overlap": {"train/val": 0, "train/test": 0, "val/test": 0},
    }


def build_pipeline(*, max_features: int, c: float, max_iter: int, seed: int) -> Pipeline:
    if type(max_features) is not int or max_features < 1:
        raise ClassicalError("max_features must be positive integer")
    if type(max_iter) is not int or max_iter < 1:
        raise ClassicalError("max_iter must be positive integer")
    if type(seed) is not int or seed < 0:
        raise ClassicalError("seed must be nonnegative integer")
    if isinstance(c, bool) or not isinstance(c, (int, float)) or not np.isfinite(c) or c <= 0:
        raise ClassicalError("C must be finite and positive")
    return Pipeline([
        ("tfidf", TfidfVectorizer(max_features=max_features, ngram_range=(1, 2), sublinear_tf=True)),
        ("classifier", LogisticRegression(C=float(c), max_iter=max_iter, random_state=seed)),
    ])


def classification_metrics(y_true: list[str], y_pred: list[str], labels: list[str]) -> dict:
    if not y_true or len(y_true) != len(y_pred):
        raise ClassicalError("aligned, nonempty predictions required")
    result: dict[str, Any] = {"accuracy": float(accuracy_score(y_true, y_pred))}
    for avg in ("macro", "weighted"):
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=labels, average=avg, zero_division=0
        )
        result.update({f"{avg}_precision": float(precision),
                       f"{avg}_recall": float(recall),
                       f"{avg}_f1": float(f1)})
    result["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=labels).tolist()
    result["labels"] = list(labels)
    result["n"] = len(y_true)
    return result


def bootstrap_macro_f1(y_true: list[str], y_pred: list[str], labels: list[str], *,
                       seed: int = 42, samples: int = 500) -> list[float]:
    """Example-level bootstrap; does NOT account for group/training variance."""
    if type(samples) is not int or samples < 1:
        raise ClassicalError("bootstrap_samples must be a positive integer")
    if not y_true or len(y_true) != len(y_pred):
        raise ClassicalError("aligned, nonempty predictions required")
    rng = np.random.default_rng(seed)
    y = np.asarray(y_true)
    p = np.asarray(y_pred)
    values = []
    for _ in range(samples):
        idx = rng.integers(0, len(y), len(y))
        values.append(float(precision_recall_fscore_support(
            y[idx], p[idx], labels=labels, average="macro", zero_division=0
        )[2]))
    return np.quantile(values, [0.025, 0.975]).tolist()


def fit_and_evaluate(datasets: dict[str, list[dict]], *, max_features: int = 18000,
                     c: float = 2.0, max_iter: int = 500, seed: int = 42,
                     bootstrap_samples: int = 500) -> tuple[Pipeline, dict, dict]:
    """Fit solely on train and evaluate fixed model on val/test once.

    Return (fitted model, aggregate report, text-free ID/prediction rows).
    The caller persists artifacts locally. No hyperparameter search is done.
    """
    audit = validate_splits(datasets)
    model = build_pipeline(max_features=max_features, c=c, max_iter=max_iter, seed=seed)
    train = datasets["train"]
    model.fit([r["context"] for r in train], [r["domain"] for r in train])
    labels = [str(v) for v in model.named_steps["classifier"].classes_]
    report: dict[str, Any] = {
        "stage": "F1.7", "fit_split": "train", "evaluation_splits": ["val", "test"],
        "algorithm": "TF-IDF (word 1-2 grams) + LogisticRegression",
        "config": {"max_features": max_features, "ngram_range": [1, 2],
                   "sublinear_tf": True, "C": float(c), "max_iter": max_iter,
                   "seed": seed, "bootstrap_samples": bootstrap_samples},
        "data_contract": audit,
        "bootstrap_limitations": "row-wise bootstrap only; not group/training uncertainty",
        "task_limitation": "labels are source sites, not technical correctness",
    }
    predictions: dict[str, list[dict]] = {}
    for split in ("val", "test"):
        rows = datasets[split]
        truth = [r["domain"] for r in rows]
        pred = [str(x) for x in model.predict([r["context"] for r in rows])]
        result = classification_metrics(truth, pred, labels)
        result["macro_f1_ci95"] = bootstrap_macro_f1(
            truth, pred, labels, seed=seed, samples=bootstrap_samples
        )
        report[split] = result
        predictions[split] = [
            {"id": row["id"], "true": y, "predicted": p}
            for row, y, p in zip(rows, truth, pred)
        ]
    return model, report, predictions
