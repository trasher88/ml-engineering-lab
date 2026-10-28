"""F2.7: сопоставление классического и нейросетевого классификаторов.

Сравнение выполняется на одинаковых ID и метках фиксированного test-набора.
Функции здесь не обучают модели, не подбирают параметры и не сохраняют тексты.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch.utils.data import DataLoader, SequentialSampler

from src.classical_ml.train import classification_metrics
from src.pytorch_f2.data_pipeline import QuestionDataset
from src.pytorch_f2.embedding_classifier import EmbeddingClassifier


class ComparisonError(ValueError):
    """Ошибка целостности выборок, предсказаний или протокола сравнения."""


def sha256_file(path: Path) -> str:
    """Вычислить SHA-256 локального файла, не выгружая его содержимое."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _aligned_predictions(rows: Sequence[dict[str, Any]],
                         predictions: Sequence[dict[str, Any]],
                         labels: Sequence[str], *, split: str) -> list[str]:
    """Проверить соответствие предсказаний реальным ID независимо от порядка JSON.

    Не допускаем дубликаты, отсутствующие/лишние ID, несовпадение эталонных
    меток и неизвестные классы. Результат располагается в порядке F1.6 JSONL.
    """
    if not isinstance(predictions, list) or len(predictions) != len(rows):
        raise ComparisonError(f"{split}: prediction count mismatch")
    if not isinstance(rows, (list, tuple)) or not rows:
        raise ComparisonError(f"{split}: missing original rows")
    expected = {row["id"]: row["domain"] for row in rows}
    if len(expected) != len(rows):
        raise ComparisonError(f"{split}: duplicate source IDs")
    by_id: dict[str, str] = {}
    for entry in predictions:
        if not isinstance(entry, dict) or set(entry) != {"id", "true", "predicted"}:
            raise ComparisonError(f"{split}: invalid prediction row")
        key, truth, guess = entry["id"], entry["true"], entry["predicted"]
        if not isinstance(key, str) or key in by_id or key not in expected:
            raise ComparisonError(f"{split}: duplicate or foreign prediction ID")
        if truth != expected[key] or guess not in labels:
            raise ComparisonError(f"{split}: label mismatch")
        by_id[key] = guess
    if set(by_id) != set(expected):
        raise ComparisonError(f"{split}: missing prediction IDs")
    return [by_id[row["id"]] for row in rows]


def verify_classical_artifacts(splits: Mapping[str, list[dict]],
                               report: Mapping[str, Any],
                               predictions: Mapping[str, Any],
                               source_hashes: Mapping[str, str]) -> dict[str, list[str]]:
    """Проверить происхождение F1.7 и восстановить упорядоченные предсказания.

    Исходный F1.7 фиксирует train-only fit и SHA каждого файла F1.6. Сверяем
    сохранённые метрики с предсказаниями, чтобы не сравнить чужой эксперимент.
    """
    names = ("train", "val", "test")
    if set(splits) != set(names) or set(source_hashes) != set(names):
        raise ComparisonError("expected exact train/val/test sources")
    if not isinstance(report, Mapping) or not isinstance(predictions, Mapping):
        raise ComparisonError("invalid F1 artifacts")
    if (report.get("stage") != "F1.7" or report.get("fit_split") != "train"
            or report.get("evaluation_splits") != ["val", "test"]):
        raise ComparisonError("F1 baseline must be fitted on train only")
    if report.get("split_source_sha256") != dict(source_hashes):
        raise ComparisonError("F1 split SHA-256 mismatch")
    contract = report.get("data_contract")
    classes = sorted({row["domain"] for row in splits["train"]})
    if (not isinstance(contract, dict)
            or contract.get("split_counts") != {name: len(splits[name]) for name in names}
            or contract.get("train_classes") != classes):
        raise ComparisonError("F1 split metadata mismatch")
    if set(predictions) != {"val", "test"}:
        raise ComparisonError("F1 predictions require val/test only")
    aligned: dict[str, list[str]] = {}
    for split in ("val", "test"):
        aligned[split] = _aligned_predictions(splits[split], predictions[split], classes, split=split)
        truth = [row["domain"] for row in splits[split]]
        actual = classification_metrics(truth, aligned[split], classes)
        recorded = report.get(split)
        if not isinstance(recorded, dict):
            raise ComparisonError(f"{split}: missing F1 metrics")
        for key in ("accuracy", "macro_f1", "weighted_f1"):
            if (not isinstance(recorded.get(key), (int, float))
                    or not math.isclose(actual[key], recorded[key], abs_tol=1e-9, rel_tol=0)):
                raise ComparisonError(f"{split}: F1 {key} is inconsistent")
        if (recorded.get("confusion_matrix") != actual["confusion_matrix"]
                or recorded.get("labels") != classes or recorded.get("n") != len(truth)):
            raise ComparisonError(f"{split}: F1 confusion matrix mismatch")
    return aligned


def predict_pytorch_test(model: EmbeddingClassifier, loader: DataLoader,
                         *, label_to_id: Mapping[str, int]) -> list[dict[str, str]]:
    """Получить предсказания только на test в фиксированном порядке строк.

    Используем inference_mode и последовательный sampler, исключаем пропуск
    последнего батча; не меняем режим и параметры переданной модели.
    """
    if not isinstance(model, EmbeddingClassifier):
        raise ComparisonError("expected EmbeddingClassifier")
    if (not isinstance(loader, DataLoader) or not isinstance(loader.dataset, QuestionDataset)
            or not isinstance(loader.sampler, SequentialSampler) or loader.drop_last):
        raise ComparisonError("test requires a complete sequential F2.2 loader")
    dataset = loader.dataset
    if any(row.get("split") != "test" for row in dataset.rows):
        raise ComparisonError("only test split may be evaluated here")
    if (dict(label_to_id) != dataset.label_to_id
            or set(label_to_id.values()) != set(range(len(label_to_id)))
            or model.num_classes != len(label_to_id)):
        raise ComparisonError("model class mapping mismatch")
    if any(parameter.device.type != "cpu" for parameter in model.parameters()):
        raise ComparisonError("F2.7 requires CPU inference")
    id_to_label = {index: name for name, index in label_to_id.items()}
    position = 0
    was_training = model.training
    predictions: list[dict[str, str]] = []
    model.eval()
    try:
        with torch.inference_mode():
            for batch in loader:
                ids, mask, labels = batch["input_ids"], batch["attention_mask"], batch["labels"]
                sample_ids = batch["sample_ids"]
                if (ids.dtype != torch.long or mask.dtype != torch.bool
                        or labels.dtype != torch.long or ids.shape != mask.shape
                        or ids.ndim != 2 or len(sample_ids) != len(labels)
                        or len(labels) != ids.shape[0]):
                    raise ComparisonError("malformed F2.2 test batch")
                scores = model(ids, mask)
                if not bool(torch.isfinite(scores).all()):
                    raise ComparisonError("nonfinite logits")
                for name, truth, predicted in zip(sample_ids, labels.tolist(),
                                                  scores.argmax(dim=1).tolist()):
                    if position >= len(dataset) or name != dataset.rows[position]["id"]:
                        raise ComparisonError("test loader reordered or duplicated examples")
                    expected_truth = dataset.rows[position]["domain"]
                    if id_to_label[truth] != expected_truth:
                        raise ComparisonError("test label mismatch")
                    predictions.append({"id": name, "true": expected_truth,
                                        "predicted": id_to_label[predicted]})
                    position += 1
    finally:
        model.train(was_training)
    if position != len(dataset):
        raise ComparisonError("not all test samples were evaluated")
    return predictions


def compare_on_test(test_rows: Sequence[dict[str, Any]], *,
                    classical_predictions: Sequence[dict[str, Any]],
                    neural_predictions: Sequence[dict[str, Any]],
                    labels: Sequence[str]) -> dict[str, Any]:
    """Вычислить парные метрики и ошибки по одним и тем же объектам.

    Идентификаторы используются только внутри сопоставления; наружу выходят
    исключительно агрегаты и confusion matrices, без текстов и ID.
    """
    names = list(labels)
    if len(names) < 2 or names != sorted(set(names)):
        raise ComparisonError("classes must be unique and sorted")
    classical = _aligned_predictions(test_rows, classical_predictions, names, split="test")
    neural = _aligned_predictions(test_rows, neural_predictions, names, split="test")
    true = [row["domain"] for row in test_rows]
    if any(label not in names for label in true):
        raise ComparisonError("unknown true class")
    first = classification_metrics(true, classical, names)
    second = classification_metrics(true, neural, names)
    both_wrong = sum(a != y and b != y for y, a, b in zip(true, classical, neural))
    only_classical = sum(a != y and b == y for y, a, b in zip(true, classical, neural))
    only_neural = sum(a == y and b != y for y, a, b in zip(true, classical, neural))
    both_right = sum(a == y and b == y for y, a, b in zip(true, classical, neural))
    disagreement = sum(a != b for a, b in zip(classical, neural))
    if both_wrong + only_classical + only_neural + both_right != len(true):
        raise ComparisonError("paired comparison count mismatch")
    return {
        "samples": len(true), "labels": names,
        "classical": first, "neural": second,
        "delta_neural_minus_classical": {
            "accuracy": second["accuracy"] - first["accuracy"],
            "macro_f1": second["macro_f1"] - first["macro_f1"],
        },
        "paired_errors": {
            "both_correct": both_right, "both_wrong": both_wrong,
            "only_classical_wrong": only_classical, "only_neural_wrong": only_neural,
            "prediction_disagreements": disagreement,
        },
        "winning_model_by_macro_f1": (
            "classical" if first["macro_f1"] > second["macro_f1"] else
            "neural" if first["macro_f1"] < second["macro_f1"] else "tie"
        ),
    }


def save_aggregate_report(result: Mapping[str, Any], output_dir: Path) -> tuple[Path, Path]:
    """Сохранить только агрегированный JSON и краткую таблицу Markdown.

    Не выгружаем ID, тексты, токены, веса и индивидуальные предсказания.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path, md_path = out / "comparison.json", out / "comparison.md"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                         encoding="utf-8")
    comparison = result["comparison"]
    f1, f2 = comparison["classical"], comparison["neural"]
    text = [
        "# F2.7 — TF-IDF + Logistic Regression против EmbeddingClassifier",
        "", "## Одна test-выборка",
        f"Объектов: {comparison['samples']}. Классы: {', '.join(comparison['labels'])}.",
        "", "| Модель | Accuracy | Macro F1 | Ошибок |",
        "|---|---:|---:|---:|",
        f"| Classical ML | {f1['accuracy']:.4f} | {f1['macro_f1']:.4f} | {comparison['samples'] - sum(f1['confusion_matrix'][i][i] for i in range(len(comparison['labels'])))} |",
        f"| EmbeddingClassifier | {f2['accuracy']:.4f} | {f2['macro_f1']:.4f} | {comparison['samples'] - sum(f2['confusion_matrix'][i][i] for i in range(len(comparison['labels'])))} |",
        "", "## Парные ошибки",
        *(f"- {name}: {value}" for name, value in comparison["paired_errors"].items()),
        "", "## Интерпретация",
        f"По test Macro F1: {comparison['winning_model_by_macro_f1']}.",
        "Это сравнение двух зафиксированных моделей на одной выборке; оно не доказывает",
        "универсального превосходства и не означает значимости наблюдаемого различия.",
        "Тексты, ID и индивидуальные предсказания не экспортируются.",
        "Не следует подбирать гиперпараметры на основании test-метрик.",
        "", "## Вычислительные ограничения",
        "Время обучения и инференса не сравнивается без сопоставимых замеров.",
        "Размер файла модели не эквивалентен занимаемой оперативной памяти.", "",
    ]
    md_path.write_text("\n".join(text), encoding="utf-8")
    return json_path, md_path
