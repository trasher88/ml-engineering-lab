"""F2.7: локальный парный benchmark F1 и F2 на одном test-наборе.

F1 — ранее сохранённые предсказания с проверкой SHA-256 F1.6.
F2 — лучший checkpoint, выбранный по validation Macro F1 до test.
Публикуются только агрегированные метрики, без исходных Q&A и ID.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

import torch
import yaml

from scripts.pytorch_batches import synthetic_splits
from src.classical_ml.train import build_pipeline, classification_metrics
from src.common.io import ROOT
from src.pytorch_f2.data_pipeline import make_loader, prepare_data, read_f1_splits
from src.pytorch_f2.embedding_classifier import EmbeddingClassifier
from src.pytorch_f2.model_comparison import (
    ComparisonError, compare_on_test, predict_pytorch_test, save_aggregate_report,
    sha256_file, verify_classical_artifacts,
)
from src.pytorch_f2.validation_checkpoint import (
    evaluate_validation, fit_with_validation, restore_best_checkpoint,
)


def _local_f1_artifacts(root: Path, *, report_path: Path,
                        predictions_path: Path) -> tuple[dict, dict]:
    """Прочитать два локальных JSON без небезопасной загрузки joblib."""
    paths = (root / report_path, root / predictions_path)
    if any(not p.is_file() for p in paths):
        raise ComparisonError("missing F1.7 report/predictions; run F1.7 locally first")
    try:
        return tuple(json.loads(p.read_text(encoding="utf-8")) for p in paths)
    except (OSError, json.JSONDecodeError) as exc:
        raise ComparisonError("cannot read F1.7 artifacts") from exc


def _synthetic_baseline(splits: dict[str, list[dict]]) -> tuple[dict, dict]:
    """Обучить фиксированный F1 baseline только на синтетическом train для CI.

    В реальном эксперименте эта функция НЕ вызывается: используются уже
    сохранённые F1-артефакты, и оценка test выполняется без повторного fit.
    """
    model = build_pipeline(max_features=18000, c=2.0, max_iter=500, seed=42)
    model.fit([r["context"] for r in splits["train"]],
              [r["domain"] for r in splits["train"]])
    labels = sorted({row["domain"] for row in splits["train"]})
    report = {
        "stage": "F1.7", "fit_split": "train", "evaluation_splits": ["val", "test"],
        "split_source_sha256": {key: f"synthetic-{key}" for key in splits},
        "data_contract": {
            "split_counts": {key: len(rows) for key, rows in splits.items()},
            "train_classes": labels,
        },
    }
    predictions: dict[str, list[dict]] = {}
    for split in ("val", "test"):
        rows = splits[split]
        guessed = model.predict([row["context"] for row in rows])
        predictions[split] = [
            {"id": row["id"], "true": row["domain"], "predicted": str(pred)}
            for row, pred in zip(rows, guessed)
        ]
        report[split] = classification_metrics(
            [row["domain"] for row in rows], [str(pred) for pred in guessed], labels
        )
    return report, predictions


def run(*, synthetic: bool = False, root: Path = ROOT,
        data_dir: Path = Path("data/processed"),
        config_path: Path = Path("configs/local_cpu/lab.yaml"),
        checkpoint_path: Path = Path("checkpoints/f2_5_best.pt"),
        baseline_report: Path = Path("reports/classical.json"),
        baseline_predictions: Path = Path("reports/classical_predictions.json"),
        output_dir: Path = Path("outputs/f2_7_comparison"),
        expect_train: int | None = None, expect_val: int | None = None,
        expect_test: int | None = None) -> dict:
    """Сопоставить F1 и F2 при строгой фиксации данных и выборе по validation.

    Real: повторное обучение отсутствует; F1 report + предсказания и F2 best
    checkpoint должны существовать заранее. Один тестовый проход выполняется
    только после проверки hashes, валидности checkpoint и val-метрики.
    Synthetic: обучаем две мини-модели на вымышленных данных для CPU CI.
    """
    root = Path(root)
    if synthetic:
        splits = synthetic_splits()
        params = {"vocab_size": 12000, "max_length": 128, "batch_size": 32,
                  "embedding_dim": 48, "hidden_dim": 48, "learning_rate": 0.03}
        seed = 42
        source_hashes = {key: f"synthetic-{key}" for key in splits}
        classical_report, classical_predictions = _synthetic_baseline(splits)
    else:
        config = yaml.safe_load((root / config_path).read_text(encoding="utf-8"))
        if not isinstance(config, dict) or not isinstance(config.get("neural"), dict):
            raise ComparisonError("invalid F1/F2 neural configuration")
        params = config["neural"]
        seed = config["seed"]
        splits = read_f1_splits(root / data_dir)
        source_hashes = {split: sha256_file(root / data_dir / f"{split}.jsonl")
                         for split in ("train", "val", "test")}
        classical_report, classical_predictions = _local_f1_artifacts(
            root, report_path=baseline_report, predictions_path=baseline_predictions)
    prepared = prepare_data(
        splits, max_vocab_size=params["vocab_size"], max_length=params["max_length"]
    )
    for split, expected in (("train", expect_train), ("val", expect_val),
                            ("test", expect_test)):
        if expected is not None and (type(expected) is not int or expected < 1
                                     or prepared.counts[split] != expected):
            raise ComparisonError(f"unexpected {split} split size")
    aligned = verify_classical_artifacts(splits, classical_report,
                                         classical_predictions, source_hashes)

    val_loader = make_loader(prepared.datasets["val"], batch_size=params["batch_size"],
                             shuffle=False, seed=seed)
    test_loader = make_loader(prepared.datasets["test"], batch_size=params["batch_size"],
                              shuffle=False, seed=seed)

    if synthetic:
        with tempfile.TemporaryDirectory(prefix="f2_7_synthetic_") as tmp:
            path = Path(tmp) / "best.pt"
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed)
                model = EmbeddingClassifier(vocab_size=len(prepared.vocab),
                                            num_classes=len(prepared.label_to_id),
                                            embedding_dim=params["embedding_dim"],
                                            hidden_dim=params["hidden_dim"])
                fit_with_validation(
                    model,
                    make_loader(prepared.datasets["train"], batch_size=params["batch_size"],
                                shuffle=True, seed=seed),
                    val_loader, vocab=prepared.vocab, label_to_id=prepared.label_to_id,
                    max_length=params["max_length"], split_counts=prepared.counts,
                    seed=seed, checkpoint_path=path, epochs=2,
                    learning_rate=params["learning_rate"],
                )
            model, metadata = restore_best_checkpoint(
                path, vocab=prepared.vocab, label_to_id=prepared.label_to_id,
                max_length=params["max_length"])
    else:
        if not (root / checkpoint_path).is_file():
            raise ComparisonError("missing F2.5 best checkpoint; recreate on validation first")
        model, metadata = restore_best_checkpoint(
            root / checkpoint_path, vocab=prepared.vocab,
            label_to_id=prepared.label_to_id, max_length=params["max_length"])
    if (metadata.get("selection_metric") != "val_macro_f1"
            or metadata.get("tie_break") != "earliest_epoch"
            or metadata.get("split_counts") != prepared.counts
            or metadata.get("val_samples") != prepared.counts["val"]
            or not isinstance(metadata.get("best_epoch"), int)
            or metadata["best_epoch"] < 1):
        raise ComparisonError("checkpoint was not selected by validation only")
    checked_validation = evaluate_validation(model, val_loader)
    if (abs(checked_validation.macro_f1 - metadata["val_macro_f1"]) > 1e-10
            or abs(checked_validation.accuracy - metadata["val_accuracy"]) > 1e-10):
        raise ComparisonError("saved best checkpoint disagrees with validation")

    # Test используется впервые для PyTorch после всех проверок происхождения.
    predictions = predict_pytorch_test(model, test_loader,
                                       label_to_id=prepared.label_to_id)
    comparison = compare_on_test(splits["test"],
                                 classical_predictions=classical_predictions["test"],
                                 neural_predictions=predictions,
                                 labels=sorted(prepared.label_to_id))
    if classical_report["test"]["macro_f1"] != comparison["classical"]["macro_f1"]:
        raise ComparisonError("F1 report mismatch at final comparison")
    result = {
        "source": "synthetic" if synthetic else "local_f1_split",
        "split_counts": prepared.counts,
        "split_sha256": source_hashes,
        "protocol": {
            "classical_model": "F1.7: fixed TF-IDF + LogisticRegression",
            "classical_fit_split": "train",
            "classical_predictions_source": "verified_saved_F1_artifact",
            "neural_model": "F2.3: EmbeddingClassifier",
            "neural_selection_split": "val",
            "neural_selection_metric": "macro_f1",
            "neural_best_epoch": metadata["best_epoch"],
            "neural_validation_macro_f1": checked_validation.macro_f1,
            "test_used_for_selection": False,
        },
        "comparison": comparison,
        "complexity": {
            "neural_parameter_count": sum(p.numel() for p in model.parameters()),
            "classical_max_features": classical_report.get("config", {}).get("max_features", 18000 if synthetic else None),
            "measured_runtime_comparison": False,
            "runtime_note": "Comparative training/inference time was not measured",
        },
        "comparison_invariants_pass": True,
    }
    json_file, md_file = save_aggregate_report(result, root / output_dir)
    return {"source": result["source"], "split_counts": result["split_counts"],
            "neural_best_epoch": metadata["best_epoch"],
            "test_accuracy_classical": comparison["classical"]["accuracy"],
            "test_macro_f1_classical": comparison["classical"]["macro_f1"],
            "test_accuracy_neural": comparison["neural"]["accuracy"],
            "test_macro_f1_neural": comparison["neural"]["macro_f1"],
            "winner": comparison["winning_model_by_macro_f1"],
            "test_used_for_selection": False,
            "comparison_invariants_pass": True,
            "output_files": [str(json_file), str(md_file)]}


def main() -> int:
    """Выполнить один фиксированный benchmark и вернуть код ошибки при нарушениях."""
    parser = argparse.ArgumentParser(description="F2.7: paired classical vs neural test")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/f2_5_best.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/f2_7_comparison"))
    for split in ("train", "val", "test"):
        parser.add_argument("--expect-" + split, type=int)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    try:
        result = run(synthetic=args.synthetic, root=args.root,
                     checkpoint_path=args.checkpoint, output_dir=args.output_dir,
                     expect_train=args.expect_train, expect_val=args.expect_val,
                     expect_test=args.expect_test)
    except (ComparisonError, ValueError, TypeError, KeyError, OSError, RuntimeError,
            yaml.YAMLError) as exc:
        print(f"F2.7 Classical vs Neural CHECK: FAIL ({exc})")
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    if args.check:
        print("F2.7 Classical vs Neural CHECK: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
