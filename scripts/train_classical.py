# Назначение файла: F1.7: CLI обучения фиксированного classical baseline.
# Проверяет разбиение и сохраняет модель/предсказания только локально.

"""F1.7: train a fixed classical baseline from locally prepared F1.6 splits.

Run from repository root: python -m scripts.train_classical --expect-test 608
Training does not download data; outputs stay local and are not for GitHub.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

import joblib
import yaml

from src.classical_ml.train import ClassicalError, fit_and_evaluate
from src.common.io import ROOT, read_jsonl


# Ошибка локального эксперимента baseline либо артефактов.
class BaselineRunError(ClassicalError):
    """Invalid inputs, configuration, or expected output sizes."""


# Вычисляет контрольную сумму файла входных выборок.
def _digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fd:
        for chunk in iter(lambda: fd.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


# Безопасно записывает текстовый отчёт через временный файл.
def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n",
                                         dir=path.parent, prefix=".classical-",
                                         suffix=".tmp", delete=False) as fd:
            temp = Path(fd.name)
            fd.write(text)
        os.replace(temp, path)
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)


# Проверяет train/val/test, обучает baseline и сохраняет модель с метриками.
def run(root: Path = ROOT, *,
        config_path: Path = Path("configs/local_cpu/lab.yaml"),
        data_dir: Path = Path("data/processed"),
        model_path: Path = Path("models/classical.joblib"),
        report_path: Path = Path("reports/classical.json"),
        predictions_path: Path = Path("reports/classical_predictions.json"),
        expect_train: int | None = None,
        expect_val: int | None = None,
        expect_test: int | None = None) -> dict:
    root = Path(root)
    cfg_path = root / config_path
    if not cfg_path.is_file():
        raise BaselineRunError(f"configuration not found: {cfg_path}")
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict) or not isinstance(cfg.get("classical"), dict):
        raise BaselineRunError("invalid classical YAML config")
    if not isinstance(cfg.get("evaluation"), dict):
        raise BaselineRunError("evaluation.bootstrap_samples missing")
    classical = cfg["classical"]
    for key in ("max_features", "C", "max_iter"):
        if key not in classical:
            raise BaselineRunError(f"missing classical.{key}")
    seed = cfg.get("seed")
    bootstrap = cfg["evaluation"].get("bootstrap_samples")
    paths = {s: root / data_dir / f"{s}.jsonl" for s in ("train", "val", "test")}
    for s, path in paths.items():
        if not path.is_file():
            raise BaselineRunError(f"missing F1.6 split {s}: {path}")
    artifacts = [root / p for p in (model_path, report_path, predictions_path)]
    all_paths = [*paths.values(), cfg_path, *artifacts]
    if len({p.resolve() for p in all_paths}) != len(all_paths):
        raise BaselineRunError("inputs and output artifacts must have distinct paths")
    expected = {"train": expect_train, "val": expect_val, "test": expect_test}
    for n in expected.values():
        if n is not None and (type(n) is not int or n < 1):
            raise BaselineRunError("expected counts must be positive integers")
    try:
        datasets = {s: read_jsonl(paths[s]) for s in ("train", "val", "test")}
    except (json.JSONDecodeError, TypeError) as exc:
        raise BaselineRunError("invalid JSONL in F1.6 split") from exc
    for s, n in expected.items():
        if n is not None and len(datasets[s]) != n:
            raise BaselineRunError(f"expected {s}={n}, found {len(datasets[s])}")
    model, report, predictions = fit_and_evaluate(
        datasets, max_features=classical["max_features"], c=classical["C"],
        max_iter=classical["max_iter"], seed=seed, bootstrap_samples=bootstrap)
    report["split_source_sha256"] = {s: _digest(paths[s]) for s in ("train", "val", "test")}
    report["artifacts"] = {"model": str(model_path).replace("\\", "/"),
                           "predictions": str(predictions_path).replace("\\", "/")}
    # Model serialization is local-only. Loading an untrusted joblib is unsafe.
    target = artifacts[0]
    target.parent.mkdir(parents=True, exist_ok=True)
    temp: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=target.parent, prefix=".classical-",
                                         suffix=".joblib", delete=False) as fd:
            temp = Path(fd.name)
        joblib.dump(model, temp)
        os.replace(temp, target)
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)
    report["model_bytes"] = target.stat().st_size
    _atomic_text(artifacts[1], json.dumps(report, ensure_ascii=False,
                                          indent=2, allow_nan=False) + "\n")
    _atomic_text(artifacts[2], json.dumps(predictions, ensure_ascii=False,
                                          indent=2, allow_nan=False) + "\n")
    return report


# Читает аргументы и выводит результаты обучения/оценки.
def main() -> int:
    parser = argparse.ArgumentParser(description="F1.7 — offline classical ML baseline")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--config", type=Path, default=Path("configs/local_cpu/lab.yaml"))
    for name in ("train", "val", "test"):
        parser.add_argument("--expect-" + name, type=int)
    args = parser.parse_args()
    try:
        report = run(root=args.root, config_path=args.config,
                     expect_train=args.expect_train, expect_val=args.expect_val,
                     expect_test=args.expect_test)
    except (ValueError, OSError, TypeError, KeyError, yaml.YAMLError) as exc:
        print(f"Classical baseline: FAIL: {exc}")
        return 1
    print("Classical baseline: PASS; fit=train; "
          f"train={report['data_contract']['split_counts']['train']}; "
          f"val={report['val']['n']}; test={report['test']['n']}")
    print(f"Validation: accuracy={report['val']['accuracy']:.4f}; "
          f"macro_F1={report['val']['macro_f1']:.4f}")
    print(f"Test: accuracy={report['test']['accuracy']:.4f}; "
          f"macro_F1={report['test']['macro_f1']:.4f}; "
          f"weighted_F1={report['test']['weighted_f1']:.4f}")
    print("Model and aggregate report: local only; error analysis reserved for F1.8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
