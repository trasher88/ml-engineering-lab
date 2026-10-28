"""F2.6: запуск F2.5 и сохранение графиков и агрегированной истории.

Не сохраняет тексты вопросов, их идентификаторы или test-предсказания.
Результаты по умолчанию находятся в игнорируемой папке outputs/.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

import torch
import yaml

from scripts.pytorch_validation import run as validation_run
from src.common.io import ROOT
from src.pytorch_f2.data_pipeline import DataPipelineError
from src.pytorch_f2.embedding_classifier import ModelInputError
from src.pytorch_f2.training_loop import TrainingError
from src.pytorch_f2.validation_checkpoint import ValidationError
from src.pytorch_f2.training_diagnostics import DiagnosticsError, export_diagnostics


def _parameters(root: Path, *, synthetic: bool) -> dict:
    """Сохранить только фактически используемые числовые гиперпараметры F2.5."""
    if synthetic:
        return {"seed": 42, "batch_size": 32, "max_length": 128,
                "max_vocab_size": 12000, "embedding_dim": 48, "hidden_dim": 48,
                "learning_rate": 0.03, "weight_decay": 0.01}
    cfg = yaml.safe_load((Path(root) / "configs/local_cpu/lab.yaml").read_text(encoding="utf-8"))
    neural = cfg["neural"]
    return {"seed": cfg["seed"], "batch_size": neural["batch_size"],
            "max_length": neural["max_length"], "max_vocab_size": neural["vocab_size"],
            "embedding_dim": neural["embedding_dim"], "hidden_dim": neural["hidden_dim"],
            "learning_rate": neural["learning_rate"], "weight_decay": 0.01}


def run(*, synthetic: bool = False, root: Path = ROOT, epochs: int = 5,
        output_dir: Path | None = None, expect_train: int | None = None,
        expect_val: int | None = None, expect_test: int | None = None) -> dict:
    """Один запуск F2.5 → неизменная история → проверяемые файлы F2.6.

    При synthetic генерируются только фиктивные вопросы; при локальном запуске
    реальный split F1 считывается внутри F2.5, но в файлы идёт лишь агрегат.
    """
    root = Path(root)
    target = Path(output_dir) if output_dir is not None else root / "outputs" / "f2_6_diagnostics"
    # F2.6 не должен перезаписывать checkpoints/f2_5_best.pt из прошлого опыта.
    # F2.5 сам проверит round-trip весов, затем временный файл удалится.
    with tempfile.TemporaryDirectory(prefix="f2_6_checkpoint_") as temporary:
        report = validation_run(
            synthetic=synthetic, root=root, epochs=epochs,
            checkpoint_path=Path(temporary) / "best.pt",
            expect_train=expect_train, expect_val=expect_val, expect_test=expect_test,
        )
    result = export_diagnostics(report, target, parameters=_parameters(root, synthetic=synthetic))
    return {
        "source": result["source"], "split_counts": result["split_counts"],
        "epochs": result["epochs"], "best_epoch": result["best_epoch"],
        "best_val_macro_f1": result["best_val_macro_f1"],
        "diagnostic_signal": result["interpretation"]["signal"],
        "test_evaluated": result["test_evaluated"],
        "output_dir": str(target), "export_invariants_pass": True,
    }


def main() -> int:
    """Запустить CPU-эксперимент и завершиться ненулевым кодом при ошибках."""
    parser = argparse.ArgumentParser(description="F2.6: PyTorch training diagnostics")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--output-dir", type=Path)
    for split in ("train", "val", "test"):
        parser.add_argument("--expect-" + split, type=int)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    try:
        result = run(synthetic=args.synthetic, root=args.root, epochs=args.epochs,
                     output_dir=args.output_dir, expect_train=args.expect_train,
                     expect_val=args.expect_val, expect_test=args.expect_test)
    except (DiagnosticsError, ValidationError, TrainingError, DataPipelineError,
            ModelInputError, OSError, ValueError, TypeError, KeyError,
            yaml.YAMLError) as exc:
        print(f"F2.6 Training Diagnostics CHECK: FAIL ({exc})")
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.check:
        print("F2.6 Training Diagnostics CHECK: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
