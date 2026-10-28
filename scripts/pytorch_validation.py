"""F2.5: CPU validation, выбор лучшей эпохи и проверка восстановления весов.

Печатает только агрегированные метрики; не выводит исходные вопросы или ID.
Реальные checkpoint-файлы остаются локальными и исключены из Git.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

import torch
import yaml

from scripts.pytorch_batches import synthetic_splits
from src.common.io import ROOT
from src.pytorch_f2.data_pipeline import DataPipelineError, make_loader, prepare_data, read_f1_splits
from src.pytorch_f2.embedding_classifier import EmbeddingClassifier, ModelInputError
from src.pytorch_f2.training_loop import TrainingError
from src.pytorch_f2.validation_checkpoint import (
    ValidationError, evaluate_validation, fit_with_validation, restore_best_checkpoint,
)


def run(*, synthetic: bool = False, root: Path = ROOT,
        config_path: Path = Path("configs/local_cpu/lab.yaml"),
        data_dir: Path = Path("data/processed"), epochs: int = 3,
        checkpoint_path: Path | None = None,
        expect_train: int | None = None, expect_val: int | None = None,
        expect_test: int | None = None) -> dict:
    """Обучить на train, выбрать лучшую val-эпоху, восстановить предсказания.

    Для synthetic checkpoint создаётся в временном каталоге и удаляется.
    Для локального F1 по умолчанию он помещается в игнорируемые checkpoints/.
    Test применяется только для проверки числа записей при prepare_data().
    """
    if synthetic:
        splits = synthetic_splits()
        vocab_size, max_length, batch_size = 12000, 128, 32
        embedding_dim, hidden_dim, seed, learning_rate = 48, 48, 42, 0.03
    else:
        config_file = Path(root) / config_path
        if not config_file.is_file():
            raise ValidationError(f"missing F1 configuration: {config_file}")
        config = yaml.safe_load(config_file.read_text(encoding="utf-8"))
        if not isinstance(config, dict) or not isinstance(config.get("neural"), dict):
            raise ValidationError("missing neural configuration")
        neural = config["neural"]
        try:
            vocab_size, max_length, batch_size = neural["vocab_size"], neural["max_length"], neural["batch_size"]
            embedding_dim, hidden_dim = neural["embedding_dim"], neural["hidden_dim"]
            seed, learning_rate = config["seed"], neural["learning_rate"]
        except KeyError as exc:
            raise ValidationError("missing neural parameter") from exc
        splits = read_f1_splits(Path(root) / data_dir)

    data = prepare_data(splits, max_vocab_size=vocab_size, max_length=max_length)
    for split, expected in (("train", expect_train), ("val", expect_val), ("test", expect_test)):
        if expected is not None and (type(expected) is not int or expected < 1 or expected != data.counts[split]):
            raise ValidationError(f"unexpected {split} split size")
    if type(seed) is not int or seed < 0:
        raise ValidationError("seed must be a nonnegative integer")

    # Отдельный временный каталог предотвращает сохранение учебных весов в проект.
    with tempfile.TemporaryDirectory(prefix="f2_5_smoke_") as temp_dir:
        if checkpoint_path is not None:
            destination = Path(checkpoint_path)
        elif synthetic:
            destination = Path(temp_dir) / "best.pt"
        else:
            destination = Path(root) / "checkpoints" / "f2_5_best.pt"
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            model = EmbeddingClassifier(
                vocab_size=len(data.vocab), num_classes=len(data.label_to_id),
                embedding_dim=embedding_dim, hidden_dim=hidden_dim,
            ).cpu()
            train_loader = make_loader(data.datasets["train"], batch_size=batch_size,
                                       shuffle=True, seed=seed)
            val_loader = make_loader(data.datasets["val"], batch_size=batch_size,
                                     shuffle=False, seed=seed)
            result = fit_with_validation(
                model, train_loader, val_loader,
                vocab=data.vocab, label_to_id=data.label_to_id,
                max_length=max_length, split_counts=data.counts, seed=seed,
                checkpoint_path=destination, epochs=epochs, learning_rate=learning_rate,
            )
            restored, metadata = restore_best_checkpoint(
                destination, vocab=data.vocab, label_to_id=data.label_to_id,
                max_length=max_length,
            )
            # Сверяем восстановленную validation-метрику и предсказания на одном
            # и том же val-батче. Не сравниваем с последней эпохой, если лучшая ранняя.
            restored_metrics = evaluate_validation(restored, val_loader)
            batch = next(iter(val_loader))
            # Контрольная модель с весами checkpoint восстанавливается повторно:
            # сравнение до/после сериализации выполнено внутри тестов _save_checkpoint.
            repeat, _ = restore_best_checkpoint(destination, vocab=data.vocab,
                                                label_to_id=data.label_to_id,
                                                max_length=max_length)
            with torch.inference_mode():
                logits1 = restored(batch["input_ids"], batch["attention_mask"])
                logits2 = repeat(batch["input_ids"], batch["attention_mask"])
            identical = torch.equal(logits1, logits2)
            metrics_match = (
                abs(restored_metrics.macro_f1 - metadata["val_macro_f1"]) < 1e-12
                and abs(restored_metrics.accuracy - metadata["val_accuracy"]) < 1e-12
                and abs(restored_metrics.mean_loss - metadata["val_mean_loss"]) < 1e-8
            )
            if not identical or not metrics_match or result.best_epoch != metadata["best_epoch"]:
                raise ValidationError("checkpoint verification failed")

        return {
            "source": "synthetic" if synthetic else "local_f1_split",
            "split_counts": data.counts,
            "trained_split": "train",
            "selection_split": "val",
            "test_evaluated": False,
            "selection_metric": "val_macro_f1",
            "epochs": epochs,
            "history": [
                {"epoch": row.train.epoch,
                 "train_loss": round(row.train.mean_loss, 8),
                 "val_loss": round(row.validation.mean_loss, 8),
                 "val_accuracy": round(row.validation.accuracy, 8),
                 "val_macro_f1": round(row.validation.macro_f1, 8)}
                for row in result.history
            ],
            "best_epoch": result.best_epoch,
            "best_val_macro_f1": round(result.best_macro_f1, 8),
            "checkpoint_saved": destination.is_file(),
            "checkpoint_location": "temporary" if synthetic and checkpoint_path is None else str(destination),
            "checkpoint_reload_match": identical and metrics_match,
            "validation_invariants_pass": bool(
                len(result.history) == epochs and restored_metrics.samples == data.counts["val"]
                and all(row.train.samples == data.counts["train"] for row in result.history)
            ),
        }


def main() -> int:
    """Выполнить демонстрацию CLI с явным кодом ошибки при нарушениях."""
    parser = argparse.ArgumentParser(description="F2.5: validation + best checkpoint (CPU)")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--checkpoint", type=Path)
    for split in ("train", "val", "test"):
        parser.add_argument("--expect-" + split, type=int)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    try:
        report = run(synthetic=args.synthetic, root=args.root,
                     epochs=args.epochs, checkpoint_path=args.checkpoint,
                     expect_train=args.expect_train, expect_val=args.expect_val,
                     expect_test=args.expect_test)
    except (ValidationError, TrainingError, DataPipelineError, ModelInputError,
            OSError, yaml.YAMLError, TypeError) as exc:
        print(f"F2.5 Validation/Checkpoint CHECK: FAIL ({exc})")
        return 1
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.check:
        if not report["validation_invariants_pass"] or not report["checkpoint_reload_match"]:
            print("F2.5 Validation/Checkpoint CHECK: FAIL (invariants)")
            return 1
        print("F2.5 Validation/Checkpoint CHECK: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
