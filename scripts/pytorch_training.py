"""F2.4: CPU smoke training на синтетике или только на train split F1.

CLI выводит исключительно числовые агрегаты, без текстов или ID вопросов.
Validation/test не используются для расчёта loss, подбора параметров или обучения.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from scripts.pytorch_batches import synthetic_splits
from src.common.io import ROOT
from src.pytorch_f2.data_pipeline import DataPipelineError, make_loader, prepare_data, read_f1_splits
from src.pytorch_f2.embedding_classifier import EmbeddingClassifier, ModelInputError
from src.pytorch_f2.training_loop import TrainingError, train_epochs


def run(*, synthetic: bool = False, root: Path = ROOT,
        config_path: Path = Path("configs/local_cpu/lab.yaml"),
        data_dir: Path = Path("data/processed"), epochs: int = 2,
        expect_train: int | None = None, expect_val: int | None = None,
        expect_test: int | None = None) -> dict:
    """Проверить один мини-процесс обучения на CPU и вернуть агрегаты.

    Args:
        synthetic: Изолированные вымышленные записи вместо локального F1.
        root: Корень проекта для поиска YAML/JSONL в локальном режиме.
        config_path/data_dir: Пути относительно корня проекта.
        epochs: Число учебных эпох; по умолчанию 2 для быстрого smoke.
        expect_train/val/test: Проверяемые размеры неизменного split F1.
    Returns:
        Число примеров, формы и losses по эпохам; без оценки held-out.
    """
    if synthetic:
        splits = synthetic_splits()
        vocab_size, max_length, batch_size = 12000, 128, 32
        embedding_dim, hidden_dim, seed, learning_rate = 48, 48, 42, 0.03
    else:
        path = Path(root) / config_path
        if not path.is_file():
            raise TrainingError(f"missing F1 configuration: {path}")
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(config, dict) or not isinstance(config.get("neural"), dict):
            raise TrainingError("missing neural configuration")
        neural = config["neural"]
        try:
            vocab_size, max_length = neural["vocab_size"], neural["max_length"]
            batch_size = neural["batch_size"]
            embedding_dim, hidden_dim = neural["embedding_dim"], neural["hidden_dim"]
            learning_rate, seed = neural["learning_rate"], config["seed"]
        except KeyError as exc:
            raise TrainingError("missing training parameter") from exc
        splits = read_f1_splits(Path(root) / data_dir)

    prepared = prepare_data(splits, max_vocab_size=vocab_size, max_length=max_length)
    for split_name, expected in (("train", expect_train), ("val", expect_val), ("test", expect_test)):
        if expected is not None and (type(expected) is not int or expected < 1
                                     or prepared.counts[split_name] != expected):
            raise TrainingError(f"unexpected {split_name} split size")
    if type(seed) is not int or seed < 0:
        raise TrainingError("seed must be a nonnegative integer")

    # fork_rng локализует инициализацию модели, не изменяя состояние RNG клиента.
    # Генератор порядка батчей живёт отдельно в DataLoader F2.2.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        net = EmbeddingClassifier(vocab_size=len(prepared.vocab),
                                  num_classes=len(prepared.label_to_id),
                                  embedding_dim=embedding_dim, hidden_dim=hidden_dim).cpu()
        loader = make_loader(prepared.datasets["train"], batch_size=batch_size,
                             shuffle=True, seed=seed)
        history = train_epochs(net, loader, epochs=epochs, learning_rate=learning_rate)

    invariant = bool(all(e.samples == prepared.counts["train"] and
                         e.nonzero_gradients and e.parameters_updated and
                         e.mean_loss >= 0 for e in history))
    if not invariant:
        raise TrainingError("training invariants failed")
    return {
        "source": "synthetic" if synthetic else "local_f1_train",
        "split_counts": prepared.counts,
        "trained_split": "train",
        "vocab_size": len(prepared.vocab),
        "class_count": len(prepared.label_to_id),
        "epochs": len(history),
        "batch_size": batch_size,
        "learning_rate": float(learning_rate),
        "parameter_count": sum(p.numel() for p in net.parameters()),
        "loss_by_epoch": [round(e.mean_loss, 8) for e in history],
        "epoch_samples": [e.samples for e in history],
        "epoch_batches": [e.batches for e in history],
        "nonzero_gradients": all(e.nonzero_gradients for e in history),
        "parameters_updated": all(e.parameters_updated for e in history),
        "train_invariants_pass": invariant,
        "validation_performed": False,
        "checkpoint_saved": False,
    }


def main() -> int:
    """Обработать параметры CLI и установить код завершения по smoke-проверке."""
    parser = argparse.ArgumentParser(description="F2.4: CPU-only training loop smoke")
    parser.add_argument("--synthetic", action="store_true", help="обучить на вымышленных вопросах")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--epochs", type=int, default=2)
    for split in ("train", "val", "test"):
        parser.add_argument("--expect-" + split, type=int)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    try:
        report = run(synthetic=args.synthetic, root=args.root, epochs=args.epochs,
                     expect_train=args.expect_train, expect_val=args.expect_val,
                     expect_test=args.expect_test)
    except (TrainingError, DataPipelineError, ModelInputError, OSError,
            yaml.YAMLError, TypeError) as exc:
        print(f"F2.4 Training Loop CHECK: FAIL ({exc})")
        return 1
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.check:
        print("F2.4 Training Loop CHECK: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
