"""F2.3: CPU-демонстрация прямого прохода Embedding Classifier без обучения.

По умолчанию использует локальный F1 split и параметры проекта. Для CI
предусмотрены только синтетические вопросы. Вывод содержит агрегаты, не тексты.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from scripts.pytorch_batches import synthetic_splits
from src.common.io import ROOT
from src.pytorch_f2.data_pipeline import (
    DataPipelineError, make_loader, prepare_data, read_f1_splits,
)
from src.pytorch_f2.embedding_classifier import EmbeddingClassifier, ModelInputError


def run(*, synthetic: bool = False, root: Path = ROOT,
        config_path: Path = Path("configs/local_cpu/lab.yaml"),
        data_dir: Path = Path("data/processed"),
        expect_train: int | None = None, expect_val: int | None = None,
        expect_test: int | None = None) -> dict:
    """Создать модель под словарь F2.2, выполнить один forward и вернуть агрегаты.

    Args:
        synthetic: Использовать исключительно вымышленные вопросы в памяти.
        root: Корень проекта для конфигурации и локального разделения F1.
        config_path: Относительный путь к настройкам модели.
        data_dir: Относительный путь к уже зафиксированным F1 JSONL.
        expect_train/val/test: Необязательное подтверждение размеров F1 split.
    Returns:
        Словарь размерностей, числа параметров и булевых инвариантов.

    Ничего не обучает, не вычисляет точность и не сохраняет веса/исходные данные.
    """
    if synthetic:
        splits = synthetic_splits()
        vocab_limit, max_length, batch_size = 12000, 128, 32
        embedding_dim, hidden_dim, seed = 48, 48, 42
    else:
        path = Path(root) / config_path
        if not path.is_file():
            raise ModelInputError(f"missing F1 configuration: {path}")
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(config, dict) or not isinstance(config.get("neural"), dict):
            raise ModelInputError("missing neural configuration")
        neural = config["neural"]
        try:
            vocab_limit = neural["vocab_size"]
            max_length = neural["max_length"]
            batch_size = neural["batch_size"]
            embedding_dim = neural["embedding_dim"]
            hidden_dim = neural["hidden_dim"]
            seed = config["seed"]
        except KeyError as exc:
            raise ModelInputError("missing classifier parameter") from exc
        splits = read_f1_splits(Path(root) / data_dir)

    prepared = prepare_data(splits, max_vocab_size=vocab_limit, max_length=max_length)
    for name, expected in (("train", expect_train), ("val", expect_val), ("test", expect_test)):
        if expected is not None and (type(expected) is not int or expected < 1 or
                                     prepared.counts[name] != expected):
            raise ModelInputError(f"unexpected {name} split size")

    batch = next(iter(make_loader(
        prepared.datasets["train"], batch_size=batch_size, shuffle=False, seed=seed
    )))
    input_ids, mask = batch["input_ids"], batch["attention_mask"]

    # Изолируем генератор: повторение demo детерминировано и не изменяет RNG клиента.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = EmbeddingClassifier(
            vocab_size=len(prepared.vocab), num_classes=len(prepared.label_to_id),
            embedding_dim=embedding_dim, hidden_dim=hidden_dim,
        ).cpu().eval()

    with torch.no_grad():
        logits = model(input_ids, mask)
        # Дополнительный PAD не должен менять усреднение реальных токенов.
        extra_ids = torch.nn.functional.pad(input_ids, (0, 3), value=0)
        extra_mask = torch.nn.functional.pad(mask, (0, 3), value=False)
        padded_logits = model(extra_ids, extra_mask)
        repeat_logits = model(input_ids, mask)

    expected_shape = (input_ids.shape[0], len(prepared.label_to_id))
    shape_ok = logits.shape == expected_shape
    finite_ok = bool(torch.isfinite(logits).all())
    padding_ok = bool(torch.allclose(logits, padded_logits, atol=1e-6, rtol=1e-6))
    repeat_ok = torch.equal(logits, repeat_logits)
    if not (shape_ok and finite_ok and padding_ok and repeat_ok):
        raise ModelInputError("forward shape/finite/padding/repeatability invariant failed")

    return {
        "source": "synthetic" if synthetic else "local_f1_split",
        "split_counts": prepared.counts,
        "vocab_size": len(prepared.vocab),
        "class_count": len(prepared.label_to_id),
        "batch_shape": list(input_ids.shape),
        "embedding_dim": embedding_dim,
        "hidden_dim": hidden_dim,
        "logits_shape": list(logits.shape),
        "logits_dtype": str(logits.dtype),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "logits_finite": finite_ok,
        "padding_invariant": padding_ok,
        "repeatable_forward": repeat_ok,
        "forward_invariants_pass": bool(shape_ok and finite_ok and padding_ok and repeat_ok),
        "trained": False,
    }


def main() -> int:
    """Запустить CLI; при некорректном батче завершить процесс с кодом 1."""
    parser = argparse.ArgumentParser(description="F2.3: Embedding Classifier (forward only)")
    parser.add_argument("--synthetic", action="store_true", help="без чтения локальных F1 данных")
    parser.add_argument("--root", type=Path, default=ROOT)
    for name in ("train", "val", "test"):
        parser.add_argument("--expect-" + name, type=int)
    parser.add_argument("--check", action="store_true", help="проверить инварианты forward")
    args = parser.parse_args()
    try:
        result = run(synthetic=args.synthetic, root=args.root,
                     expect_train=args.expect_train, expect_val=args.expect_val,
                     expect_test=args.expect_test)
    except (DataPipelineError, ModelInputError, OSError, yaml.YAMLError) as exc:
        print(f"F2.3 Embedding Classifier CHECK: FAIL ({exc})")
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.check:
        print("F2.3 Embedding Classifier CHECK: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
