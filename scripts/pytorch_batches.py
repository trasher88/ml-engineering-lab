"""F2.2: воспроизводимая CLI-демонстрация Dataset/DataLoader без обучения.

По умолчанию читает локальные JSONL F1.6. Флаг --synthetic создаёт данные
только в памяти и не раскрывает реальные вопросы в выводе или GitHub Actions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml
import torch

from src.common.io import ROOT
from src.pytorch_f2.data_pipeline import (
    DataPipelineError, PAD_ID, UNK_ID, make_loader, prepare_data, read_f1_splits,
)


def synthetic_splits() -> dict[str, list[dict]]:
    """Создать несколько непересекающихся синтетических групп для smoke test."""
    def row(index: int, split: str, domain: str, question: str) -> dict:
        """Построить запись с контрактом F1.6 на вымышленных данных."""
        return {"id": f"synthetic-{index}", "context": question, "domain": domain,
                "group_id": index, "split": split}

    return {
        "train": [
            row(0, "train", "linux", "linux service command"),
            row(1, "train", "systems", "system memory status"),
            row(2, "train", "linux", "linux kernel status"),
            row(3, "train", "systems", "system service memory"),
        ],
        "val": [row(4, "val", "linux", "unseenlinux service")],
        "test": [row(5, "test", "systems", "unseensystem memory")],
    }


def run(*, synthetic: bool = False, root: Path = ROOT,
        config_path: Path = Path("configs/local_cpu/lab.yaml"),
        data_dir: Path = Path("data/processed"),
        expect_train: int | None = None, expect_val: int | None = None,
        expect_test: int | None = None) -> dict:
    """Подготовить три Dataset, получить батч и вернуть только агрегаты.

    В CLI не печатаются токены, тексты, ID реальных вопросов или содержимое
    файлов: отчёт безопасен для журнала команд и демонстрации в CI.
    """
    if synthetic:
        vocab_size, max_length, batch_size, seed = 12000, 128, 32, 42
        splits = synthetic_splits()
    else:
        config_file = Path(root) / config_path
        if not config_file.is_file():
            raise DataPipelineError(f"missing F1 config: {config_file}")
        config = yaml.safe_load(config_file.read_text(encoding="utf-8"))
        if not isinstance(config, dict) or not isinstance(config.get("neural"), dict):
            raise DataPipelineError("missing neural section in config")
        params = config["neural"]
        try:
            vocab_size = params["vocab_size"]
            max_length = params["max_length"]
            batch_size = params["batch_size"]
            seed = config["seed"]
        except KeyError as exc:
            raise DataPipelineError("missing neural parameter") from exc
        splits = read_f1_splits(Path(root) / data_dir)

    prepared = prepare_data(splits, max_vocab_size=vocab_size, max_length=max_length)
    expected = {"train": expect_train, "val": expect_val, "test": expect_test}
    for name, count in expected.items():
        if count is not None and (type(count) is not int or count < 1 or
                                  prepared.counts[name] != count):
            raise DataPipelineError(f"unexpected {name} size")

    batch = next(iter(make_loader(
        prepared.datasets["train"], batch_size=batch_size, shuffle=False, seed=seed
    )))
    ids, mask, labels = batch["input_ids"], batch["attention_mask"], batch["labels"]
    # В каждой настоящей позиции маска истинна; PAD всегда закрывается маской.
    valid = (ids.shape == mask.shape and ids.shape[0] == labels.shape[0]
             and ids.dtype == labels.dtype == torch.long
             and mask.dtype == torch.bool and ids.device.type == "cpu"
             and ((ids == PAD_ID) == (~mask)).all().item()
             and bool(mask[:, 0].all())
             and len(batch["sample_ids"]) == ids.shape[0])
    # Validation/test намеренно содержат слова, отсутствующие в train.
    unknown_in_heldout = sum(
        UNK_ID in prepared.vocab.encode(row["context"], max_length=max_length)
        for name in ("val", "test") for row in splits[name]
    )
    if not valid:
        raise DataPipelineError("batch shape/mask/dtype invariant failed")
    return {
        "source": "synthetic" if synthetic else "local_f1_split",
        "split_counts": prepared.counts,
        "vocab_size": len(prepared.vocab),
        "class_count": len(prepared.label_to_id),
        "batch_shape": list(ids.shape),
        "input_dtype": str(ids.dtype),
        "mask_dtype": str(mask.dtype),
        "label_dtype": str(labels.dtype),
        "heldout_questions_with_unk": unknown_in_heldout,
        "batch_invariants_pass": bool(valid),
    }


def main() -> int:
    """Запустить smoke-демонстрацию, завершить процесс с ошибкой при нарушении."""
    parser = argparse.ArgumentParser(description="F2.2: Dataset, DataLoader, Tokenization")
    parser.add_argument("--synthetic", action="store_true", help="без локальных данных F1")
    parser.add_argument("--root", type=Path, default=ROOT)
    for name in ("train", "val", "test"):
        parser.add_argument("--expect-" + name, type=int)
    parser.add_argument("--check", action="store_true", help="проверить batch-инварианты")
    args = parser.parse_args()
    try:
        result = run(synthetic=args.synthetic, root=args.root,
                     expect_train=args.expect_train, expect_val=args.expect_val,
                     expect_test=args.expect_test)
    except (DataPipelineError, OSError, yaml.YAMLError) as exc:
        print(f"F2.2 Dataset/DataLoader CHECK: FAIL ({exc})")
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.check:
        print("F2.2 Dataset/DataLoader CHECK: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
