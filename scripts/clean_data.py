# Назначение файла: F1.4: CLI для очистки subset и записи агрегированного аудита.
# Не загружает данные из сети; исходные тексты не попадают в отчёт.

"""F1.4: clean the F1.3 local subset without accessing the network.

Run: python -m scripts.clean_data --expect-input 3066 --expect-output 3040
No user-generated Q&A are printed; text outputs stay in ignored local paths.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path

import yaml

from src.common.io import ROOT, read_jsonl
from src.data.cleaning import clean_records


# Ошибка локальной очистки или проверки ожидаемых чисел.
class CleaningError(ValueError):
    """Input, configuration or historical count check failed."""


# Вычисляет хеш входного файла для связи очистки с источником.
def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# Сохраняет результат через временный файл для защиты от обрыва записи.
def _atomic_write(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n", suffix=".tmp",
            prefix=".clean-", dir=path.parent, delete=False,
        ) as writer:
            temp = Path(writer.name)
            writer.write(payload)
        os.replace(temp, path)
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)


# Загружает subset, очищает записи, сверяет объёмы и пишет аудит.
def run(root: Path = ROOT, *, config_path: Path = Path("configs/local_cpu/lab.yaml"),
        input_path: Path | None = None, output_path: Path = Path("data/interim/cleaned.jsonl"),
        audit_path: Path = Path("reports/data_cleaning_audit.json"),
        expect_input: int | None = None, expect_output: int | None = None) -> dict:
    """Execute isolated F1.4; never fall back to the 240-row smoke sample."""
    root = Path(root)
    cfg_file = root / config_path
    cfg = yaml.safe_load(cfg_file.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict) or not isinstance(cfg.get("data"), dict):
        raise CleaningError("invalid data configuration")
    min_words = cfg["data"].get("min_words")
    if not isinstance(min_words, int) or isinstance(min_words, bool) or min_words < 1:
        raise CleaningError("data.min_words must be a positive integer")
    chosen_input = input_path if input_path is not None else Path(cfg["data"]["input"])
    source = root / chosen_input
    output = root / output_path
    audit_file = root / audit_path
    if not source.is_file():
        raise CleaningError(f"missing full F1.3 subset: {source} (run F1.3 first)")
    if source.resolve() in {output.resolve(), audit_file.resolve()} or output.resolve() == audit_file.resolve():
        raise CleaningError("input, output, and audit must have different paths")
    original = read_jsonl(source)
    if expect_input is not None and len(original) != expect_input:
        raise CleaningError(f"input row count mismatch: expected {expect_input}, found {len(original)}")
    cleaned, counts = clean_records(original, min_words=min_words)
    if expect_output is not None and len(cleaned) != expect_output:
        raise CleaningError(f"output row count mismatch: expected {expect_output}, found {len(cleaned)}")

    audit = {
        "stage": "F1.4", "source_file": str(chosen_input).replace("\\", "/"),
        "source_sha256": _sha256(source), "min_words": min_words,
        "cleaning": counts,
        "by_domain": dict(sorted(Counter(row["domain"] for row in cleaned).items())),
        "near_duplicate_grouping": "NOT RUN (F1.6)",
        "train_val_test_split": "NOT RUN (F1.6)",
    }
    # Only scalar metrics, never full question/answer contents, in the audit.
    output_rows = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in cleaned)
    _atomic_write(output, output_rows)
    _atomic_write(audit_file, json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    return audit


# Читает CLI-параметры и выводит статистику очистки.
def main() -> int:
    parser = argparse.ArgumentParser(description="F1.4: deterministic technical Q&A cleaning")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--config", type=Path, default=Path("configs/local_cpu/lab.yaml"))
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path, default=Path("data/interim/cleaned.jsonl"))
    parser.add_argument("--audit", type=Path, default=Path("reports/data_cleaning_audit.json"))
    parser.add_argument("--expect-input", type=int)
    parser.add_argument("--expect-output", type=int)
    args = parser.parse_args()
    try:
        audit = run(args.root, config_path=args.config, input_path=args.input,
                    output_path=args.output, audit_path=args.audit,
                    expect_input=args.expect_input, expect_output=args.expect_output)
    except (OSError, ValueError, KeyError, json.JSONDecodeError, yaml.YAMLError) as exc:
        print(f"Data cleaning: FAIL: {exc}")
        return 1
    stats = audit["cleaning"]
    print(f"Data cleaning: PASS; input={stats['input_rows']}; "
          f"short_removed={stats['short_removed']}; "
          f"duplicates_removed={stats['duplicates_removed']}; "
          f"invalid_or_missing={stats['invalid_or_missing']}; "
          f"output={stats['output_rows']}")
    print("Near-duplicate grouping / train-val-test split: NOT RUN (F1.6)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
