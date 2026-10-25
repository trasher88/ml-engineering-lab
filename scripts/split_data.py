# Назначение файла: F1.6: CLI для группового разделения очищенного корпуса.
# Сохраняет train/val/test локально и аудит без исходных текстов.

"""F1.6: offline, grouped train/val/test split of cleaned F1.4 JSONL.

Run: python -m scripts.split_data --expect-rows 3040
All text-bearing artifacts remain local and excluded from Git.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

import yaml

from src.common.io import ROOT, read_jsonl
from src.data.splitting import SplitError, split_cleaned


# Ошибка подготовки split или проверки ожидаемых характеристик.
class SplitRunError(SplitError):
    """Invalid file/config or mismatch to a caller's expected counts."""


# Привязывает отчёт разбиения к точным байтам очищенного корпуса.
def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


# Записывает локальные выборки либо аудит через временный файл.
def _write_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n",
                                         dir=path.parent, prefix=".split-",
                                         suffix=".tmp", delete=False) as output:
            temp = Path(output.name)
            output.write(content)
        os.replace(temp, path)
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)


# Выполняет групповое разбиение, проверяет контракт и сохраняет файлы.
def run(root: Path = ROOT, *,
        config_path: Path = Path("configs/local_cpu/lab.yaml"),
        input_path: Path = Path("data/interim/cleaned.jsonl"),
        output_dir: Path = Path("data/processed"),
        audit_path: Path = Path("reports/data_split_audit.json"),
        expect_rows: int | None = None, expect_train: int | None = None,
        expect_val: int | None = None, expect_test: int | None = None,
        expect_edges: int | None = None) -> dict:
    root = Path(root)
    source = root / input_path
    cfg_file = root / config_path
    if not source.is_file():
        raise SplitRunError(f"missing F1.4 cleaned corpus: {source}")
    if not cfg_file.is_file():
        raise SplitRunError(f"missing configuration: {cfg_file}")
    cfg = yaml.safe_load(cfg_file.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict) or not isinstance(cfg.get("data"), dict):
        raise SplitRunError("invalid split configuration")
    seed = cfg.get("seed")
    threshold = cfg["data"].get("near_duplicate_threshold")
    if type(seed) is not int or seed < 0:
        raise SplitRunError("seed must be a non-negative integer")
    if (isinstance(threshold, bool) or not isinstance(threshold, (int, float))
            or not 0 < threshold <= 1):
        raise SplitRunError("data.near_duplicate_threshold must be in (0, 1]")
    for value in (expect_rows, expect_train, expect_val, expect_test, expect_edges):
        if value is not None and (type(value) is not int or value < 0):
            raise SplitRunError("expected counts must be non-negative integers")
    out_dir = root / output_dir
    out_audit = root / audit_path
    targets = [out_dir / f"{split}.jsonl" for split in ("train", "val", "test")]
    if len({p.resolve() for p in [source, out_audit, *targets]}) != 5:
        raise SplitRunError("input, split files, and audit must have distinct paths")

    try:
        rows = read_jsonl(source)
    except json.JSONDecodeError as exc:
        raise SplitRunError("invalid JSONL input") from exc
    if expect_rows is not None and len(rows) != expect_rows:
        raise SplitRunError(f"expected {expect_rows} rows, got {len(rows)}")
    assigned, audit = split_cleaned(rows, seed=seed, threshold=float(threshold))
    for name, expected in (("train", expect_train), ("val", expect_val), ("test", expect_test)):
        if expected is not None and audit["split_counts"][name] != expected:
            raise SplitRunError(f"expected {name}={expected}, got {audit['split_counts'][name]}")
    if expect_edges is not None and audit["near_duplicate_edges"] != expect_edges:
        raise SplitRunError(f"expected near edges={expect_edges}, got {audit['near_duplicate_edges']}")
    audit["source_file"] = str(input_path).replace("\\", "/")
    audit["source_sha256"] = _sha256(source)
    audit["output_files"] = [str(output_dir / f"{name}.jsonl").replace("\\", "/")
                             for name in ("train", "val", "test")]

    # Only after all validation and expectation checks do we write outputs.
    for name, target in zip(("train", "val", "test"), targets):
        _write_atomic(target, "".join(json.dumps(row, ensure_ascii=False) + "\n"
                                      for row in assigned if row["split"] == name))
    _write_atomic(out_audit, json.dumps(audit, ensure_ascii=False, indent=2,
                                       allow_nan=False) + "\n")
    return audit


# Принимает параметры разбиения и выводит статистику групп.
def main() -> int:
    parser = argparse.ArgumentParser(description="F1.6 — grouped, leakage-audited split")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--config", type=Path, default=Path("configs/local_cpu/lab.yaml"))
    parser.add_argument("--input", type=Path, default=Path("data/interim/cleaned.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--audit", type=Path, default=Path("reports/data_split_audit.json"))
    for flag in ("rows", "train", "val", "test", "edges"):
        parser.add_argument("--expect-" + flag, type=int)
    args = parser.parse_args()
    try:
        audit = run(root=args.root, config_path=args.config, input_path=args.input,
                    output_dir=args.output_dir, audit_path=args.audit,
                    expect_rows=args.expect_rows, expect_train=args.expect_train,
                    expect_val=args.expect_val, expect_test=args.expect_test,
                    expect_edges=args.expect_edges)
    except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError) as exc:
        print(f"Data splitting: FAIL: {exc}")
        return 1
    parts = audit["split_counts"]
    print(f"Data splitting: PASS; rows={audit['total_rows']}; "
          f"train={parts['train']}; val={parts['val']}; test={parts['test']}")
    print(f"Near-duplicate edges={audit['near_duplicate_edges']}; "
          f"groups={audit['group_count']}; threshold={audit['near_duplicate_threshold']}")
    print(f"Group overlap: {audit['group_overlap']}")
    print("Files: data/processed/{train,val,test}.jsonl and aggregate-only local audit")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
