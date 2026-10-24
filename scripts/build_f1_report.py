"""F1.9: rebuild a public-safe report from local, aggregate-only F1 audits.

Usage:
  python -m scripts.build_f1_report --expect-rows 3040 --expect-test 608
  python -m scripts.build_f1_report --check
No network, model fitting, or source Q&A in the generated Markdown.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

from src.common.io import ROOT
from src.reporting.f1 import F1ReportError, render_markdown, validate_audits

AUDIT_FILES = {
    "cleaning": Path("reports/data_cleaning_audit.json"),
    "eda": Path("reports/data_eda_audit.json"),
    "split": Path("reports/data_split_audit.json"),
    "classical": Path("reports/classical.json"),
    "errors": Path("reports/classical_error_analysis.json"),
    "manifest": Path("data/raw/manifest.json"),
}
OUTFILE = Path("docs/reports/F1_PHASE_REPORT.md")


def sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as fd:
        for buf in iter(lambda: fd.read(1024 * 1024), b""):
            result.update(buf)
    return result.hexdigest()


def read_report_inputs(root: Path) -> dict:
    result = {}
    for name, relpath in AUDIT_FILES.items():
        path = root / relpath
        if not path.is_file():
            raise F1ReportError(f"missing {name} audit: {relpath}")
        try:
            result[name] = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, UnicodeError) as exc:
            raise F1ReportError(f"invalid JSON in {name} audit") from exc
    return result


def build_report(root: Path, *, expected_rows: int | None = None,
                 expected_test: int | None = None) -> str:
    root = Path(root)
    docs = read_report_inputs(root)
    summary = validate_audits(**docs)
    if expected_rows is not None and summary["cleaned"] != expected_rows:
        raise F1ReportError(f"expected cleaned={expected_rows}, found {summary['cleaned']}")
    if expected_test is not None and summary["sizes"]["test"] != expected_test:
        raise F1ReportError(f"expected test={expected_test}, found {summary['sizes']['test']}")

    # Stronger provenance than merely comparing recorded hashes against each other:
    # verify that all saved split files and cleaned JSONL still match the audits.
    local_hashes = {"cleaned": (Path("data/interim/cleaned.jsonl"), summary["cleaned_sha256"])}
    for split in ("train", "val", "test"):
        local_hashes[split] = (Path(f"data/processed/{split}.jsonl"),
                               docs["classical"]["split_source_sha256"][split])
    for name, (relpath, expected) in local_hashes.items():
        path = root / relpath
        if not path.is_file() or sha256(path) != expected:
            raise F1ReportError(f"{name} source SHA-256 differs from saved audit: {relpath}")
    return render_markdown(summary)


def save_report(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n", suffix=".tmp",
                                         prefix=".f19-", dir=path.parent, delete=False) as f:
            tmp = Path(f.name)
            f.write(content)
        os.replace(tmp, path)
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)


def run(root: Path = ROOT, *, output: Path = OUTFILE, check: bool = False,
        expected_rows: int | None = None, expected_test: int | None = None) -> Path:
    root = Path(root)
    destination = root / output
    # Reports must not overwrite audits/raw data, and output belongs to docs/reports.
    if destination.resolve() != (root / OUTFILE).resolve():
        raise F1ReportError("report output must be docs/reports/F1_PHASE_REPORT.md")
    content = build_report(root, expected_rows=expected_rows, expected_test=expected_test)
    if check:
        if not destination.is_file() or destination.read_text(encoding="utf-8") != content:
            raise F1ReportError("published F1 report missing or differs from regenerated evidence")
    else:
        save_report(destination, content)
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description="F1.9 public-safe reproducible Markdown report")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--expect-rows", type=int)
    parser.add_argument("--expect-test", type=int)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    try:
        if args.expect_rows is not None and (type(args.expect_rows) is not int or args.expect_rows < 1):
            raise F1ReportError("expect-rows must be positive")
        if args.expect_test is not None and (type(args.expect_test) is not int or args.expect_test < 1):
            raise F1ReportError("expect-test must be positive")
        path = run(root=args.root, check=args.check, expected_rows=args.expect_rows,
                   expected_test=args.expect_test)
    except (F1ReportError, OSError, KeyError, TypeError) as exc:
        print(f"F1.9 Final Phase Report: FAIL: {exc}")
        return 1
    print(f"F1.9 Final Phase Report: PASS; {'verified' if args.check else 'generated'}={path}")
    print("Source checks: audits F1.4-F1.8, source SHA-256, train-only contract, confusion matrix")
    print("Published content: aggregate metrics only; original Q&A and record IDs excluded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
