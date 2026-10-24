"""F1.8: offline, post-training analysis of stored F1.7 test predictions.

Does not load serialized models or train. Results contain aggregate metrics;
case-level IDs are stored only in a gitignored file for local review.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from scripts.train_classical import _atomic_text
from src.classical_ml.error_analysis import ErrorAnalysisError, analyze_errors, draw_confusion_matrix
from src.common.io import ROOT, read_jsonl


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for buf in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(buf)
    return digest.hexdigest()


def run(root: Path = ROOT, *,
        test_path: Path = Path("data/processed/test.jsonl"),
        predictions_path: Path = Path("reports/classical_predictions.json"),
        baseline_path: Path = Path("reports/classical.json"),
        output_path: Path = Path("reports/classical_error_analysis.json"),
        cases_path: Path = Path("reports/classical_error_ids.json"),
        plot_path: Path = Path("docs/visuals/f1/confusion_matrix.png"),
        expect_test: int | None = None) -> dict:
    root = Path(root)
    paths = [root / p for p in (test_path, predictions_path, baseline_path,
                                output_path, cases_path, plot_path)]
    if len({p.resolve() for p in paths}) != len(paths):
        raise ErrorAnalysisError("input and output paths must be distinct")
    test_file, prediction_file, baseline_file, report_file, cases_file, plot_file = paths
    for path in (test_file, prediction_file, baseline_file):
        if not path.is_file():
            raise ErrorAnalysisError(f"missing F1.7 input: {path}")
    if expect_test is not None and (type(expect_test) is not int or expect_test < 1):
        raise ErrorAnalysisError("expect-test must be positive integer")
    rows = read_jsonl(test_file)
    frozen = json.loads(baseline_file.read_text(encoding="utf-8"))
    predictions = json.loads(prediction_file.read_text(encoding="utf-8"))
    recorded = frozen.get("split_source_sha256") if isinstance(frozen, dict) else None
    if (not isinstance(recorded, dict) or recorded.get("test") != _digest(test_file)):
        raise ErrorAnalysisError("F1.7 report/test SHA-256 mismatch; rerun F1.7")
    report, cases = analyze_errors(rows, predictions, frozen)
    if expect_test is not None and report["n"] != expect_test:
        raise ErrorAnalysisError(f"expected test={expect_test}, found {report['n']}")
    report["test_source_sha256"] = recorded["test"]
    report["cases_local_only"] = str(cases_path).replace("\\", "/")
    report["figure_local_only"] = str(plot_path).replace("\\", "/")
    # Both outputs are gitignored. A local-only ID list is intentionally not
    # part of the published report or screenshot; the plot uses counts only.
    content = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    case_content = json.dumps(cases, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    plot_file.parent.mkdir(parents=True, exist_ok=True)
    draw_confusion_matrix(report, plot_file)
    _atomic_text(report_file, content)
    _atomic_text(cases_file, case_content)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="F1.8 — offline error analysis of saved F1.7 predictions")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--expect-test", type=int, default=None)
    args = parser.parse_args()
    try:
        report = run(root=args.root, expect_test=args.expect_test)
    except (ErrorAnalysisError, OSError, json.JSONDecodeError, TypeError,
            KeyError, ValueError) as exc:
        print(f"Error analysis: FAIL: {exc}")
        return 1
    print(f"Error analysis: PASS; test={report['n']}; correct={report['correct']}; "
          f"errors={report['errors']}; accuracy={report['accuracy']:.4f}; "
          f"macro_F1={report['macro_f1']:.4f}")
    if report["confusion_pairs"]:
        top = report["confusion_pairs"][0]
        print(f"Top confusion: {top['true']} -> {top['predicted']} ({top['count']})")
    print("Report, ID-only error cases and confusion plot: LOCAL ONLY")
    print("No model fitting, no Q&A texts written to report")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
