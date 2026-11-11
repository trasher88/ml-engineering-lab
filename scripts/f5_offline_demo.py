"""F5.8: однокомандная офлайн-демонстрация и чтение результатов без записи в Git."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from src.demo_f5.offline_demo import OfflineDemoError, JSON_NAME, MD_NAME, run
from src.api_f5.demo_service import APIFixtureError
from src.evaluation_f5.unified_report import EvaluationError

ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="F5.8: offline CLI demo и audit F1–F5")
    parser.add_argument("command", choices=("run", "check"), nargs="?", default="run")
    parser.add_argument("--root", type=Path, default=ROOT,
                        help="Корень checkout, по умолчанию определяется по пути скрипта")
    parser.add_argument("--output-dir", type=Path,
                        default=Path.home() / "Documents" / "ML_Engineering_Lab_F5_08_Demo",
                        help="Новый каталог вне репозитория для двух отчётов")
    args = parser.parse_args(argv)
    try:
        result = run(args.root, args.output_dir, check=args.command == "check")
    except (OfflineDemoError, APIFixtureError, EvaluationError, OSError, UnicodeError) as exc:
        print(f"F5.8 FAIL: {exc}", file=sys.stderr)
        return 2
    summary = result["case_summary"]
    audit = result["lineage"]["F5_01"]
    print(f"F5.8 PASS: synthetic={summary['passed']}/{summary['total']}; "
          f"F1-F4 sources={audit['sources_with_metrics']}/{audit['source_count']}; "
          f"F5.1={audit['status']}; mode={args.command}; "
          "network=NOT USED; LLM=NOT EXECUTED; Docker=NOT EXECUTED; MLflow=NOT EXECUTED")
    print(f"Files: {args.output_dir.expanduser().resolve() / JSON_NAME}; "
          f"{args.output_dir.expanduser().resolve() / MD_NAME}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
