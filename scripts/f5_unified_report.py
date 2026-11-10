"""CLI единого отчёта F5.1: python -m scripts.f5_unified_report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.evaluation_f5.unified_report import EvaluationError, numeric_paths, run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="F5.1: единый отчёт по сохранённым метрикам F1–F4")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Корень репозитория")
    parser.add_argument("--config", default="configs/f5_01_unified_evaluation.json", help="Относительный путь к конфигурации")
    parser.add_argument("--check", action="store_true", help="Проверить отчёт без перезаписи")
    parser.add_argument("--require-complete", action="store_true", help="Не допускать PARTIAL")
    parser.add_argument("--inspect-paths", action="store_true", help="Показать пути числовых полей без значений")
    args = parser.parse_args(argv)
    if args.check and args.inspect_paths:
        parser.error("--check и --inspect-paths несовместимы")
    try:
        if args.inspect_paths:
            print(json.dumps(numeric_paths(args.root, args.config), ensure_ascii=False, indent=2))
            return 0
        report = run(args.root, args.config, check=args.check, require_complete=args.require_complete)
    except (EvaluationError, OSError) as exc:
        print(f"F5.1 FAIL: {exc}", file=sys.stderr)
        return 2
    print(f"F5.1 {report['status']}: источники с метриками {report['summary']['sources_with_metrics']}/{report['summary']['source_count']}; метрик {report['summary']['metric_count']}; режим {'check' if args.check else 'build'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
