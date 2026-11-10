"""CLI F5.3 — офлайн-проверка ограниченного synthetic safety-контракта."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from src.evaluation_f5.safety_benchmark import SafetyBenchmarkError, run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="F5.3: frozen synthetic Safety Benchmark (не LLM-аудит)")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Корень репозитория")
    parser.add_argument("--config", default="configs/f5_03_safety_benchmark.json", help="Конфигурация от корня")
    parser.add_argument("--predictions", default=None, help="Внешний JSONL в data/processed/f5_03/")
    parser.add_argument("--system-id", default=None, help="Объявленный идентификатор внешней системы")
    parser.add_argument("--check", action="store_true", help="Сравнить существующие JSON/MD без перезаписи")
    args = parser.parse_args(argv)
    try:
        report = run(args.root, config=args.config, predictions=args.predictions,
                     system_id=args.system_id, check=args.check)
    except (SafetyBenchmarkError, OSError, UnicodeError) as exc:
        print(f"F5.3 FAIL: {exc}", file=sys.stderr)
        return 2
    metrics = report["metrics"]
    print(f"F5.3 {report['status']} / contract {report['contract_status']}: "
          f"primary {metrics['primary']['passed']}/16, "
          f"adversarial {metrics['adversarial']['passed']}/12, "
          f"mode={report['run']['mode']}; "
          f"action={'check' if args.check else 'build'}")
    return 0 if report["contract_status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
