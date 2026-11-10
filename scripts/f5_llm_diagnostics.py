"""CLI F5.2: анализ сохранённых F4.7 Base/LoRA ответов."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from src.evaluation_f5.llm_diagnostics import DiagnosticsError, run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="F5.2: LLM Evaluation & Diagnostics без загрузки модели")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Корень репозитория")
    parser.add_argument("--config", default="configs/f5_02_llm_diagnostics.json", help="Путь к конфигурации от корня")
    parser.add_argument("--input", dest="input_path", default=None, help="Точный путь к JSON/JSONL/MD F4.7 от корня")
    parser.add_argument("--check", action="store_true", help="Сверить существующие выходы без записи")
    parser.add_argument("--require-expert", action="store_true", help="Не разрешать PARTIAL по экспертным оценкам")
    args = parser.parse_args(argv)
    try:
        report = run(args.root, config=args.config, input_path=args.input_path,
                     check=args.check, require_expert=args.require_expert)
    except (DiagnosticsError, OSError, UnicodeError) as exc:
        print(f"F5.2 FAIL: {exc}", file=sys.stderr)
        return 2
    s = report["summary"]
    print(f"F5.2 {report['status']}: {s['cases']} cases, {s['criteria_total']} criteria, "
          f"{s['changed_pairs']} changed pairs, "
          f"expert complete {s['human_review']['complete_cases']}/{s['cases']}; "
          f"mode={'check' if args.check else 'build'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
