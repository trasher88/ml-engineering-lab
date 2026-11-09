"""CLI F4.9: план / сборка / проверка итогового отчёта без ML-весов."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.llm_f4.final_phase_report import (ReportError, load_config, plan,
                                           build_report, verify_report)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="F4.9: verified phase report from F4.1 to F4.8; no training")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--plan", action="store_true", help="Check frozen inputs and show available artifacts")
    modes.add_argument("--build", choices=("cpu",), help="Build report from verified CPU experiments")
    modes.add_argument("--verify", choices=("cpu",), help="Recompute report and check every hash")
    args = parser.parse_args(argv)
    try:
        conf = load_config(ROOT / "configs/f4_09_final_phase_report.json", root=ROOT)
        if args.build:
            result = build_report(conf)
            print("F4.9 BUILD CPU: PASS (verified evidence, no training)")
            print(f"SFT steps: {result['sft']['optimizer_steps']}; DPO steps: {result['dpo']['optimizer_steps']}")
            print(f"Paired responses: {result['comparison']['cases']}/32")
            print("Human grading: " + ("VERIFIED" if result['comparison']['human_grades'] else "NOT EVALUATED"))
            print(f"Report: {conf.output_dir / conf.profile / 'final_report.md'}")
        elif args.verify:
            result = verify_report(conf)
            print("F4.9 VERIFY CPU: PASS")
            print(f"Model: {result['model']['id']} | baseline: {result['base']['responses']}/32")
            print("Quality improvement: " + ("REVIEW SCORED" if result['comparison']['human_grades'] else "NOT PROVEN"))
        else:
            result = plan(conf)
            print("F4.9 PLAN: PASS (metadata only)")
            print(f"Model: {result['model_id']} | revision={result['model_revision']}")
            print(f"Data: {result['instruction_records']} instruction rows | {result['evaluation_cases']} evaluation cases")
            for name, available in result['artifacts_present'].items():
                print(f"{name}: {'PRESENT' if available else 'MISSING'} (not verified by PLAN)")
            print("Weights / training / inference: NOT RUN")
    except (ReportError, ValueError, OSError, RuntimeError, KeyError, TypeError) as exc:
        encoding = sys.stderr.encoding or "utf-8"
        message = str(exc).encode(encoding, "backslashreplace").decode(encoding, "replace")
        print(f"F4.9 FAIL: {message}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
