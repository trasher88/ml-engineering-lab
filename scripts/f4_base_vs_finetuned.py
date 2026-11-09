"""CLI F4.7: plan, реальные LoRA ответы, попарное сравнение, ручная оценка."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.llm_f4.base_vs_finetuned import (  # noqa: E402
    ComparisonError, compare, grades_template, load_config, plan, run,
    score_grades, verify, verify_comparison,
)
from src.llm_f4.base_model_evaluation import BaselineError  # noqa: E402
from src.llm_f4.sft_lora import SFTError  # noqa: E402
from src.llm_f4.evaluation_set import EvaluationError  # noqa: E402
from src.llm_f4.instruction_dataset import DatasetError  # noqa: E402
from src.llm_f4.model_selection import ModelSelectionError  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="F4.7: Qwen Base versus saved Qwen LoRA on the same evaluation cases")
    parser.add_argument("--config", type=Path, default=Path("configs/f4_07_base_vs_finetuned.json"))
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--plan", action="store_true", help="Check frozen configs and prompt-only evaluation; no weights")
    modes.add_argument("--run", choices=("smoke", "full"), help="Generate real adapted answers on CPU")
    modes.add_argument("--compare", choices=("smoke", "full"), help="Pair saved Base and LoRA responses; no weights")
    modes.add_argument("--verify", choices=("smoke", "full"), help="Verify adapted run and optional paired comparison")
    modes.add_argument("--grades-template", action="store_true", help="Create expert review form after full comparison")
    modes.add_argument("--score-grades", type=Path, help="Validate human grades and compute paired rubric scores")
    parser.add_argument("--resume", action="store_true", help="Resume interrupted adapted generation")
    parser.add_argument("--allow-download", action="store_true", help="Explicitly permit Hub downloads")
    args = parser.parse_args(argv)
    if args.resume and args.run is None:
        parser.error("--resume requires --run")
    if args.allow_download and args.run is None:
        parser.error("--allow-download requires --run")
    try:
        path = args.config if args.config.is_absolute() else ROOT / args.config
        conf = load_config(path, root=ROOT)
        if args.run:
            data = run(conf, root=ROOT, profile=args.run, allow_download=args.allow_download, resume=args.resume)
            print(f"F4.7 RUN {args.run.upper()}: PASS (REAL LoRA inference)")
            print(f"Responses: {data['completed_cases']}/{data['expected_cases']}")
            print("Quality: NOT HUMAN GRADED")
        elif args.compare:
            data = compare(conf, root=ROOT, profile=args.compare)
            print(f"F4.7 COMPARE {args.compare.upper()}: PASS (text comparison only)")
            print(f"Changed: {data['changed_text_count']}/{data['cases']}; unchanged: {data['unchanged_text_count']}")
            print("Quality: NOT HUMAN GRADED")
        elif args.verify:
            data = verify(conf, root=ROOT, profile=args.verify)
            print(f"F4.7 VERIFY {args.verify.upper()}: PASS")
            print(f"Responses: {data['completed_cases']}/{data['expected_cases']}")
            if (conf.output_dir / args.verify / "comparison.json").exists():
                verify_comparison(conf, root=ROOT, profile=args.verify)
                print("Paired comparison: VERIFIED")
        elif args.grades_template:
            grade, review = grades_template(conf, root=ROOT)
            print("F4.7 GRADES TEMPLATE: PASS")
            print(f"Form: {grade}\nHuman review: {review}")
            print("Quality: NOT YET GRADED")
        elif args.score_grades is not None:
            grades = args.score_grades if args.score_grades.is_absolute() else ROOT / args.score_grades
            result = score_grades(conf, root=ROOT, grades_file=grades)
            print("F4.7 HUMAN SCORE: PASS")
            print(f"Base macro: {result['base_macro']:.4f}; Fine-tuned macro: {result['finetuned_macro']:.4f}")
            print(f"Delta: {result['delta_macro']:+.4f}; paired W/L/T: {result['paired_wins']}/{result['paired_losses']}/{result['paired_ties']}")
        else:
            p = plan(conf, root=ROOT)
            print("F4.7 PLAN: PASS (metadata and frozen prompts only)")
            print(f"Model: {p['model']} | revision={p['revision']}")
            print(f"Evaluation: {p['evaluation_id']} | full={p['evaluation_cases']} | smoke={p['smoke_cases']}")
            print(f"Adapter: F4.5/{p['sft_profile']}; inference/quality: NOT RUN")
        return 0
    except (ComparisonError, BaselineError, SFTError, EvaluationError, DatasetError,
            ModelSelectionError, OSError, RuntimeError) as exc:
        # Печатать ошибки в ASCII, чтобы subprocess под Windows cp1251/cp866 не падал сам.
        msg = str(exc).encode("ascii", "backslashreplace").decode("ascii")
        print(f"F4.7 FAIL: {msg}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
