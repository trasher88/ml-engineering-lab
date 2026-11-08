"""F4.4 CLI: план, реальный CPU inference, проверка и ручная оценка."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.llm_f4.base_model_evaluation import (  # noqa: E402
    BaselineError, RealCPUGenerator, execute, load_config, load_prompts,
    real_runtime, score_grades, selection, template_grades, verify,
)
from src.llm_f4.evaluation_set import EvaluationError  # noqa: E402
from src.llm_f4.model_selection import ModelSelectionError  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="F4.4: pinned Qwen Base CPU inference and reproducible records")
    parser.add_argument("--config", type=Path, default=Path("configs/f4_04_base_model_evaluation.json"))
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--plan", action="store_true", help="Validate plan; no model loading")
    group.add_argument("--run", choices=("smoke", "full"), help="Run real CPU inference; weights required")
    group.add_argument("--verify", choices=("smoke", "full"), help="Verify existing responses; no inference")
    group.add_argument("--grades-template", action="store_true", help="Create human scoring form after FULL")
    group.add_argument("--score-grades", type=Path, help="Score filled human grades after FULL")
    parser.add_argument("--allow-download", action="store_true", help="Explicitly permit downloads from Hub")
    parser.add_argument("--resume", action="store_true", help="Resume incomplete real inference run")
    args = parser.parse_args(argv)
    if args.allow_download and args.run is None:
        parser.error("--allow-download requires --run")
    if args.resume and args.run is None:
        parser.error("--resume requires --run")
    try:
        conf_path = args.config if args.config.is_absolute() else ROOT / args.config
        conf = load_config(conf_path, root=ROOT)
        if args.run is not None:
            runtime = real_runtime(conf)
            # Load the real weights only when an actual generation request requires them.
            output = execute(
                conf, root=ROOT, profile=args.run, runtime=runtime,
                backend_factory=lambda: RealCPUGenerator(conf, allow_download=args.allow_download),
                resume=args.resume,
            )
            print(f"F4.4 RUN {args.run.upper()}: PASS")
            print(f"Model: {output['model_id']} | revision={output['model_revision']}")
            print(f"Responses: {output['completed_cases']}/{output['expected_cases']}")
            print(f"Generated tokens: {output['metrics']['total_generated_tokens']}")
            print("Quality score: NOT HUMAN GRADED")
        elif args.verify is not None:
            result = verify(conf, root=ROOT, profile=args.verify)
            print(f"F4.4 VERIFY {args.verify.upper()}: PASS")
            print(f"Responses: {result['completed_cases']}/{result['expected_cases']}")
            print(f"Responses SHA256: {result['responses_sha256']}")
        elif args.grades_template:
            path = template_grades(conf, root=ROOT)
            print("F4.4 GRADES TEMPLATE: PASS")
            print(f"File: {path}")
            print("Human scores: NOT PROVIDED")
        elif args.score_grades is not None:
            path = args.score_grades if args.score_grades.is_absolute() else ROOT / args.score_grades
            result = score_grades(conf, root=ROOT, grades_file=path)
            print("F4.4 HUMAN GRADING: PASS")
            print(f"Macro score: {result['macro_over_cases']:.4f}")
            print("Human score proven by annotations, not automated lexical matches")
        else:
            prompts, info = load_prompts(conf, root=ROOT)
            smoke = selection(prompts, conf, "smoke")
            print("F4.4 PLAN: PASS (metadata and prompt export only)")
            print(f"Model: {conf.model_id} | revision={conf.model_revision}")
            print(f"Evaluation: {info['evaluation_id']} | full={len(prompts)} | smoke={len(smoke)}")
            print(f"Prompt export SHA256: {info['prompts_sha256']}")
            print(f"Input cap: {conf.max_input_tokens} | new token cap: {conf.max_new_tokens}")
            print("Weights / generation / expert grading: NOT RUN")
        return 0
    except (BaselineError, ModelSelectionError, EvaluationError, OSError, RuntimeError) as exc:
        message = str(exc).encode("ascii", "backslashreplace").decode("ascii")
        print(f"F4.4 FAIL: {message}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
