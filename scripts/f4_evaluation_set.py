"""F4.3: CPU CLI для подготовки sealed evaluation prompts и rubric preview."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.llm_f4.evaluation_set import (  # noqa: E402
    EvaluationError,
    export_artifacts,
    load_cases,
    load_config,
    model_prompt,
    verify_artifacts,
)
from src.llm_f4.instruction_dataset import DatasetError  # noqa: E402
from src.llm_f4.model_selection import ModelSelectionError  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    # ASCII-only help and routine stdout are compatible with Windows pipes cp1251/cp866/cp1252.
    parser = argparse.ArgumentParser(description="F4.3: frozen Russian technical evaluation set and prompt-only export")
    parser.add_argument("--config", type=Path, default=Path("configs/f4_03_evaluation_set.json"))
    parser.add_argument("--verify", action="store_true", help="Compare exported files byte-for-byte with pinned sources")
    parser.add_argument("--show-case", metavar="CASE_ID", help="Show model prompt and human rubric for one case (no inference)")
    args = parser.parse_args(argv)
    try:
        path = args.config if args.config.is_absolute() else ROOT / args.config
        config = load_config(path, root=ROOT)
        payloads = verify_artifacts(config) if args.verify else export_artifacts(config)
        manifest = json.loads(payloads["manifest.json"])
        print(f"F4.3 {'VERIFY' if args.verify else 'BUILD'}: PASS")
        print(f"Evaluation: {manifest['evaluation_id']} | {manifest['total_cases']} cases")
        print("Domains: " + ", ".join(f"{d}={n}" for d, n in manifest["domain_counts"].items()))
        print(f"Required human-graded facts: {manifest['required_facts_total']}")
        print(f"Selected model: {manifest['model_id']} | revision={manifest['model_revision']}")
        print(f"Prompt-only SHA256: {manifest['prompts_sha256'][:12]}")
        print(f"Manifest: {config.output_dir / 'manifest.json'}")
        print("Model weights / inference / grading: NOT RUN")
        if args.show_case:
            cases = load_cases(config.source_file)
            matched = next((case for case in cases if case.case_id == args.show_case), None)
            if matched is None:
                raise EvaluationError(f"Неизвестный case_id: {args.show_case}")
            # ASCII-escaped JSON for console compatibility; only explicit flag shows gold.
            print("EXPERT-ONLY CASE PREVIEW (do not feed this to model):")
            print(json.dumps({"model_prompt": model_prompt(matched),
                              "reference_answer": matched.reference_answer,
                              "required_facts": [{"fact_id": f.fact_id, "description": f.description}
                                                 for f in matched.required_facts],
                              "review_checks": matched.review_checks},
                             ensure_ascii=True, indent=2))
        return 0
    except (EvaluationError, DatasetError, ModelSelectionError, OSError) as exc:
        safe = str(exc).encode("ascii", errors="backslashreplace").decode("ascii")
        print(f"F4.3: FAIL: {safe}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
