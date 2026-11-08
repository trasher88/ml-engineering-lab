"""F4.2: выбор checkpoint, оценка памяти и необязательная проверка tokenizer."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.llm_f4.model_selection import (  # noqa: E402
    ModelSelectionError,
    create_report,
    load_catalog,
    probe_tokenizer,
    verify_report,
    write_report,
)


def main(argv: list[str] | None = None) -> int:
    # ASCII в help и stdout: Windows cp1251/cp866/cp1252, subprocess capture_output.
    parser = argparse.ArgumentParser(description="F4.2: pinned base model selection and CPU resource estimates")
    parser.add_argument("--config", type=Path, default=Path("configs/f4_02_model_selection.json"))
    parser.add_argument("--write-report", action="store_true", help="Save deterministic local report")
    parser.add_argument("--verify-report", action="store_true", help="Verify previously written report")
    parser.add_argument("--probe-tokenizer", action="store_true", help="Check real tokenizer, cache-only by default")
    parser.add_argument("--allow-download", action="store_true", help="Explicitly allow tokenizer-only download (no weights)")
    args = parser.parse_args(argv)
    if args.allow_download and not args.probe_tokenizer:
        parser.error("--allow-download requires --probe-tokenizer")
    if args.write_report and args.verify_report:
        parser.error("--write-report and --verify-report are mutually exclusive")
    try:
        path = args.config if args.config.is_absolute() else ROOT / args.config
        catalog = load_catalog(path)
        report = create_report(catalog)
        report_path = ROOT / "data/processed/f4_02/model_selection_report.json"
        if args.write_report:
            write_report(report_path, report)
        if args.verify_report:
            verify_report(report_path, report)
        selected = report["selected_model"]
        print("F4.2 MODEL SELECTION: PASS (metadata only)")
        print(f"Selected: {selected['model_id']}")
        print(f"Revision: {selected['revision']}")
        print(f"Tokenizer revision: {selected['tokenizer_revision']}")
        for item in report["candidate_assessments"]:
            r = item["resources"]
            print(f"Candidate: {item['model_id']} | eligible={item['eligible']} | reasons={','.join(item['reasons']) or '-'}")
            print(f"  parameters_estimated={r['estimated_parameters']}; FP32_weights_MiB={r['fp32_weights_mib']}; KV_MiB={r['kv_cache_fp32_mib']}")
        runtime = report["runtime"]
        print(f"Runtime ready for optional tokenizer probe: {runtime['ready_for_optional_probe']}")
        print(f"Runtime issues: {','.join(runtime['issues']) or '-'}")
        if args.write_report or args.verify_report:
            print(f"Report: {report_path}")
        if args.probe_tokenizer:
            model = next(m for m in catalog.candidates if m.model_id == catalog.selected_model_id)
            sample = "Что такое градиентный спуск в машинном обучении?"
            result = probe_tokenizer(model, sample, allow_download=args.allow_download)
            print(f"Tokenizer probe: PASS | token_count={result['token_count']} | roundtrip=True")
            print("Weights: NOT LOADED")
        print("Inference / model training: NOT RUN")
        return 0
    except (ModelSelectionError, OSError) as exc:
        # ASCII conversion prevents UnicodeEncodeError for arbitrary Windows code pages.
        message = str(exc).encode("ascii", errors="backslashreplace").decode("ascii")
        print(f"F4.2: FAIL: {message}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
