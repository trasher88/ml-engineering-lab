"""CLI F4.6: план, визуализация подтверждённого F4.5 и верификация файлов."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.llm_f4.training_visualization import (  # noqa: E402
    VisualizationError, build_report, get_sft, load_config, plan, verify_report,
)
from src.llm_f4.sft_lora import SFTError  # noqa: E402
from src.llm_f4.instruction_dataset import DatasetError  # noqa: E402
from src.llm_f4.evaluation_set import EvaluationError  # noqa: E402
from src.llm_f4.model_selection import ModelSelectionError  # noqa: E402
from src.llm_f4.base_model_evaluation import BaselineError  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="F4.6: offline training metrics visualization (verified F4.5 only)")
    parser.add_argument("--config", type=Path, default=Path("configs/f4_06_training_visualization.json"))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--plan", action="store_true", help="Validate upstream configuration without rendering")
    mode.add_argument("--build", choices=("cpu", "gpu"), help="Render real verified F4.5 run to HTML/SVG/CSV")
    mode.add_argument("--verify", choices=("cpu", "gpu"), help="Verify generated charts and upstream hashes")
    args = parser.parse_args(argv)
    try:
        path = args.config if args.config.is_absolute() else ROOT / args.config
        conf = load_config(path, root=ROOT)
        if args.build:
            stats = build_report(conf, root=ROOT, profile=args.build)
            print(f"F4.6 BUILD {args.build.upper()}: PASS")
            print(f"Optimizer steps: {stats['optimizer_steps']}; trend: {stats['trend_interpretation']}")
            print(f"Validation loss before/after: {stats['validation_loss_before']:.6f} -> {stats['validation_loss_after']:.6f}")
            print(f"Open offline report: {conf.output_dir / args.build / 'report.html'}")
            print("Quality benchmark / model training: NOT RUN IN F4.6")
        elif args.verify:
            stats = verify_report(conf, root=ROOT, profile=args.verify)
            print(f"F4.6 VERIFY {args.verify.upper()}: PASS")
            print(f"Optimizer steps: {stats['optimizer_steps']}; source F4.5: VERIFIED")
        else:
            info = plan(conf, root=ROOT)
            print("F4.6 PLAN: PASS (metadata only)")
            print(f"Selected: {info['model']} | revision={info['revision']}")
            print(f"F4.1 splits: {info['splits']}")
            print(f"Source manifests present: {info['profiles']}")
            print("Model weights / training / charts: NOT RUN")
        return 0
    except (VisualizationError, SFTError, DatasetError, EvaluationError,
            ModelSelectionError, BaselineError, OSError, RuntimeError, ValueError) as exc:
        err = str(exc).encode("ascii", "backslashreplace").decode("ascii")
        print(f"F4.6 FAIL: {err}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
