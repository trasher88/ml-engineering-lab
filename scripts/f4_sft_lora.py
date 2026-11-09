"""CLI F4.5: план, синтетический autograd smoke, реальный SFT, verify."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.llm_f4.sft_lora import (  # noqa: E402
    SFTError, load_config, make_plan, run_real, torch_smoke, verify_run,
)
from src.llm_f4.instruction_dataset import DatasetError  # noqa: E402
from src.llm_f4.evaluation_set import EvaluationError  # noqa: E402
from src.llm_f4.base_model_evaluation import BaselineError  # noqa: E402
from src.llm_f4.model_selection import ModelSelectionError  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="F4.5: reproducible instruction SFT and LoRA")
    parser.add_argument("--config", type=Path, default=Path("configs/f4_05_sft_lora.json"))
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--plan", action="store_true", help="Validate datasets and metadata; no model")
    modes.add_argument("--torch-smoke", action="store_true", help="Synthetic torch LoRA autograd, no LLM")
    modes.add_argument("--run", choices=("cpu", "gpu"), help="Real LoRA training on pinned model")
    modes.add_argument("--verify", choices=("cpu", "gpu"), help="Verify saved training artifacts")
    parser.add_argument("--allow-download", action="store_true", help="Allow fetching pinned model weights")
    args = parser.parse_args(argv)
    if args.allow_download and args.run is None:
        parser.error("--allow-download requires --run")
    try:
        config_path = args.config if args.config.is_absolute() else ROOT / args.config
        conf = load_config(config_path, root=ROOT)
        if args.torch_smoke:
            result = torch_smoke()
            print("F4.5 TORCH SMOKE: PASS (SYNTHETIC, NO LLM TRAINING)")
            print(f"Loss: {result['initial_loss']:.6f}; adapter_changed={result['adapter_changed']}")
        elif args.run:
            result = run_real(conf, root=ROOT, profile=args.run, allow_download=args.allow_download)
            print(f"F4.5 REAL SFT {args.run.upper()}: PASS")
            print(f"Optimizer steps: {result['optimizer_steps']}; LoRA trainable: {result['trainable_parameters']}")
            print(f"Validation loss: {result['initial_validation_loss']:.6f} -> {result['final_validation_loss']:.6f}")
            print("Evaluation holdout / manual quality: NOT RUN")
        elif args.verify:
            result = verify_run(conf, root=ROOT, profile=args.verify)
            print(f"F4.5 VERIFY {args.verify.upper()}: PASS")
            print(f"Optimizer steps: {result['optimizer_steps']}")
        else:
            plan, _ = make_plan(conf, root=ROOT)
            print("F4.5 PLAN: PASS (metadata and verified F4.1/F4.3 only)")
            print(f"Selected: {plan['model_id']} | revision={plan['revision']}")
            print(f"Splits: {plan['dataset_counts']}")
            print(f"LoRA: rank={conf.rank} alpha={conf.alpha} modules={','.join(conf.target_modules)}")
            print("Model weights / optimizer / training: NOT RUN")
        return 0
    except (SFTError, DatasetError, EvaluationError, BaselineError, ModelSelectionError, OSError,
            RuntimeError, ValueError) as exc:
        message = str(exc).encode("ascii", "backslashreplace").decode("ascii")
        print(f"F4.5 FAIL: {message}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
