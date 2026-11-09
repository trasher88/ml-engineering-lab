"""CLI F4.8: план, независимый эталон, torch smoke и необязательный real DPO."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.llm_f4.dpo_experiment import DPOError, load_config, make_plan, dpo_loss, run_real, verify_run


def main() -> int:
    parser = argparse.ArgumentParser(description="F4.8: preference dataset, exact DPO loss and optional real training")
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--plan", action="store_true", help="validate datasets and plans without model weights")
    actions.add_argument("--math-demo", action="store_true", help="reproducible numeric DPO example")
    actions.add_argument("--torch-smoke", action="store_true", help="synthetic torch autograd test; no LLM")
    actions.add_argument("--run", choices=("cpu", "gpu"), help="real DPO on verified F4.5 adapter")
    actions.add_argument("--verify", choices=("cpu", "gpu"), help="verify saved real DPO artifacts")
    parser.add_argument("--allow-download", action="store_true", help="permit download of pinned Qwen weights")
    args = parser.parse_args()
    if args.allow_download and not args.run:
        parser.error("--allow-download requires --run")
    try:
        if args.math_demo:
            loss, margin, prob = dpo_loss(-2.0, -3.0, -2.5, -3.0, 0.1)
            print(f"F4.8 MATH DEMO: PASS | loss={loss:.6f} margin={margin:.6f} preferred_probability={prob:.6f}")
        elif args.torch_smoke:
            try:
                import torch
                from src.llm_f4.dpo_experiment import torch_dpo_loss
            except ImportError as exc:
                raise DPOError(f"Нужен PyTorch для smoke: {exc}") from exc
            torch.manual_seed(42)
            policy_chosen = torch.tensor(-2.0, requires_grad=True)
            policy_rejected = torch.tensor(-3.0, requires_grad=True)
            loss = torch_dpo_loss(policy_chosen, policy_rejected, torch.tensor(-2.5), torch.tensor(-3.0), beta=0.1)
            loss.backward()
            if policy_chosen.grad is None or policy_rejected.grad is None or not torch.isfinite(loss):
                raise DPOError("Torch gradient check failed")
            if policy_chosen.grad.item() >= 0 or policy_rejected.grad.item() <= 0:
                raise DPOError("DPO gradients point in wrong direction")
            print(f"F4.8 TORCH SMOKE: PASS (SYNTHETIC, NO LLM TRAINING) | loss={loss.item():.6f}")
        else:
            conf = load_config(ROOT / "configs/f4_08_dpo_experiment.json", root=ROOT)
            if args.plan:
                plan, _ = make_plan(conf, root=ROOT)
                print("F4.8 PLAN: PASS (metadata and provenance only)")
                print(f"Model: {plan['model_id']} | reference: {plan['reference']}")
                print(f"Preference pairs: {plan['split_counts']} | beta={plan['beta']}")
                print("Weights / optimizer / real DPO: NOT RUN")
            elif args.run:
                result = run_real(conf, root=ROOT, profile=args.run, allow_download=args.allow_download)
                print(f"F4.8 REAL DPO {args.run.upper()}: PASS")
                print(f"Optimizer steps: {result['optimizer_steps']} | train pairs: {result['train_pairs_used']}")
                print("Answer quality: NOT EVALUATED")
            else:
                result = verify_run(conf, root=ROOT, profile=args.verify)
                print(f"F4.8 VERIFY {args.verify.upper()}: PASS")
                print(f"Optimizer steps: {result['optimizer_steps']} | source reference: VERIFIED")
    except (DPOError, ValueError, OSError, RuntimeError) as exc:
        message = str(exc).encode(sys.stderr.encoding or "utf-8", errors="backslashreplace").decode(sys.stderr.encoding or "utf-8", errors="replace")
        print(f"F4.8 FAIL: {message}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
