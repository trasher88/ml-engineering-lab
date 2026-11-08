"""F4.1: CPU-демонстрация подготовки и проверки instruction dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.llm_f4.instruction_dataset import (  # noqa: E402
    DatasetError,
    SPLIT_NAMES,
    export_artifacts,
    load_config,
    load_examples,
    verify_artifacts,
)


def main(argv: list[str] | None = None) -> int:
    # Справка argparse должна кодироваться даже в Windows cp1251/cp866/cp1252:
    # при capture_output=True она пишется в pipe, а не в Unicode-консоль.
    parser = argparse.ArgumentParser(description="F4.1: instruction/response -> split JSONL + manifest")
    parser.add_argument("--config", type=Path, default=Path("configs/f4_01_instruction_dataset.json"))
    parser.add_argument("--verify", action="store_true", help="Verify previously generated split artifacts")
    args = parser.parse_args(argv)
    try:
        config_path = args.config if args.config.is_absolute() else ROOT / args.config
        config = load_config(config_path, root=ROOT)
        payloads = verify_artifacts(config) if args.verify else export_artifacts(config)
        manifest = json.loads(payloads["manifest.json"])
        print(f"F4.1 {'VERIFY' if args.verify else 'BUILD'}: PASS")
        print(f"Source: {manifest['total_records']} records, {manifest['total_groups']} groups")
        for name in SPLIT_NAMES:
            info = manifest["splits"][name]
            print(f"{name}: {info['records']} records, {info['groups']} groups; sha256={info['sha256'][:12]}")
        print(f"Manifest: {config.output_dir / 'manifest.json'}")
        train = load_examples(config.output_dir / "train.jsonl")
        print(f"Sample train: [{train[0].domain}] {train[0].instruction}")
        print("Model training / inference: NOT RUN")
        return 0
    except DatasetError as exc:
        print(f"F4.1: FAIL — {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
