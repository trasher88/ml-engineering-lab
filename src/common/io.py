import hashlib
import json
import logging
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]

def read_jsonl(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]

def save_json(path: str | Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')

def save_jsonl(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows), encoding='utf-8')

def config(path: str = 'configs/local_cpu/lab.yaml') -> dict:
    cfg = yaml.safe_load((ROOT / path).read_text(encoding='utf-8'))
    if 'inherits' in cfg:
        base = config(cfg.pop('inherits'))
        for key, value in cfg.items():
            if isinstance(value, dict):
                base[key].update(value)
            else:
                base[key] = value
        return base
    return cfg

def seed_everything(seed: int) -> None:
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)

def digest(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def log_event(step: str, status: str, command: str, detail: str) -> None:
    path = ROOT / 'docs/EXECUTION_LOG.md'
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text('# Журнал выполнения\n\nВремя UTC; локальное время пользователя UTC+3.\n\n', encoding='utf-8')
    now = datetime.now(timezone.utc).isoformat()
    with path.open('a', encoding='utf-8') as stream:
        stream.write(f'- {now} — **{step} — {status}**. `{command}`. {detail}\n')
    logging.info('%s %s %s', step, status, detail)

