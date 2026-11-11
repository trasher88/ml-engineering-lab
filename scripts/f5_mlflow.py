"""F5.7: plan, import, verify, list — MLflow tracking из зафиксированных F5.1 JSON.

Стороннее хранилище MLflow размещается вне репозитория по умолчанию.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from src.tracking_f5.mlflow_tracking import TrackingError, build_plan, sync_runs, tracking_storage

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STORE = Path.home() / "ML_Engineering_Lab_MLflow"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="F5.7: MLflow tracking по сохранённым F1–F4 метрикам")
    parser.add_argument("command", choices=("plan", "import", "verify", "list"))
    parser.add_argument("--store-dir", type=Path, default=DEFAULT_STORE)
    parser.add_argument("--registry-output", type=Path, help="Сохранить реестр runs (не включать в Git)")
    args = parser.parse_args(argv)
    try:
        plan = build_plan(ROOT)
        if args.command == "plan":
            for record in plan.records:
                print(f"READY {record.name}: phase={record.phase}; metrics={','.join(record.metrics)}")
            for item in plan.skipped:
                print(f"SKIP {item['name']}: {item['reason']}")
            print(f"F5.7 PLAN {plan.status}: {len(plan.records)}/{plan.expected} available; MLflow=NOT EXECUTED")
            return 0 if plan.records else 2
        try:
            from mlflow.tracking import MlflowClient  # только для реального import/verify
        except ImportError as exc:
            raise TrackingError("Пакет MLflow не установлен; pip install -r requirements-f5_07.txt") from exc
        uri, artifact_location = tracking_storage(args.store_dir)
        client = MlflowClient(tracking_uri=uri)
        if args.command in {"import", "verify"}:
            summary = sync_runs(client, plan, artifact_location, create=args.command == "import")
        else:
            experiment = client.get_experiment_by_name(plan.experiment)
            if experiment is None:
                raise TrackingError("Нет runs: сначала выполни import")
            rows = client.search_runs(experiment_ids=[experiment.experiment_id], max_results=5000)
            summary = {"stage": "F5.7", "status": "INVENTORY", "experiment": plan.experiment,
                       "runs": [{"name": r.data.tags.get("f5_07.name"), "run_id": r.info.run_id,
                                 "status": r.info.status, "metrics": sorted(r.data.metrics)} for r in rows]}
        if args.registry_output:
            path = args.registry_output.expanduser().resolve()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
            print(f"Registry: {path}")
        if args.command == "list":
            for row in summary["runs"]:
                print(f"{row['status']}: {row['name']}; run_id={row['run_id']}")
            print(f"F5.7 INVENTORY: {len(summary['runs'])} runs")
        else:
            for row in summary["runs"]:
                print(f"{row['action']} {row['name']}: run_id={row['run_id']}; metrics={row['metric_count']}")
            for row in summary["skipped"]:
                print(f"SKIP {row['name']}: {row['reason']}")
            print(f"F5.7 {summary['status']}: verified={summary['verified_runs']}/{summary['expected']}; "
                  f"mode={args.command}; backend=local_sqlite; model=NOT EXECUTED")
        return 0
    except (TrackingError, OSError, ValueError, RuntimeError) as exc:
        print(f"F5.7 FAIL: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
