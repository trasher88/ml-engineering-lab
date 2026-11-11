"""F5.7 — provenance-aware перенос измеренных метрик F5.1 в MLflow.

Запрещено подставлять отсутствующие измерения, копировать исходные JSON
с приватными текстами и создавать фиктивные runs ради заданного количества.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.evaluation_f5.unified_report import build_report


class TrackingError(ValueError):
    """Неоднозначный источник, некорректный протокол либо ошибка реестра."""


ID = re.compile(r"^[a-z][a-z0-9_]*$")
NAME = re.compile(r"^[a-zA-Z][a-zA-Z0-9_.-]*$")
HEX = re.compile(r"^[a-f0-9]{64}$")
CONFIG = "configs/f5_07_mlflow_tracking.json"
ALLOWED_PHASES = {"F1", "F2", "F3", "F4"}
ALLOWED_TASKS = {"classification", "retrieval", "llm"}


def canonical(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def safe_path(root: Path, name: str) -> Path:
    if not isinstance(name, str) or not name or "\\" in name or name.startswith("/") or ":" in name:
        raise TrackingError("Путь должен быть относительным POSIX-путём без drive letter")
    if any(part in {"", ".", ".."} for part in name.split("/")):
        raise TrackingError("Недопустимый компонент пути")
    base = root.resolve()
    target = (base / name).resolve()
    if not target.is_relative_to(base):
        raise TrackingError("Файл находится за пределами checkout")
    return target


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"), parse_constant=lambda v: (_ for _ in ()).throw(TrackingError(f"Недопустимое значение JSON: {v}")))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TrackingError(f"Не удалось прочитать {path.name}: {exc}") from exc
    if not isinstance(value, dict):
        raise TrackingError("Требуется JSON object")
    return value


@dataclass(frozen=True)
class Record:
    name: str
    source_id: str
    phase: str
    task: str
    dataset: str
    split: str
    protocol: str
    source_path: str
    source_sha256: str
    metrics: dict[str, float]
    units: dict[str, str]
    selectors: dict[str, str]
    signature: str

    def summary(self) -> dict[str, Any]:
        """Только безопасные агрегаты; без исходных ответов и текстов."""
        return {
            "schema_version": 1,
            "stage": "F5.7",
            "source_id": self.source_id,
            "phase": self.phase,
            "task": self.task,
            "dataset": self.dataset,
            "split": self.split,
            "protocol": self.protocol,
            "source_path": self.source_path,
            "source_sha256": self.source_sha256,
            "metrics": self.metrics,
            "units": self.units,
            "selectors": self.selectors,
            "signature": self.signature,
            "claim": "imported_observations_not_new_training_or_model_evaluation",
        }


@dataclass(frozen=True)
class Plan:
    records: tuple[Record, ...]
    skipped: tuple[dict[str, str], ...]
    source_report_sha256: str
    config_sha256: str
    expected: int
    experiment: str

    @property
    def status(self) -> str:
        return "PASS" if len(self.records) == self.expected else "PARTIAL"


def _validated_config(root: Path, name: str) -> tuple[dict[str, Any], str]:
    file = safe_path(root, name)
    raw = file.read_bytes()
    cfg = load_json(file)
    if type(cfg.get("schema_version")) is not int or cfg["schema_version"] != 1:
        raise TrackingError("Неверная версия config")
    if not isinstance(cfg.get("experiment_name"), str) or not re.fullmatch(r"[a-zA-Z0-9_.-]{3,100}", cfg["experiment_name"]):
        raise TrackingError("Некорректное имя эксперимента")
    for key in ("f5_01_report", "f5_01_config"):
        if not isinstance(cfg.get(key), str):
            raise TrackingError(f"Отсутствует {key}")
        safe_path(root, cfg[key])
    runs = cfg.get("runs")
    if not isinstance(runs, list) or not runs:
        raise TrackingError("runs должен быть непустым массивом")
    seen: set[str] = set()
    for run in runs:
        if not isinstance(run, dict) or set(run) != {"id", "source_id", "required_metrics", "metric_prefix"}:
            raise TrackingError("Неверный формат runs[]")
        for key in ("id", "source_id"):
            if not isinstance(run[key], str) or not ID.fullmatch(run[key]):
                raise TrackingError(f"Некорректное {key}")
        if run["id"] in seen:
            raise TrackingError("Дублирован run id")
        seen.add(run["id"])
        prefix = run["metric_prefix"]
        if not isinstance(prefix, str) or (prefix and not ID.fullmatch(prefix)):
            raise TrackingError("Некорректный префикс метрики")
        required = run["required_metrics"]
        if not isinstance(required, list) or not required or any(not isinstance(v, str) or not ID.fullmatch(v) for v in required) or len(set(required)) != len(required):
            raise TrackingError("Некорректные required_metrics")
    return cfg, digest(raw)


def build_plan(root: Path, config: str = CONFIG) -> Plan:
    """Независимо проверить JSON F5.1 и вычислить строго подтверждённые runs."""
    root = root.resolve()
    cfg, config_sha = _validated_config(root, config)
    report_path = safe_path(root, cfg["f5_01_report"])
    if not report_path.is_file():
        raise TrackingError("Нет отчёта F5.1: выполни python -m scripts.f5_unified_report")
    raw = report_path.read_bytes()
    actual = load_json(report_path)
    expected = build_report(root, cfg["f5_01_config"])
    if actual != expected:
        raise TrackingError("Отчёт F5.1 устарел или расходится с исходными артефактами; сначала обнови F5.1")
    sources = {item["source_id"]: item for item in actual["sources"]}
    records: list[Record] = []
    skipped: list[dict[str, str]] = []
    for entry in cfg["runs"]:
        src = sources.get(entry["source_id"])
        if src is None:
            raise TrackingError(f"Нет источника {entry['source_id']} в F5.1")
        if src["phase"] not in ALLOWED_PHASES or src["task"] not in ALLOWED_TASKS:
            raise TrackingError("Неизвестные phase/task")
        if src["status"] not in {"EXTRACTED", "MISSING_SOURCE", "NO_MATCHING_METRICS"}:
            raise TrackingError("Неизвестный source status")
        prefix = entry["metric_prefix"]
        metrics: dict[str, float] = {}
        units: dict[str, str] = {}
        selectors: dict[str, str] = {}
        for metric in src["metrics"]:
            name = metric["name"]
            if not isinstance(name, str) or not name.startswith(prefix):
                continue
            key = name[len(prefix):]
            if not key or not ID.fullmatch(key) or not NAME.fullmatch(metric["unit"]):
                raise TrackingError("Некорректное имя/единица метрики")
            value = metric["value"]
            if type(value) not in (int, float) or not math.isfinite(value):
                raise TrackingError("Метрики должны быть конечными числами, не bool")
            if key in metrics:
                raise TrackingError("Дублирован metric key")
            metrics[key] = float(value)
            units[key] = metric["unit"]
            selectors[key] = metric["selector"]
        missing = sorted(set(entry["required_metrics"]) - set(metrics))
        if src["status"] != "EXTRACTED" or missing:
            skipped.append({"name": entry["id"], "reason": "source unavailable" if src["status"] != "EXTRACTED" else "missing required metrics: " + ", ".join(missing)})
            continue
        sha = src["sha256"]
        if not isinstance(sha, str) or not HEX.fullmatch(sha):
            raise TrackingError("Отсутствует SHA исходного артефакта")
        upstream = safe_path(root, src["path"])
        if not upstream.is_file() or digest(upstream.read_bytes()) != sha:
            raise TrackingError("Изменился исходный файл после расчёта F5.1")
        base = {"name": entry["id"], "source_id": src["source_id"], "phase": src["phase"], "task": src["task"],
                "dataset": src["dataset"], "split": src["split"], "protocol": src["protocol"],
                "source_path": src["path"], "source_sha256": sha,
                "metrics": metrics, "units": units, "selectors": selectors, "config_sha256": config_sha}
        signature = digest(canonical(base))
        records.append(Record(**{k: v for k, v in base.items() if k != "config_sha256"}, signature=signature))
    return Plan(tuple(records), tuple(skipped), digest(raw), config_sha, len(cfg["runs"]), cfg["experiment_name"])


def tracking_storage(store_dir: Path) -> tuple[str, str]:
    """SQLite metadata и отдельный локальный каталог артефактов."""
    base = store_dir.expanduser().resolve()
    if not base.is_absolute():
        raise TrackingError("Некорректный storage path")
    base.mkdir(parents=True, exist_ok=True)
    (base / "artifacts").mkdir(parents=True, exist_ok=True)
    db = (base / "tracking.db").as_posix()
    return "sqlite:///" + db, (base / "artifacts").as_uri()


def _experiment(client: Any, plan: Plan, artifact_location: str) -> str:
    existing = client.get_experiment_by_name(plan.experiment)
    if existing is not None:
        if existing.lifecycle_stage != "active":
            raise TrackingError("Эксперимент MLflow помечен удалённым")
        if str(existing.artifact_location).rstrip("/") != artifact_location.rstrip("/"):
            raise TrackingError("Существующий experiment использует другой artifact store")
        return existing.experiment_id
    return str(client.create_experiment(plan.experiment, artifact_location=artifact_location))


def _find_prior(client: Any, experiment_id: str, name: str, signature: str) -> Any | None:
    # Поисковый список ограничен; существующий эксперимент предназначен только для F5.7.
    runs = client.search_runs(experiment_ids=[experiment_id], max_results=5000)
    matches = [r for r in runs if r.data.tags.get("f5_07.name") == name and r.data.tags.get("f5_07.signature") == signature]
    if len(matches) > 1:
        raise TrackingError("Обнаружены дубли run с одинаковой сигнатурой")
    return matches[0] if matches else None


def _readback(client: Any, run: Any, record: Record, *, check_artifact: bool = True) -> None:
    if run.info.status != "FINISHED":
        raise TrackingError(f"Run {record.name} не завершён: {run.info.status}")
    if run.data.tags.get("f5_07.signature") != record.signature or run.data.tags.get("f5_07.name") != record.name:
        raise TrackingError("Run не совпадает по identity tags")
    for key, expected in record.metrics.items():
        found = run.data.metrics.get(key)
        if found is None or not math.isclose(found, expected, rel_tol=1e-12, abs_tol=1e-12):
            raise TrackingError(f"MLflow metrics.{key} не соответствует F5.1")
    if set(run.data.metrics) != set(record.metrics):
        raise TrackingError("MLflow содержит неучтённые метрики")
    for k, v in {"phase": record.phase, "task": record.task, "dataset": record.dataset,
                 "split": record.split, "protocol": record.protocol,
                 "source_sha256": record.source_sha256}.items():
        if run.data.params.get(k) != v:
            raise TrackingError(f"MLflow param {k} не совпадает с источником")
    if check_artifact:
        with tempfile.TemporaryDirectory(prefix="f5_07_artifact_") as td:
            downloaded = Path(client.download_artifacts(run.info.run_id, "evidence.json", td))
            if downloaded.read_bytes() != canonical(record.summary()):
                raise TrackingError("MLflow artifact не соответствует ожидаемому evidence.json")


def sync_runs(client: Any, plan: Plan, artifact_location: str, *, create: bool) -> dict[str, Any]:
    """Идемпотентный импорт / независимое считывание через MLflow client."""
    if not plan.records:
        raise TrackingError("Ни один исходный эксперимент не содержит необходимых метрик")
    experiment = client.get_experiment_by_name(plan.experiment)
    if create:
        experiment_id = _experiment(client, plan, artifact_location)
    elif experiment is None:
        raise TrackingError("Не найден MLflow эксперимент. Сначала выполни import")
    else:
        experiment_id = experiment.experiment_id
    summary: list[dict[str, Any]] = []
    for record in plan.records:
        prior = _find_prior(client, experiment_id, record.name, record.signature)
        if prior is not None:
            _readback(client, client.get_run(prior.info.run_id), record)
            action, run_id = "REUSED", prior.info.run_id
        elif not create:
            raise TrackingError(f"Нет ранее импортированного run: {record.name}")
        else:
            tags = {"f5_07.name": record.name, "f5_07.signature": record.signature,
                    "f5_07.stage": "F5.7", "f5_07.origin": "F5.1_extracted_metrics_only",
                    "f5_07.model_execution": "NOT_EXECUTED"}
            run = client.create_run(experiment_id, tags=tags, run_name=record.name)
            run_id = run.info.run_id
            try:
                for k, v in {"phase": record.phase, "task": record.task, "dataset": record.dataset,
                             "split": record.split, "protocol": record.protocol,
                             "source_sha256": record.source_sha256}.items():
                    client.log_param(run_id, k, v)
                for k, v in record.metrics.items():
                    client.log_metric(run_id, k, v)
                with tempfile.TemporaryDirectory(prefix="f5_07_payload_") as td:
                    artifact = Path(td) / "evidence.json"
                    artifact.write_bytes(canonical(record.summary()))
                    client.log_artifact(run_id, str(artifact))
                client.set_terminated(run_id, status="FINISHED")
                _readback(client, client.get_run(run_id), record)
            except Exception:
                client.set_terminated(run_id, status="FAILED")
                raise
            action = "CREATED"
        summary.append({"name": record.name, "run_id": run_id, "action": action,
                        "source_sha256": record.source_sha256, "metric_count": len(record.metrics)})
    return {"schema_version": 1, "stage": "F5.7", "status": plan.status,
            "expected": plan.expected, "verified_runs": len(summary), "skipped": list(plan.skipped),
            "source_report_sha256": plan.source_report_sha256, "config_sha256": plan.config_sha256,
            "runs": summary, "experiment": plan.experiment, "model_execution": "NOT_EXECUTED"}
