"""F5.1: сбор фактических метрик F1–F4 без переобучения и общего score.

Протокол определяет источники и разрешённые числовые поля в JSON-конфигурации.
Отчёт не переносит произвольные строки и записи входных наборов данных.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class EvaluationError(ValueError):
    """Нарушение контракта источника, конфигурации или показателя."""


_MISSING = object()
_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
_LABEL_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_@.\-]*$")
_VALID_PHASES = {"F1", "F2", "F3", "F4"}
_VALID_TASKS = {"classification", "retrieval", "llm"}
_VALID_UNITS = {"fraction", "count", "loss", "learning_rate", "margin", "percent", "scalar"}
_STATUS_VALUES = {"PASS", "PARTIAL"}


def _validate_id(value: Any, description: str) -> str:
    if not isinstance(value, str) or not _ID_PATTERN.fullmatch(value):
        raise EvaluationError(f"{description}: ожидается идентификатор [a-z][a-z0-9_]*")
    return value


def _validate_label(value: Any, description: str) -> str:
    if not isinstance(value, str) or not _LABEL_PATTERN.fullmatch(value):
        raise EvaluationError(f"{description}: недопустимый идентификатор/метка")
    return value


def _validate_relative_path(value: Any, description: str) -> Path:
    if not isinstance(value, str) or not value or "\\" in value:
        raise EvaluationError(f"{description}: требуется относительный POSIX-путь")
    if re.match(r"^[A-Za-z]:", value) or value.startswith("/"):
        raise EvaluationError(f"{description}: абсолютные пути запрещены")
    chunks = value.split("/")
    if any(chunk in {"", ".", ".."} for chunk in chunks):
        raise EvaluationError(f"{description}: сегменты '.', '..' и пустые запрещены")
    if any("\x00" in chunk or ":" in chunk for chunk in chunks):
        raise EvaluationError(f"{description}: недопустимые символы пути")
    return Path(*chunks)


def _safe_path(root: Path, value: Any, description: str) -> Path:
    relative = _validate_relative_path(value, description)
    root_resolved = root.resolve()
    path = (root_resolved / relative).resolve()
    if not path.is_relative_to(root_resolved):
        raise EvaluationError(f"{description}: путь выходит за пределы репозитория")
    return path


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"), parse_constant=lambda x: (_ for _ in ()).throw(EvaluationError(f"Нечисловая константа JSON: {x}")))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise EvaluationError(f"Невозможно прочитать корректный JSON '{path.name}': {exc}") from exc


def _metric_value(node: Any, unit: str, description: str) -> int | float:
    if isinstance(node, bool) or not isinstance(node, (int, float)):
        raise EvaluationError(f"{description}: требуется числовое значение, не bool/строка")
    number = float(node)
    if not math.isfinite(number):
        raise EvaluationError(f"{description}: NaN и бесконечность запрещены")
    if unit == "fraction" and not 0 <= number <= 1:
        raise EvaluationError(f"{description}: доля должна находиться в диапазоне [0, 1]")
    if unit == "percent" and not 0 <= number <= 100:
        raise EvaluationError(f"{description}: проценты должны находиться в диапазоне [0, 100]")
    if unit in {"loss", "learning_rate", "count"} and number < 0:
        raise EvaluationError(f"{description}: значение должно быть неотрицательным")
    if unit == "count":
        if not number.is_integer():
            raise EvaluationError(f"{description}: число объектов должно быть целым")
        return int(number)
    return number


def _resolve_selector(document: Any, selector: str) -> Any:
    """Разрешить только точный путь `key.subkey.0.value` в JSON-дереве."""
    node: Any = document
    for part in selector.split("."):
        if not part or part.startswith("_"):
            raise EvaluationError(f"Недопустимый selector: {selector}")
        if isinstance(node, dict):
            if part not in node:
                return _MISSING
            node = node[part]
        elif isinstance(node, list) and part.isascii() and part.isdecimal():
            index = int(part)
            if index >= len(node):
                return _MISSING
            node = node[index]
        else:
            return _MISSING
    return node


def _load_config(root: Path, config_path: str) -> dict[str, Any]:
    path = _safe_path(root, config_path, "config")
    config = _read_json(path)
    if not isinstance(config, dict) or config.get("schema_version") != 1:
        raise EvaluationError("config: требуется объект с schema_version=1")
    if not isinstance(config.get("sources"), list) or not config["sources"]:
        raise EvaluationError("config: sources должен быть непустым массивом")
    outputs = config.get("outputs")
    if not isinstance(outputs, dict) or set(outputs) != {"json", "markdown"}:
        raise EvaluationError("config: outputs должен содержать только json и markdown")
    for name, value in outputs.items():
        _safe_path(root, value, f"outputs.{name}")
    if not outputs["json"].startswith("reports/public/f5_01/") or not outputs["json"].endswith(".json"):
        raise EvaluationError("config: JSON разрешён только в reports/public/f5_01/")
    if not outputs["markdown"].startswith("docs/reports/F5_01_") or not outputs["markdown"].endswith(".md"):
        raise EvaluationError("config: Markdown разрешён только под docs/reports/F5_01_*.md")
    return config


@dataclass(frozen=True)
class SourceResult:
    source_id: str
    phase: str
    task: str
    path: str
    status: str
    sha256: str | None
    split: str
    dataset: str
    protocol: str
    metrics: tuple[dict[str, Any], ...]
    missing_metrics: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "phase": self.phase,
            "task": self.task,
            "path": self.path,
            "status": self.status,
            "sha256": self.sha256,
            "split": self.split,
            "dataset": self.dataset,
            "protocol": self.protocol,
            "metrics": list(self.metrics),
            "missing_metrics": list(self.missing_metrics),
        }


def _parse_source(root: Path, spec: Any) -> SourceResult:
    if not isinstance(spec, dict):
        raise EvaluationError("sources[] должен быть объектом")
    source_id = _validate_id(spec.get("id"), "sources[].id")
    phase = spec.get("phase")
    task = spec.get("task")
    if phase not in _VALID_PHASES or task not in _VALID_TASKS:
        raise EvaluationError(f"{source_id}: неизвестная phase/task")
    if (phase in {"F1", "F2"} and task != "classification") or (phase == "F3" and task != "retrieval") or (phase == "F4" and task != "llm"):
        raise EvaluationError(f"{source_id}: task не соответствует фазе")
    path_str = spec.get("path")
    source_path = _safe_path(root, path_str, f"sources.{source_id}.path")
    metadata: dict[str, str] = {}
    for key in ("split", "dataset", "protocol"):
        metadata[key] = _validate_label(spec.get(key), f"{source_id}.{key}")
    mappings = spec.get("metrics")
    if not isinstance(mappings, list):
        raise EvaluationError(f"{source_id}: metrics должен быть массивом")
    if not source_path.exists():
        return SourceResult(source_id, phase, task, path_str, "MISSING_SOURCE", None, metadata["split"], metadata["dataset"], metadata["protocol"], (), tuple(x.get("name", "invalid") for x in mappings if isinstance(x, dict)))
    if not source_path.is_file():
        raise EvaluationError(f"{source_id}: источник не является файлом")
    blob = source_path.read_bytes()
    fingerprint = hashlib.sha256(blob).hexdigest()
    document = _read_json(source_path)
    if not isinstance(document, dict):
        raise EvaluationError(f"{source_id}: корнем JSON должен быть объект")
    found: list[dict[str, Any]] = []
    missing: list[str] = []
    used_names: set[str] = set()
    for mapping in mappings:
        if not isinstance(mapping, dict):
            raise EvaluationError(f"{source_id}: metrics[] должен быть объектом")
        name = _validate_id(mapping.get("name"), f"{source_id}.metrics[].name")
        if name in used_names:
            raise EvaluationError(f"{source_id}: повторная метрика {name}")
        used_names.add(name)
        unit = mapping.get("unit")
        if unit not in _VALID_UNITS:
            raise EvaluationError(f"{source_id}.{name}: некорректная единица измерения")
        selectors = mapping.get("selectors")
        if not isinstance(selectors, list) or not selectors or not all(isinstance(x, str) and x for x in selectors):
            raise EvaluationError(f"{source_id}.{name}: необходимы selectors")
        matches: list[tuple[str, int | float]] = []
        for selector in selectors:
            value = _resolve_selector(document, selector)
            if value is _MISSING:
                continue
            matches.append((selector, _metric_value(value, unit, f"{source_id}.{name}@{selector}")))
        if not matches:
            missing.append(name)
            continue
        unique_values = {v for _, v in matches}
        if len(unique_values) != 1:
            raise EvaluationError(f"{source_id}.{name}: неоднозначные значения по разным selectors")
        found.append({"name": name, "value": matches[0][1], "unit": unit, "selector": matches[0][0]})
    return SourceResult(source_id, phase, task, path_str, "EXTRACTED" if found else "NO_MATCHING_METRICS", fingerprint, metadata["split"], metadata["dataset"], metadata["protocol"], tuple(found), tuple(missing))


def build_report(root: Path, config_path: str = "configs/f5_01_unified_evaluation.json") -> dict[str, Any]:
    """Сформировать только структурированный отчёт, без побочных записей."""
    root = root.resolve()
    config = _load_config(root, config_path)
    seen: set[str] = set()
    sources: list[SourceResult] = []
    for spec in config["sources"]:
        result = _parse_source(root, spec)
        if result.source_id in seen:
            raise EvaluationError(f"Повторный source_id: {result.source_id}")
        seen.add(result.source_id)
        sources.append(result)
    phases_with_metrics = {r.phase for r in sources if r.metrics}
    missing_phases = sorted(_VALID_PHASES - phases_with_metrics)
    status = "PASS" if not missing_phases and all(r.status == "EXTRACTED" for r in sources) and all(not r.missing_metrics for r in sources) else "PARTIAL"
    assert status in _STATUS_VALUES
    return {
        "schema_version": 1,
        "stage": "F5.1",
        "status": status,
        "summary": {
            "source_count": len(sources),
            "sources_with_metrics": sum(bool(r.metrics) for r in sources),
            "metric_count": sum(len(r.metrics) for r in sources),
            "phases_with_metrics": sorted(phases_with_metrics),
            "phases_without_metrics": missing_phases,
        },
        "sources": [s.to_dict() for s in sources],
        "methodology": {
            "aggregation": "none_cross_task",
            "reported_values": "copied_from_verified_json_paths_not_recomputed",
            "quality_claim": "not_established_by_collection",
        },
        "limitations": [
            "SHA256 подтверждает целостность прочитанных файлов, но не подлинность исходного эксперимента.",
            "Метрики относятся к собственным протоколам и разным наборам данных; общего score нет.",
            "F4.5/F4.6: один optimizer step не доказывает тенденцию обучения или улучшение качества ответов.",
            "Автоматическая выжимка не заменяет F1/F3/F4 аудит исходных отчётов.",
            "Отсутствующие источники и неподходящие selectors отражаются как PARTIAL, значения не подставляются.",
        ],
    }


def serialize_json(report: dict[str, Any]) -> str:
    return json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"


def _md_escape(value: str) -> str:
    return value.replace("|", "\\|").replace("`", "\\`").replace("\n", " ")


def render_markdown(report: dict[str, Any]) -> str:
    """Сформировать русскоязычный отчёт только из разрешённой схемы."""
    summary = report["summary"]
    lines = [
        "# F5.1 — Unified Evaluation / Report",
        "",
        f"**Статус сборки:** `{report['status']}`. Данные считываются из сохранённых JSON; обучение и inference не выполняются.",
        "",
        "## 1. Охват источников",
        "",
        f"- Источников в протоколе: **{summary['source_count']}**; с извлечёнными метриками: **{summary['sources_with_metrics']}**.",
        f"- Извлечено метрик: **{summary['metric_count']}**.",
        f"- Фазы без подходящих метрик: **{', '.join(summary['phases_without_metrics']) or 'нет'}**.",
        "",
        "## 2. Метрики по независимым протоколам",
        "",
    ]
    for phase, title in (("F1", "Классический ML"), ("F2", "PyTorch"), ("F3", "Retrieval"), ("F4", "LLM / Fine-Tuning")):
        lines.extend([f"### {phase} — {title}", ""])
        for source in report["sources"]:
            if source["phase"] != phase:
                continue
            lines.extend([
                f"**Источник:** `{_md_escape(source['path'])}` · `{source['status']}` · split `{_md_escape(source['split'])}` · dataset `{_md_escape(source['dataset'])}` · protocol `{_md_escape(source['protocol'])}`.",
                "",
                f"SHA-256: `{source['sha256'] or 'источник отсутствует'}`.",
                "",
            ])
            if source["metrics"]:
                lines.extend(["| Метрика | Значение | Единица | JSON selector |", "|---|---:|---|---|"])
                for metric in source["metrics"]:
                    lines.append(f"| `{metric['name']}` | {metric['value']:.10g} | `{metric['unit']}` | `{metric['selector']}` |")
                lines.append("")
            if source["missing_metrics"]:
                lines.extend([f"Не извлечены: {', '.join('`' + x + '`' for x in source['missing_metrics'])}. Значения не подставляются.", ""])
    lines.extend([
        "## 3. Интерпретация и ограничения",
        "",
        "Метрики разных задач **не суммируются и не усредняются**. Доля корректных классификаций, доля релевантных документов в top-k и validation loss имеют разные знаменатели и разные направления интерпретации.",
        "",
        "`PASS` означает только полноту извлечения **по данному конфигурационному контракту**, не повторную независимую оценку модели, не production readiness и не качество генерации.",
        "",
    ])
    lines.extend(f"- {_md_escape(text)}" for text in report["limitations"])
    lines.extend(["", "## 4. Метод воспроизведения", "", "`python -m scripts.f5_unified_report --check` сверяет текущие файлы с сохранённым отчётом. Несовпадение SHA-256 или значений требует повторной сборки.", ""])
    return "\n".join(lines)


def report_paths(root: Path, config_path: str) -> tuple[Path, Path]:
    config = _load_config(root, config_path)
    return (_safe_path(root, config["outputs"]["json"], "outputs.json"), _safe_path(root, config["outputs"]["markdown"], "outputs.markdown"))


def run(root: Path, config_path: str, *, check: bool = False, require_complete: bool = False) -> dict[str, Any]:
    """Записать отчёт либо проверить его побайтовое соответствие источникам."""
    report = build_report(root, config_path)
    if require_complete and report["status"] != "PASS":
        raise EvaluationError("Неполный охват источников: статус PARTIAL")
    json_path, md_path = report_paths(root, config_path)
    artifacts = ((json_path, serialize_json(report)), (md_path, render_markdown(report)))
    if check:
        for path, expected in artifacts:
            if not path.is_file() or path.read_bytes() != expected.encode("utf-8"):
                raise EvaluationError(f"Отчёт отсутствует или устарел: {path.relative_to(root.resolve())}")
    else:
        for path, output in artifacts:
            path.parent.mkdir(parents=True, exist_ok=True)
            # Временный файл и атомарная замена исключают неполную публикацию при записи.
            temporary = path.with_name(path.name + ".tmp")
            try:
                temporary.write_bytes(output.encode("utf-8"))
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
    return report


def numeric_paths(root: Path, config_path: str) -> dict[str, list[str]]:
    """Показать только адреса числовых полей (не их значения) для уточнения selectors."""
    config = _load_config(root, config_path)
    output: dict[str, list[str]] = {}
    for source in config["sources"]:
        key = _validate_id(source.get("id"), "id")
        path = _safe_path(root, source.get("path"), "source.path")
        if not path.is_file():
            output[key] = []
            continue
        data = _read_json(path)
        result: list[str] = []

        def visit(node: Any, prefix: str = "", depth: int = 0) -> None:
            if depth > 9 or len(result) >= 200:
                return
            if isinstance(node, dict):
                for subkey, value in node.items():
                    if isinstance(subkey, str) and not subkey.startswith("_") and "." not in subkey:
                        visit(value, f"{prefix}.{subkey}" if prefix else subkey, depth + 1)
            elif isinstance(node, list):
                for i, value in enumerate(node[:5]):
                    visit(value, f"{prefix}.{i}", depth + 1)
            elif isinstance(node, (int, float)) and not isinstance(node, bool) and math.isfinite(node):
                result.append(prefix)

        visit(data)
        output[key] = result
    return output
