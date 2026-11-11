"""F5.8 — CLI-демонстрация на замороженной синтетике без сетевых вызовов.

Одна команда выполняет настоящие алгоритмы F5.4 и читает сводные
метрики F1–F4 через F5.1. Никакого обучения LLM, HTTP, Docker, MLflow
или перезаписи прежних экспериментов здесь нет.
"""

from __future__ import annotations

from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any

from src.api_f5.demo_service import DemoEngine, load_demo
from src.evaluation_f5.unified_report import EvaluationError, build_report

CONFIG_PATH = "configs/f5_08_offline_demo.json"
JSON_NAME = "F5_08_offline_demo.json"
MD_NAME = "F5_08_offline_demo.md"
_SHA = re.compile(r"^[0-9a-f]{64}$")
_ID = re.compile(r"^[a-z][a-z0-9_]{2,79}$")
_EXPECTED_IDS = (
    "classify_python", "classify_linux", "classify_sql", "classify_ml",
    "classify_unknown", "search_sql", "search_linux", "search_unknown",
    "ask_python", "ask_unknown",
)
_ALLOWED = {"classify", "search", "ask"}
_MAX_CONFIG_BYTES = 32_768


class OfflineDemoError(ValueError):
    """Ошибка конфигурации, проверки данных или сохранённых результатов."""


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Отвергать повторяющиеся ключи JSON."""
    result: dict[str, Any] = {}
    for key, val in pairs:
        if key in result:
            raise OfflineDemoError(f"Повторён JSON-ключ: {key}")
        result[key] = val
    return result


def _invalid_constant(value: str) -> None:
    raise OfflineDemoError(f"Нечисловая JSON-константа: {value}")


def _json_file(path: Path, size_limit: int = 512_000) -> Any:
    try:
        if path.stat().st_size > size_limit:
            raise OfflineDemoError(f"Файл слишком большой: {path.name}")
        return json.loads(path.read_text(encoding="utf-8-sig"),
                          object_pairs_hook=_pairs, parse_constant=_invalid_constant)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise OfflineDemoError(f"Не удалось прочитать JSON {path.name}: {exc}") from exc


def load_config(root: Path) -> tuple[dict[str, Any], str]:
    """Прочитать фиксированный профиль и проверить строгий контракт сценариев."""
    root = root.resolve()
    path = (root / CONFIG_PATH).resolve()
    if not path.is_relative_to(root):
        raise OfflineDemoError("Конфигурация вышла за корень репозитория")
    cfg = _json_file(path, _MAX_CONFIG_BYTES)
    if not isinstance(cfg, dict) or set(cfg) != {"schema_version", "demo_id", "fixture_id", "fixture_sha256", "cases"}:
        raise OfflineDemoError("Неподдерживаемая схема F5.8")
    if (type(cfg["schema_version"]) is not int or cfg["schema_version"] != 1
            or cfg["demo_id"] != "f5_08_offline_synthetic_v1"
            or cfg["fixture_id"] != "f5_04_api_synthetic_v1"
            or not isinstance(cfg["fixture_sha256"], str)
            or not _SHA.fullmatch(cfg["fixture_sha256"])):
        raise OfflineDemoError("Неверная версия профиля / fixture")
    cases = cfg["cases"]
    if not isinstance(cases, list) or len(cases) != len(_EXPECTED_IDS):
        raise OfflineDemoError("Требуются ровно 10 фиксированных сценариев")
    if tuple(x.get("id") for x in cases if isinstance(x, dict)) != _EXPECTED_IDS:
        raise OfflineDemoError("Порядок и идентификаторы сценариев изменены")
    for case in cases:
        if not isinstance(case, dict) or not isinstance(case.get("id"), str) or not _ID.fullmatch(case["id"]):
            raise OfflineDemoError("Неверный идентификатор сценария")
        task = case.get("task")
        if task not in _ALLOWED:
            raise OfflineDemoError(f"Неверный тип сценария {case['id']}")
        expected_keys = {"id", "task", "text", "expect"} | ({"top_k"} if task == "search" else set())
        if set(case) != expected_keys:
            raise OfflineDemoError(f"Неправильная схема сценария {case['id']}")
        if not isinstance(case["text"], str) or not 1 <= len(case["text"]) <= 500 or case["text"].strip() != case["text"]:
            raise OfflineDemoError(f"Неверный запрос {case['id']}")
        exp = case["expect"]
        if not isinstance(exp, dict):
            raise OfflineDemoError(f"Неверные ожидаемые данные {case['id']}")
        if task == "classify":
            if (set(exp) != {"status", "label"} or exp["status"] not in {"predicted", "no_features"}
                    or exp["label"] not in {None, "python", "linux", "sql", "ml"}
                    or (exp["status"] == "no_features") != (exp["label"] is None)):
                raise OfflineDemoError(f"Неверная проверка классификации {case['id']}")
        elif task == "search":
            if (type(case["top_k"]) is not int or not 1 <= case["top_k"] <= 5
                    or set(exp) != {"first_doc_id"} or
                    exp["first_doc_id"] is not None and
                    (not isinstance(exp["first_doc_id"], str) or not _ID.fullmatch(exp["first_doc_id"]))):
                raise OfflineDemoError(f"Неверная проверка поиска {case['id']}")
        else:
            if (set(exp) != {"mode", "doc_id"} or exp["mode"] not in {"extractive_demo", "no_evidence"}
                    or (exp["mode"] == "no_evidence") != (exp["doc_id"] is None)
                    or (exp["doc_id"] is not None and
                        (not isinstance(exp["doc_id"], str) or not _ID.fullmatch(exp["doc_id"])))):
                raise OfflineDemoError(f"Неверная проверка ответа {case['id']}")
    return cfg, sha256(path.read_bytes()).hexdigest()


def _finite_number(value: Any) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        raise OfflineDemoError("Нечисловой результат inference")
    return round(float(value), 8)


def evaluate_cases(engine: DemoEngine, cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Выполнить 10 сценариев и независимо проверить дословное доказательство."""
    rows: list[dict[str, Any]] = []
    docs = {d.doc_id: d for d in engine.docs}
    for case in cases:
        task, exp, text = case["task"], case["expect"], case["text"]
        if task == "classify":
            prediction = engine.classify(text)
            if prediction["backend"] != "demo_tfidf_logreg" or prediction["fixture_id"] != engine.fixture_id:
                raise OfflineDemoError("Некорректный backend классификации")
            actual = {"status": prediction["status"], "label": prediction["label"],
                      "score": None if prediction["score"] is None else _finite_number(prediction["score"])}
            passed = all(actual[k] == v for k, v in exp.items())
        elif task == "search":
            hits = engine.search(text, case["top_k"])
            for hit in hits:
                if hit.doc.doc_id not in docs or not math.isfinite(hit.score) or hit.score <= 0:
                    raise OfflineDemoError("Поиск вернул неизвестный документ/невалидный score")
            actual = {"top_doc_id": hits[0].doc.doc_id if hits else None,
                      "hits": [{"doc_id": hit.doc.doc_id, "score": _finite_number(hit.score)} for hit in hits]}
            passed = actual["top_doc_id"] == exp["first_doc_id"]
        else:
            answer = engine.ask(text)
            if answer["backend"] != "demo_extractive_bm25" or answer["fixture_id"] != engine.fixture_id:
                raise OfflineDemoError("Некорректный backend ответа")
            evidence = answer["evidence"]
            verified_evidence = False
            if answer["mode"] == "extractive_demo" and len(evidence) == 1:
                e = evidence[0]
                doc = docs.get(e.get("doc_id"))
                verified_evidence = bool(
                    doc is not None and e.get("source_id") == doc.source_id
                    and isinstance(e.get("start"), int) and not isinstance(e.get("start"), bool)
                    and isinstance(e.get("end"), int) and not isinstance(e.get("end"), bool)
                    and 0 <= e["start"] < e["end"] <= len(doc.text)
                    and doc.text[e["start"]:e["end"]] == e.get("quote") == answer["answer"]
                    and sha256(doc.text.encode("utf-8")).hexdigest() == e.get("text_sha256")
                )
            elif answer["mode"] == "no_evidence":
                verified_evidence = answer["answer"] is None and evidence == []
            actual = {"mode": answer["mode"],
                      "doc_id": evidence[0]["doc_id"] if evidence else None,
                      "answer": answer["answer"], "evidence_valid": verified_evidence}
            passed = (actual["mode"] == exp["mode"] and actual["doc_id"] == exp["doc_id"]
                      and verified_evidence)
        rows.append({"case_id": case["id"], "task": task, "passed": bool(passed), "expected": exp, "actual": actual})
    return rows


def _lineage(root: Path) -> dict[str, Any]:
    """Проверить только агрегаты F5.1, без F1–F4 обучения и без MLflow I/O."""
    current = build_report(root)
    path = root / "reports/public/f5_01/F5_01_unified_evaluation.json"
    if not path.is_file():
        f5_01 = "NOT_AVAILABLE"
    else:
        previous = _json_file(path)
        f5_01 = "VERIFIED" if previous == current else "STALE"
    phases: dict[str, dict[str, Any]] = {}
    for phase in ("F1", "F2", "F3", "F4"):
        sources = [s for s in current["sources"] if s["phase"] == phase]
        phases[phase] = {
            "status": "METRICS_PRESENT" if sources and all(s["status"] == "EXTRACTED" and not s["missing_metrics"] for s in sources) else "PARTIAL",
            "sources": [{"source_id": s["source_id"], "status": s["status"],
                         "sha256": s["sha256"], "metric_count": len(s["metrics"]),
                         "split": s["split"], "protocol": s["protocol"]}
                        for s in sources],
        }
    return {"F1_to_F4": phases, "F5_01": {"status": f5_01,
                                           "source_count": current["summary"]["source_count"],
                                           "sources_with_metrics": current["summary"]["sources_with_metrics"],
                                           "metric_count": current["summary"]["metric_count"]},
            "other_F5": {
                "F5_02": "NOT_EXECUTED_IN_DEMO",
                "F5_03": "NOT_EXECUTED_IN_DEMO",
                "F5_04": "DEMO_SCENARIOS_EXECUTED",
                "F5_05": "DOCKER_NOT_EXECUTED_IN_DEMO",
                "F5_06": "PYTEST_NOT_EXECUTED_IN_DEMO",
                "F5_07": "MLFLOW_NOT_EXECUTED_IN_DEMO",
            }}


def build_demo(root: Path) -> dict[str, Any]:
    """Рассчитать все результаты, не создавая файлы и не используя сеть."""
    root = root.resolve()
    cfg, config_sha = load_config(root)
    api_cfg, fixture = load_demo(root)
    if api_cfg["fixture_sha256"] != cfg["fixture_sha256"] or fixture["fixture_id"] != cfg["fixture_id"]:
        raise OfflineDemoError("Конфигурация F5.8 не соответствует F5.4")
    engine = DemoEngine(api_cfg, fixture)
    rows = evaluate_cases(engine, cfg["cases"])
    passed = sum(r["passed"] for r in rows)
    lineage = _lineage(root)
    return {
        "schema_version": 1,
        "stage": "F5.8",
        "demo_id": cfg["demo_id"],
        "backend": "synthetic_demo",
        "network": "NOT_USED_BY_PIPELINE",
        "llm_inference": "NOT_EXECUTED",
        "config_sha256": config_sha,
        "fixture_sha256": cfg["fixture_sha256"],
        "case_summary": {"total": len(rows), "passed": passed, "failed": len(rows)-passed,
                         "status": "PASS" if passed == len(rows) else "FAIL"},
        "cases": rows,
        "lineage": lineage,
        "interpretation": {
            "demo": "Алгоритмы F5.4 исполнены на синтетических данных без HTTP и скачивания моделей.",
            "f1_f4": "Прочитаны только агрегаты сохранённых метрик; обучение и инференс F1–F4 не повторялись.",
            "f5_01": "Статус VERIFIED означает совпадение сохранённого и заново рассчитанного отчётов F5.1.",
            "limits": "Результат 10/10 не подтверждает пригодность модели для внешних данных или безопасность LLM.",
        },
    }


def json_bytes(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def markdown_bytes(report: dict[str, Any]) -> bytes:
    lines = [
        "# F5.8 — Offline CLI Demo: результаты",
        "",
        f"**Демонстрация:** `{report['case_summary']['status']}`; проверок: "
        f"{report['case_summary']['passed']}/{report['case_summary']['total']}; "
        "backend: `synthetic_demo`, LLM: `NOT EXECUTED`.",
        "",
        "## 1. Демонстрационные сценарии",
        "",
        "| ID | Метод | Проверка |",
        "|---|---|---|",
    ]
    for row in report["cases"]:
        lines.append(f"| `{row['case_id']}` | `{row['task']}` | {'PASS' if row['passed'] else 'FAIL'} |")
    lines += ["", "## 2. Связь F1–F4 с проверяемыми метриками", "",
              "| Фаза | Статус | Источников |", "|---|---|---:|"]
    for phase, entry in report["lineage"]["F1_to_F4"].items():
        lines.append(f"| {phase} | `{entry['status']}` | {len(entry['sources'])} |")
    f5 = report["lineage"]["F5_01"]
    lines += ["", f"F5.1: `{f5['status']}`; источников с метриками: "
              f"{f5['sources_with_metrics']}/{f5['source_count']}; метрик: {f5['metric_count']}.",
              "", "## 3. Не выполнялось в CLI", "",
              "- Не запускались Qwen, Docker, MLflow, HTTP-сервер и полная регрессия pytest.",
              "- F5.3 safety-контракт не является доказательством безопасности LLM.",
              "- Сводные метрики F1–F4 не объединяются в единый score и не пересчитываются на новых данных.",
              "", "## 4. Воспроизводимость", "",
              f"- SHA-256 конфигурации F5.8: `{report['config_sha256']}`.",
              f"- SHA-256 synthetic fixture F5.4: `{report['fixture_sha256']}`.",
              "- `--check` пересчитывает результат и сравнивает побайтово JSON и Markdown.",
              "- Результаты сохраняются вне репозитория; `run` не заменяет существующий каталог.",
              ""]
    return "\n".join(lines).encode("utf-8")


def _output_path(root: Path, output: Path) -> Path:
    """Запретить запись в репозиторий и в источники через symlink."""
    base = root.resolve()
    absolute = output.expanduser().resolve()
    if absolute == base or absolute.is_relative_to(base):
        raise OfflineDemoError("Выходной каталог должен находиться вне репозитория")
    if absolute == base.parent:
        raise OfflineDemoError("Нельзя использовать родительскую директорию репозитория как выходной каталог")
    if output.is_symlink():
        raise OfflineDemoError("Выходной каталог не должен быть symlink")
    return absolute


def run(root: Path, output: Path, *, check: bool = False) -> dict[str, Any]:
    """Создать новую изолированную поставку или проверить её без записи."""
    root = root.resolve()
    destination = _output_path(root, output)
    report = build_demo(root)
    if report["case_summary"]["status"] != "PASS":
        raise OfflineDemoError("Контрольные сценарии не прошли; выдача результата заблокирована")
    expected = {JSON_NAME: json_bytes(report), MD_NAME: markdown_bytes(report)}
    if check:
        for name, content in expected.items():
            target = destination / name
            try:
                if not target.is_file() or target.read_bytes() != content:
                    raise OfflineDemoError(f"Отчёт отсутствует, изменён или устарел: {name}")
            except OSError as exc:
                raise OfflineDemoError(f"Ошибка проверки {name}: {exc}") from exc
        return report
    if destination.exists():
        raise OfflineDemoError("Выходной каталог уже существует: используйте check или новый --output-dir")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".f5_08_tmp_", dir=destination.parent))
    try:
        for name, data in expected.items():
            (temporary / name).write_bytes(data)
        temporary.replace(destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return report
