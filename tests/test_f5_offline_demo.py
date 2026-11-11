"""F5.8 — воспроизводимость, офлайн-исполнение и отрицательные проверки."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import shutil
import socket

import pytest

from scripts.f5_offline_demo import main as cli_main
from src.api_f5.demo_service import DemoEngine, load_demo
from src.demo_f5.offline_demo import (
    CONFIG_PATH, JSON_NAME, MD_NAME, OfflineDemoError, _lineage, build_demo,
    evaluate_cases, json_bytes, load_config, markdown_bytes, run,
)
from src.evaluation_f5.unified_report import build_report


ROOT = Path(__file__).resolve().parents[1]
API_CONFIG = "configs/f5_04_api.json"
API_FIXTURE = "tests/fixtures/f5_04_api_demo_v1.json"
EVAL_CONFIG = "configs/f5_01_unified_evaluation.json"


def sandbox(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    for relative in (CONFIG_PATH, API_CONFIG, API_FIXTURE, EVAL_CONFIG):
        dest = root / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, dest)
    return root


def edit_config(root: Path, fn) -> None:
    target = root / CONFIG_PATH
    obj = json.loads(target.read_text(encoding="utf-8"))
    fn(obj)
    target.write_text(json.dumps(obj, ensure_ascii=False) + "\n", encoding="utf-8")


def mini_metrics(root: Path) -> None:
    """Изолированные тестовые метрики, никогда не выдаваемые за реальный F1–F4."""
    names = (
        ("f1_fixture", "F1", "classification", "reports/f1_fixture.json", "test"),
        ("f2_fixture", "F2", "classification", "reports/f2_fixture.json", "validation"),
        ("f3_fixture", "F3", "retrieval", "reports/f3_fixture.json", "frozen_test"),
        ("f4_fixture", "F4", "llm", "reports/f4_fixture.json", "validation"),
    )
    specs = []
    for ident, phase, task, path, split in names:
        p = root / path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text('{"score":0.5}\n', encoding="utf-8")
        specs.append({"id": ident, "phase": phase, "task": task, "path": path,
                      "split": split, "dataset": "fixture", "protocol": "synthetic_test",
                      "metrics": [{"name": "score", "unit": "fraction", "selectors": ["score"]}]})
    config = {"schema_version": 1, "sources": specs, "outputs": {
        "json": "reports/public/f5_01/F5_01_unified_evaluation.json",
        "markdown": "docs/reports/F5_01_UNIFIED_EVALUATION.md"}}
    (root / EVAL_CONFIG).write_text(json.dumps(config, ensure_ascii=False), encoding="utf-8")


@pytest.fixture(scope="module")
def engine() -> DemoEngine:
    cfg, fixture = load_demo(ROOT)
    return DemoEngine(cfg, fixture)


def test_config_fixed_case_count() -> None:
    cfg, digest = load_config(ROOT)
    assert len(cfg["cases"]) == 10
    assert digest == sha256((ROOT / CONFIG_PATH).read_bytes()).hexdigest()


@pytest.mark.parametrize("case_id", [
    "classify_python", "classify_linux", "classify_sql", "classify_ml",
    "classify_unknown", "search_sql", "search_linux", "search_unknown",
    "ask_python", "ask_unknown",
])
def test_each_case_success(engine: DemoEngine, case_id: str) -> None:
    cfg, _ = load_config(ROOT)
    case = next(x for x in cfg["cases"] if x["id"] == case_id)
    result = evaluate_cases(engine, [case])
    assert result[0]["passed"] is True


def test_output_has_no_faked_llm_success() -> None:
    report = build_demo(ROOT)
    assert report["case_summary"] == {"total": 10, "passed": 10, "failed": 0, "status": "PASS"}
    assert report["llm_inference"] == "NOT_EXECUTED"
    assert report["backend"] == "synthetic_demo"
    assert report["network"] == "NOT_USED_BY_PIPELINE"
    assert "overall_ml_score" not in report


def test_evidence_independently_verified() -> None:
    report = build_demo(ROOT)
    row = next(r for r in report["cases"] if r["case_id"] == "ask_python")
    assert row["actual"]["evidence_valid"] is True
    assert row["actual"]["answer"]


def test_no_evidence_does_not_hallucinate() -> None:
    report = build_demo(ROOT)
    row = next(r for r in report["cases"] if r["case_id"] == "ask_unknown")
    assert row["actual"]["answer"] is None
    assert row["actual"]["evidence_valid"] is True


def test_lineage_missing_is_explicit(tmp_path: Path) -> None:
    root = sandbox(tmp_path)
    entry = _lineage(root)
    assert entry["F5_01"]["status"] == "NOT_AVAILABLE"
    assert entry["F5_01"]["sources_with_metrics"] == 0
    assert all(x["status"] == "PARTIAL" for x in entry["F1_to_F4"].values())


def test_lineage_validated_not_assumed(tmp_path: Path) -> None:
    root = sandbox(tmp_path)
    mini_metrics(root)
    before = _lineage(root)
    assert before["F5_01"]["status"] == "NOT_AVAILABLE"
    p = root / "reports/public/f5_01/F5_01_unified_evaluation.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(build_report(root), ensure_ascii=False), encoding="utf-8")
    after = _lineage(root)
    assert after["F5_01"]["status"] == "VERIFIED"
    assert after["F5_01"]["sources_with_metrics"] == 4
    assert all(x["status"] == "METRICS_PRESENT" for x in after["F1_to_F4"].values())
    # Любое изменение источника делает прежний F5.1-отчёт устаревшим.
    (root / "reports/f2_fixture.json").write_text('{"score":0.6}\n', encoding="utf-8")
    assert _lineage(root)["F5_01"]["status"] == "STALE"


def test_no_network_calls(monkeypatch, tmp_path: Path) -> None:
    def forbidden(*args, **kwargs):
        raise AssertionError("Сетевое соединение запрещено для F5.8")
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    root = sandbox(tmp_path)
    result = run(root, tmp_path / "out")
    assert result["case_summary"]["status"] == "PASS"


def test_run_check_idempotent(tmp_path: Path) -> None:
    root = sandbox(tmp_path)
    output = tmp_path / "out"
    first = run(root, output)
    assert (output / JSON_NAME).is_file() and (output / MD_NAME).is_file()
    assert run(root, output, check=True) == first
    assert (output / JSON_NAME).read_bytes() == json_bytes(first)
    assert (output / MD_NAME).read_bytes() == markdown_bytes(first)


def test_reject_overwrite(tmp_path: Path) -> None:
    root = sandbox(tmp_path)
    output = tmp_path / "out"
    run(root, output)
    with pytest.raises(OfflineDemoError, match="уже существует"):
        run(root, output)


def test_check_missing_reports(tmp_path: Path) -> None:
    root = sandbox(tmp_path)
    with pytest.raises(OfflineDemoError, match="отсутствует"):
        run(root, tmp_path / "absent", check=True)


@pytest.mark.parametrize("name", [JSON_NAME, MD_NAME])
def test_check_tamper_reports(tmp_path: Path, name: str) -> None:
    root = sandbox(tmp_path)
    output = tmp_path / "out"
    run(root, output)
    target = output / name
    target.write_bytes(target.read_bytes() + b"#altered")
    with pytest.raises(OfflineDemoError, match="изменён"):
        run(root, output, check=True)


def test_check_detects_config_edit(tmp_path: Path) -> None:
    root = sandbox(tmp_path)
    output = tmp_path / "out"
    run(root, output)
    edit_config(root, lambda c: c["cases"][0].update(text="Python enumerate start индекс?"))
    with pytest.raises(OfflineDemoError, match="устарел"):
        run(root, output, check=True)


def test_check_detects_f1_to_f4_metric_change(tmp_path: Path) -> None:
    root = sandbox(tmp_path)
    mini_metrics(root)
    output = tmp_path / "out"
    run(root, output)
    (root / "reports/f2_fixture.json").write_text('{"score":0.75}\n', encoding="utf-8")
    with pytest.raises(OfflineDemoError, match="устарел"):
        run(root, output, check=True)


def test_reject_output_inside_repo(tmp_path: Path) -> None:
    root = sandbox(tmp_path)
    with pytest.raises(OfflineDemoError, match="вне репозитория"):
        run(root, root / "reports" / "output")
    assert not (root / "reports/output").exists()


def test_reject_symlink_output(tmp_path: Path) -> None:
    root = sandbox(tmp_path)
    real = tmp_path / "real_out"
    real.mkdir()
    link = tmp_path / "link_out"
    try:
        link.symlink_to(real, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("Создание symlink недоступно в данном окружении")
    with pytest.raises(OfflineDemoError, match="symlink"):
        run(root, link)


def test_reject_fixture_mutation(tmp_path: Path) -> None:
    root = sandbox(tmp_path)
    fixture = root / API_FIXTURE
    fixture.write_bytes(fixture.read_bytes() + b"\n")
    with pytest.raises(Exception, match="SHA-256"):
        run(root, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_reject_wrong_expected_class(tmp_path: Path) -> None:
    root = sandbox(tmp_path)
    edit_config(root, lambda c: c["cases"][0]["expect"].update(label="linux"))
    report = build_demo(root)
    assert report["case_summary"]["status"] == "FAIL"
    with pytest.raises(OfflineDemoError, match="не прошли"):
        run(root, tmp_path / "out")


def test_reject_duplicate_json_keys(tmp_path: Path) -> None:
    root = sandbox(tmp_path)
    path = root / CONFIG_PATH
    path.write_text(path.read_text(encoding="utf-8").replace('"schema_version": 1,',
        '"schema_version": 1, "schema_version": 1,'), encoding="utf-8")
    with pytest.raises(OfflineDemoError, match="Повторён JSON-ключ"):
        load_config(root)


@pytest.mark.parametrize("bad", ["NaN", "Infinity"])
def test_reject_invalid_json_constants(tmp_path: Path, bad: str) -> None:
    root = sandbox(tmp_path)
    path = root / CONFIG_PATH
    path.write_text(path.read_text(encoding="utf-8").replace('"schema_version": 1', f'"schema_version": {bad}'), encoding="utf-8")
    with pytest.raises(OfflineDemoError, match="Нечисловая"):
        load_config(root)


def test_reject_unknown_demo_id(tmp_path: Path) -> None:
    root = sandbox(tmp_path)
    edit_config(root, lambda c: c.update(demo_id="other"))
    with pytest.raises(OfflineDemoError, match="Неверная версия"):
        load_config(root)


def test_reject_bad_case_count(tmp_path: Path) -> None:
    root = sandbox(tmp_path)
    edit_config(root, lambda c: c["cases"].pop())
    with pytest.raises(OfflineDemoError, match="ровно 10"):
        load_config(root)


def test_reject_duplicate_case_ids(tmp_path: Path) -> None:
    root = sandbox(tmp_path)
    edit_config(root, lambda c: c["cases"][1].update(id="classify_python"))
    with pytest.raises(OfflineDemoError, match="Порядок"):
        load_config(root)


def test_reject_extra_case_property(tmp_path: Path) -> None:
    root = sandbox(tmp_path)
    edit_config(root, lambda c: c["cases"][0].update(forbidden="value"))
    with pytest.raises(OfflineDemoError, match="схема"):
        load_config(root)


def test_reject_bool_topk(tmp_path: Path) -> None:
    root = sandbox(tmp_path)
    edit_config(root, lambda c: c["cases"][5].update(top_k=True))
    with pytest.raises(OfflineDemoError, match="проверка поиска"):
        load_config(root)


def test_reject_oversized_config(tmp_path: Path) -> None:
    root = sandbox(tmp_path)
    (root / CONFIG_PATH).write_bytes(b"x" * 32769)
    with pytest.raises(OfflineDemoError, match="слишком большой"):
        load_config(root)


def test_cli_run_and_check(tmp_path: Path, capsys) -> None:
    root = sandbox(tmp_path)
    output = tmp_path / "out"
    assert cli_main(["run", "--root", str(root), "--output-dir", str(output)]) == 0
    assert "synthetic=10/10" in capsys.readouterr().out
    assert cli_main(["check", "--root", str(root), "--output-dir", str(output)]) == 0
    assert "mode=check" in capsys.readouterr().out


def test_cli_rejects_overwrite(tmp_path: Path, capsys) -> None:
    root = sandbox(tmp_path)
    output = tmp_path / "out"
    assert cli_main(["run", "--root", str(root), "--output-dir", str(output)]) == 0
    assert cli_main(["run", "--root", str(root), "--output-dir", str(output)]) == 2
    assert "уже существует" in capsys.readouterr().err


def test_no_previous_artifacts_modified(tmp_path: Path) -> None:
    root = sandbox(tmp_path)
    mini_metrics(root)
    source = root / "reports/f4_fixture.json"
    before = sha256(source.read_bytes()).hexdigest()
    run(root, tmp_path / "out")
    assert sha256(source.read_bytes()).hexdigest() == before
    assert not (root / "reports/public/f5_08").exists()


def test_output_contains_only_two_reports(tmp_path: Path) -> None:
    root = sandbox(tmp_path)
    dest = tmp_path / "out"
    run(root, dest)
    assert {x.name for x in dest.iterdir()} == {JSON_NAME, MD_NAME}


def test_marker_not_real_benchmark(tmp_path: Path) -> None:
    root = sandbox(tmp_path)
    report = build_demo(root)
    assert report["lineage"]["other_F5"]["F5_07"] == "MLFLOW_NOT_EXECUTED_IN_DEMO"
    assert report["lineage"]["other_F5"]["F5_05"] == "DOCKER_NOT_EXECUTED_IN_DEMO"


def test_full_synthetic_lineage_without_network(tmp_path: Path, monkeypatch) -> None:
    root = sandbox(tmp_path)
    mini_metrics(root)
    evidence = root / "reports/public/f5_01/F5_01_unified_evaluation.json"
    evidence.parent.mkdir(parents=True, exist_ok=True)
    evidence.write_text(json.dumps(build_report(root)), encoding="utf-8")
    monkeypatch.setattr(socket.socket, "connect", lambda *a, **k: (_ for _ in ()).throw(AssertionError("network")))
    report = run(root, tmp_path / "out")
    assert report["lineage"]["F5_01"]["status"] == "VERIFIED"
    assert report["case_summary"]["passed"] == 10
