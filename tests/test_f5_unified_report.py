"""Unit/contract-тесты F5.1. Все численные входы синтетические."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.f5_unified_report import main
from src.evaluation_f5.unified_report import EvaluationError, build_report, numeric_paths, run


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def make_test_config() -> dict:
    return {
        "schema_version": 1,
        "outputs": {"json": "reports/public/f5_01/F5_01_unified_evaluation.json", "markdown": "docs/reports/F5_01_UNIFIED_EVALUATION.md"},
        "sources": [
            {
                "id": phase.lower(),
                "phase": phase,
                "task": task,
                "path": f"artifacts/{phase.lower()}.json",
                "split": "test", "dataset": "synthetic_fixture", "protocol": "synthetic_contract",
                "metrics": [{"name": metric, "unit": unit, "selectors": [selector]}],
            }
            for phase, task, metric, unit, selector in [
                ("F1", "classification", "accuracy", "fraction", "test.accuracy"),
                ("F2", "classification", "macro_f1", "fraction", "test.macro_f1"),
                ("F3", "retrieval", "recall_at_5", "fraction", "retrieval.recall_at_5"),
                ("F4", "llm", "validation_loss_after", "loss", "validation.loss_after"),
            ]
        ],
    }


def setup_fixture(tmp_path: Path) -> dict:
    config = make_test_config()
    write_json(tmp_path / "configs/f5_01_unified_evaluation.json", config)
    for phase, data in {
        "f1": {"test": {"accuracy": 0.75}, "private_text": "не публиковать"},
        "f2": {"test": {"macro_f1": 0.5}},
        "f3": {"retrieval": {"recall_at_5": 0.25}},
        "f4": {"validation": {"loss_after": 2.0}},
    }.items():
        write_json(tmp_path / f"artifacts/{phase}.json", data)
    return config


def test_four_phases_and_hash_provenance(tmp_path: Path):
    setup_fixture(tmp_path)
    report = build_report(tmp_path)
    assert report["status"] == "PASS"
    assert report["summary"]["phases_with_metrics"] == ["F1", "F2", "F3", "F4"]
    assert report["summary"]["metric_count"] == 4
    assert report["methodology"]["aggregation"] == "none_cross_task"
    values = [s["metrics"][0]["value"] for s in report["sources"]]
    assert values == [0.75, 0.5, 0.25, 2.0]
    src = tmp_path / "artifacts/f1.json"
    assert report["sources"][0]["sha256"] == hashlib.sha256(src.read_bytes()).hexdigest()
    assert "private_text" not in json.dumps(report, ensure_ascii=False)
    assert "не публиковать" not in json.dumps(report, ensure_ascii=False)


def test_write_idempotence_and_check(tmp_path: Path):
    setup_fixture(tmp_path)
    run(tmp_path, "configs/f5_01_unified_evaluation.json")
    out = tmp_path / "reports/public/f5_01/F5_01_unified_evaluation.json"
    doc = tmp_path / "docs/reports/F5_01_UNIFIED_EVALUATION.md"
    before = (out.read_bytes(), doc.read_bytes())
    assert b'"status": "PASS"' in before[0]
    assert "не суммируются" in before[1].decode("utf-8")
    assert run(tmp_path, "configs/f5_01_unified_evaluation.json", check=True)["status"] == "PASS"
    run(tmp_path, "configs/f5_01_unified_evaluation.json")
    assert before == (out.read_bytes(), doc.read_bytes())
    write_json(tmp_path / "artifacts/f1.json", {"test": {"accuracy": 0.9}})
    with pytest.raises(EvaluationError, match="устарел"):
        run(tmp_path, "configs/f5_01_unified_evaluation.json", check=True)


def test_missing_artifact_is_partial_not_fabricated(tmp_path: Path):
    setup_fixture(tmp_path)
    (tmp_path / "artifacts/f3.json").unlink()
    report = build_report(tmp_path)
    assert report["status"] == "PARTIAL"
    assert report["summary"]["phases_without_metrics"] == ["F3"]
    assert report["sources"][2]["metrics"] == []
    assert report["sources"][2]["status"] == "MISSING_SOURCE"
    with pytest.raises(EvaluationError, match="PARTIAL"):
        run(tmp_path, "configs/f5_01_unified_evaluation.json", require_complete=True)


def test_missing_selector_is_partial(tmp_path: Path):
    config = setup_fixture(tmp_path)
    config["sources"][0]["metrics"][0]["selectors"] = ["wrong.path"]
    write_json(tmp_path / "configs/f5_01_unified_evaluation.json", config)
    report = build_report(tmp_path)
    assert report["status"] == "PARTIAL"
    assert report["sources"][0]["missing_metrics"] == ["accuracy"]


@pytest.mark.parametrize("bad", [True, "0.8", -0.1, 1.1, float("nan")])
def test_fraction_rejects_invalid_input(tmp_path: Path, bad):
    setup_fixture(tmp_path)
    write_json(tmp_path / "artifacts/f1.json", {"test": {"accuracy": bad}})
    with pytest.raises(EvaluationError):
        build_report(tmp_path)


def test_conflicting_selectors_rejected(tmp_path: Path):
    config = setup_fixture(tmp_path)
    config["sources"][0]["metrics"][0]["selectors"] = ["test.accuracy", "other.accuracy"]
    write_json(tmp_path / "configs/f5_01_unified_evaluation.json", config)
    write_json(tmp_path / "artifacts/f1.json", {"test": {"accuracy": 0.75}, "other": {"accuracy": 0.80}})
    with pytest.raises(EvaluationError, match="неоднозначные"):
        build_report(tmp_path)


@pytest.mark.parametrize("path", ["../escape.json", "/tmp/absolute.json", "C:/external.json", "reports/../escape.json", "sub//file.json"])
def test_source_path_traversal_rejected(tmp_path: Path, path: str):
    config = setup_fixture(tmp_path)
    config["sources"][0]["path"] = path
    write_json(tmp_path / "configs/f5_01_unified_evaluation.json", config)
    with pytest.raises(EvaluationError):
        build_report(tmp_path)


def test_symlink_leaves_root_rejected(tmp_path: Path):
    config = setup_fixture(tmp_path)
    outsider = tmp_path.parent / f"f5_outside_{tmp_path.name}.json"
    outsider.write_text("{}", encoding="utf-8")
    try:
        link = tmp_path / "artifacts/outside.json"
        try:
            link.symlink_to(outsider)
        except (OSError, NotImplementedError):
            pytest.skip("Создание symlink недоступно в данном окружении")
        config["sources"][0]["path"] = "artifacts/outside.json"
        write_json(tmp_path / "configs/f5_01_unified_evaluation.json", config)
        with pytest.raises(EvaluationError, match="за пределы"):
            build_report(tmp_path)
    finally:
        outsider.unlink(missing_ok=True)


def test_wrong_task_phase_rejected(tmp_path: Path):
    config = setup_fixture(tmp_path)
    config["sources"][1]["task"] = "llm"
    write_json(tmp_path / "configs/f5_01_unified_evaluation.json", config)
    with pytest.raises(EvaluationError, match="соответствует фазе"):
        build_report(tmp_path)


def test_cli_check_and_inspection(tmp_path: Path, capsys):
    setup_fixture(tmp_path)
    cli = ["--root", str(tmp_path)]
    assert main(cli) == 0
    assert main(cli + ["--check"]) == 0
    assert main(cli + ["--inspect-paths"]) == 0
    capture = capsys.readouterr()
    assert "test.accuracy" in capture.out
    assert "0.75" not in capture.out  # Режим inspection скрывает значения


def test_output_path_traversal_rejected(tmp_path: Path):
    config = setup_fixture(tmp_path)
    config["outputs"]["json"] = "../wrong.json"
    write_json(tmp_path / "configs/f5_01_unified_evaluation.json", config)
    with pytest.raises(EvaluationError):
        run(tmp_path, "configs/f5_01_unified_evaluation.json")


def test_published_f2_f3_selectors_match_actual_saved_json_shapes(tmp_path: Path):
    """Регрессия схем F2.6/F3.7: реальные имена полей, числа — фикстуры."""
    root = Path(__file__).resolve().parents[1]
    config_path = 'configs/f5_01_unified_evaluation.json'
    dst = tmp_path / config_path
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes((root / config_path).read_bytes())
    sources = {
        'reports/classical.json': {'test': {'accuracy': .8, 'macro_f1': .75,
                                            'macro_precision': .76, 'macro_recall': .74}},
        'outputs/f2_6_diagnostics/metrics.json': {
            'format_version': 1, 'best_val_macro_f1': .69, 'best_epoch': 5,
            'epochs': 5, 'test': {'macro_f1': .99},
            'history': [{'epoch': 5, 'val_accuracy': .7, 'val_macro_f1': .69}]},
        'reports/public/f3_07/F3_07_comparison.json': {
            'n_queries': 80, 'metrics': {
                'bm25': {'5': {'recall_at_k': .4, 'mrr_at_k': .3}},
                'vector': {'5': {'recall_at_k': .6, 'mrr_at_k': .45}},
                'hybrid': {'5': {'recall_at_k': .5, 'mrr_at_k': .35}}}},
        'data/processed/f4_06/cpu/summary.json': {
            'optimizer_steps': 1, 'train_loss_last': 2.39,
            'validation_loss_before': 2.28, 'validation_loss_after': 2.27,
            'trainable_parameters': 540672, 'learning_rate_configured': .0002},
    }
    for path, content in sources.items():
        write_json(tmp_path / path, content)
    report = build_report(tmp_path)
    assert report['status'] == 'PASS'
    assert report['summary']['sources_with_metrics'] == 4
    by_source = {src['source_id']: src for src in report['sources']}
    f2 = by_source['f2_pytorch_diagnostics']
    assert f2['split'] == 'validation'
    assert not f2['missing_metrics']
    assert {m['name']: m['value'] for m in f2['metrics']} == {
        'macro_f1': .69, 'best_epoch': 5, 'epochs': 5,
    }
    f3 = by_source['f3_retrieval_comparison']
    assert not f3['missing_metrics']
    assert len(f3['metrics']) == 6
    assert {m['name']: m['value'] for m in f3['metrics']} == {
        'bm25_recall_at_5': .4, 'vector_recall_at_5': .6,
        'hybrid_recall_at_5': .5, 'bm25_mrr_at_5': .3,
        'vector_mrr_at_5': .45, 'hybrid_mrr_at_5': .35,
    }
    assert all('.5.' in m['selector'] for m in f3['metrics'])
