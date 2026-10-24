# Назначение файла: F1.9: проверки согласования аудит-файлов и формирования публичного отчёта.
# Примечание: комментарии не влияют на работу программы.

"""F1.9 offline synthetic tests. No real Q&A or model weights."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from scripts.build_f1_report import AUDIT_FILES, build_report, run, sha256
from src.reporting.f1 import F1ReportError, render_markdown, validate_audits

LABELS = ["databases", "linux", "programming", "systems"]


# Формирует согласованный синтетический комплект локальных аудитов F1.
def payload():
    matrix = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0]]
    # F1=1.0, 2/3, 1.0, 0.0 -> macro 2/3.
    digest = "a" * 64
    hashes = {s: digest for s in ("train", "val", "test")}
    return {
        "manifest": {"sources": [{"file": f"s{i}", "selected_rows": 3} for i in range(4)]},
        "cleaning": {"stage": "F1.4", "cleaning": {"input_rows": 12,
                     "output_rows": 12, "short_removed": 0,
                     "duplicates_removed": 0, "invalid_or_missing": 0},
                     "by_domain": {name: 3 for name in LABELS}},
        "eda": {"stage": "F1.5", "total_rows": 12, "source_sha256": digest,
                "class_distribution": {name: 3 for name in LABELS}},
        "split": {"stage": "F1.6", "total_rows": 12, "source_sha256": digest,
                  "group_count": 12, "near_duplicate_edges": 0, "near_duplicate_threshold": 0.92,
                  "split_counts": {s: 4 for s in ("train", "val", "test")},
                  "group_overlap": {s: 0 for s in ("train/test", "train/val", "val/test")},
                  "splits": {name: {s: 1 for s in ("train", "val", "test")} for name in LABELS}},
        "classical": {"stage": "F1.7", "fit_split": "train", "evaluation_splits": ["val", "test"],
                      "data_contract": {"split_counts": {s: 4 for s in ("train", "val", "test")}},
                      "split_source_sha256": hashes,
                      "test": {"n": 4, "labels": LABELS.copy(), "confusion_matrix": matrix,
                               "accuracy": .75, "macro_f1": 2 / 3, "weighted_f1": 2 / 3},
                      "val": {"n": 4, "labels": LABELS.copy(), "accuracy": .75, "macro_f1": 2/3}},
        "errors": {"stage": "F1.8", "frozen_model": "F1.7", "data_split": "test",
                   "n": 4, "correct": 3, "errors": 1, "accuracy": .75, "macro_f1": 2/3,
                   "labels": LABELS.copy(), "confusion_matrix": matrix,
                   "test_source_sha256": digest,
                   "confusion_pairs": [{"true": "systems", "predicted": "linux", "count": 1}]},
    }


# Проверяет успешную сверку всех аудит-файлов на согласованных данных.
def test_valid_reconciliation():
    report = validate_audits(**payload())
    assert report["cleaned"] == 12
    assert report["errors"] == 1
    assert report["top_confusion"]["count"] == 1


# Проверяет отсутствие исходных Q&A и ID в публичном Markdown.
def test_render_no_ids_no_raw_text():
    reports = payload()
    reports["errors"]["private_case_ids"] = ["SHOULD_NOT_LEAK"]
    text = render_markdown(validate_audits(**reports))
    assert "SHOULD_NOT_LEAK" not in text
    assert "Source" not in text or "F1" in text
    assert "systems → linux" in text
    assert "SHA-256" in text


# Проверяет отказ, когда аудит помечен неверным этапом.
@pytest.mark.parametrize("stage", ["cleaning", "eda", "split", "classical", "errors"])
def test_wrong_stage_rejected(stage):
    d = payload(); d[stage]["stage"] = "wrong"
    with pytest.raises(F1ReportError): validate_audits(**d)


# Проверяет отказ строить отчёт по противоречивым аудитам.
@pytest.mark.parametrize("case", [
    "bad_totals", "class_mismatch", "hash_mismatch", "group_leakage",
    "split_count", "train_only", "wrong_test", "matrix_mismatch", "wrong_accuracy",
    "wrong_f1", "wrong_pairs", "missing_pairs", "invalid_label", "not_finite",
    "wrong_test_hash", "duplicate_source",
])
def test_invalid_audit_fails_closed(case):
    d = payload()
    if case == "bad_totals": d["cleaning"]["cleaning"]["short_removed"] = 1
    if case == "class_mismatch": d["eda"]["class_distribution"]["linux"] += 1
    if case == "hash_mismatch": d["split"]["source_sha256"] = "b" * 64
    if case == "group_leakage": d["split"]["group_overlap"]["train/test"] = 1
    if case == "split_count": d["split"]["split_counts"]["train"] += 1
    if case == "train_only": d["classical"]["fit_split"] = "test"
    if case == "wrong_test": d["classical"]["test"]["n"] = 5
    if case == "matrix_mismatch": d["errors"]["confusion_matrix"][3][1] = 0
    if case == "wrong_accuracy": d["errors"]["accuracy"] = .3
    if case == "wrong_f1": d["errors"]["macro_f1"] = .5
    if case == "wrong_pairs": d["errors"]["confusion_pairs"][0]["count"] = 2
    if case == "missing_pairs": d["errors"]["confusion_pairs"] = []
    if case == "invalid_label": d["classical"]["test"]["labels"][0] = "<script>"
    if case == "not_finite": d["classical"]["test"]["accuracy"] = float("nan")
    if case == "wrong_test_hash": d["errors"]["test_source_sha256"] = "b" * 64
    if case == "duplicate_source": d["manifest"]["sources"][0]["file"] = "s1"
    with pytest.raises(F1ReportError): validate_audits(**d)


# Создаёт временные исходные отчёты и структуру проекта.
def fixture_root(tmp_path):
    d = payload()
    contents = b"SYNTHETIC fixture for F1.9 without any original questions\n"
    hexdigest = hashlib.sha256(contents).hexdigest()
    d["eda"]["source_sha256"] = hexdigest
    d["split"]["source_sha256"] = hexdigest
    d["errors"]["test_source_sha256"] = hexdigest
    d["classical"]["split_source_sha256"] = {x: hexdigest for x in ("train", "val", "test")}
    for n, rel in AUDIT_FILES.items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(d[n]), encoding="utf-8")
    for rel in ["data/interim/cleaned.jsonl", *[f"data/processed/{s}.jsonl" for s in ("train", "val", "test")]]:
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(contents)
    return d


# Проверяет построение отчёта и проверку --check без изменений.
def test_build_and_check_round_trip(tmp_path):
    fixture_root(tmp_path)
    assert run(root=tmp_path, expected_rows=12, expected_test=4).is_file()
    assert run(root=tmp_path, check=True).is_file()
    contents = (tmp_path / "docs/reports/F1_PHASE_REPORT.md").read_text(encoding="utf-8")
    assert "Train" in contents
    assert "test" in contents


# Проверяет одинаковый Markdown при одинаковых исходных аудитах.
def test_report_deterministic(tmp_path):
    fixture_root(tmp_path)
    first = build_report(tmp_path)
    second = build_report(tmp_path)
    assert first == second


# Проверяет обнаружение подмены контрольной суммы источника.
def test_hash_tamper_rejected(tmp_path):
    fixture_root(tmp_path)
    (tmp_path / "data/processed/test.jsonl").write_text("changed", encoding="utf-8")
    with pytest.raises(F1ReportError, match="test source SHA-256"):
        build_report(tmp_path)


# Проверяет отказ при отсутствии одного из обязательных отчётов.
def test_missing_file_rejected(tmp_path):
    fixture_root(tmp_path)
    (tmp_path / "reports/classical_error_analysis.json").unlink()
    with pytest.raises(F1ReportError, match="missing errors audit"):
        build_report(tmp_path)


# Проверяет обнаружение ручных изменений в сгенерированном документе.
def test_generated_report_check_detects_edits(tmp_path):
    fixture_root(tmp_path)
    p = run(root=tmp_path)
    p.write_text(p.read_text(encoding="utf-8") + "MODIFIED", encoding="utf-8")
    with pytest.raises(F1ReportError, match="differs"):
        run(root=tmp_path, check=True)


# Проверяет соответствие размеров корпуса и test заданным ограничениям.
def test_expected_counts_are_enforced(tmp_path):
    fixture_root(tmp_path)
    with pytest.raises(F1ReportError, match="expected cleaned"):
        build_report(tmp_path, expected_rows=3040)
    with pytest.raises(F1ReportError, match="expected test"):
        build_report(tmp_path, expected_test=608)
