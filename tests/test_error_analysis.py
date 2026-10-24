# Назначение файла: F1.8: проверки анализа зафиксированных прогнозов без реальных вопросов.
# Примечание: комментарии не влияют на работу программы.

"""F1.8 offline tests — all examples are invented, no Stack Exchange texts."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from scripts.analyze_classical import run
from src.classical_ml.error_analysis import (
    ErrorAnalysisError, analyze_errors, draw_confusion_matrix,
)
from src.classical_ml.train import classification_metrics


# Создаёт выдуманные прогнозы и правильные ответы для анализа.
def example():
    labels = ["databases", "linux", "systems"]
    truth = ["databases", "linux", "systems", "linux", "systems", "databases"]
    predicted = ["databases", "systems", "linux", "linux", "systems", "linux"]
    rows = [
        {"id": f"synthetic:test:{i}", "domain": y, "split": "test",
         "context": f"Synthetic question {i} about {y} demo placeholder"}
        for i, y in enumerate(truth)
    ]
    prediction = {"test": [
        {"id": x["id"], "true": y, "predicted": p}
        for x, y, p in zip(rows, truth, predicted)
    ]}
    frozen = {"stage": "F1.7", "test": classification_metrics(truth, predicted, labels)}
    return rows, prediction, frozen


# Проверяет число ошибок, Accuracy и F1 в сводке.
def test_analysis_reports_correct_counts_and_metrics():
    rows, preds, f17 = example()
    aggregate, errors = analyze_errors(rows, preds, f17)
    assert aggregate["n"] == 6
    assert aggregate["correct"] == 3
    assert aggregate["errors"] == 3
    assert aggregate["error_rate"] == 0.5
    assert aggregate["confusion_matrix"] == f17["test"]["confusion_matrix"]
    assert aggregate["macro_f1"] == pytest.approx(f17["test"]["macro_f1"])
    assert aggregate["per_class"]["linux"]["support"] == 2
    assert aggregate["per_class"]["linux"]["false_positives"] == 2
    assert len(errors) == 3
    assert all(set(x) == {"id", "true", "predicted", "question_words"} for x in errors)


# Проверяет наиболее частую пару ошибок и стабильное разрешение ничьих.
def test_most_confused_pair_and_deterministic_tie_order():
    rows, preds, report = example()
    a, _ = analyze_errors(rows, preds, report)
    assert a["confusion_pairs"] == [
        {"true": "databases", "predicted": "linux", "count": 1},
        {"true": "linux", "predicted": "systems", "count": 1},
        {"true": "systems", "predicted": "linux", "count": 1},
    ]


# Проверяет агрегирование длин без публикации текста вопроса.
def test_question_lengths_no_raw_question_text():
    rows, preds, report = example()
    stats, errors = analyze_errors(rows, preds, report)
    assert stats["question_lengths"]["incorrect"]["n"] == 3
    assert isinstance(stats["question_lengths"]["correct"]["median_words"], float)
    payload = json.dumps({"summary": stats, "cases": errors})
    assert "Synthetic question" not in payload
    assert "demo placeholder" not in payload


# Проверяет неизменность переданных списков предсказаний.
def test_input_lists_are_not_mutated():
    rows, preds, report = example()
    frozen = copy.deepcopy((rows, preds, report))
    analyze_errors(rows, preds, report)
    assert (rows, preds, report) == frozen


# Проверяет корректную статистику при отсутствии ошибок.
def test_no_error_cases_and_empty_wrong_length_stats():
    rows, preds, report = example()
    for row, pred in zip(rows, preds["test"]):
        pred["predicted"] = row["domain"]
    y = [x["domain"] for x in rows]
    report["test"] = classification_metrics(y, y, report["test"]["labels"])
    aggregate, cases = analyze_errors(rows, preds, report)
    assert aggregate["errors"] == 0 and cases == []
    assert aggregate["confusion_pairs"] == []
    assert aggregate["question_lengths"]["incorrect"]["median_words"] is None


# Проверяет отказ при пустых или повреждённых тестовых записях.
@pytest.mark.parametrize("bad", [[], None, {}, "not rows"])
def test_rejects_empty_or_malformed_test_rows(bad):
    _, preds, report = example()
    with pytest.raises(ErrorAnalysisError):
        analyze_errors(bad, preds, report)


# Проверяет обнаружение некорректных либо изменённых предсказаний.
@pytest.mark.parametrize("corrupt", [
    lambda r,p,f: p.pop("test"),
    lambda r,p,f: p["test"].pop(),
    lambda r,p,f: p["test"][0].update({"context": "Leaked text"}),
    lambda r,p,f: p["test"][0].update({"id": "different"}),
    lambda r,p,f: p["test"][0].update({"true": "systems"}),
    lambda r,p,f: p["test"][0].update({"predicted": "unknown"}),
    lambda r,p,f: r[1].update({"id": r[0]["id"]}),
    lambda r,p,f: r[0].update({"split": "train"}),
    lambda r,p,f: f["test"].update({"n": 10}),
    lambda r,p,f: f["test"].update({"accuracy": 0.8}),
    lambda r,p,f: f["test"].update({"macro_f1": 0.9}),
    lambda r,p,f: f["test"].update({"confusion_matrix": [[1, 0], [0, 1]]}),
    lambda r,p,f: f["test"].update({"labels": ["linux", "linux"]}),
])
def test_rejects_invalid_or_tampered_inputs(corrupt):
    rows, preds, report = example()
    corrupt(rows, preds, report)
    with pytest.raises(ErrorAnalysisError):
        analyze_errors(rows, preds, report)


# Создаёт временные файлы модели, аудита и фиктивных прогнозов.
def make_project(tmp_path: Path):
    rows, preds, frozen = example()
    data = tmp_path / "data/processed/test.jsonl"
    data.parent.mkdir(parents=True)
    data.write_text("".join(json.dumps(x) + "\n" for x in rows), encoding="utf-8")
    expected = hashlib.sha256(data.read_bytes()).hexdigest()
    frozen["split_source_sha256"] = {"test": expected}
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "classical_predictions.json").write_text(json.dumps(preds), encoding="utf-8")
    (reports / "classical.json").write_text(json.dumps(frozen), encoding="utf-8")
    return data


# Проверяет генерацию агрегатов, ID ошибок и рисунка.
def test_run_generates_aggregate_report_id_only_cases_and_figure(tmp_path):
    make_project(tmp_path)
    report = run(tmp_path, expect_test=6)
    assert report["errors"] == 3
    aggregate_path = tmp_path / "reports/classical_error_analysis.json"
    cases_path = tmp_path / "reports/classical_error_ids.json"
    figure_path = tmp_path / "docs/visuals/f1/confusion_matrix.png"
    assert figure_path.stat().st_size > 1000
    persisted = json.loads(aggregate_path.read_text(encoding="utf-8"))
    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    assert persisted == report and len(cases) == 3
    assert "Synthetic question" not in aggregate_path.read_text(encoding="utf-8")
    assert "Synthetic question" not in cases_path.read_text(encoding="utf-8")
    assert report["test_source_sha256"] == hashlib.sha256((tmp_path/"data/processed/test.jsonl").read_bytes()).hexdigest()


# Проверяет повторяемость и неизменность входных прогнозов.
def test_run_is_repeatable_and_does_not_change_predictions(tmp_path):
    make_project(tmp_path)
    pred_path = tmp_path/"reports/classical_predictions.json"
    before = pred_path.read_bytes()
    r1 = run(tmp_path)
    outputs1 = [(tmp_path/x).read_bytes() for x in [
        "reports/classical_error_analysis.json", "reports/classical_error_ids.json"]]
    r2 = run(tmp_path)
    outputs2 = [(tmp_path/x).read_bytes() for x in [
        "reports/classical_error_analysis.json", "reports/classical_error_ids.json"]]
    assert r1 == r2 and outputs1 == outputs2 and before == pred_path.read_bytes()


# Проверяет отсутствие артефактов при несовпадении SHA-256.
def test_hash_mismatch_prevents_any_output(tmp_path):
    data = make_project(tmp_path)
    data.write_text(data.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ErrorAnalysisError, match="SHA-256"):
        run(tmp_path)
    assert not (tmp_path / "reports/classical_error_analysis.json").exists()
    assert not (tmp_path / "docs/visuals/f1/confusion_matrix.png").exists()


# Проверяет блокировку анализа при неправильном размере теста.
def test_wrong_expected_size_prevents_output(tmp_path):
    make_project(tmp_path)
    with pytest.raises(ErrorAnalysisError, match="expected test"):
        run(tmp_path, expect_test=608)
    assert not (tmp_path / "reports/classical_error_ids.json").exists()


# Проверяет запрет анализа без артефактов F1.7.
def test_missing_baseline_inputs_block(tmp_path):
    with pytest.raises(ErrorAnalysisError, match="missing F1.7"):
        run(tmp_path)


# Проверяет, что выходные пути не должны совпадать с входными.
def test_paths_cannot_overlap(tmp_path):
    make_project(tmp_path)
    with pytest.raises(ErrorAnalysisError, match="distinct"):
        run(tmp_path, output_path=Path("data/processed/test.jsonl"))


# Проверяет корректную генерацию изображения PNG.
def test_chart_writes_valid_png(tmp_path):
    rows, preds, f17 = example()
    stats, _ = analyze_errors(rows, preds, f17)
    png = tmp_path / "out.png"
    draw_confusion_matrix(stats, png)
    assert png.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
