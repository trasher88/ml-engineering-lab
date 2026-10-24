# Назначение файла: F1.7: проверка модели и метрик на синтетических вопросах; без настоящих Q&A.
# Примечание: комментарии не влияют на работу программы.

"""F1.7 offline tests: synthetic labels and text, no downloaded dataset."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import joblib
import pytest

from src.classical_ml.train import (
    ClassicalError, bootstrap_macro_f1, build_pipeline,
    classification_metrics, fit_and_evaluate, validate_splits,
)
from scripts.train_classical import BaselineRunError, run


# Создаёт синтетические train, val и test для тестирования классификатора.
def samples() -> dict[str, list[dict]]:
    result = {}
    for split, n in (("train", 12), ("val", 4), ("test", 4)):
        rows = []
        for i in range(n):
            label = "linux" if i % 2 == 0 else "databases"
            token = "kernel shell process" if label == "linux" else "sql table query"
            rows.append({"id": f"synthetic:{split}:{i}", "context": f"{token} task {i} {split}",
                         "domain": label, "group_id": f"{split}-{i}", "split": split})
        result[split] = rows
    return result


# Проверяет число элементов, метки классов и независимость групп.
def test_validate_splits_expected_counts_and_groups():
    summary = validate_splits(samples())
    assert summary["split_counts"] == {"train": 12, "val": 4, "test": 4}
    assert summary["train_classes"] == ["databases", "linux"]
    assert not any(summary["group_overlap"].values())


# Проверяет отказ при неверном наборе train/val/test.
@pytest.mark.parametrize("invalid", [None, {}, {"train": []}, [1, 2, 3]])
def test_rejects_wrong_split_structure(invalid):
    with pytest.raises(ClassicalError):
        validate_splits(invalid)


# Проверяет обязательные поля и допустимые типы записей.
@pytest.mark.parametrize("field,value", [
    ("id", ""), ("context", "  "), ("domain", None), ("split", "test"),
    ("group_id", None), ("group_id", True),
])
def test_rejects_bad_input_fields(field, value):
    d = samples()
    d["train"][0][field] = value
    with pytest.raises(ClassicalError):
        validate_splits(d)


# Проверяет, что ID не могут повторяться между выборками.
def test_rejects_duplicate_ids_across_splits():
    d = samples()
    d["test"][0]["id"] = d["train"][0]["id"]
    with pytest.raises(ClassicalError, match="duplicate"):
        validate_splits(d)


# Проверяет блокировку утечки одной группы в разные выборки.
def test_rejects_cross_split_group_leakage():
    d = samples()
    d["test"][0]["group_id"] = d["train"][0]["group_id"]
    with pytest.raises(ClassicalError, match="group leakage"):
        validate_splits(d)


# Проверяет отказ от классов в val/test, отсутствующих в train.
def test_rejects_classes_unknown_in_train():
    d = samples()
    d["val"][0]["domain"] = "unknown"
    with pytest.raises(ClassicalError, match="unseen classes"):
        validate_splits(d)


# Проверяет границы параметров классификатора.
@pytest.mark.parametrize("bad", [
    {"max_features": 0}, {"max_iter": -1}, {"seed": -1},
    {"c": 0}, {"c": float("nan")}, {"c": True},
])
def test_rejects_bad_model_config(bad):
    params = {"max_features": 100, "max_iter": 100, "seed": 42, "c": 2.0}
    params.update(bad)
    with pytest.raises(ClassicalError):
        build_pipeline(**params)


# Проверяет, что словарь TF-IDF обучается только на train.
def test_train_only_fit_and_test_vocabulary_not_seen():
    d = samples()
    for row in d["val"] + d["test"]:
        row["context"] += " NEVERINTRAINUNIQUEWORD"
    before = copy.deepcopy(d)
    model, report, predictions = fit_and_evaluate(d, max_features=300, bootstrap_samples=10)
    assert d == before
    vocab = model.named_steps["tfidf"].vocabulary_
    assert "neverintrainuniqueword" not in vocab
    assert report["fit_split"] == "train"
    assert report["evaluation_splits"] == ["val", "test"]
    assert report["val"]["n"] == 4 and report["test"]["n"] == 4
    assert report["test"]["macro_f1_ci95"][0] <= report["test"]["macro_f1_ci95"][1]
    assert set(predictions) == {"val", "test"}
    assert all(set(p) == {"id", "true", "predicted"} for p in predictions["test"])
    assert "context" not in json.dumps(report)  # audit contains no Q&A texts
    assert report["config"]["ngram_range"] == [1, 2]
    assert report["config"]["sublinear_tf"] is True


# Проверяет повторяемость оценок на одном и том же наборе.
def test_deterministic_metric_outputs_on_same_data():
    a, m1, p1 = fit_and_evaluate(samples(), bootstrap_samples=10)
    b, m2, p2 = fit_and_evaluate(samples(), bootstrap_samples=10)
    assert m1 == m2 and p1 == p2
    assert list(a.named_steps["classifier"].classes_) == list(b.named_steps["classifier"].classes_)


# Проверяет Accuracy и confusion matrix на примере с известным ответом.
def test_metrics_known_answer_and_matrix():
    out = classification_metrics(["A", "A", "B", "B"], ["A", "B", "B", "B"], ["A", "B"])
    assert out["accuracy"] == 0.75
    assert out["confusion_matrix"] == [[1, 1], [0, 2]]
    assert out["n"] == 4
    assert 0 <= out["macro_f1"] <= 1


# Проверяет валидацию длины прогнозов и числа bootstrap-повторов.
def test_bad_predictions_and_bootstrap_counts():
    with pytest.raises(ClassicalError):
        classification_metrics([], [], ["A"])
    with pytest.raises(ClassicalError):
        bootstrap_macro_f1(["A"], ["A"], ["A"], samples=0)


# Проверяет воспроизводимость доверительного интервала Bootstrap.
def test_bootstrap_is_repeatable():
    a = bootstrap_macro_f1(["a", "a", "b", "b"], ["a", "b", "b", "b"], ["a", "b"], samples=30)
    b = bootstrap_macro_f1(["a", "a", "b", "b"], ["a", "b", "b", "b"], ["a", "b"], samples=30)
    assert a == b and 0 <= a[0] <= a[1] <= 1


# Создаёт временные конфигурации и файлы выборок для CLI.
@pytest.fixture
def sandbox(tmp_path: Path):
    cfg = tmp_path / "configs/local_cpu/lab.yaml"
    cfg.parent.mkdir(parents=True)
    cfg.write_text("seed: 42\nclassical:\n  max_features: 100\n  C: 2.0\n  max_iter: 200\nevaluation:\n  bootstrap_samples: 10\n", encoding="utf-8")
    for name, rows in samples().items():
        path = tmp_path / f"data/processed/{name}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return tmp_path


# Проверяет локальное сохранение модели и отчёта без Q&A-текстов.
def test_run_local_smoke_artifacts_and_text_free_report(sandbox):
    report = run(sandbox, expect_train=12, expect_val=4, expect_test=4)
    model_path = sandbox / "models/classical.joblib"
    report_path = sandbox / "reports/classical.json"
    predictions_path = sandbox / "reports/classical_predictions.json"
    assert model_path.stat().st_size > 0
    assert model_path.stat().st_size == report["model_bytes"]
    stored = json.loads(report_path.read_text(encoding="utf-8"))
    assert stored == report
    assert "kernel shell" not in report_path.read_text(encoding="utf-8")
    saved = json.loads(predictions_path.read_text(encoding="utf-8"))
    assert set(saved) == {"val", "test"} and len(saved["test"]) == 4
    loaded = joblib.load(model_path)  # test-created trusted artifact only
    assert list(loaded.predict(["kernel shell process", "sql table query"])) == ["linux", "databases"]


# Проверяет повторяемость результатов двух запусков обучения.
def test_repeat_run_is_deterministic_except_serialized_size(sandbox):
    first = run(sandbox)
    second = run(sandbox)
    assert first == second


# Проверяет отказ сохранять артефакты при отсутствии одной выборки.
def test_missing_split_does_not_write_artifacts(sandbox):
    (sandbox / "data/processed/test.jsonl").unlink()
    with pytest.raises(BaselineRunError, match="missing F1.6"):
        run(sandbox)
    assert not (sandbox / "models/classical.joblib").exists()


# Проверяет остановку при несовпадении ожидаемого размера выборок.
def test_expected_count_blocks_training(sandbox):
    with pytest.raises(BaselineRunError, match="expected test"):
        run(sandbox, expect_test=608)
    assert not (sandbox / "reports").exists()


# Проверяет недопустимость совпадения путей выходных файлов.
def test_output_collision_blocks_writes(sandbox):
    with pytest.raises(BaselineRunError, match="distinct paths"):
        run(sandbox, report_path=Path("data/processed/train.jsonl"))
    assert not (sandbox / "models/classical.joblib").exists()


# Проверяет запрет обучения при обнаруженной утечке групп.
def test_data_group_leakage_blocks_training(sandbox):
    p = sandbox / "data/processed/test.jsonl"
    rows = [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines()]
    rows[0]["group_id"] = "train-0"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    with pytest.raises(ClassicalError, match="group leakage"):
        run(sandbox)
    assert not (sandbox / "models/classical.joblib").exists()
