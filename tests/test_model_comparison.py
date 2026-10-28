"""F2.7: защита от утечки test, расхождений ID и подмены артефактов.

Все примеры синтетические. Проверяется равенство определений метрик F1/F2,
а также запрет повторного выбора моделей по результатам тестирования.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch

from scripts.pytorch_batches import synthetic_splits
from scripts.pytorch_comparison import _synthetic_baseline, run
from src.classical_ml.train import classification_metrics
from src.pytorch_f2.data_pipeline import make_loader, prepare_data
from src.pytorch_f2.embedding_classifier import EmbeddingClassifier
from src.pytorch_f2.model_comparison import (
    ComparisonError, compare_on_test, predict_pytorch_test,
    save_aggregate_report, sha256_file, verify_classical_artifacts,
)


@pytest.fixture
def sample():
    """Создать F1.6 split из уникальных синтетических ID и групп."""
    return synthetic_splits()


@pytest.fixture
def baseline(sample):
    """Воспроизвести фиксированный F1 baseline на вымышленных текстах."""
    return _synthetic_baseline(sample)


def hashes(sample):
    """Вернуть контрольные метки синтетических источников smoke-процедуры."""
    return {key: f"synthetic-{key}" for key in sample}


def test_verifies_unmodified_baseline(sample, baseline):
    """Оригинальные F1-артефакты проходят сверку до neural test."""
    report, predictions = baseline
    ordered = verify_classical_artifacts(sample, report, predictions, hashes(sample))
    assert len(ordered["test"]) == 1


def test_permuted_prediction_order_remains_aligned(sample, baseline):
    """Порядок записей в файле предсказаний не должен влиять на результат."""
    report, predictions = baseline
    assert verify_classical_artifacts(sample, report, predictions, hashes(sample))["val"]


@pytest.mark.parametrize("field,new", [
    ("stage", "F9"), ("fit_split", "test"),
    ("evaluation_splits", ["train", "test"]),
])
def test_wrong_baseline_selection_protocol_is_rejected(sample, baseline, field, new):
    """Нельзя принимать baseline, выбранный или обученный с использованием test."""
    report, predictions = copy.deepcopy(baseline)
    report[field] = new
    with pytest.raises(ComparisonError):
        verify_classical_artifacts(sample, report, predictions, hashes(sample))


@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_source_hash_mismatch_is_rejected(sample, baseline, split):
    """Даже одно изменение split-файла делает сравнение недействительным."""
    report, predictions = baseline
    wrong = hashes(sample)
    wrong[split] = "changed"
    with pytest.raises(ComparisonError, match="SHA-256"):
        verify_classical_artifacts(sample, report, predictions, wrong)


def test_source_count_mismatch_is_rejected(sample, baseline):
    """Чужое число test-строк нельзя считать тем же экспериментом."""
    report, predictions = copy.deepcopy(baseline)
    report["data_contract"]["split_counts"]["test"] += 1
    with pytest.raises(ComparisonError):
        verify_classical_artifacts(sample, report, predictions, hashes(sample))


@pytest.mark.parametrize("tamper", ["missing", "duplicate", "foreign", "wrong_truth", "wrong_prediction"])
def test_corrupt_ids_and_labels_are_rejected(sample, baseline, tamper):
    """Проверяем лишние/чужие ID, дубликаты и несовпадения меток."""
    report, predictions = copy.deepcopy(baseline)
    record = predictions["test"][0]
    if tamper == "missing":
        predictions["test"] = []
    elif tamper == "duplicate":
        predictions["test"].append(record.copy())
    elif tamper == "foreign":
        record["id"] = "someone-else"
    elif tamper == "wrong_truth":
        record["true"] = "linux"
    else:
        record["predicted"] = "missing-class"
    with pytest.raises(ComparisonError):
        verify_classical_artifacts(sample, report, predictions, hashes(sample))


@pytest.mark.parametrize("field", ["accuracy", "macro_f1", "weighted_f1"])
def test_falsified_saved_metric_is_rejected(sample, baseline, field):
    """Сохранённое число должно пересчитываться из исходных предсказаний."""
    report, predictions = copy.deepcopy(baseline)
    report["test"][field] *= 0.5
    with pytest.raises(ComparisonError):
        verify_classical_artifacts(sample, report, predictions, hashes(sample))


def test_wrong_confusion_matrix_is_rejected(sample, baseline):
    """Матрица ошибок не может не совпадать с сохранёнными предсказаниями."""
    report, predictions = copy.deepcopy(baseline)
    report["test"]["confusion_matrix"][0][0] += 1
    with pytest.raises(ComparisonError):
        verify_classical_artifacts(sample, report, predictions, hashes(sample))


def test_paired_metrics_and_error_partition():
    """Разбиваем ошибки на четыре взаимоисключающие группы без повторного учёта."""
    truth = ["a", "a", "b", "b", "a"]
    classical = ["a", "b", "a", "b", "a"]
    neural = ["a", "a", "a", "a", "b"]
    rows = [{"id": str(i), "domain": value} for i, value in enumerate(truth)]
    to_rows = lambda p: [{"id": str(i), "true": y, "predicted": guess}
                         for i, (y, guess) in enumerate(zip(truth, p))]
    result = compare_on_test(rows, classical_predictions=to_rows(classical),
                             neural_predictions=to_rows(neural), labels=["a", "b"])
    assert result["paired_errors"] == {
        "both_correct": 1, "both_wrong": 1, "only_classical_wrong": 1,
        "only_neural_wrong": 2, "prediction_disagreements": 3,
    }
    assert result["classical"]["macro_f1"] == pytest.approx(
        classification_metrics(truth, classical, ["a", "b"])["macro_f1"])
    assert result["samples"] == 5


def test_paired_metrics_deltas_and_winner():
    """Разность нейросеть минус baseline может быть как отрицательной, так и положительной."""
    rows = [{"id": str(i), "domain": y} for i, y in enumerate(["a", "b", "a", "b"])]
    def pred(values):
        return [{"id": row["id"], "true": row["domain"], "predicted": guess}
                for row, guess in zip(rows, values)]
    first = compare_on_test(rows, classical_predictions=pred(["a", "b", "a", "b"]),
                            neural_predictions=pred(["a", "b", "a", "a"]), labels=["a", "b"])
    assert first["winning_model_by_macro_f1"] == "classical"
    assert first["delta_neural_minus_classical"]["macro_f1"] < 0
    second = compare_on_test(rows, classical_predictions=pred(["a", "b", "a", "a"]),
                             neural_predictions=pred(["a", "b", "a", "b"]), labels=["a", "b"])
    assert second["winning_model_by_macro_f1"] == "neural"
    assert second["delta_neural_minus_classical"]["macro_f1"] > 0


def test_equal_predictions_means_tie():
    """Равные метрики не должны превращаться в искусственный рейтинг моделей."""
    rows = [{"id": "x", "domain": "a"}, {"id": "y", "domain": "b"}]
    predictions = [{"id": row["id"], "true": row["domain"], "predicted": row["domain"]}
                   for row in rows]
    result = compare_on_test(rows, classical_predictions=predictions,
                             neural_predictions=predictions, labels=["a", "b"])
    assert result["winning_model_by_macro_f1"] == "tie"
    assert result["paired_errors"]["prediction_disagreements"] == 0


def test_duplicate_neural_id_is_rejected():
    """Нельзя дважды использовать одну запись при подсчёте Macro F1."""
    rows = [{"id": "x", "domain": "a"}, {"id": "y", "domain": "b"}]
    prediction = [{"id": "x", "true": "a", "predicted": "a"}] * 2
    with pytest.raises(ComparisonError):
        compare_on_test(rows, classical_predictions=prediction,
                        neural_predictions=prediction, labels=["a", "b"])


def test_test_loader_rejects_train_and_shuffling(sample):
    """Неправильный split и перестановка на test должны приводить к отказу."""
    data = prepare_data(sample)
    model = EmbeddingClassifier(vocab_size=len(data.vocab), num_classes=2)
    for split, shuffle in (("train", False), ("test", True)):
        with pytest.raises(ComparisonError):
            predict_pytorch_test(model, make_loader(data.datasets[split], shuffle=shuffle),
                                 label_to_id=data.label_to_id)


def test_test_inference_restores_training_mode(sample):
    """Инференс не меняет исходный режим модели и не вычисляет градиенты."""
    data = prepare_data(sample)
    model = EmbeddingClassifier(vocab_size=len(data.vocab), num_classes=2)
    model.train()
    before = {k: p.detach().clone() for k, p in model.state_dict().items()}
    rows = predict_pytorch_test(model, make_loader(data.datasets["test"]),
                                label_to_id=data.label_to_id)
    assert rows[0]["id"] == sample["test"][0]["id"]
    assert model.training is True
    assert all(torch.equal(t, before[k]) for k, t in model.state_dict().items())
    assert all(p.grad is None for p in model.parameters())


def test_test_inference_does_not_skip_last_batch(sample):
    """Один неполный test-батч также полностью обрабатывается."""
    data = prepare_data(sample)
    model = EmbeddingClassifier(vocab_size=len(data.vocab), num_classes=2)
    results = predict_pytorch_test(model, make_loader(data.datasets["test"], batch_size=32),
                                   label_to_id=data.label_to_id)
    assert len(results) == 1


def test_sha256_reads_actual_bytes(tmp_path):
    """Контрольная сумма реагирует на побайтовые изменения файла."""
    path = tmp_path / "input.jsonl"
    path.write_bytes(b"a\n")
    first = sha256_file(path)
    path.write_bytes(b"a\r\n")
    assert len(first) == 64 and first != sha256_file(path)


def test_private_predictions_not_exported(tmp_path, sample, baseline):
    """В JSON и Markdown не должны сохраняться отдельные ID, тексты, ответы."""
    report, predictions = baseline
    result = compare_on_test(sample["test"], classical_predictions=predictions["test"],
                             neural_predictions=predictions["test"], labels=["linux", "systems"])
    paths = save_aggregate_report({"comparison": result}, tmp_path)
    output = "\n".join(p.read_text(encoding="utf-8") for p in paths)
    assert all(row["id"] not in output and row["context"] not in output
               for row in sample["test"])
    assert json.loads(paths[0].read_text(encoding="utf-8"))["comparison"]["samples"] == 1


def test_synthetic_end_to_end(tmp_path):
    """Два классификатора сравниваются на одинаковом отложенном synthetic test."""
    result = run(synthetic=True, root=tmp_path)
    assert result["comparison_invariants_pass"]
    assert not result["test_used_for_selection"]
    assert result["split_counts"] == {"train": 4, "val": 1, "test": 1}
    assert result["neural_best_epoch"] >= 1
    assert all(Path(path).is_file() for path in result["output_files"])


def test_synthetic_is_reproducible(tmp_path):
    """Фиксированный seed и один test дают одинаковые пары метрик."""
    first = run(synthetic=True, root=tmp_path / "run1")
    second = run(synthetic=True, root=tmp_path / "run2")
    for key in ("test_accuracy_classical", "test_macro_f1_classical",
                "test_accuracy_neural", "test_macro_f1_neural", "neural_best_epoch"):
        assert first[key] == second[key]


def test_wrong_expected_count_fails_before_test(tmp_path):
    """Несовпадение размеров выборки блокирует финальную оценку."""
    with pytest.raises(ComparisonError, match="unexpected test"):
        run(synthetic=True, root=tmp_path, expect_test=608)


def _build_local_fixture(root: Path) -> None:
    """Собрать локальные F1/F2 артефакты на synthetic split без публикации."""
    from scripts.pytorch_comparison import _synthetic_baseline
    from src.pytorch_f2.validation_checkpoint import fit_with_validation
    import yaml

    sample = synthetic_splits()
    data_dir = root / "data/processed"
    data_dir.mkdir(parents=True)
    for split, rows in sample.items():
        file = data_dir / f"{split}.jsonl"
        file.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                        encoding="utf-8")
    cfg = {"seed": 42, "neural": {"vocab_size": 12000, "max_length": 128,
                                   "batch_size": 32, "embedding_dim": 48,
                                   "hidden_dim": 48, "learning_rate": 0.03}}
    config = root / "configs/local_cpu/lab.yaml"
    config.parent.mkdir(parents=True)
    config.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    baseline_report, predictions = _synthetic_baseline(sample)
    baseline_report["split_source_sha256"] = {split: sha256_file(data_dir / f"{split}.jsonl")
                                               for split in sample}
    reports = root / "reports"
    reports.mkdir()
    (reports / "classical.json").write_text(json.dumps(baseline_report), encoding="utf-8")
    (reports / "classical_predictions.json").write_text(json.dumps(predictions), encoding="utf-8")
    prepared = prepare_data(sample)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        model = EmbeddingClassifier(vocab_size=len(prepared.vocab), num_classes=2)
        fit_with_validation(
            model, make_loader(prepared.datasets["train"], batch_size=32, shuffle=True, seed=42),
            make_loader(prepared.datasets["val"], batch_size=32, shuffle=False, seed=42),
            vocab=prepared.vocab, label_to_id=prepared.label_to_id, max_length=128,
            split_counts=prepared.counts, seed=42,
            checkpoint_path=root / "checkpoints/f2_5_best.pt", epochs=2, learning_rate=0.03,
        )


def test_real_artifact_path_end_to_end(tmp_path):
    """Реальный путь чтения F1 JSONL/отчётов и F2 checkpoint работает совместно."""
    _build_local_fixture(tmp_path)
    result = run(root=tmp_path, expect_train=4, expect_val=1, expect_test=1)
    assert result["source"] == "local_f1_split"
    assert result["comparison_invariants_pass"]
    assert not result["test_used_for_selection"]
    assert result["test_macro_f1_classical"] == 0.5


def test_real_artifact_hash_tampering_blocked(tmp_path):
    """Любое изменение test-файла после F1 делает прежние прогнозы недействительными."""
    _build_local_fixture(tmp_path)
    path = tmp_path / "data/processed/test.jsonl"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ComparisonError, match="SHA-256"):
        run(root=tmp_path)


def test_real_artifact_absent_checkpoint_blocked(tmp_path):
    """Отсутствие validation-selected checkpoint не запускает обучение по test."""
    _build_local_fixture(tmp_path)
    (tmp_path / "checkpoints/f2_5_best.pt").unlink()
    with pytest.raises(ComparisonError, match="missing F2.5"):
        run(root=tmp_path)


def test_real_path_does_not_refit_or_save_models(tmp_path, monkeypatch):
    """Сравнение реальных артефактов только читает обе уже выбранные модели."""
    import scripts.pytorch_comparison as module
    _build_local_fixture(tmp_path)
    def forbidden(*args, **kwargs):
        """Любой повторный fit нарушает границу между train/val и test."""
        raise AssertionError("fit is forbidden in real comparison")
    monkeypatch.setattr(module, "fit_with_validation", forbidden)
    monkeypatch.setattr(module, "build_pipeline", forbidden)
    result = module.run(root=tmp_path)
    assert result["comparison_invariants_pass"]
