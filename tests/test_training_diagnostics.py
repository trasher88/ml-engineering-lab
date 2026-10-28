"""F2.6: проверка источников графиков, интерпретации и сохранения агрегатов.

Все данные фиктивные. Реальный корпус F1, его ID и checkpoint не нужны.
"""

from __future__ import annotations

import csv
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.pytorch_diagnostics import run
from src.pytorch_f2.training_diagnostics import (
    DiagnosticsError, export_diagnostics, interpret_history, normalize_history, verify_export,
)


def report():
    """Вернуть минимальную корректную историю F2.5 только с агрегатами."""
    return {
        "source": "synthetic", "split_counts": {"train": 4, "val": 1, "test": 1},
        "trained_split": "train", "selection_split": "val", "test_evaluated": False,
        "selection_metric": "val_macro_f1", "checkpoint_reload_match": True,
        "validation_invariants_pass": True, "epochs": 3, "best_epoch": 2,
        "best_val_macro_f1": 0.75,
        "history": [
            {"epoch": 1, "train_loss": 1.0, "val_loss": 0.9, "val_accuracy": 0.5, "val_macro_f1": 0.5},
            {"epoch": 2, "train_loss": 0.8, "val_loss": 0.7, "val_accuracy": 0.75, "val_macro_f1": 0.75},
            {"epoch": 3, "train_loss": 0.6, "val_loss": 0.9, "val_accuracy": 0.5, "val_macro_f1": 0.5},
        ],
    }


def test_normalization_keeps_only_numeric_aggregates():
    """Пути, исходные вопросы и дополнительные поля не попадают в экспорт."""
    source = report()
    source["private_question"] = "private question"
    source["checkpoint_location"] = "C:/secret.pt"
    source["history"][0]["sample_id"] = "private-id"
    normalized = normalize_history(source)
    as_json = json.dumps(normalized)
    assert "private" not in as_json and "secret" not in as_json
    assert set(normalized["history"][0]) == {
        "epoch", "train_loss", "val_loss", "val_accuracy", "val_macro_f1"}


@pytest.mark.parametrize("key,value", [
    ("trained_split", "val"), ("selection_split", "test"),
    ("test_evaluated", True), ("selection_metric", "train_loss"),
    ("checkpoint_reload_match", False), ("validation_invariants_pass", False),
])
def test_invalid_f2_5_invariants_rejected(key, value):
    """Графики нельзя строить из эксперимента с утечкой или сбойным checkpoint."""
    source = report()
    source[key] = value
    with pytest.raises(DiagnosticsError, match="invariants"):
        normalize_history(source)


@pytest.mark.parametrize("key,value", [
    ("train_loss", math.nan), ("val_loss", math.inf),
    ("val_accuracy", 1.01), ("val_macro_f1", -0.1),
    ("train_loss", "0.2"), ("val_loss", True),
])
def test_invalid_metrics_rejected(key, value):
    """Нельзя рисовать NaN, inf, неправильные вероятности и нечисловые значения."""
    source = report()
    source["history"][1][key] = value
    with pytest.raises(DiagnosticsError):
        normalize_history(source)


@pytest.mark.parametrize("field,value", [
    ("epochs", 2), ("epochs", 0), ("epochs", True),
    ("best_epoch", 0), ("best_epoch", 4),
    ("best_val_macro_f1", 0.99),
    ("source", "test"), ("split_counts", {"train": 4, "val": 1}),
])
def test_invalid_history_metadata_rejected(field, value):
    """Несоответствие длины, лучшей эпохи или split нельзя скрыть графиком."""
    source = report()
    source[field] = value
    with pytest.raises(DiagnosticsError):
        normalize_history(source)


def test_missing_epoch_rejected():
    """Пропуск номера эпохи искажает ось времени."""
    source = report()
    source["history"][1]["epoch"] = 5
    with pytest.raises(DiagnosticsError, match="consecutive"):
        normalize_history(source)


def test_rounding_of_f2_5_is_accepted():
    """Округлённый F2.5 Macro F1 допускает расхождение в последнем разряде."""
    source = report()
    source["best_val_macro_f1"] = 0.750000005
    assert normalize_history(source)["best_epoch"] == 2


def test_stale_best_epoch_rejected():
    """Лучшей не может быть эпоха с меньшей F1, чем зафиксированный максимум."""
    source = report()
    source["best_epoch"] = 1
    source["best_val_macro_f1"] = 0.5
    with pytest.raises(DiagnosticsError, match="best checkpoint"):
        normalize_history(source)


def test_possible_overfitting_signal():
    """Падение train loss при росте val loss — лишь возможный сигнал."""
    rows = report()["history"]
    rows[-1]["val_loss"] = 1.2
    result = interpret_history(rows, best_epoch=2)
    assert result["signal"] == "possible_overfitting"
    assert "не доказательство" in result["caveat"]


def test_no_overfitting_signal_with_joint_loss_decrease():
    """Синхронное улучшение loss не позволяет объявить переобучение."""
    result = interpret_history(report()["history"], best_epoch=2)
    assert result["signal"] == "no_clear_overfitting_signal"


def test_short_history_does_not_overclaim():
    """Одна-две эпохи недостаточны для устойчивого вывода."""
    assert interpret_history(report()["history"][:2], 1)["signal"] == "insufficient_epochs"


def test_no_clear_progress_signal():
    """При отсутствии улучшения train и F1 сообщаем только наблюдение."""
    rows = report()["history"]
    rows[-1]["train_loss"] = 1.2
    rows[-1]["val_macro_f1"] = 0.4
    assert interpret_history(rows, 2)["signal"] == "no_clear_learning_progress"


def test_export_produces_3_plots_csv_json_report(tmp_path):
    """Точки PNG берутся из одного проверенного CSV, который описан JSON."""
    data = export_diagnostics(report(), tmp_path)
    expected = {"metrics.json", "history.csv", "analysis.md", "training_loss.png",
                "validation_loss.png", "validation_macro_f1.png"}
    assert {p.name for p in tmp_path.iterdir()} == expected
    verify_export(tmp_path)
    with (tmp_path / "history.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 3
    assert float(rows[1]["val_macro_f1"]) == 0.75
    assert data["best_epoch"] == 2
    assert "best_epoch" in (tmp_path / "metrics.json").read_text(encoding="utf-8")
    assert "test не оценивался" in (tmp_path / "analysis.md").read_text(encoding="utf-8")


def test_corrupt_csv_detected(tmp_path):
    """При расхождении числовой истории CSV и JSON контроль SHA-256 должен упасть."""
    export_diagnostics(report(), tmp_path)
    path = tmp_path / "history.csv"
    path.write_text(path.read_text(encoding="utf-8").replace("0.75", "0.85"), encoding="utf-8")
    with pytest.raises(DiagnosticsError, match="hash mismatch"):
        verify_export(tmp_path)


def test_corrupt_plot_detected(tmp_path):
    """Пустой/неверный файл не считается созданным PNG-графиком."""
    export_diagnostics(report(), tmp_path)
    (tmp_path / "training_loss.png").write_text("not png", encoding="utf-8")
    with pytest.raises(DiagnosticsError, match="invalid PNG"):
        verify_export(tmp_path)


def test_invalid_parameters_rejected(tmp_path):
    """Экспорт метаданных не должен сохранять произвольные приватные поля."""
    with pytest.raises(DiagnosticsError, match="unexpected"):
        export_diagnostics(report(), tmp_path, parameters={"private_path": "C:/private"})


def test_parameter_numeric_validation(tmp_path):
    """В JSON нельзя записать NaN, который не входит в JSON-стандарт."""
    with pytest.raises(DiagnosticsError, match="finite"):
        export_diagnostics(report(), tmp_path, parameters={"learning_rate": math.nan})


def test_export_deterministic_csv_and_interpretation(tmp_path):
    """Повторная сборка из одних агрегатов даёт те же точки и диагностику."""
    first = export_diagnostics(report(), tmp_path / "a")
    second = export_diagnostics(report(), tmp_path / "b")
    assert first == second
    assert (tmp_path / "a/history.csv").read_bytes() == (tmp_path / "b/history.csv").read_bytes()


@pytest.fixture(scope="module")
def synthetic_run(tmp_path_factory):
    """Один реальный синтетический прогон F2.5, чтобы не дублировать обучение в тестах."""
    directory = tmp_path_factory.mktemp("f2_6_synthetic")
    result = run(synthetic=True, epochs=3, output_dir=directory)
    return result, directory


def test_synthetic_run_passes(synthetic_run):
    """Реальный F2.5 → диагностика без test на CPU."""
    result, directory = synthetic_run
    assert result["source"] == "synthetic"
    assert result["test_evaluated"] is False
    assert result["export_invariants_pass"] is True
    assert result["epochs"] == 3
    verify_export(directory)


def test_synthetic_run_documents_effective_hyperparameters(synthetic_run):
    """JSON хранит настройки того же эксперимента, что нарисован на графиках."""
    _, directory = synthetic_run
    data = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
    assert data["parameters"]["seed"] == 42
    assert data["parameters"]["learning_rate"] == 0.03
    assert data["split_counts"] == {"train": 4, "val": 1, "test": 1}


def test_cli_synthetic_smoke(tmp_path):
    """Запуск как python -m работает без F1 dataset и внешнего сервиса."""
    command = [sys.executable, "-m", "scripts.pytorch_diagnostics", "--synthetic",
               "--epochs", "2", "--output-dir", str(tmp_path), "--check"]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr + result.stdout
    assert "F2.6 Training Diagnostics CHECK: PASS" in result.stdout
    verify_export(tmp_path)


def test_cli_rejects_zero_epochs(tmp_path):
    """Некорректный запуск не выводит фиктивный PASS и не создаёт графиков."""
    command = [sys.executable, "-m", "scripts.pytorch_diagnostics", "--synthetic",
               "--epochs", "0", "--output-dir", str(tmp_path), "--check"]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode != 0
    assert "CHECK: FAIL" in result.stdout
    assert not (tmp_path / "metrics.json").exists()


def test_valid_png_tampering_detected_by_hash(tmp_path):
    """Даже файл с корректной PNG-сигнатурой не проходит при подмене байтов."""
    export_diagnostics(report(), tmp_path)
    image = tmp_path / "training_loss.png"
    image.write_bytes(image.read_bytes() + b"extra-data")
    with pytest.raises(DiagnosticsError, match="PNG hash mismatch"):
        verify_export(tmp_path)


def test_diagnostics_uses_disposable_checkpoint_not_f2_5_file(tmp_path, monkeypatch):
    """F2.6 не имеет права перезаписывать ранее сохранённый best F2.5."""
    import scripts.pytorch_diagnostics as entry
    existing = tmp_path / "checkpoints" / "f2_5_best.pt"
    existing.parent.mkdir()
    existing.write_bytes(b"important previous checkpoint")
    targets = []
    def fake_run(**kwargs):
        path = kwargs["checkpoint_path"]
        targets.append(path)
        assert path != existing
        path.write_bytes(b"temporary checkpoint")
        return report()
    monkeypatch.setattr(entry, "validation_run", fake_run)
    result = entry.run(synthetic=True, root=tmp_path, epochs=3)
    assert result["export_invariants_pass"]
    assert existing.read_bytes() == b"important previous checkpoint"
    assert targets and not targets[0].exists()
