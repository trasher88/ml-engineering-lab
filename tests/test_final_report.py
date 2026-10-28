"""F2.8: проверка агрегированных метрик, происхождения и безопасной публикации.

Тестовые данные вымышленные: ни реальные вопросы, ни test-ID не используются.
"""

from __future__ import annotations

from copy import deepcopy
import json
import hashlib
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.build_f2_report import run
from src.pytorch_f2.final_report import (
    PhaseReportError, build_from_artifacts, render_final_report,
    sha256_report, update_readme, validate_comparison, validate_diagnostics,
    verify_source_fingerprints,
)
from src.pytorch_f2.training_diagnostics import export_diagnostics


_LABELS = ["databases", "linux", "programming", "systems"]
_CLASSICAL_CM = [
    [130, 10, 6, 6],
    [8, 134, 5, 5],
    [4, 4, 140, 4],
    [8, 6, 7, 131],
]
_NEURAL_CM = [
    [124, 10, 9, 9],
    [9, 128, 6, 9],
    [7, 11, 126, 8],
    [7, 7, 9, 129],
]


def _metric(cm: list[list[int]]) -> dict:
    """Создать аналитически согласованные метрики для вымышленных классов."""
    total = sum(map(sum, cm))
    right = sum(cm[i][i] for i in range(len(cm)))
    per_class = []
    for i in range(len(cm)):
        tp = cm[i][i]
        fp = sum(cm[r][i] for r in range(len(cm))) - tp
        fn = sum(cm[i]) - tp
        per_class.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0)
    return {"accuracy": right / total, "macro_f1": sum(per_class) / len(cm),
            "labels": _LABELS, "n": total, "confusion_matrix": cm}


def comparison_fixture() -> dict:
    """Построить real-like агрегат по 608 фиктивным объектам без текстов и ID."""
    first, second = _metric(deepcopy(_CLASSICAL_CM)), _metric(deepcopy(_NEURAL_CM))
    return {
        "source": "local_f1_split",
        "split_counts": {"train": 1824, "val": 608, "test": 608},
        "split_sha256": {"train": "a" * 64, "val": "b" * 64, "test": "c" * 64},
        "comparison_invariants_pass": True,
        "protocol": {"test_used_for_selection": False, "classical_fit_split": "train",
                     "neural_selection_split": "val", "neural_selection_metric": "macro_f1",
                     "neural_best_epoch": 2, "neural_validation_macro_f1": 0.71},
        "comparison": {
            "samples": 608, "labels": _LABELS, "classical": first, "neural": second,
            "delta_neural_minus_classical": {
                "accuracy": second["accuracy"] - first["accuracy"],
                "macro_f1": second["macro_f1"] - first["macro_f1"],
            },
            "paired_errors": {"both_correct": 484, "both_wrong": 50,
                              "only_classical_wrong": 23, "only_neural_wrong": 51,
                              "prediction_disagreements": 90},
            "winning_model_by_macro_f1": "classical",
        },
        "complexity": {"neural_parameter_count": 578548,
                       "measured_runtime_comparison": False},
    }


def diagnostic_fixture() -> dict:
    """История трёх эпох отдельного фиктивного диагностического запуска."""
    return {
        "format_version": 1, "source": "local_f1_split",
        "split_counts": {"train": 1824, "val": 608, "test": 608},
        "trained_split": "train", "selection_split": "val",
        "test_evaluated": False, "selection_metric": "val_macro_f1",
        "epochs": 3, "best_epoch": 1, "best_val_macro_f1": 0.6,
        "history": [
            {"epoch": 1, "train_loss": 0.8, "val_loss": 0.9,
             "val_accuracy": 0.7, "val_macro_f1": 0.6},
            {"epoch": 2, "train_loss": 0.6, "val_loss": 0.85,
             "val_accuracy": 0.65, "val_macro_f1": 0.55},
            {"epoch": 3, "train_loss": 0.4, "val_loss": 0.82,
             "val_accuracy": 0.6, "val_macro_f1": 0.50},
        ],
        "interpretation": {"signal": "no_clear_overfitting_signal"},
    }


def test_verified_baseline_accuracy_and_f1() -> None:
    """Точность и F1 подтверждаются матрицами, а не только полями JSON."""
    summary = validate_comparison(comparison_fixture())
    assert summary["classical"]["errors"] == 73
    assert summary["neural"]["errors"] == 101
    assert summary["winner"] == "classical"
    assert summary["split_counts"]["test"] == 608


def test_diagnostic_epoch_is_not_comparison_checkpoint() -> None:
    """Отдельные запуски F2.6 и F2.7 не смешивают номера лучших эпох."""
    report = render_final_report(comparison_fixture(), diagnostic_fixture())
    assert "checkpoint, участвовавшего в сравнении: **2**" in report
    assert "диагностического запуска: **1**" in report
    assert "**Это отдельный запуск**" in report


def test_report_omits_untrusted_content() -> None:
    """Не допускать переноса исходных текстов, ID и абсолютных путей."""
    sample = comparison_fixture()
    sample["private_record"] = "SECRET_QUESTION_001"
    sample["comparison"]["labels"][0] = "databases"  # метки не подменяются
    output = render_final_report(sample, diagnostic_fixture())
    assert "SECRET_QUESTION_001" not in output
    assert "C:\\Users" not in output


def test_render_is_byte_identical_for_same_inputs() -> None:
    """Воспроизводимость: два рендера дают одинаковые байты и SHA-256."""
    first = render_final_report(comparison_fixture(), diagnostic_fixture())
    second = render_final_report(comparison_fixture(), diagnostic_fixture())
    assert first == second
    assert sha256_report(first) == sha256_report(second)


@pytest.mark.parametrize("change", [
    lambda x: x.update(source="synthetic"),
    lambda x: x.update(comparison_invariants_pass=False),
    lambda x: x["split_counts"].update(test=609),
    lambda x: x["split_sha256"].update(test="invalid"),
    lambda x: x["protocol"].update(test_used_for_selection=True),
    lambda x: x["protocol"].update(classical_fit_split="test"),
    lambda x: x["protocol"].update(neural_selection_split="test"),
    lambda x: x["comparison"].update(samples=609),
    lambda x: x["comparison"].update(labels=["a", "b", "c", "d"]),
    lambda x: x["comparison"]["classical"].update(accuracy=0.1),
    lambda x: x["comparison"]["classical"].update(macro_f1=0.1),
    lambda x: x["comparison"]["classical"]["confusion_matrix"][0].__setitem__(0, 129),
    lambda x: x["comparison"]["paired_errors"].update(both_wrong=49),
    lambda x: x["comparison"]["paired_errors"].update(prediction_disagreements=2),
    lambda x: x["comparison"].update(winning_model_by_macro_f1="neural"),
    lambda x: x["comparison"]["delta_neural_minus_classical"].update(macro_f1=0.0),
    lambda x: x["complexity"].update(neural_parameter_count=-1),
])
def test_rejects_corrupted_comparison(change) -> None:
    """Повреждение происхождения, матрицы или метрик не проходит аудит."""
    sample = comparison_fixture()
    change(sample)
    with pytest.raises(PhaseReportError):
        validate_comparison(sample)


@pytest.mark.parametrize("change", [
    lambda x: x.update(source="synthetic"),
    lambda x: x.update(test_evaluated=True),
    lambda x: x["split_counts"].update(train=10),
    lambda x: x["history"][1].update(epoch=4),
    lambda x: x["history"][1].update(train_loss=-0.1),
    lambda x: x.update(best_val_macro_f1=0.9),
    lambda x: x["interpretation"].update(signal="certain_overfitting"),
])
def test_rejects_corrupted_diagnostics(change) -> None:
    """История не должна маскировать утечку или подмену лучшей эпохи."""
    history = diagnostic_fixture()
    change(history)
    with pytest.raises(PhaseReportError):
        validate_diagnostics(history)


def test_readme_patch_preserves_user_text() -> None:
    """Навигация F2 добавляется точечно и не повторяется при повторном запуске."""
    original = "# ML Engineering Lab\nCUSTOM USER SECTION\n\nПлан последовательного раскрытия модулей: ...\n"
    edited = update_readme(original)
    assert "CUSTOM USER SECTION" in edited
    assert "docs/reports/F2_PHASE_REPORT.md" in edited
    assert edited.count("### PyTorch и нейросети (F2)") == 1
    assert update_readme(edited) == edited


def test_readme_rejects_missing_anchor() -> None:
    """Скрипт должен остановиться, а не переписывать незнакомый README."""
    with pytest.raises(PhaseReportError):
        update_readme("# совершенно другая структура")


def test_real_and_synthetic_are_not_interchangeable() -> None:
    """Проверяем запрет на замену реального test синтетическими примерами."""
    comp = comparison_fixture()
    comp["source"] = "synthetic"
    assert validate_comparison(comp, require_real=False)["source"] == "synthetic"
    with pytest.raises(PhaseReportError):
        validate_comparison(comp)


@pytest.fixture
def saved_outputs(tmp_path: Path) -> Path:
    """Создать правдоподобные публичные агрегаты, без настоящих данных F1."""
    root = tmp_path
    (root / "docs" / "reports").mkdir(parents=True)
    (root / "README.md").write_text(
        "# ML Engineering Lab\n\nПлан последовательного раскрытия модулей: ...\n",
        encoding="utf-8",
    )
    checkpoint_history = diagnostic_fixture()
    report = {**checkpoint_history,
              "checkpoint_reload_match": True,
              "validation_invariants_pass": True,
              "best_epoch": 1}
    export_diagnostics(report, root / "outputs" / "f2_6_diagnostics")
    folder = root / "outputs" / "f2_7_comparison"
    folder.mkdir(parents=True)
    # Тексты фиктивные. Исходные данные не публикуются; проверяем только SHA.
    fake_data = root / "data" / "processed"
    fake_data.mkdir(parents=True)
    fixture = comparison_fixture()
    for part in ("train", "val", "test"):
        raw = (f"dummy-{part}-only-for-test\n").encode("utf-8")
        (fake_data / f"{part}.jsonl").write_bytes(raw)
        fixture["split_sha256"][part] = hashlib.sha256(raw).hexdigest()
    (folder / "comparison.json").write_text(
        json.dumps(fixture, ensure_ascii=False), encoding="utf-8")
    return root


def test_end_to_end_export_from_aggregate_files(saved_outputs: Path) -> None:
    """На валидных JSON/CSV/PNG формируется согласованный публичный Markdown."""
    output = run(root=saved_outputs, update_navigation=True)
    assert output["no_training"] is True
    public = saved_outputs / "docs/reports/F2_PHASE_REPORT.md"
    assert public.is_file()
    assert "73 |" in public.read_text(encoding="utf-8")
    assert "PyTorch EmbeddingClassifier" in public.read_text(encoding="utf-8")
    assert "### PyTorch и нейросети (F2)" in (saved_outputs / "README.md").read_text(encoding="utf-8")
    a = public.read_bytes()
    run(root=saved_outputs, update_navigation=True)
    assert public.read_bytes() == a


def test_missing_comparison_does_not_touch_report(saved_outputs: Path) -> None:
    """Отсутствие F2.7 не должно затирать предыдущий хороший отчёт."""
    target = saved_outputs / "docs/reports/F2_PHASE_REPORT.md"
    target.write_text("EXISTING REPORT", encoding="utf-8")
    (saved_outputs / "outputs/f2_7_comparison/comparison.json").unlink()
    with pytest.raises(PhaseReportError):
        run(root=saved_outputs, update_navigation=True)
    assert target.read_text(encoding="utf-8") == "EXISTING REPORT"


def test_source_sha_mismatch_blocks_publication(saved_outputs: Path) -> None:
    """Подмена локального train после F2.7 должна блокировать итоговый отчёт."""
    (saved_outputs / "data/processed/train.jsonl").write_bytes(b"tampered data")
    with pytest.raises(PhaseReportError, match="train SHA-256"):
        run(root=saved_outputs, update_navigation=True)


def test_source_split_missing_blocks_publication(saved_outputs: Path) -> None:
    """Отсутствие локального test нельзя заменить сохранённой таблицей агрегатов."""
    (saved_outputs / "data/processed/test.jsonl").unlink()
    with pytest.raises(PhaseReportError, match="missing local test"):
        run(root=saved_outputs)


def test_corrupted_f2_6_png_blocks_publication(saved_outputs: Path) -> None:
    """Изменение одной картинки обнаруживается по PNG SHA до записи отчёта."""
    path = saved_outputs / "outputs/f2_6_diagnostics/training_loss.png"
    path.write_bytes(path.read_bytes() + b"modified")
    with pytest.raises(PhaseReportError):
        run(root=saved_outputs)


def test_update_readme_keeps_windows_crlf(saved_outputs: Path) -> None:
    """Точечное обновление не должно менять переводы строк всего README."""
    filename = saved_outputs / "README.md"
    filename.write_bytes(b"# ML Engineering Lab\r\n\r\n" +
                         "План последовательного раскрытия модулей: ...\r\n".encode("utf-8"))
    run(root=saved_outputs, update_navigation=True)
    raw = filename.read_bytes()
    assert raw.count(b"\r\n") > 3
    assert raw.replace(b"\r\n", b"").find(b"\n") == -1


def test_corrupted_f2_6_csv_blocks_publication(saved_outputs: Path) -> None:
    """Испорченный history.csv обнаруживается раньше, чем запись отчёта."""
    csv_path = saved_outputs / "outputs/f2_6_diagnostics/history.csv"
    csv_path.write_bytes(csv_path.read_bytes() + b"unexpected-data")
    with pytest.raises(PhaseReportError):
        build_from_artifacts(csv_path.parent,
                             saved_outputs / "outputs/f2_7_comparison/comparison.json")


def test_nonmatching_split_sizes_block_publication(saved_outputs: Path) -> None:
    """Нельзя складывать агрегаты от разных train/val/test разбиений."""
    filename = saved_outputs / "outputs/f2_7_comparison/comparison.json"
    data = json.loads(filename.read_text(encoding="utf-8"))
    data["split_counts"]["val"] = 607
    filename.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(PhaseReportError):
        build_from_artifacts(saved_outputs / "outputs/f2_6_diagnostics", filename)


def test_cli_self_check() -> None:
    """CLI smoke выполняется без реального датасета и обучения на CI."""
    process = subprocess.run(
        [sys.executable, "-m", "scripts.build_f2_report", "--self-check", "--check"],
        capture_output=True, text=True, timeout=30,
    )
    assert process.returncode == 0, process.stderr + process.stdout
    assert "F2.8 Final Phase Report CHECK: PASS" in process.stdout
