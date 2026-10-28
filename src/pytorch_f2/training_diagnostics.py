"""F2.6: проверяемая визуализация агрегированной истории train/validation.

Модуль не получает тексты вопросов, DataLoader или test-метрики. Источник
всех точек — история F2.5; никаких дополнительных проходов модели здесь нет.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")  # Рендеринг в файл без GUI, пригодный для CI и Windows.
from matplotlib import pyplot as plt


class DiagnosticsError(ValueError):
    """История, её метаданные или путь вывода нарушают контракт F2.6."""


_COLUMNS = ("epoch", "train_loss", "val_loss", "val_accuracy", "val_macro_f1")
_IMAGES = (
    ("training_loss.png", "train_loss", "Training loss", None),
    ("validation_loss.png", "val_loss", "Validation loss", None),
    ("validation_macro_f1.png", "val_macro_f1", "Validation Macro F1", (0.0, 1.0)),
)


def _numeric(value: Any, name: str, *, minimum: float = 0.0, maximum: float | None = None) -> float:
    """Проверить конечность и допустимый диапазон метрики без неявного bool→int."""
    if type(value) not in (float, int) or not math.isfinite(value):
        raise DiagnosticsError(f"{name} must be a finite number")
    if value < minimum or (maximum is not None and value > maximum):
        raise DiagnosticsError(f"{name} is outside its valid range")
    return float(value)


def normalize_history(report: Mapping[str, Any]) -> dict[str, Any]:
    """Преобразовать F2.5 JSON-отчёт в строгую безопасную структуру F2.6.

    Принимаются исключительно агрегированные числовые метрики. Проверяем
    номера эпох, разделы данных, выбор лучшей эпохи и отсутствие test-оценки.
    Все необязательные исходные поля (пути, ID и тексты) отбрасываются.
    """
    if not isinstance(report, Mapping):
        raise DiagnosticsError("F2.5 report must be a mapping")
    if (report.get("trained_split") != "train" or report.get("selection_split") != "val"
            or report.get("test_evaluated") is not False
            or report.get("selection_metric") != "val_macro_f1"
            or report.get("checkpoint_reload_match") is not True
            or report.get("validation_invariants_pass") is not True):
        raise DiagnosticsError("F2.5 split, selection or checkpoint invariants failed")
    if report.get("source") not in ("synthetic", "local_f1_split"):
        raise DiagnosticsError("unknown F2.5 source")
    split_counts = report.get("split_counts")
    if not isinstance(split_counts, Mapping) or set(split_counts) != {"train", "val", "test"}:
        raise DiagnosticsError("invalid split counts")
    if any(type(split_counts[name]) is not int or split_counts[name] < 1 for name in split_counts):
        raise DiagnosticsError("split counts must be positive integers")
    count = report.get("epochs")
    history = report.get("history")
    if type(count) is not int or count < 1 or not isinstance(history, (list, tuple)) or len(history) != count:
        raise DiagnosticsError("incorrect history length")
    cleaned = []
    for index, row in enumerate(history, start=1):
        if not isinstance(row, Mapping) or type(row.get("epoch")) is not int or row["epoch"] != index:
            raise DiagnosticsError("epochs must be consecutive starting from 1")
        cleaned.append({
            "epoch": index,
            "train_loss": _numeric(row.get("train_loss"), "train_loss"),
            "val_loss": _numeric(row.get("val_loss"), "val_loss"),
            "val_accuracy": _numeric(row.get("val_accuracy"), "val_accuracy", maximum=1.0),
            "val_macro_f1": _numeric(row.get("val_macro_f1"), "val_macro_f1", maximum=1.0),
        })
    best_epoch = report.get("best_epoch")
    if type(best_epoch) is not int or not 1 <= best_epoch <= count:
        raise DiagnosticsError("best_epoch outside history")
    best_score = _numeric(report.get("best_val_macro_f1"), "best_val_macro_f1", maximum=1.0)
    # Исходный CLI округляет метрики до 8 знаков; tolerances не должны
    # ошибочно объявлять несогласованной историю из-за последнего разряда.
    if (abs(cleaned[best_epoch - 1]["val_macro_f1"] - best_score) > 1e-8
            or max(row["val_macro_f1"] for row in cleaned) - best_score > 1e-8):
        raise DiagnosticsError("best checkpoint does not match validation history")
    return {
        "format_version": 1,
        "source": report["source"],
        "split_counts": {name: split_counts[name] for name in ("train", "val", "test")},
        "trained_split": "train",
        "selection_split": "val",
        "test_evaluated": False,
        "selection_metric": "val_macro_f1",
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_score,
        "epochs": count,
        "history": cleaned,
    }


def interpret_history(history: list[dict[str, float]], best_epoch: int) -> dict[str, Any]:
    """Зафиксировать наблюдаемые тренды без выдачи эвристики за диагноз.

    Рост validation loss при падении train loss — лишь сигнал возможного
    переобучения. На коротких рядах и малой validation выборке выводы
    ограничены; изменение F1 не обязано быть монотонным.
    """
    first, last = history[0], history[-1]
    train_change = last["train_loss"] - first["train_loss"]
    val_change = last["val_loss"] - first["val_loss"]
    f1_change = last["val_macro_f1"] - first["val_macro_f1"]
    # Для двух точек трудно делать вывод об устойчивости тренда.
    if len(history) < 3:
        signal = "insufficient_epochs"
    elif train_change < -1e-8 and val_change > 1e-8:
        signal = "possible_overfitting"
    elif train_change >= -1e-8 and f1_change <= 1e-8:
        signal = "no_clear_learning_progress"
    else:
        signal = "no_clear_overfitting_signal"
    return {
        "signal": signal,
        "train_loss_change": round(train_change, 8),
        "val_loss_change": round(val_change, 8),
        "val_macro_f1_change": round(f1_change, 8),
        "best_epoch": best_epoch,
        "caveat": "Эвристика по train/val-кривым; не оценка test и не доказательство причин.",
    }


def _csv_bytes(rows: list[dict[str, Any]]) -> bytes:
    """Экспортировать ровно точки графиков в детерминированном порядке."""
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=_COLUMNS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def _plot(rows: list[dict[str, Any]], *, field: str, title: str,
          best_epoch: int, bounds: tuple[float, float] | None, target: Path) -> None:
    """Построить одну PNG-кривую строго по проверенным точкам CSV.

    Линия лучшей эпохи относится к val Macro F1, а не к минимуму loss.
    """
    figure, axis = plt.subplots(figsize=(7.6, 4.3), dpi=140)
    try:
        epochs = [item["epoch"] for item in rows]
        values = [item[field] for item in rows]
        axis.plot(epochs, values, marker="o", linewidth=2)
        axis.axvline(best_epoch, linestyle="--", linewidth=1.2,
                     label="Best val Macro F1 epoch")
        axis.set_title(title)
        axis.set_xlabel("Epoch")
        axis.set_ylabel(title)
        axis.set_xticks(epochs)
        if bounds:
            axis.set_ylim(*bounds)
        axis.grid(alpha=0.25)
        axis.legend(loc="best")
        figure.tight_layout()
        figure.savefig(target, dpi=140, metadata={"Software": "ML Engineering Lab F2.6"})
    finally:
        plt.close(figure)


def _report_markdown(document: dict[str, Any]) -> str:
    """Короткий локальный технический отчёт только по агрегатам."""
    hist = document["history"]
    head = [
        "# F2.6 — Диагностика обучения",
        "",
        f"Источник: {document['source']}. Выбор лучшей эпохи: val Macro F1.",
        f"Эпох: {document['epochs']}. Лучшая эпоха: {document['best_epoch']}; "
        f"Macro F1: {document['best_val_macro_f1']:.8f}.",
        "Train применяется для обновления весов; validation — для выбора модели; test не оценивался.",
        "",
        "| Эпоха | Train loss | Val loss | Val Accuracy | Val Macro F1 |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in hist:
        head.append(f"| {row['epoch']} | {row['train_loss']:.8f} | "
                    f"{row['val_loss']:.8f} | {row['val_accuracy']:.8f} | "
                    f"{row['val_macro_f1']:.8f} |")
    head.extend(["", "## Диагностическая интерпретация", "",
                 f"Сигнал: `{document['interpretation']['signal']}`.",
                 document["interpretation"]["caveat"], "",
                 "Графики: `training_loss.png`, `validation_loss.png`, `validation_macro_f1.png`.",
                 "Их точки совпадают с `history.csv`; контроль SHA-256 CSV находится в `metrics.json`.", ""])
    return "\n".join(head)


def export_diagnostics(f2_5_report: Mapping[str, Any], output_dir: Path, *,
                       parameters: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Записать метрики, CSV, три PNG и текстовый отчёт из одной истории.

    Args:
        f2_5_report: Готовый отчёт F2.5 без повторного чтения данных.
        output_dir: Изолированная папка для локальных результатов.
        parameters: Нечувствительные параметры эксперимента (только whitelist).
    Returns:
        Проверенную структуру metrics.json, включая хеш исходных точек.
    """
    document = normalize_history(f2_5_report)
    if not isinstance(output_dir, (str, Path)):
        raise DiagnosticsError("output_dir must be a path")
    folder = Path(output_dir)
    if folder.exists() and not folder.is_dir():
        raise DiagnosticsError("output_dir is not a directory")
    if parameters is not None:
        allowed = {"seed", "batch_size", "max_length", "max_vocab_size", "embedding_dim",
                   "hidden_dim", "learning_rate", "weight_decay"}
        if not isinstance(parameters, Mapping) or not set(parameters) <= allowed:
            raise DiagnosticsError("unexpected experiment parameters")
        for name, value in parameters.items():
            _numeric(value, name)
        document["parameters"] = dict(parameters)
    document["interpretation"] = interpret_history(document["history"], document["best_epoch"])
    csv_bytes = _csv_bytes(document["history"])
    document["history_csv_sha256"] = hashlib.sha256(csv_bytes).hexdigest()
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "history.csv").write_bytes(csv_bytes)
    image_hashes = {}
    for filename, field, label, bounds in _IMAGES:
        _plot(document["history"], field=field, title=label,
              best_epoch=document["best_epoch"], bounds=bounds, target=folder / filename)
        image_hashes[filename] = hashlib.sha256((folder / filename).read_bytes()).hexdigest()
    # Хеши PNG защищают связь экспорта с сохранёнными изображениями от
    # последующей незаметной замены, но сами по себе не доказывают адекватность анализа.
    document["images_sha256"] = image_hashes
    (folder / "metrics.json").write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (folder / "analysis.md").write_text(_report_markdown(document), encoding="utf-8")
    verify_export(folder)
    return document


def verify_export(output_dir: Path) -> None:
    """Сверить CSV, JSON, хеш и PNG после записи; исключить тихую подмену точек."""
    folder = Path(output_dir)
    document = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
    if document.get("format_version") != 1:
        raise DiagnosticsError("unknown diagnostics format")
    csv_bytes = (folder / "history.csv").read_bytes()
    if hashlib.sha256(csv_bytes).hexdigest() != document.get("history_csv_sha256"):
        raise DiagnosticsError("history CSV hash mismatch")
    if csv_bytes != _csv_bytes(document["history"]):
        raise DiagnosticsError("CSV does not match metrics.json")
    for filename, _, _, _ in _IMAGES:
        raw = (folder / filename).read_bytes()
        if raw[:8] != b"\x89PNG\r\n\x1a\n":
            raise DiagnosticsError(f"invalid PNG: {filename}")
        if hashlib.sha256(raw).hexdigest() != document.get("images_sha256", {}).get(filename):
            raise DiagnosticsError(f"PNG hash mismatch: {filename}")
