"""F2.8: проверка агрегатов F2.6/F2.7 и воспроизводимый итоговый отчёт.

Модуль не читает исходные Q&A, не загружает веса и не выполняет обучение.
В публичный отчёт попадают только проверенные числовые показатели.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

from src.pytorch_f2.training_diagnostics import verify_export, DiagnosticsError


class PhaseReportError(ValueError):
    """Нарушен формат, происхождение или согласованность результатов F2."""


def _count(value: Any, field: str, *, minimum: int = 0) -> int:
    """Проверить неотрицательный целочисленный счётчик без преобразования bool."""
    if type(value) is not int or value < minimum:
        raise PhaseReportError(f"invalid {field}")
    return value


def _score(value: Any, field: str) -> float:
    """Принять только конечное значение метрики в диапазоне [0, 1]."""
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
        raise PhaseReportError(f"invalid {field}")
    return float(value)


def _load_json(path: Path) -> dict:
    """Загрузить JSON-объект с понятной ошибкой при отсутствии или повреждении."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PhaseReportError(f"cannot read {Path(path).name}") from exc
    if not isinstance(data, dict):
        raise PhaseReportError("artifact root must be an object")
    return data


def _validated_counts(document: Mapping[str, Any]) -> dict[str, int]:
    """Сверить размеры разбиений с зафиксированным контрактом F1."""
    counts = document.get("split_counts")
    if not isinstance(counts, dict) or set(counts) != {"train", "val", "test"}:
        raise PhaseReportError("invalid split_counts")
    clean = {part: _count(counts[part], part, minimum=1) for part in ("train", "val", "test")}
    return clean


def _metric_row(row: Any, labels: list[str], n: int, name: str) -> dict[str, Any]:
    """Проверить, что Accuracy согласована с матрицей ошибок и числом объектов.

    Macro F1 пересчитывается по строкам confusion matrix; это не зависит
    от внешней версии scikit-learn и исключает подмену только итоговой метрики.
    """
    if not isinstance(row, dict):
        raise PhaseReportError(f"missing {name} metrics")
    accuracy = _score(row.get("accuracy"), f"{name}.accuracy")
    macro = _score(row.get("macro_f1"), f"{name}.macro_f1")
    matrix = row.get("confusion_matrix")
    k = len(labels)
    if (not isinstance(matrix, list) or len(matrix) != k or
            any(not isinstance(line, list) or len(line) != k for line in matrix)):
        raise PhaseReportError(f"invalid {name} confusion matrix shape")
    if any(type(value) is not int or value < 0 for line in matrix for value in line):
        raise PhaseReportError(f"invalid {name} confusion matrix cell")
    if sum(sum(line) for line in matrix) != n:
        raise PhaseReportError(f"wrong {name} confusion matrix sample count")
    if row.get("labels") != labels or _count(row.get("n"), f"{name}.n") != n:
        raise PhaseReportError(f"mismatched {name} labels or sample count")
    right = sum(matrix[i][i] for i in range(k))
    if abs(accuracy - right / n) > 1e-8:
        raise PhaseReportError(f"{name} accuracy disagrees with confusion matrix")
    per_class = []
    for i in range(k):
        tp = matrix[i][i]
        fp = sum(matrix[r][i] for r in range(k)) - tp
        fn = sum(matrix[i]) - tp
        per_class.append(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0)
    if abs(macro - sum(per_class) / k) > 1e-8:
        raise PhaseReportError(f"{name} macro F1 disagrees with confusion matrix")
    return {"accuracy": accuracy, "macro_f1": macro, "errors": n - right}


def validate_comparison(report: Mapping[str, Any], *, require_real: bool = True) -> dict:
    """Проверить честность сравнения и вернуть только безопасные агрегаты.

    Не принимать произвольные тексты или пути из входного JSON. Сначала
    проверяются split/протокол, затем confusion matrices и парные ошибки.
    """
    if not isinstance(report, Mapping):
        raise PhaseReportError("comparison must be an object")
    source = report.get("source")
    if source not in ("synthetic", "local_f1_split") or (require_real and source != "local_f1_split"):
        raise PhaseReportError("real F2.7 comparison required")
    if report.get("comparison_invariants_pass") is not True:
        raise PhaseReportError("F2.7 invariants are not confirmed")
    counts = _validated_counts(report)
    if require_real and counts != {"train": 1824, "val": 608, "test": 608}:
        raise PhaseReportError("unexpected real F1 split size")
    protocol = report.get("protocol")
    if (not isinstance(protocol, dict) or protocol.get("test_used_for_selection") is not False
            or protocol.get("classical_fit_split") != "train"
            or protocol.get("neural_selection_split") != "val"
            or protocol.get("neural_selection_metric") != "macro_f1"):
        raise PhaseReportError("test leakage or unsupported selection protocol")
    epoch = _count(protocol.get("neural_best_epoch"), "best_epoch", minimum=1)
    val_f1 = _score(protocol.get("neural_validation_macro_f1"), "validation F1")
    sha = report.get("split_sha256")
    if not isinstance(sha, dict) or set(sha) != set(counts):
        raise PhaseReportError("missing split fingerprints")
    if require_real and any(not isinstance(sha[x], str) or len(sha[x]) != 64 or
                            any(c not in "0123456789abcdef" for c in sha[x])
                            for x in counts):
        raise PhaseReportError("invalid real split SHA-256")
    comp = report.get("comparison")
    if not isinstance(comp, dict) or _count(comp.get("samples"), "samples", minimum=1) != counts["test"]:
        raise PhaseReportError("comparison sample count mismatch")
    labels = comp.get("labels")
    if (not isinstance(labels, list) or len(labels) < 2 or
            not all(isinstance(x, str) and x for x in labels) or labels != sorted(set(labels))):
        raise PhaseReportError("invalid classes")
    if require_real and labels != ["databases", "linux", "programming", "systems"]:
        raise PhaseReportError("real F1 class set differs from fixed four-class task")
    classical = _metric_row(comp.get("classical"), labels, counts["test"], "classical")
    neural = _metric_row(comp.get("neural"), labels, counts["test"], "neural")
    paired = comp.get("paired_errors")
    names = ("both_correct", "both_wrong", "only_classical_wrong", "only_neural_wrong", "prediction_disagreements")
    if not isinstance(paired, dict) or set(paired) != set(names):
        raise PhaseReportError("invalid paired errors")
    pair = {name: _count(paired[name], name) for name in names}
    if (sum(pair[name] for name in names[:-1]) != counts["test"] or
            classical["errors"] != pair["both_wrong"] + pair["only_classical_wrong"] or
            neural["errors"] != pair["both_wrong"] + pair["only_neural_wrong"] or
            not (pair["only_classical_wrong"] + pair["only_neural_wrong"] <=
                 pair["prediction_disagreements"] <= counts["test"] - pair["both_correct"])):
        raise PhaseReportError("paired errors disagree with confusion matrices")
    delta = comp.get("delta_neural_minus_classical")
    if not isinstance(delta, dict) or any(
        type(delta.get(key)) not in (int, float) or not math.isfinite(delta[key]) or
        abs(delta[key] - (neural[key] - classical[key])) > 1e-8
        for key in ("accuracy", "macro_f1")
    ):
        raise PhaseReportError("invalid F2 minus F1 deltas")
    winner = ("neural" if neural["macro_f1"] > classical["macro_f1"] else
              "classical" if neural["macro_f1"] < classical["macro_f1"] else "tie")
    if comp.get("winning_model_by_macro_f1") != winner:
        raise PhaseReportError("winner mismatches macro F1")
    complexity = report.get("complexity")
    if not isinstance(complexity, dict):
        raise PhaseReportError("missing complexity metadata")
    parameters = _count(complexity.get("neural_parameter_count"), "parameter count", minimum=1)
    return {"source": source, "split_counts": counts, "labels_count": len(labels),
            "classical": classical, "neural": neural, "paired_errors": pair,
            "best_epoch": epoch, "validation_macro_f1": val_f1,
            "winner": winner, "neural_parameter_count": parameters,
            "test_sha256": sha["test"],
            "measured_runtime_comparison": complexity.get("measured_runtime_comparison") is True}


def validate_diagnostics(report: Mapping[str, Any], *, require_real: bool = True) -> dict:
    """Проверить метрики F2.6 без трактовки отдельных эпох как F2.5 checkpoint.

    F2.6 проводит самостоятельное обучение и может выбрать иную эпоху.
    Её значения не должны выдавать за свойства checkpoint из F2.7.
    """
    if not isinstance(report, Mapping) or report.get("format_version") != 1:
        raise PhaseReportError("invalid F2.6 diagnostic schema")
    source = report.get("source")
    if source not in ("synthetic", "local_f1_split") or (require_real and source != "local_f1_split"):
        raise PhaseReportError("real F2.6 diagnostics required")
    if (report.get("trained_split") != "train" or report.get("selection_split") != "val" or
            report.get("test_evaluated") is not False or report.get("selection_metric") != "val_macro_f1"):
        raise PhaseReportError("F2.6 violates train/validation/test isolation")
    counts = _validated_counts(report)
    if require_real and counts != {"train": 1824, "val": 608, "test": 608}:
        raise PhaseReportError("unexpected F2.6 split sizes")
    epochs = _count(report.get("epochs"), "epochs", minimum=1)
    best_epoch = _count(report.get("best_epoch"), "best_epoch", minimum=1)
    if best_epoch > epochs:
        raise PhaseReportError("F2.6 best epoch out of range")
    best_score = _score(report.get("best_val_macro_f1"), "best F2.6 F1")
    history = report.get("history")
    if not isinstance(history, list) or len(history) != epochs:
        raise PhaseReportError("F2.6 history length mismatch")
    clean = []
    for expected, row in enumerate(history, start=1):
        if not isinstance(row, dict) or _count(row.get("epoch"), "epoch") != expected:
            raise PhaseReportError("out-of-order epoch")
        vals = {key: _score(row.get(key), key) if key in ("val_accuracy", "val_macro_f1") else
                _nonnegative(row.get(key), key)
                for key in ("train_loss", "val_loss", "val_accuracy", "val_macro_f1")}
        clean.append({"epoch": expected, **vals})
    if (abs(clean[best_epoch - 1]["val_macro_f1"] - best_score) > 1e-8 or
            max(r["val_macro_f1"] for r in clean) > best_score + 1e-8):
        raise PhaseReportError("F2.6 best F1 disagrees with history")
    signal = report.get("interpretation")
    if not isinstance(signal, dict) or signal.get("signal") not in {
        "insufficient_epochs", "possible_overfitting", "no_clear_learning_progress",
        "no_clear_overfitting_signal"}:
        raise PhaseReportError("invalid diagnostics interpretation")
    return {"source": source, "split_counts": counts, "epochs": epochs,
            "best_epoch": best_epoch, "best_val_macro_f1": best_score,
            "history": clean, "signal": signal["signal"]}


def _nonnegative(value: Any, name: str) -> float:
    """Проверить конечность loss, не ограничивая его сверху единицей."""
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise PhaseReportError(f"invalid {name}")
    return float(value)



def verify_source_fingerprints(root: Path, comparison: Mapping[str, Any]) -> None:
    """Пересчитать SHA-256 локальных F1 JSONL без чтения их содержимого в отчёт.

    Сохранённый отчёт F2.7 сам подтверждает хеши. Здесь дополнительно
    проверяется, что текущие локальные train/val/test остались неизменными.
    """
    source = comparison.get("split_sha256")
    if not isinstance(source, dict) or set(source) != {"train", "val", "test"}:
        raise PhaseReportError("missing source split fingerprints")
    for part in ("train", "val", "test"):
        filename = Path(root) / "data" / "processed" / f"{part}.jsonl"
        if not filename.is_file():
            raise PhaseReportError(f"missing local {part} split")
        sha = hashlib.sha256()
        try:
            with filename.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    sha.update(block)
        except OSError as exc:
            raise PhaseReportError(f"cannot read local {part} split") from exc
        if sha.hexdigest() != source[part]:
            raise PhaseReportError(f"local {part} SHA-256 mismatches F2.7")


def render_final_report(comparison: Mapping[str, Any], diagnostics: Mapping[str, Any]) -> str:
    """Сформировать Markdown строго из проверенных агрегатов и фиксированного текста.

    Ни локальные пути, ни идентификаторы объектов, ни произвольные поля
    входных JSON в отчёт не копируются.
    """
    a = validate_comparison(comparison)
    b = validate_diagnostics(diagnostics)
    if a["split_counts"] != b["split_counts"]:
        raise PhaseReportError("F2.6/F2.7 split sizes mismatch")
    pa = a["paired_errors"]
    train, val, test = (a["split_counts"][name] for name in ("train", "val", "test"))
    text = [
        "# F2 — PyTorch & Neural Networks: итоговый технический отчёт",
        "",
        "> Сформировано F2.8 из локальных агрегатов F2.6 и F2.7; не результат CI-обучения.",
        "",
        "## Задача и архитектура",
        "",
        "F2 расширяет подготовленный в F1 корпус технических вопросов до небольшого PyTorch-классификатора.",
        "Модель предсказывает **категорию вопроса**, а не достоверность технического ответа.",
        "",
        "```mermaid",
        "flowchart TD",
        "    A[\"F1.6: train / validation / test\"] --> B[\"F2.2: токенизация и train-only словарь\"]",
        "    B --> C[\"Embedding → Masked mean pooling\"]",
        "    C --> D[\"Linear → ReLU → Linear logits\"]",
        "    D --> E[\"F2.4: CrossEntropy + backward + AdamW\"]",
        "    E --> F[\"F2.5: Validation Macro F1 → checkpoint\"]",
        "    F --> G[\"F2.6: диагностика\"]",
        "    F --> H[\"F2.7: test на зафиксированной модели\"]",
        "    I[\"F1.7: сохранённые TF-IDF + Logistic Regression predictions\"] --> H",
        "    H --> J[\"F2.8: парное сравнение и отчёт\"]",
        "    G --> J",
        "```",
        "",
        "## Этапы и инженерные результаты",
        "",
        "| Этап | Проверенный механизм |",
        "|---|---|",
        "| F2.1 | Tensor shapes, computation graph, autograd, градиенты и их накопление |",
        "| F2.2 | Dataset/DataLoader, padding, truncation, маски, train-only vocabulary |",
        "| F2.3 | EmbeddingClassifier с masked pooling и логитами |",
        "| F2.4 | CrossEntropyLoss, обратный проход и AdamW |",
        "| F2.5 | Оценка на validation, выбор по Macro F1, восстановление checkpoint |",
        "| F2.6 | Графики train/val, история и диагностические проверки |",
        "| F2.7 | Контроль SHA-256/ID и парное сравнение на одном test |",
        "| F2.8 | Проверка согласованности агрегатов и настоящий итоговый отчёт |",
        "",
        "## Данные и изоляция разбиений",
        "",
        f"Train: **{train}**, validation: **{val}**, test: **{test}**; число классов: **{a['labels_count']}**.",
        f"SHA-256 локального test JSONL: `{a['test_sha256']}`.",
        "Словарь и параметры обеих моделей определены без обращения к test.",
        "F1 использует сохранённые test-предсказания; F2 — checkpoint, заранее выбранный по validation.",
        "Test был прочитан для итоговой оценки, **не для настройки модели**.",
        "",
        "## Сравнение моделей на фиксированном test",
        "",
        "| Модель | Accuracy | Macro F1 | Ошибок из test |",
        "|---|---:|---:|---:|",
        f"| TF-IDF + Logistic Regression | {a['classical']['accuracy']:.4f} | {a['classical']['macro_f1']:.4f} | {a['classical']['errors']} |",
        f"| PyTorch EmbeddingClassifier | {a['neural']['accuracy']:.4f} | {a['neural']['macro_f1']:.4f} | {a['neural']['errors']} |",
        "",
        f"Δ Accuracy (PyTorch − Classical) = **{a['neural']['accuracy'] - a['classical']['accuracy']:+.4f}**; "
        f"Δ Macro F1 = **{a['neural']['macro_f1'] - a['classical']['macro_f1']:+.4f}**.",
        f"Лучшая эпоха checkpoint, участвовавшего в сравнении: **{a['best_epoch']}**, "
        f"его validation Macro F1: **{a['validation_macro_f1']:.4f}**.",
        f"Количество параметров F2: **{a['neural_parameter_count']}**.",
        "",
        "## Парный анализ ошибок",
        "",
        "| Ситуация | Количество |",
        "|---|---:|",
        f"| Обе модели верны | {pa['both_correct']} |",
        f"| Обе модели ошибаются | {pa['both_wrong']} |",
        f"| Только классическая ошибается | {pa['only_classical_wrong']} |",
        f"| Только нейросеть ошибается | {pa['only_neural_wrong']} |",
        f"| Различающиеся предсказанные классы | {pa['prediction_disagreements']} |",
        "",
        "## Диагностика F2.6 — самостоятельный запуск обучения",
        "",
        f"Эпох: **{b['epochs']}**. Лучшая эпоха этого диагностического запуска: "
        f"**{b['best_epoch']}**, validation Macro F1: **{b['best_val_macro_f1']:.4f}**.",
        f"Эвристический сигнал: `{b['signal']}`.",
        "**Это отдельный запуск**, не обязательно совпадающий с checkpoint F2.5/F2.7. "
        "Нельзя переносить его лучшую эпоху или F1 в описание другой модели.",
        "",
        "| Эпоха | Training loss | Validation loss | Val Accuracy | Val Macro F1 |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in b["history"]:
        text.append(f"| {row['epoch']} | {row['train_loss']:.6f} | {row['val_loss']:.6f} | "
                    f"{row['val_accuracy']:.4f} | {row['val_macro_f1']:.4f} |")
    text.extend([
        "",
        "## Выводы и ограничения",
        "",
        ("На этом эксперименте classical baseline точнее по Macro F1."
         if a["winner"] == "classical" else
         "На этом эксперименте PyTorch точнее по Macro F1."
         if a["winner"] == "neural" else
         "На этом эксперименте модели имеют одинаковый Macro F1."),
        "Это сравнение только на одном фиксированном test; устойчивость по другим seed/split не проверялась.",
        "Обучающие данные не включаются в публичный репозиторий из-за нерешённых вопросов атрибуции.",
        "CPU-наблюдения не подтверждают производительность на GPU и не заменяют latency benchmark.",
        ("Сопоставимые затраты времени не измерялись и в итоговую таблицу не включены."
         if not a["measured_runtime_comparison"] else
         "В артефакте есть флаг измерения времени; без протокола и единиц здесь не сравниваем затраты."),
        "Macro F1 описывает классификацию по форумным разделам, не фактическую корректность ответа.",
        "Не допускается улучшать архитектуру после просмотра test и выдавать это за ту же независимую оценку.",
        "",
        "## Файлы, с которыми стоит познакомиться",
        "",
        "| Файл | Что изучить |",
        "|---|---|",
        "| `src/pytorch_f2/data_pipeline.py` | Источник разбиений, словарь и батчи |",
        "| `src/pytorch_f2/embedding_classifier.py` | Embedding, pooling, logits |",
        "| `src/pytorch_f2/training_loop.py` | Градиенты, loss и optimizer |",
        "| `src/pytorch_f2/validation_checkpoint.py` | Selection/checkpoint без test |",
        "| `src/pytorch_f2/training_diagnostics.py` | Проверка истории и графиков |",
        "| `src/pytorch_f2/model_comparison.py` | Выравнивание двух моделей на test |",
        "| `src/pytorch_f2/final_report.py` | Итоговые инварианты, связи метрик и безопасный экспорт |",
        "| `tests/test_final_report.py` | Порча входных артефактов, воспроизводимость и контроль публикации |",
        "",
    ])
    return "\n".join(text)


def build_from_artifacts(diagnostics_dir: Path, comparison_file: Path, *,
                         require_real: bool = True) -> str:
    """Проверить целостность файлов F2.6 и F2.7 и получить текст отчёта.

    Сначала проверяется набор PNG/CSV/JSON F2.6 по уже существующему алгоритму,
    затем F2.7. При ошибке результат не записывается.
    """
    folder = Path(diagnostics_dir)
    try:
        verify_export(folder)
    except (DiagnosticsError, OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        raise PhaseReportError("invalid or damaged F2.6 export") from exc
    compare = _load_json(comparison_file)
    diag = _load_json(folder / "metrics.json")
    if require_real:
        return render_final_report(compare, diag)
    # Синтетический режим разрешён лишь для проверок: реальные выводы
    # не должны бездумно подменяться синтетическими в публичном отчёте.
    validate_comparison(compare, require_real=False)
    validate_diagnostics(diag, require_real=False)
    return "F2.8 synthetic artifact validation: PASS\n"


def update_readme(text: str) -> str:
    """Добавить ссылку на F2 без перезаписи остальных разделов README.

    Обрабатываются только однозначные якоря, повторное применение идемпотентно.
    Не использовать готовую копию README из ZIP: пользователь мог менять его
    на GitHub после формирования архивов предыдущих этапов.
    """
    marker = "### PyTorch и нейросети (F2)"
    if marker in text:
        if text.count(marker) != 1:
            raise PhaseReportError("ambiguous F2 section")
        return text
    anchor = "План последовательного раскрытия модулей:"
    if text.count(anchor) != 1:
        raise PhaseReportError("README navigation anchor is missing or ambiguous")
    section = (
        "### PyTorch и нейросети (F2)\n\n"
        "Восемь учебных этапов F2.1–F2.8: тензоры, загрузка данных, классификатор, "
        "обучение, validation/checkpoint, диагностика и сравнение с F1.\n\n"
        "[Итоговый отчёт Фазы 2](docs/reports/F2_PHASE_REPORT.md) · "
        "[Алгоритмы F2.8](docs/steps/F2_08_final_phase_report.md).\n\n"
    )
    return text.replace(anchor, section + anchor, 1)


def sha256_report(markdown: str) -> str:
    """Вернуть SHA-256 публичного отчёта для проверки детерминированности."""
    return hashlib.sha256(markdown.encode("utf-8")).hexdigest()
