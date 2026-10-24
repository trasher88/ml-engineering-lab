"""F1.9: aggregate-only final report, validated against local F1.4–F1.8 audits.

No source Q&A, record IDs, model weights or row-level predictions are read
into the output report. Test split byte hashes are checked by the CLI wrapper.
"""
from __future__ import annotations

from collections import Counter
import math
import re
from typing import Any


class F1ReportError(ValueError):
    """Missing/inconsistent audit evidence; report must not be published."""


SPLITS = ("train", "val", "test")
OVERLAPS = ("train/test", "train/val", "val/test")


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise F1ReportError(f"invalid {label}: expected integer >= {minimum}")
    return value


def _number(value: Any, label: str, *, lower: float = 0, upper: float = 1) -> float:
    if type(value) not in (float, int) or not math.isfinite(value) or not lower <= value <= upper:
        raise F1ReportError(f"invalid {label}: expected finite value in [{lower}, {upper}]")
    return float(value)


def _digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise F1ReportError(f"invalid {label} SHA-256")
    return value


def _labels(value: Any) -> list[str]:
    if not isinstance(value, list) or len(value) != 4 or any(
        not isinstance(x, str) or re.fullmatch(r"[a-z][a-z_]*", x) is None for x in value
    ) or set(value) != {"databases", "linux", "programming", "systems"}:
        raise F1ReportError("unexpected class labels")
    return value


def _dict(value: Any, label: str) -> dict:
    if not isinstance(value, dict):
        raise F1ReportError(f"missing/invalid {label}")
    return value


def validate_audits(cleaning: dict, eda: dict, split: dict,
                    classical: dict, errors: dict, manifest: dict) -> dict:
    """Reconcile five stage audits; return ONLY allowlisted aggregate metrics."""
    for name, doc, stage in (("cleaning", cleaning, "F1.4"), ("eda", eda, "F1.5"),
                             ("split", split, "F1.6"), ("classical", classical, "F1.7"),
                             ("errors", errors, "F1.8")):
        if _dict(doc, name).get("stage") != stage:
            raise F1ReportError(f"stage mismatch: {name}")
    sources = manifest.get("sources")
    if not isinstance(sources, list) or len(sources) != 4:
        raise F1ReportError("unexpected manifest source count")
    selected = sum(_integer(_dict(x, "source").get("selected_rows"), "source.selected_rows", minimum=1) for x in sources)
    if len({x.get("file") for x in sources}) != 4:
        raise F1ReportError("duplicate manifest sources")
    clean = _dict(cleaning.get("cleaning"), "cleaning counters")
    inp = _integer(clean.get("input_rows"), "input_rows", minimum=1)
    out = _integer(clean.get("output_rows"), "output_rows", minimum=1)
    short = _integer(clean.get("short_removed"), "short_removed")
    duplicate = _integer(clean.get("duplicates_removed"), "duplicates_removed")
    invalid = _integer(clean.get("invalid_or_missing"), "invalid_or_missing")
    if inp != selected or inp != out + short + duplicate + invalid:
        raise F1ReportError("F1.4 cleaning/manifest row totals inconsistent")
    cleaned_sha = _digest(eda.get("source_sha256"), "EDA cleaned source")
    if split.get("source_sha256") != cleaned_sha:
        raise F1ReportError("F1.5/F1.6 cleaned source hashes differ")
    if eda.get("total_rows") != out or split.get("total_rows") != out:
        raise F1ReportError("F1.4–F1.6 row totals differ")
    domains = _dict(eda.get("class_distribution"), "EDA domain counts")
    if set(domains) != {"databases", "linux", "programming", "systems"}:
        raise F1ReportError("unexpected domain classes")
    domain_counts = {x: _integer(domains[x], f"domain {x}", minimum=1) for x in sorted(domains)}
    if sum(domain_counts.values()) != out or domain_counts != cleaning.get("by_domain"):
        raise F1ReportError("EDA/cleaning class distribution mismatch")
    split_counts = _dict(split.get("split_counts"), "split_counts")
    sizes = {name: _integer(split_counts.get(name), name, minimum=1) for name in SPLITS}
    if sum(sizes.values()) != out:
        raise F1ReportError("split sizes do not add to cleaned corpus")
    overlaps = _dict(split.get("group_overlap"), "split overlap")
    if any(_integer(overlaps.get(name), f"overlap {name}") != 0 for name in OVERLAPS):
        raise F1ReportError("group leakage detected")
    edges = _integer(split.get("near_duplicate_edges"), "near_duplicate_edges")
    groups = _integer(split.get("group_count"), "group_count", minimum=5)
    threshold = _number(split.get("near_duplicate_threshold"), "near_duplicate_threshold")
    by_class = _dict(split.get("splits"), "split class matrix")
    if set(by_class) != set(domains):
        raise F1ReportError("split class labels differ from EDA")
    for label in sorted(domains):
        class_splits = _dict(by_class[label], "class split")
        if sum(_integer(class_splits.get(s), f"{label}/{s}") for s in SPLITS) != domain_counts[label]:
            raise F1ReportError("class split sum mismatch")
    for s in SPLITS:
        if sum(by_class[label][s] for label in by_class) != sizes[s]:
            raise F1ReportError("per-class split does not match split totals")

    contract = _dict(classical.get("data_contract"), "F1.7 data contract")
    if contract.get("split_counts") != sizes:
        raise F1ReportError("F1.7 trained on different split counts")
    if classical.get("fit_split") != "train" or classical.get("evaluation_splits") != ["val", "test"]:
        raise F1ReportError("F1.7 train-only evaluation contract failed")
    for name in ("val", "test"):
        if _integer(_dict(classical.get(name), f"F1.7 {name}").get("n"), name, minimum=1) != sizes[name]:
            raise F1ReportError(f"{name} sample size differs between F1.6 and F1.7")
    hashes = _dict(classical.get("split_source_sha256"), "F1.7 split SHA-256")
    for name in SPLITS:
        _digest(hashes.get(name), f"F1.7 {name}")
    if errors.get("test_source_sha256") != hashes["test"]:
        raise F1ReportError("F1.8/F1.7 test SHA-256 differ")
    if errors.get("frozen_model") != "F1.7" or errors.get("data_split") != "test":
        raise F1ReportError("F1.8 must analyze frozen test predictions")
    test = classical["test"]
    validation = classical["val"]
    labels = _labels(test.get("labels"))
    if _labels(validation.get("labels")) != labels or _labels(errors.get("labels")) != labels:
        raise F1ReportError("label lists differ")
    matrix = test.get("confusion_matrix")
    if not isinstance(matrix, list) or len(matrix) != len(labels):
        raise F1ReportError("invalid test confusion matrix")
    clean_matrix = []
    for r in matrix:
        if not isinstance(r, list) or len(r) != len(labels):
            raise F1ReportError("invalid test confusion matrix row")
        clean_matrix.append([_integer(x, "confusion count") for x in r])
    if clean_matrix != errors.get("confusion_matrix"):
        raise F1ReportError("F1.7/F1.8 confusion matrices differ")
    n = sizes["test"]
    correct = sum(clean_matrix[i][i] for i in range(len(labels)))
    if sum(map(sum, clean_matrix)) != n or errors.get("n") != n:
        raise F1ReportError("confusion matrix/test size mismatch")
    err_n = _integer(errors.get("errors"), "errors")
    if errors.get("correct") != correct or err_n != n - correct:
        raise F1ReportError("F1.8 correct/errors totals inconsistent")
    accuracy = _number(test.get("accuracy"), "test accuracy")
    if abs(accuracy - correct / n) > 1e-10 or abs(_number(errors.get("accuracy"), "F1.8 accuracy") - accuracy) > 1e-10:
        raise F1ReportError("accuracy does not reconcile with confusion matrix")
    macro = _number(test.get("macro_f1"), "test macro F1")
    if abs(_number(errors.get("macro_f1"), "F1.8 macro F1") - macro) > 1e-10:
        raise F1ReportError("F1.7/F1.8 Macro F1 mismatch")
    weighted = _number(test.get("weighted_f1"), "test weighted F1")
    val_macro = _number(validation.get("macro_f1"), "validation macro F1")
    val_accuracy = _number(validation.get("accuracy"), "validation accuracy")
    pairs = errors.get("confusion_pairs")
    if not isinstance(pairs, list):
        raise F1ReportError("missing F1.8 confusion pairs")
    expected_pairs = Counter({(labels[i], labels[j]): clean_matrix[i][j]
                             for i in range(len(labels)) for j in range(len(labels))
                             if i != j and clean_matrix[i][j] > 0})
    found = Counter()
    for p in pairs:
        item = _dict(p, "confusion pair")
        t, pred = item.get("true"), item.get("predicted")
        if t not in labels or pred not in labels or t == pred or (t, pred) in found:
            raise F1ReportError("invalid/duplicate confusion pair")
        found[(t, pred)] = _integer(item.get("count"), "pair count", minimum=1)
    if found != expected_pairs:
        raise F1ReportError("confusion pair audit disagrees with confusion matrix")
    first_pair = sorted(found.items(), key=lambda p: (-p[1], p[0][0], p[0][1]))[0] if found else None

    return {
        "source_count": len(sources), "selected": inp, "cleaned": out,
        "short_removed": short, "duplicates_removed": duplicate,
        "invalid_or_missing": invalid, "domains": domain_counts,
        "sizes": sizes, "groups": groups, "edges": edges, "threshold": threshold,
        "group_overlap": 0, "validation_accuracy": val_accuracy,
        "validation_macro_f1": val_macro, "test_accuracy": accuracy,
        "test_macro_f1": macro, "test_weighted_f1": weighted,
        "correct": correct, "errors": err_n,
        "top_confusion": ({"true": first_pair[0][0], "predicted": first_pair[0][1],
                           "count": first_pair[1]} if first_pair else None),
        "cleaned_sha256": cleaned_sha,
        "test_sha256": hashes["test"],
    }


def render_markdown(result: dict) -> str:
    """Render public report exclusively from typed, allowlisted aggregates."""
    d = result
    sizes = d["sizes"]
    lines = [
        "# F1 — Data Engineering & Classical ML: итоговый отчёт",
        "",
        "> **Статус:** отчёт сформирован из локальных аудит-артефактов F1.4–F1.8;",
        "> GitHub Actions проверяет автономные тесты, но **не обучает модель на реальных данных**.",
        "",
        "## Резюме",
        "",
        f"Из {d['selected']:,} отобранных технических Q&A получено {d['cleaned']:,} очищенных записей. ".replace(',', ' ') +
        "Данные разделены на непересекающиеся по вычисленным группам выборки " +
        f"train={sizes['train']}, validation={sizes['val']}, test={sizes['test']}.",
        f"Классическая модель **TF-IDF + Logistic Regression** показала на отложенном test "
        f"**Macro F1={d['test_macro_f1']:.4f}**, **Accuracy={d['test_accuracy']:.4f}** "
        f"({d['correct']} правильных из {sizes['test']}).",
        "",
        "## Происхождение и подготовка данных",
        "",
        "- Источник: технические Q&A из Stack Exchange через зафиксированную ревизию Hugging Face; "
        "метаданные и контрольные суммы — в [`data/raw/manifest.json`](../../data/raw/manifest.json).",
        f"- Исходных источников: **{d['source_count']}**, выбранных строк: **{d['selected']}**.",
        f"- Очистка: исключены короткие **{d['short_removed']}**, точные дубликаты **{d['duplicates_removed']}**, "
        f"некорректные/пустые **{d['invalid_or_missing']}**; осталось **{d['cleaned']}**.",
        "- Распределение очищенного корпуса:",
        "",
        "| Класс | Записей |",
        "|---|---:|",
    ]
    for label, count in d["domains"].items():
        lines.append(f"| `{label}` | {count} |")
    lines += [
        "", "## Разделение без прямой лексической утечки", "",
        "| Train | Validation | Test | Групп | Близких связей | Пересечение групп |",
        "|---:|---:|---:|---:|---:|---:|",
        f"| {sizes['train']} | {sizes['val']} | {sizes['test']} | {d['groups']} | {d['edges']} | 0 |",
        "",
        f"Группировка выполнена по `context` и `answer`: символьные TF-IDF n-граммы 3–5, cosine ≥ "
        f"{d['threshold']:.2f}, union-find; разделение — StratifiedGroupKFold, seed=42.",
        "Нулевое пересечение вычисленных групп **не гарантирует** отсутствия всех семантических, "
        "авторских или временных утечек.",
        *( ["Исторический отчёт первоначального проекта указывает **535** близких связей; "
           f"в повторном запуске — **{d['edges']}**. Причина различия не установлена; "
           "данные не подгонялись под прежний результат."]
           if d["selected"] == 3066 and d["cleaned"] == 3040 else [] ),
        "",
        "## Обучение и оценка модели", "",
        "- Признаки: TF-IDF (словные n-граммы 1–2, `max_features=18000`, `sublinear_tf=True`).",
        "- Классификатор: Logistic Regression (`C=2.0`, `max_iter=500`, `random_state=42`).",
        "- TF-IDF и классификатор обучены только на train; validation/test используются для оценки.",
        "",
        "| Выборка | Accuracy | Macro F1 | Weighted F1 |",
        "|---|---:|---:|---:|",
        f"| Validation | {d['validation_accuracy']:.4f} | {d['validation_macro_f1']:.4f} | — |",
        f"| Test | {d['test_accuracy']:.4f} | {d['test_macro_f1']:.4f} | {d['test_weighted_f1']:.4f} |",
        "",
        "## Анализ ошибок", "",
        f"На test: верных прогнозов — **{d['correct']}**, ошибок — **{d['errors']}** "
        f"из **{sizes['test']}** примеров.",
    ]
    p = d["top_confusion"]
    if p is not None:
        lines.append(f"Самая частая направленная ошибка: **`{p['true']} → {p['predicted']}` — {p['count']} случай(ев)**.")
    lines += [
        "Это наблюдение, а не доказанная причина. Возможное тематическое перекрытие категорий "
        "требует отдельной ручной проверки локальных примеров.",
        "", "## Воспроизводимость и ограничения", "",
        "- Основа отчёта: локальные JSON-аудиты F1.4–F1.8 с согласованием размеров, "
        "SHA-256, confusion matrix и сохранённых метрик; генератор: "
        "[`scripts/build_f1_report.py`](../../scripts/build_f1_report.py).",
        "- SHA-256 очищенного корпуса: `" + d["cleaned_sha256"] + "`.",
        "- SHA-256 test: `" + d["test_sha256"] + "`.",
        "- Лицензия карточки датасета — CC BY-NC-SA 4.0; атрибуция отдельных исходных публикаций "
        "не восстановлена. Исходные тексты, ID примеров и словарь модели в GitHub не публикуются.",
        "- Метки отражают форум-источник, а не достоверность ответа. Это классификация "
        "категории вопроса, не оценка способности решать технические задачи.",
        "- Оценка выполнена на одной версии данных и одном разбиении; исходный отчёт имеет "
        "немного отличающиеся метрики. Изменение не доказано статистически значимым.",
        "- Confusion pairs и лексические группы не дают причинного объяснения ошибок.",
        "- GitHub CI работает на синтетических данных. Полный эксперимент "
        "воспроизводился на Windows вне CI; тяжелые артефакты хранятся локально.",
        "", "## Схема и навигация", "",
        "- [Архитектура F1](../visuals/f1.md)",
        "- [Шаги F1.1–F1.9](../steps/F1_09_report.md)",
        "- [Происхождение данных](../DATA_PROVENANCE.md)",
        "", "## Вывод", "",
        "В проверенном локальном запуске сформирован воспроизводимый конвейер "
        "загрузки, очистки, исследования, группового разбиения и классификации технических вопросов. "
        "Ограничения измерения и лицензирования сохранены явно. Следующая фаза — "
        "сравнение с PyTorch-классификатором (F2).",
        "",
    ]
    return "\n".join(lines)
