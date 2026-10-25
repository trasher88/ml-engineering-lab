# Назначение файла: F1.5: описательная статистика очищенного корпуса без публикации текстов.
# Разбиение на train/validation/test и поиск близких дубликатов здесь не выполняются.

"""F1.5: text-free, reproducible descriptive statistics for cleaned Q&A.

Split-aware analysis is intentionally deferred to F1.6. The API does not
include examples or any text-derived n-grams in its returned summaries.
"""
from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
import pandas as pd


REQUIRED = ("id", "domain", "context", "answer")


# Ошибка входного корпуса либо параметров статистического анализа.
class EDAError(ValueError):
    """The F1.4 cleaned dataset violates the F1.5 input contract."""


# Считает статистику длин текстов по количеству слов.
def _length_statistics(lengths: pd.Series) -> dict[str, int | float]:
    values = lengths.to_numpy(dtype=np.int64)
    return {
        "min": int(values.min()),
        "max": int(values.max()),
        "mean": round(float(values.mean()), 3),
        "median": round(float(np.median(values)), 3),
        "p90": round(float(np.percentile(values, 90)), 3),
        "p95": round(float(np.percentile(values, 95)), 3),
    }


# Проверяет данные и возвращает агрегаты классов и длин без Q&A.
def summarize_cleaned(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute aggregate-only statistics, without modifying input rows.

    Word counts deliberately use whitespace splitting, as in the original
    pipeline. They must not be confused with BPE/subword token lengths.
    """
    if not rows:
        raise EDAError("cleaned dataset is empty")
    for index, item in enumerate(rows):
        if not isinstance(item, dict) or any(
            not isinstance(item.get(key), str) or not item[key].strip()
            for key in REQUIRED
        ):
            raise EDAError(f"invalid F1.4 cleaned record at index {index}")
        if "split" in item or "group_id" in item:
            raise EDAError("unexpected split/group_id: F1.6 has not run")

    ids = [item["id"] for item in rows]
    if len(set(ids)) != len(ids):
        raise EDAError("duplicate IDs in cleaned dataset")
    contexts = [item["context"].casefold() for item in rows]
    if len(set(contexts)) != len(contexts):
        raise EDAError("duplicate normalized contexts in cleaned dataset")

    # pandas is used to inspect the corpus as in the original EDA phase.
    frame = pd.DataFrame(rows)
    domains = {str(domain): int(count) for domain, count in
               sorted(Counter(frame["domain"]).items())}
    context_words = frame["context"].str.split().str.len()
    answer_words = frame["answer"].str.split().str.len()
    counts = list(domains.values())
    imbalance_ratio = round(max(counts) / min(counts), 3)

    # No original context/answer/rejected strings or row-level metadata leaves
    # this function; reports contain aggregates only.
    return {
        "stage": "F1.5",
        "status": "EDA_BEFORE_SPLIT",
        "total_rows": int(len(frame)),
        "class_distribution": domains,
        "class_imbalance_ratio_max_min": imbalance_ratio,
        "required_field_missing_or_empty": {
            name: int(sum(not isinstance(row.get(name), str) or not row[name].strip()
                          for row in rows)) for name in REQUIRED
        },
        "context_words": _length_statistics(context_words),
        "answer_words": _length_statistics(answer_words),
        "length_measure": "whitespace-separated words, not model tokens",
        "split_distribution": "NOT RUN (F1.6)",
        "near_duplicate_grouping": "NOT RUN (F1.6)",
    }


# Формирует числовые массивы длин для построения гистограмм.
def length_arrays(rows: list[dict[str, Any]]) -> tuple[list[int], list[int]]:
    """Private plot inputs; do not serialize or print per-row values."""
    return ([len(row["context"].split()) for row in rows],
            [len(row["answer"].split()) for row in rows])
