# Назначение файла: F1.6: группировка лексически похожих Q&A и разделение train/val/test.
# Алгоритм защищает от пересечений найденных групп, но не всех смысловых перефразировок.
# Примечание: комментарии не влияют на работу программы.

"""F1.6: group near-duplicate technical Q&A before train/val/test split.

Based on the original project's char_wb TF-IDF (3..5 grams), union-find,
and StratifiedGroupKFold workflow. No user text is logged or included in
aggregate reports. This is a lexical heuristic, not semantic deduplication.
"""
from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedGroupKFold


# Ошибка целостности входных записей или группового разбиения.
class SplitError(ValueError):
    """Invalid cleaned data or a split that violates grouping requirements."""


# Проверяет схему данных, уникальность ID и отсутствие готовых split-меток.
def validate_cleaned(rows: list[dict]) -> None:
    """Reject malformed, already-split, or exactly repeated cleaned records."""
    if not isinstance(rows, list) or not rows:
        raise SplitError("cleaned dataset must be a nonempty list")
    seen_ids: set[str] = set()
    seen_contexts: set[str] = set()
    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            raise SplitError(f"invalid record at index {i}")
        if any(not isinstance(row.get(k), str) or not row[k].strip()
               for k in ("id", "domain", "context", "answer")):
            raise SplitError(f"missing required cleaned fields at index {i}")
        if "group_id" in row or "split" in row:
            raise SplitError("input must be pre-split F1.4 cleaned data")
        if row["id"] in seen_ids:
            raise SplitError("duplicate IDs in cleaned input")
        normalized_context = row["context"].casefold()
        if normalized_context in seen_contexts:
            raise SplitError("duplicate exact contexts in cleaned input")
        seen_ids.add(row["id"])
        seen_contexts.add(normalized_context)


# Находит TF-IDF сходство и объединяет связанные записи через union-find.
def group_near_duplicates(rows: list[dict], *, threshold: float,
                          max_features: int = 18000, batch_size: int = 128
                          ) -> tuple[list[int], int]:
    """Connect lexically similar contexts OR answers with deterministic DSU.

    Count *matched pair/field edges*, not removed rows and not components.
    The pairwise comparison is done in sparse blocks to cap peak memory.
    """
    validate_cleaned(rows)
    if (isinstance(threshold, bool) or not isinstance(threshold, (int, float))
            or not np.isfinite(threshold) or not 0 < threshold <= 1):
        raise SplitError("near_duplicate_threshold must be in (0, 1]")
    if type(max_features) is not int or max_features < 1:
        raise SplitError("max_features must be a positive integer")
    if type(batch_size) is not int or batch_size < 1:
        raise SplitError("batch_size must be a positive integer")

    parent = list(range(len(rows)))

    # Находит представителя группы с сжатием пути в структуре union-find.
    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    near_edges = 0
    for field in ("context", "answer"):
        vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                                     max_features=max_features)
        matrix = vectorizer.fit_transform([row[field].lower() for row in rows])
        for start in range(0, len(rows), batch_size):
            sims = (matrix[start:start + batch_size] @ matrix.T).tocoo()
            for p in np.flatnonzero(sims.data >= threshold):
                i = int(sims.row[p]) + start
                j = int(sims.col[p])
                if i < j:
                    # Preserve the original pipeline's union-root rule.
                    parent[find(j)] = find(i)
                    near_edges += 1
    return [find(i) for i in range(len(rows))], near_edges


# Размещает целые группы по train/val/test и проверяет пересечения.
def split_cleaned(rows: list[dict], *, seed: int = 42,
                  threshold: float = 0.92, max_features: int = 18000
                  ) -> tuple[list[dict], dict[str, Any]]:
    """Return new records with split/group_id and aggregate leakage audit.

    The original pipeline uses folds 0 and 1 as val/test; other 3 folds
    form train. StratifiedGroupKFold is approximate, not a 60/20/20 quota.
    """
    if type(seed) is not int or seed < 0:
        raise SplitError("seed must be a non-negative integer")
    groups, edges = group_near_duplicates(rows, threshold=threshold,
                                         max_features=max_features)
    distinct_groups = len(set(groups))
    if distinct_groups < 5:
        raise SplitError("at least five independent groups are needed for five folds")
    labels = [row["domain"] for row in rows]
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    try:
        folds = list(sgkf.split(np.zeros(len(rows)), labels, groups))
    except ValueError as exc:
        raise SplitError(f"group-stratified split failed: {exc}") from exc
    val_indices = set(int(i) for i in folds[0][1])
    test_indices = set(int(i) for i in folds[1][1])
    if val_indices & test_indices:
        raise SplitError("validation and test fold overlap")

    result: list[dict] = []
    for i, original in enumerate(rows):
        split = "val" if i in val_indices else "test" if i in test_indices else "train"
        result.append({**original, "split": split, "group_id": groups[i]})

    split_counts = dict(Counter(item["split"] for item in result))
    if set(split_counts) != {"train", "val", "test"}:
        raise SplitError("train, val, and test must all be nonempty")
    group_sets = {name: {r["group_id"] for r in result if r["split"] == name}
                  for name in ("train", "val", "test")}
    overlap = {f"{a}/{b}": len(group_sets[a] & group_sets[b])
               for a, b in (("train", "test"), ("train", "val"), ("val", "test"))}
    if any(overlap.values()):
        raise SplitError("near-duplicate groups cross split boundaries")
    class_counts = {label: {split: sum(r["domain"] == label and r["split"] == split
                                       for r in result)
                            for split in ("train", "val", "test")}
                    for label in sorted(set(labels))}
    group_sizes = Counter(groups)
    audit: dict[str, Any] = {
        "stage": "F1.6",
        "total_rows": len(rows),
        "seed": seed,
        "near_duplicate_threshold": float(threshold),
        "char_tfidf_analyzer": "char_wb",
        "ngram_range": [3, 5],
        "max_features": max_features,
        "near_duplicate_edges": edges,
        "group_count": distinct_groups,
        "max_group_size": max(group_sizes.values()),
        "split_counts": {name: split_counts[name] for name in ("train", "val", "test")},
        "splits": class_counts,  # same class->split orientation as original audit
        "group_overlap": overlap,
        "leakage_scope": "lexical char_wb TF-IDF groups only; not all semantic paraphrases",
    }
    return result, audit
