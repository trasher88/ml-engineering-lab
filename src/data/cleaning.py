"""F1.4: deterministic text cleaning from the original ML Engineering Lab.

Keep this stage separate from near-duplicate grouping and train/val/test
allocation (F1.5). No input examples are logged or committed.
"""
from __future__ import annotations

import html
import re
from collections import Counter
from typing import Any, Iterable

REQUIRED_FIELDS = ("id", "domain", "context", "answer")


def clean_text(text: str) -> str:
    """Reproduce the historical F1 text normalization rules.

    WARNING: these rules intentionally discard Markdown/code symbols and URLs.
    They are a teaching baseline, not a production-grade code-aware parser.
    """
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"[`#*]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def clean_records(records: Iterable[Any], *, min_words: int) -> tuple[list[dict], dict[str, int]]:
    """Clean, filter, and deduplicate by ID / casefolded question.

    Filter order matches the historical pipeline: invalid raw fields ->
    normalized exact duplicates -> too-short question/answer. Accepted
    IDs/contexts are recorded only *after* the minimum-length filter.
    Near-duplicates are deliberately NOT removed here (see F1.5).
    """
    if not isinstance(min_words, int) or isinstance(min_words, bool) or min_words < 1:
        raise ValueError("min_words must be a positive integer")

    counts = Counter({"input_rows": 0, "invalid_or_missing": 0,
                      "duplicates_removed": 0, "short_removed": 0,
                      "output_rows": 0})
    seen_ids: set[str] = set()
    seen_texts: set[str] = set()
    cleaned: list[dict] = []
    for record in records:
        counts["input_rows"] += 1
        if not isinstance(record, dict) or any(
            not isinstance(record.get(key), str) or not record[key].strip()
            for key in REQUIRED_FIELDS
        ):
            counts["invalid_or_missing"] += 1
            continue

        item = dict(record)  # preserve provenance and never mutate the input
        item["context"] = clean_text(item["context"])
        item["answer"] = clean_text(item["answer"])
        rejected = item.get("rejected", "")
        item["rejected"] = clean_text(rejected) if isinstance(rejected, str) else ""

        key = item["context"].casefold()
        if item["id"] in seen_ids or key in seen_texts:
            counts["duplicates_removed"] += 1
            continue
        if min(len(item["context"].split()), len(item["answer"].split())) < min_words:
            counts["short_removed"] += 1
            continue
        seen_ids.add(item["id"])
        seen_texts.add(key)
        cleaned.append(item)

    counts["output_rows"] = len(cleaned)
    assert (counts["input_rows"] == counts["invalid_or_missing"]
            + counts["duplicates_removed"] + counts["short_removed"]
            + counts["output_rows"])
    return cleaned, dict(counts)
