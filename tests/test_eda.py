"""F1.5: deterministic EDA with synthetic records only (never remote data)."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from scripts.eda_data import EDARunError, run
from src.data.eda import EDAError, length_arrays, summarize_cleaned


def row(n: int, domain: str = "linux", *, context: str | None = None,
        answer: str | None = None) -> dict:
    return {
        "id": str(n), "domain": domain,
        "context": context or f"technical question unique number {n} has enough words here",
        "answer": answer or f"technical answer unique number {n} has enough words here",
        "source": "synthetic", "rejected": "",
    }


def test_class_distribution_sorted_and_imbalance_ratio():
    rows = [row(1, "systems"), row(2, "linux"), row(3, "linux")]
    report = summarize_cleaned(rows)
    assert report["class_distribution"] == {"linux": 2, "systems": 1}
    assert report["class_imbalance_ratio_max_min"] == 2
    assert report["total_rows"] == 3


def test_length_stats_use_whitespace_words_not_tokens():
    rows = [row(1, context="a b", answer="w x y z"),
            row(2, context="a b c d e f", answer="h i")]
    report = summarize_cleaned(rows)
    assert report["context_words"]["min"] == 2
    assert report["context_words"]["max"] == 6
    assert report["context_words"]["mean"] == 4
    assert report["context_words"]["median"] == 4
    assert report["answer_words"]["mean"] == 3
    assert report["length_measure"].startswith("whitespace-separated")


def test_no_split_or_near_duplicate_results_claimed():
    report = summarize_cleaned([row(1)])
    assert report["split_distribution"] == "NOT RUN (F1.6)"
    assert report["near_duplicate_grouping"] == "NOT RUN (F1.6)"


def test_no_input_mutation_and_no_example_text_in_report():
    rows = [row(1, context="SECRET-TRAINING-TEXT-123 is a private fake test string")]
    copy_before = copy.deepcopy(rows)
    report = summarize_cleaned(rows)
    assert rows == copy_before
    assert "SECRET-TRAINING-TEXT-123" not in json.dumps(report)
    assert "technical answer" not in json.dumps(report)


@pytest.mark.parametrize("bad", [[], [None], [{"id": "a"}], [row(1, answer=" ")]])
def test_invalid_cleaned_input_rejected(bad):
    with pytest.raises(EDAError):
        summarize_cleaned(bad)


def test_duplicate_id_rejected():
    with pytest.raises(EDAError, match="duplicate IDs"):
        summarize_cleaned([row(1), row(1, context="another question that differs a lot")])


def test_duplicate_context_rejected_even_casefolded():
    with pytest.raises(EDAError, match="duplicate normalized contexts"):
        summarize_cleaned([row(1), row(2, context=row(1)["context"].upper())])


def test_preexisting_split_or_group_rejected():
    with pytest.raises(EDAError, match="F1.6"):
        summarize_cleaned([dict(row(1), split="train")])
    with pytest.raises(EDAError, match="F1.6"):
        summarize_cleaned([dict(row(1), group_id=0)])


def test_length_arrays():
    assert length_arrays([row(1, context="x y", answer="z")]) == ([2], [1])


@pytest.fixture
def temp_project(tmp_path: Path):
    source = tmp_path / "data/interim/cleaned.jsonl"
    source.parent.mkdir(parents=True)
    records = [row(1, "linux"), row(2, "databases"), row(3, "linux")]
    source.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    return tmp_path, source


def test_full_offline_run_saves_aggregate_only_artifacts(temp_project):
    root, source = temp_project
    summary = run(root, expect_rows=3)
    output = root / "reports/data_eda_audit.json"
    class_plot = root / "docs/visuals/f1/class_distribution.png"
    length_plot = root / "docs/visuals/f1/text_length.png"
    assert class_plot.is_file() and class_plot.stat().st_size > 100
    assert length_plot.is_file() and length_plot.stat().st_size > 100
    assert summary["source_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved == summary
    assert "technical question" not in output.read_text(encoding="utf-8")
    assert saved["class_distribution"] == {"databases": 1, "linux": 2}
    first_bytes = output.read_bytes()
    assert run(root, expect_rows=3) == summary
    assert output.read_bytes() == first_bytes


def test_missing_input_never_uses_sample(tmp_path):
    with pytest.raises(EDARunError, match="missing full F1.4"):
        run(tmp_path, expect_rows=3040)
    assert not (tmp_path / "reports/data_eda_audit.json").exists()


def test_count_mismatch_fails_before_writing(temp_project):
    root, _ = temp_project
    with pytest.raises(EDARunError, match="row count mismatch"):
        run(root, expect_rows=3040)
    assert not (root / "reports/data_eda_audit.json").exists()
    assert not (root / "docs/visuals/f1").exists()


def test_input_output_collision_fails(temp_project):
    root, _ = temp_project
    with pytest.raises(EDARunError, match="different"):
        run(root, audit_path=Path("data/interim/cleaned.jsonl"))


def test_invalid_expected_count_fails(temp_project):
    root, _ = temp_project
    for count in [0, -1, True]:
        with pytest.raises(EDARunError, match="positive integer"):
            run(root, expect_rows=count)


def test_reject_bad_jsonl_before_outputs(temp_project):
    root, source = temp_project
    source.write_text("{BAD JSON}\n", encoding="utf-8")
    with pytest.raises(EDARunError, match="invalid JSONL"):
        run(root)
    assert not (root / "reports/data_eda_audit.json").exists()
