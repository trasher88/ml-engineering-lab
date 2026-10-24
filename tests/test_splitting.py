"""F1.6 offline tests: synthetic Q&A only; no raw Stack Exchange material."""
from __future__ import annotations

import copy
import hashlib
import json
import random
import string
from pathlib import Path

import pytest

from scripts.split_data import SplitRunError, run
from src.data.splitting import (
    SplitError, group_near_duplicates, split_cleaned, validate_cleaned,
)


def unique_text(i: int, salt: int) -> str:
    generator = random.Random(i * 128 + salt)
    token = "".join(generator.choice(string.ascii_lowercase) for _ in range(80))
    return f"synthetic technical example {token}"


def row(i: int, domain: str = "linux", *, context: str | None = None,
        answer: str | None = None) -> dict:
    return {"id": f"synthetic:{i}", "domain": domain,
            "context": context if context is not None else unique_text(i, 5),
            "answer": answer if answer is not None else unique_text(i, 17),
            "rejected": "", "source": "synthetic"}


def dataset(n: int = 30) -> list[dict]:
    domains = ("linux", "databases", "systems")
    return [row(i, domains[i % len(domains)]) for i in range(n)]


def test_reject_empty_dataset():
    with pytest.raises(SplitError, match="nonempty"):
        validate_cleaned([])


@pytest.mark.parametrize("bad", [None, "not a record", {"id": "x"}])
def test_reject_invalid_records(bad):
    with pytest.raises(SplitError):
        validate_cleaned([bad])


def test_reject_duplicates_by_id_and_exact_question():
    with pytest.raises(SplitError, match="duplicate IDs"):
        validate_cleaned([row(1), row(1, context="different example")])
    with pytest.raises(SplitError, match="duplicate exact contexts"):
        validate_cleaned([row(1), row(2, context=row(1)["context"].upper())])


@pytest.mark.parametrize("column", ["split", "group_id"])
def test_reject_already_assigned_input(column):
    with pytest.raises(SplitError, match="pre-split"):
        validate_cleaned([{**row(0), column: 0}])


@pytest.mark.parametrize("value", [0, -0.1, 1.1, float('nan'), float('inf'), True])
def test_reject_invalid_threshold(value):
    with pytest.raises(SplitError, match="threshold"):
        group_near_duplicates(dataset(5), threshold=value)


def test_reject_invalid_parameters():
    with pytest.raises(SplitError, match="max_features"):
        group_near_duplicates(dataset(5), threshold=.92, max_features=0)
    with pytest.raises(SplitError, match="batch_size"):
        group_near_duplicates(dataset(5), threshold=.92, batch_size=0)
    with pytest.raises(SplitError, match="seed"):
        split_cleaned(dataset(15), seed=True)


def test_same_answer_connects_different_questions():
    rows = [row(1), row(2, answer=row(1)["answer"]), row(3)]
    groups, edges = group_near_duplicates(rows, threshold=.92)
    assert groups[0] == groups[1]
    assert groups[2] != groups[0]
    assert edges >= 1


def test_near_identical_contexts_connect_even_if_answers_differ():
    context = "configuration example involving alpha beta gamma delta epsilon zeta"
    rows = [row(1, context=context), row(2, context=context + " fine")]
    groups, edges = group_near_duplicates(rows, threshold=.85)
    assert groups[0] == groups[1]
    assert edges >= 1


def test_transitive_groups_preserve_connections():
    common_answer = unique_text(100, 5)
    shared_question = "diagnostic alpha beta gamma delta epsilon zeta delta epsilon"
    rows = [row(0, context=shared_question, answer=unique_text(0, 17)),
            row(1, context=shared_question + " gamma", answer=common_answer),
            row(2, context=unique_text(2, 5), answer=common_answer)]
    groups, edges = group_near_duplicates(rows, threshold=.85)
    assert len(set(groups)) == 1
    assert edges >= 2


def test_edges_are_not_number_of_removed_rows():
    answer = unique_text(90, 17)
    rows = [row(i, answer=answer) for i in range(4)]
    groups, edges = group_near_duplicates(rows, threshold=.92)
    assert len(set(groups)) == 1
    assert edges >= 6  # four equal answers yield six matching answer pairs
    assert len(rows) == 4  # nothing removed


def test_insufficient_independent_groups_fails():
    answer = unique_text(80, 17)
    rows = [row(i, answer=answer) for i in range(7)]
    with pytest.raises(SplitError, match="five independent groups"):
        split_cleaned(rows)


def test_split_is_deterministic_group_disjoint_and_does_not_mutate():
    rows = dataset(36)
    # Tie two question records by an identical answer, even across domains.
    rows[1]["answer"] = rows[0]["answer"]
    before = copy.deepcopy(rows)
    first, audit = split_cleaned(rows)
    again, again_audit = split_cleaned(rows)
    assert rows == before
    assert first == again and audit == again_audit
    assert sum(audit["split_counts"].values()) == len(rows)
    assert all(count > 0 for count in audit["split_counts"].values())
    assert all(count == 0 for count in audit["group_overlap"].values())
    assert first[0]["group_id"] == first[1]["group_id"]
    assert first[0]["split"] == first[1]["split"]
    assert sum(sum(by_split.values()) for by_split in audit["splits"].values()) == len(rows)
    assert set(audit["splits"]) == {"linux", "databases", "systems"}
    assert audit["group_count"] < len(rows)
    assert set(r["split"] for r in first) == {"train", "val", "test"}
    assert "semantic" in audit["leakage_scope"]


def test_audit_never_contains_question_answer_or_ids():
    rows = dataset(20)
    rows[0]["context"] = "UNIQUE-SENSITIVE-TOKEN-abc123"
    _, audit = split_cleaned(rows)
    blob = json.dumps(audit)
    assert "UNIQUE-SENSITIVE-TOKEN" not in blob
    assert "synthetic:0" not in blob
    assert unique_text(1, 17) not in blob


@pytest.fixture
def sandbox(tmp_path: Path):
    conf = tmp_path / "configs/local_cpu/lab.yaml"
    conf.parent.mkdir(parents=True)
    conf.write_text("seed: 42\ndata:\n  near_duplicate_threshold: 0.92\n", encoding="utf-8")
    source = tmp_path / "data/interim/cleaned.jsonl"
    source.parent.mkdir(parents=True)
    examples = dataset(30)
    examples[1]["answer"] = examples[0]["answer"]
    source.write_text("".join(json.dumps(item) + "\n" for item in examples), encoding="utf-8")
    return tmp_path, source


def test_local_run_creates_split_files_and_aggregate_only_audit(sandbox):
    root, source = sandbox
    audit = run(root, expect_rows=30)
    by_split = {}
    for name in ("train", "val", "test"):
        path = root / f"data/processed/{name}.jsonl"
        assert path.is_file()
        by_split[name] = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]
        assert all(x["split"] == name for x in by_split[name])
    assert sum(map(len, by_split.values())) == 30
    records = sum(by_split.values(), [])
    assert len(set(x["id"] for x in records)) == 30
    assert len({x["group_id"] for x in records}) == audit["group_count"]
    assert {x["split"] for x in records if x["id"] in {"synthetic:0", "synthetic:1"}}.__len__() == 1
    report_path = root / "reports/data_split_audit.json"
    saved = json.loads(report_path.read_text(encoding="utf-8"))
    assert saved == audit
    assert saved["source_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert "synthetic technical" not in report_path.read_text(encoding="utf-8")
    original_outputs = [path.read_bytes() for path in [report_path] + [root / f"data/processed/{n}.jsonl" for n in ("train", "val", "test")]]
    assert run(root, expect_rows=30) == audit
    assert original_outputs == [path.read_bytes() for path in [report_path] + [root / f"data/processed/{n}.jsonl" for n in ("train", "val", "test")]]


def test_missing_cleaned_dataset_does_not_fall_back_to_sample(tmp_path):
    with pytest.raises(SplitRunError, match="missing F1.4"):
        run(tmp_path)


def test_expected_row_count_blocks_outputs(sandbox):
    root, _ = sandbox
    with pytest.raises(SplitRunError, match="expected 3040"):
        run(root, expect_rows=3040)
    assert not (root / "data/processed").exists()
    assert not (root / "reports/data_split_audit.json").exists()


def test_expected_split_count_blocks_outputs(sandbox):
    root, _ = sandbox
    with pytest.raises(SplitRunError, match="expected train"):
        run(root, expect_train=999)
    assert not (root / "data/processed").exists()


def test_expected_edge_count_blocks_outputs(sandbox):
    root, _ = sandbox
    with pytest.raises(SplitRunError, match="expected near edges"):
        run(root, expect_edges=999)
    assert not (root / "data/processed").exists()


def test_output_collision_blocks_processing(sandbox):
    root, _ = sandbox
    with pytest.raises(SplitRunError, match="distinct paths"):
        run(root, audit_path=Path("data/processed/train.jsonl"))
    assert not (root / "data/processed").exists()


def test_bad_config_rejected(sandbox):
    root, _ = sandbox
    with pytest.raises(SplitRunError, match="configuration"):
        run(root, config_path=Path("not-here.yaml"))
    cfg = root / "configs/local_cpu/lab.yaml"
    cfg.write_text("seed: 42\ndata:\n  near_duplicate_threshold: 1.4\n", encoding="utf-8")
    with pytest.raises(SplitRunError, match="near_duplicate_threshold"):
        run(root)
