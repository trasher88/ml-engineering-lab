"""F1.3 ingestion checks with synthetic gzip fixtures only; never reach HF."""
from __future__ import annotations

import copy
import gzip
import hashlib
import io
import json
from pathlib import Path

import pytest

from scripts import download_data as ingestion
from src.data.provenance import load_manifest, validate_manifest


@pytest.fixture
def synthetic_lab(tmp_path: Path):
    manifest = copy.deepcopy(load_manifest(ingestion.ROOT / "data/raw/manifest.json"))
    for index, record in enumerate(manifest["sources"]):
        path = tmp_path / record["file"].replace("\\", "/")
        path.parent.mkdir(parents=True, exist_ok=True)
        entries = [
            {"title_body": f"Synthetic question {index}-0", "upvoted_answer": "Answer 0",
             "downvoted_answer": "Rejected 0"},
            [f"Synthetic question {index}-1", "Answer 1", "Rejected 1"],
            ["", "No question", "Rejected 2"],  # invalid row; preserve physical line numbers
            {"title_body": f"Synthetic question {index}-3", "upvoted_answer": "Answer 3"},
            [f"Synthetic question {index}-4", "Answer 4", "Rejected 4"],
        ]
        payload = gzip.compress(
            "".join(json.dumps(entry) + "\n" for entry in entries).encode("utf-8"),
            mtime=0,
        )
        path.write_bytes(payload)
        record["sha256"] = hashlib.sha256(payload).hexdigest()
        record["available_rows"] = 4
        record["selected_rows"] = 3
    manifest_path = tmp_path / "data/raw/manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                             encoding="utf-8")
    validate_manifest(manifest)
    return tmp_path, manifest


def test_reproducible_offline_and_manifest_read_only(synthetic_lab, monkeypatch):
    root, manifest = synthetic_lab
    before = (root / "data/raw/manifest.json").read_bytes()

    def no_network(*_args, **_kwargs):
        raise AssertionError("unexpected network access")

    monkeypatch.setattr(ingestion.urllib.request, "urlopen", no_network)
    first = ingestion.run(root, offline=True)
    subset = (root / "data/raw/subset.jsonl").read_bytes()
    sample = (root / "data/sample/technical_qa.jsonl").read_bytes()
    assert first == {"selected": 12, "sample": 12, "sources_verified": 4}
    assert ingestion.run(root, offline=True) == first
    assert subset == (root / "data/raw/subset.jsonl").read_bytes()
    assert sample == (root / "data/sample/technical_qa.jsonl").read_bytes()
    assert (root / "data/raw/manifest.json").read_bytes() == before
    assert (root / "data/sample/manifest.json").read_bytes() == before
    assert len(subset.splitlines()) == 12
    assert len(sample.splitlines()) == 12
    rows = [json.loads(line) for line in subset.splitlines()]
    assert all(r["source_row"] in {0, 1, 3, 4} for r in rows)
    assert all(r["id"].split(":")[1] == str(r["source_row"]) for r in rows)
    assert {r["domain"] for r in rows} == set(ingestion.select_records(manifest, root)[0][i]["domain"] for i in range(12))


def test_sample_limit(synthetic_lab):
    root, manifest = synthetic_lab
    selected, sample = ingestion.select_records(manifest, root, sample_per_domain=2)
    assert len(selected) == 12
    assert len(sample) == 8


def test_existing_corrupt_source_rejected(synthetic_lab):
    root, manifest = synthetic_lab
    path = root / manifest["sources"][0]["file"].replace("\\", "/")
    path.write_bytes(b"tampered source")
    with pytest.raises(ingestion.IngestionError, match="cached file"):
        ingestion.run(root, offline=True)
    assert not (root / "data/raw/subset.jsonl").exists()


def test_offline_missing_fails_without_writing_output(synthetic_lab):
    root, manifest = synthetic_lab
    path = root / manifest["sources"][0]["file"].replace("\\", "/")
    path.unlink()
    with pytest.raises(ingestion.IngestionError, match="Missing local file"):
        ingestion.run(root, offline=True)
    assert not (root / "data/raw/subset.jsonl").exists()


def test_validated_download_only_from_pinned_url(synthetic_lab, monkeypatch):
    root, manifest = synthetic_lab
    rec = manifest["sources"][0]
    dest = root / rec["file"].replace("\\", "/")
    payload = dest.read_bytes()
    dest.unlink()
    calls = []

    def fake_urlopen(url, timeout):
        calls.append((url, timeout))
        return io.BytesIO(payload)

    monkeypatch.setattr(ingestion.urllib.request, "urlopen", fake_urlopen)
    ingestion.ensure_sources(manifest, root)
    assert calls == [(rec["url"], 120)]
    assert dest.read_bytes() == payload


def test_bad_download_is_rejected_and_temp_deleted(synthetic_lab, monkeypatch):
    root, manifest = synthetic_lab
    rec = manifest["sources"][0]
    dest = root / rec["file"].replace("\\", "/")
    dest.unlink()
    monkeypatch.setattr(ingestion.urllib.request, "urlopen",
                        lambda *_args, **_kwargs: io.BytesIO(b"WRONG"))
    with pytest.raises(ingestion.IngestionError, match="downloaded file"):
        ingestion.ensure_sources(manifest, root)
    assert not dest.exists()
    assert not list(dest.parent.glob(".download-*.tmp"))


def test_row_count_mismatch_is_detected(synthetic_lab):
    root, manifest = synthetic_lab
    manifest["sources"][0]["available_rows"] = 999
    with pytest.raises(ingestion.IngestionError, match="Row count mismatch"):
        ingestion.select_records(manifest, root)


def test_invalid_json_is_detected(synthetic_lab):
    root, manifest = synthetic_lab
    rec = manifest["sources"][0]
    path = root / rec["file"].replace("\\", "/")
    payload = gzip.compress(b'{bad json}\n', mtime=0)
    path.write_bytes(payload)
    rec["sha256"] = hashlib.sha256(payload).hexdigest()
    with pytest.raises(ingestion.IngestionError, match="Cannot parse pinned gzip"):
        ingestion.select_records(manifest, root)


def test_invalid_sample_size(synthetic_lab):
    root, manifest = synthetic_lab
    with pytest.raises(ingestion.IngestionError, match="nonnegative"):
        ingestion.select_records(manifest, root, sample_per_domain=-1)


def test_original_manifest_unmodified_after_invalid_download(synthetic_lab, monkeypatch):
    root, manifest = synthetic_lab
    before = (root / "data/raw/manifest.json").read_bytes()
    rec = manifest["sources"][0]
    (root / rec["file"].replace("\\", "/")).unlink()
    monkeypatch.setattr(ingestion.urllib.request, "urlopen",
                        lambda *_args, **_kwargs: io.BytesIO(b"incorrect bytes"))
    with pytest.raises(ingestion.IngestionError):
        ingestion.run(root)
    assert (root / "data/raw/manifest.json").read_bytes() == before
