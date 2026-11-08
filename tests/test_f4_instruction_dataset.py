"""F4.1: тесты загрузки, группового split, контроля утечки и целостности."""

from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path
import json
import os
import subprocess
import sys

import pytest

from src.llm_f4.instruction_dataset import (
    DatasetError,
    InstructionExample,
    SplitConfig,
    allocate_group_counts,
    create_artifacts,
    export_artifacts,
    load_config,
    load_examples,
    normalized_prompt,
    split_examples,
    validate_examples,
    validate_splits,
    verify_artifacts,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/synthetic/f4_01_instruction_seed.jsonl"
CONFIG = ROOT / "configs/f4_01_instruction_dataset.json"


@pytest.fixture
def rows() -> list[InstructionExample]:
    return load_examples(SOURCE)


@pytest.fixture
def config() -> SplitConfig:
    return load_config(CONFIG, root=ROOT)


def test_dataset_has_all_expected_authored_groups(rows: list[InstructionExample]) -> None:
    assert len(rows) == 80
    assert len({row.group_id for row in rows}) == 40
    assert {row.domain for row in rows} == {"python", "linux", "sql", "ml"}
    assert {row.origin for row in rows} == {"authored_synthetic"}
    assert all(len([e for e in rows if e.group_id == row.group_id]) == 2 for row in rows)


def test_reader_returns_typed_records(rows: list[InstructionExample]) -> None:
    assert all(isinstance(row, InstructionExample) for row in rows)
    assert rows[0].id == "f4q001_a"
    assert rows[0].input == ""
    assert next(row for row in rows if row.id == "f4q034_a").input


def test_exact_split_counts_and_coverage(rows: list[InstructionExample], config: SplitConfig) -> None:
    splits = split_examples(rows, seed=config.seed, ratios=config.ratios)
    assert {name: len(splits[name]) for name in splits} == {"train": 56, "validation": 12, "test": 12}
    assert {name: len({r.group_id for r in split}) for name, split in splits.items()} == {
        "train": 28, "validation": 6, "test": 6,
    }
    validate_splits(splits, source=rows)


def test_groups_and_prompt_duplicates_disjoint(rows: list[InstructionExample], config: SplitConfig) -> None:
    splits = split_examples(rows, seed=config.seed, ratios=config.ratios)
    grouped = [{row.group_id for row in splits[name]} for name in ("train", "validation", "test")]
    assert not (grouped[0] & grouped[1] or grouped[0] & grouped[2] or grouped[1] & grouped[2])
    keys = [set(normalized_prompt(r) for r in splits[name]) for name in ("train", "validation", "test")]
    assert not (keys[0] & keys[1] or keys[0] & keys[2] or keys[1] & keys[2])


def test_split_invariant_to_input_row_order(rows: list[InstructionExample], config: SplitConfig) -> None:
    forward = split_examples(rows, seed=config.seed, ratios=config.ratios)
    backward = split_examples(list(reversed(rows)), seed=config.seed, ratios=config.ratios)
    assert forward == backward


def test_different_seed_can_change_assignment(rows: list[InstructionExample], config: SplitConfig) -> None:
    first = split_examples(rows, seed=config.seed, ratios=config.ratios)
    second = split_examples(rows, seed=config.seed + 1, ratios=config.ratios)
    assert {r.group_id for r in first["train"]} != {r.group_id for r in second["train"]}


@pytest.mark.parametrize("total,counts", [
    (3, (1, 1, 1)), (4, (2, 1, 1)), (5, (3, 1, 1)), (40, (28, 6, 6)),
])
def test_integer_group_allocation(total: int, counts: tuple[int, ...], config: SplitConfig) -> None:
    actual = allocate_group_counts(total, config.ratios)
    assert tuple(actual[name] for name in ("train", "validation", "test")) == counts


def test_too_few_groups_is_error(config: SplitConfig) -> None:
    with pytest.raises(DatasetError, match="минимум"):
        allocate_group_counts(2, config.ratios)


def test_duplicate_ids_rejected(rows: list[InstructionExample]) -> None:
    with pytest.raises(DatasetError, match="Повторный id"):
        validate_examples(rows + [rows[0]])


def test_duplicate_instruction_rejected_even_with_new_id(rows: list[InstructionExample]) -> None:
    duplicate = replace(rows[0], id="f4q999_x", instruction=rows[0].instruction.upper())
    with pytest.raises(DatasetError, match="Повторный instruction"):
        validate_examples(rows + [duplicate])


def test_inconsistent_group_domain_rejected(rows: list[InstructionExample]) -> None:
    other = replace(rows[0], domain="sql")
    with pytest.raises(DatasetError, match="разные domain"):
        validate_examples([rows[1], other])


def test_group_leak_rejected(rows: list[InstructionExample], config: SplitConfig) -> None:
    splits = split_examples(rows, seed=config.seed, ratios=config.ratios)
    moved = splits["train"].pop(0)
    splits["validation"].append(moved)
    with pytest.raises(DatasetError, match="Утечка группы"):
        validate_splits(splits)


def test_source_record_omitted_rejected(rows: list[InstructionExample], config: SplitConfig) -> None:
    splits = split_examples(rows, seed=config.seed, ratios=config.ratios)
    splits["train"].pop(0)
    with pytest.raises(DatasetError, match="не покрывает"):
        validate_splits(splits, source=rows)


def test_source_content_mutation_rejected(rows: list[InstructionExample], config: SplitConfig) -> None:
    splits = split_examples(rows, seed=config.seed, ratios=config.ratios)
    splits["train"][0] = replace(splits["train"][0], response="Подменённый ответ, который не должен пройти проверку.")
    with pytest.raises(DatasetError, match="изменил содержимое"):
        validate_splits(splits, source=rows)


@pytest.mark.parametrize("mutation,fragment", [
    ({"id": "invalid id"}, "некорректный id"),
    ({"origin": "external_corpus"}, "origin"),
    ({"instruction": "    "}, "instruction"),
    ({"response": 41}, "response"),
    ({"input": "неверный\\rсимвол".replace("\\r", "\r")}, "управляющие"),
    ({"unexpected": "extra"}, "требуются поля"),
])
def test_bad_jsonl_record_rejected(tmp_path: Path, rows: list[InstructionExample],
                                   mutation: dict[str, object], fragment: str) -> None:
    record = asdict(rows[0]) | mutation
    file = tmp_path / "bad.jsonl"
    file.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
    with pytest.raises(DatasetError, match=fragment):
        load_examples(file)


@pytest.mark.parametrize("content,fragment", [
    ("\n", "пустая строка"),
    ("{broken}\n", "ошибка JSON"),
    ('{"x":NaN}\n', "Некорректная JSON-константа"),
    ("", "Dataset пуст"),
])
def test_bad_jsonl_structure_rejected(tmp_path: Path, content: str, fragment: str) -> None:
    file = tmp_path / "bad.jsonl"
    file.write_text(content, encoding="utf-8")
    with pytest.raises(DatasetError, match=fragment):
        load_examples(file)


def test_invalid_utf8_rejected(tmp_path: Path) -> None:
    file = tmp_path / "invalid.jsonl"
    file.write_bytes(b"\xff\xfe")
    with pytest.raises(DatasetError, match="Не удалось прочитать JSONL"):
        load_examples(file)


@pytest.mark.parametrize("modifier,fragment", [
    (lambda obj: obj.update({"seed": True}), "seed"),
    (lambda obj: obj.update({"ratios": {"train": 70, "validation": 15, "test": 20}}), "ratios"),
    (lambda obj: obj.update({"source_file": "../outside.jsonl"}), "source_file"),
    (lambda obj: obj.update({"source_file": "/tmp/file.jsonl"}), "source_file"),
    (lambda obj: obj.update({"schema_version": 7}), "Конфигурация"),
])
def test_invalid_config_rejected(tmp_path: Path, modifier, fragment: str) -> None:
    settings = json.loads(CONFIG.read_text(encoding="utf-8"))
    modifier(settings)
    path = tmp_path / "wrong.json"
    path.write_text(json.dumps(settings), encoding="utf-8")
    with pytest.raises(DatasetError, match=fragment):
        load_config(path, root=ROOT)


def test_export_and_verify_are_byte_identical(tmp_path: Path, config: SplitConfig) -> None:
    local = replace(config, output_dir=tmp_path / "splits")
    first = export_artifacts(local)
    actual = verify_artifacts(local)
    assert first == actual
    second = export_artifacts(local)
    assert first == second
    assert set(first) == {"train.jsonl", "validation.jsonl", "test.jsonl", "manifest.json"}
    metadata = json.loads(first["manifest.json"])
    assert metadata["source_sha256"] and len(metadata["source_sha256"]) == 64
    assert "instruction" not in metadata  # manifest не содержит исходный текст
    assert all(data.endswith(b"\n") for data in first.values())
    assert b"\r\n" not in first["train.jsonl"]


def test_verify_rejects_corrupted_output(tmp_path: Path, config: SplitConfig) -> None:
    local = replace(config, output_dir=tmp_path / "splits")
    export_artifacts(local)
    path = local.output_dir / "train.jsonl"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(DatasetError, match="не соответствует"):
        verify_artifacts(local)


def test_verify_rejects_corrupted_manifest(tmp_path: Path, config: SplitConfig) -> None:
    local = replace(config, output_dir=tmp_path / "splits")
    export_artifacts(local)
    path = local.output_dir / "manifest.json"
    path.write_text('{}\n', encoding="utf-8")
    with pytest.raises(DatasetError, match="не соответствует"):
        verify_artifacts(local)


def test_verify_rejects_missing_file(tmp_path: Path, config: SplitConfig) -> None:
    local = replace(config, output_dir=tmp_path / "splits")
    export_artifacts(local)
    (local.output_dir / "test.jsonl").unlink()
    with pytest.raises(DatasetError, match="Нет файла"):
        verify_artifacts(local)


def test_source_change_invalidates_manifest(tmp_path: Path, config: SplitConfig) -> None:
    local_source = tmp_path / "source.jsonl"
    local_source.write_bytes(SOURCE.read_bytes())
    local = replace(config, source_file=local_source, output_dir=tmp_path / "splits")
    export_artifacts(local)
    local_source.write_bytes(local_source.read_bytes() + b"\n")
    with pytest.raises(DatasetError, match="пустая строка"):
        verify_artifacts(local)


def test_create_artifacts_independent_from_row_order(rows: list[InstructionExample], config: SplitConfig) -> None:
    data = SOURCE.read_bytes()
    first = create_artifacts(rows, config, source_bytes=data)
    second = create_artifacts(list(reversed(rows)), config, source_bytes=data)
    assert first == second


def test_demo_help_works_without_ml_dependencies() -> None:
    call = subprocess.run(
        [sys.executable, str(ROOT / "scripts/f4_instruction_dataset.py"), "--help"],
        cwd=ROOT, capture_output=True, text=True, check=True,
    )
    assert "--verify" in call.stdout
    assert "--config" in call.stdout


@pytest.mark.parametrize("codepage", ["cp1251", "cp866", "cp1252"])
def test_demo_help_supported_under_windows_legacy_codepages(codepage: str) -> None:
    """В pipe кодировка Windows может не поддерживать Unicode-стрелки и кириллицу."""
    env = {**os.environ, "PYTHONIOENCODING": codepage}
    call = subprocess.run(
        [sys.executable, str(ROOT / "scripts/f4_instruction_dataset.py"), "--help"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        encoding=codepage,
        check=True,
    )
    assert "--verify" in call.stdout
    assert "--config" in call.stdout
    assert "instruction/response -> split JSONL" in call.stdout
