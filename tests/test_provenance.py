# Назначение файла: F1.2: проверки происхождения данных на манифестах и синтетических архивах.
# Примечание: комментарии не влияют на работу программы.

"""Offline F1.2 checks; fixtures contain no third-party Q&A text."""
import copy
import hashlib
from pathlib import Path

import pytest

from src.data.provenance import (
    ManifestError, ROOT, load_manifest, validate_manifest, verify_raw_files,
)


# Формирует синтетический манифест для проверки правил происхождения данных.
@pytest.fixture
def manifest():
    return load_manifest(ROOT / "data/raw/manifest.json")


# Проверяет основные поля закреплённого манифеста проекта.
def test_real_manifest_contract(manifest):
    info = validate_manifest(manifest)
    assert info["sources"] == 4
    assert info["selected_total"] == 3066
    assert info["by_domain"] == {
        "linux": 800, "databases": 800, "systems": 800, "programming": 666,
    }


# Проверяет корректное чтение путей Windows с обратными слешами.
def test_windows_paths_are_accepted(manifest):
    # The pinned manifest stores Windows separators, but CI uses Ubuntu.
    assert "\\" in manifest["sources"][0]["file"]
    validate_manifest(manifest)


# Проверяет отказ при подмене идентификатора датасета или ревизии.
@pytest.mark.parametrize("field,value", [
    ("revision", "main"),
    ("seed", 43),
    ("dataset", "untrusted/other"),
    ("fields", ["title_body"]),
    ("license", "unknown"),
])
def test_identity_changes_rejected(manifest, field, value):
    changed = copy.deepcopy(manifest)
    changed[field] = value
    with pytest.raises(ManifestError):
        validate_manifest(changed)


# Проверяет отказ при URL, не соответствующем закреплённой ревизии.
def test_wrong_pinned_source_url_rejected(manifest):
    changed = copy.deepcopy(manifest)
    changed["sources"][0]["url"] = changed["sources"][0]["url"].replace(
        changed["revision"], "main")
    with pytest.raises(ManifestError, match="not pinned"):
        validate_manifest(changed)


# Проверяет отказ при неверном формате SHA-256.
def test_invalid_sha_rejected(manifest):
    changed = copy.deepcopy(manifest)
    changed["sources"][0]["sha256"] = "bad-sha"
    with pytest.raises(ManifestError, match="SHA-256"):
        validate_manifest(changed)


# Проверяет обнаружение повторяющегося источника.
def test_duplicate_source_rejected(manifest):
    changed = copy.deepcopy(manifest)
    changed["sources"][1] = copy.deepcopy(changed["sources"][0])
    with pytest.raises(ManifestError, match="duplicate"):
        validate_manifest(changed)


# Проверяет блокировку небезопасного или неожиданного пути к архиву.
def test_unsafe_path_rejected(manifest):
    changed = copy.deepcopy(manifest)
    changed["sources"][0]["file"] = "data/raw/../secret.jsonl.gz"
    with pytest.raises(ManifestError):
        validate_manifest(changed)


# Проверяет ограничения на число исходных и выбранных записей.
def test_invalid_counts_rejected(manifest):
    changed = copy.deepcopy(manifest)
    changed["sources"][0]["selected_rows"] = 999
    with pytest.raises(ManifestError):
        validate_manifest(changed)


# Проверяет сообщение об отсутствии архивов при строгой сверке.
def test_missing_raw_files_reported(manifest, tmp_path):
    with pytest.raises(ManifestError, match="missing raw file"):
        verify_raw_files(manifest, tmp_path)


# Проверяет SHA-256 на синтетических файлах и выявление повреждения.
def test_synthetic_files_sha_success_and_corruption(manifest, tmp_path):
    # Create four short *synthetic* gzip-named byte fixtures; no Q&A text.
    synthetic = copy.deepcopy(manifest)
    for i, rec in enumerate(synthetic["sources"]):
        path = tmp_path / rec["file"].replace("\\", "/")
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = f"synthetic-test-source-{i}".encode()
        path.write_bytes(payload)
        rec["sha256"] = hashlib.sha256(payload).hexdigest()
    assert verify_raw_files(synthetic, tmp_path) == 4
    first = tmp_path / synthetic["sources"][0]["file"].replace("\\", "/")
    first.write_bytes(b"corrupted")
    with pytest.raises(ManifestError, match="SHA-256 mismatch"):
        verify_raw_files(synthetic, tmp_path)
