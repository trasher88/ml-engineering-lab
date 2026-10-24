# Назначение файла: F1.2: проверка неизменяемых метаданных датасета Stack Exchange.
# Без флага --verify-files проверяется манифест, а не содержимое gzip-архивов.
# Примечание: комментарии не влияют на работу программы.

"""Offline validation of the pinned Stack Exchange dataset manifest (F1.2).

This module reads metadata only. It does not download or redistribute Q&A.
To validate downloaded raw files by SHA-256, use --verify-files after F1.3.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DATASET = "flax-sentence-embeddings/stackexchange_titlebody_best_and_down_voted_answer_jsonl"
SITES = {
    "unix.stackexchange.com": "linux",
    "dba.stackexchange.com": "databases",
    "serverfault.com": "systems",
    "codereview.stackexchange.com": "programming",
}
FIELDS = ["title_body", "upvoted_answer", "downvoted_answer"]
SHA40 = re.compile(r"[0-9a-f]{40}\Z")
SHA64 = re.compile(r"[0-9a-f]{64}\Z")


# Ошибка в схеме либо значениях закреплённого манифеста.
class ManifestError(ValueError):
    """The pinned dataset metadata violates the F1.2 contract."""


# Останавливает проверку с понятной ошибкой при нарушении условия.
def _require(condition: bool, detail: str) -> None:
    if not condition:
        raise ManifestError(detail)


# Читает JSON-манифест без запросов к сети или исходным архивам.
def load_manifest(path: Path) -> dict[str, Any]:
    """Load JSON metadata. No source files or network calls are involved."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), "manifest must be a JSON object")
    return payload


# Проверяет источник, revision, четыре домена, пути, SHA-256 и объёмы.
def validate_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    """Check identity, pin, source list, paths, SHA format and sample counts.

    A SHA stored *in* a manifest is just a claim. This function cannot prove
    file integrity until verify_raw_files is run against the actual bytes.
    """
    _require(isinstance(manifest, dict), "manifest must be a JSON object")
    _require(manifest.get("dataset") == DATASET, "unexpected dataset")
    revision = manifest.get("revision")
    _require(isinstance(revision, str) and SHA40.fullmatch(revision) is not None,
             "revision must be a 40-char lowercase Git SHA")
    _require(type(manifest.get("seed")) is int and manifest["seed"] == 42,
             "this recorded experiment expects seed=42")
    _require(manifest.get("fields") == FIELDS, "unexpected dataset field schema")
    license_note = manifest.get("license", "")
    _require(isinstance(license_note, str) and "cc-by-nc-sa-4.0" in license_note.lower(),
             "license metadata is missing or changed")
    sources = manifest.get("sources")
    _require(isinstance(sources, list) and len(sources) == len(SITES),
             "expected exactly four source records")
    seen = set()
    selected_counts: dict[str, int] = {}
    for record in sources:
        _require(isinstance(record, dict), "source record must be an object")
        url, name, digest = record.get("url"), record.get("file"), record.get("sha256")
        _require(isinstance(name, str), "source file path is missing")
        # The historical manifest was generated on Windows: normalize for GitHub/Linux.
        normalized = name.replace("\\", "/")
        basename = normalized.removeprefix("data/raw/")
        _require(basename.endswith(".jsonl.gz"), "not a recognized raw filename")
        site = basename.removesuffix(".jsonl.gz")
        _require(site in SITES, f"unknown source site: {site}")
        _require(normalized == f"data/raw/{site}.jsonl.gz", "unsafe or unexpected raw path")
        _require(site not in seen, "duplicate source site")
        seen.add(site)
        expected_url = (f"https://huggingface.co/datasets/{DATASET}"
                        f"/resolve/{revision}/{site}.jsonl.gz")
        _require(url == expected_url, "source URL is not pinned to manifest revision")
        _require(isinstance(digest, str) and SHA64.fullmatch(digest) is not None,
                 "invalid expected SHA-256 format")
        available, selected = record.get("available_rows"), record.get("selected_rows")
        _require(type(available) is int and type(selected) is int
                 and 0 < selected <= available and selected <= 800,
                 "invalid available/selected row counts")
        selected_counts[SITES[site]] = selected
    _require(seen == set(SITES), "missing source sites")
    return {"revision": revision, "dataset": DATASET,
            "sources": len(sources), "selected_total": sum(selected_counts.values()),
            "by_domain": selected_counts}


# Сравнивает реальные SHA-256 всех gzip-файлов с манифестом.
def verify_raw_files(manifest: dict[str, Any], root: Path) -> int:
    """Optional strict SHA check of all pinned gzip files downloaded in F1.3."""
    validate_manifest(manifest)
    checked = 0
    root = Path(root)
    for record in manifest["sources"]:
        normalized = record["file"].replace("\\", "/")
        path = root / normalized
        _require(path.is_file(), f"missing raw file: {normalized}")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        _require(digest.hexdigest() == record["sha256"],
                 f"SHA-256 mismatch for: {normalized}")
        checked += 1
    return checked


# Разбирает аргументы CLI и выводит результат валидации.
def main() -> int:
    parser = argparse.ArgumentParser(description="Offline dataset provenance check (F1.2)")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/raw/manifest.json")
    parser.add_argument("--verify-files", action="store_true",
                        help="Require all local raw gzip files and compare their SHA-256")
    parser.add_argument("--root", type=Path, default=ROOT,
                        help="Root containing data/raw (only used with --verify-files)")
    args = parser.parse_args()
    try:
        manifest = load_manifest(args.manifest)
        summary = validate_manifest(manifest)
        if args.verify_files:
            count = verify_raw_files(manifest, args.root)
            print(f"Raw files SHA-256: PASS ({count}/{count})")
        else:
            print("Raw files SHA-256: NOT CHECKED (use --verify-files after F1.3)")
        print(f"Dataset provenance: PASS; sources={summary['sources']}; "
              f"selected={summary['selected_total']}; seed=42")
        return 0
    except (OSError, json.JSONDecodeError, ManifestError) as exc:
        print(f"Dataset provenance: FAIL: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
