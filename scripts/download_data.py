# Назначение файла: F1.3: скачивание архивов по закреплённому манифесту и воспроизводимый отбор Q&A.
# Сверяет SHA-256, поддерживает offline-режим; исходные тексты остаются локально.
# Примечание: комментарии не влияют на работу программы.

"""F1.3: reproduce the pinned technical Q&A subset from verified source files.

No downloaded Q&A or derivatives belong in the public Git repository.
The historical manifest produced in the original run is a *read-only input*.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import random
import shutil
import tempfile
import time
import urllib.request
from pathlib import Path

from src.data.provenance import (
    ROOT, ManifestError, load_manifest, validate_manifest, verify_raw_files,
)

INSTRUCTION = "Answer the technical question using the supplied context."
COLUMNS = ("title_body", "upvoted_answer", "downvoted_answer")


# Ошибка загрузки, целостности архива или схемы JSONL.
class IngestionError(ValueError):
    """Downloaded bytes or parsed source rows violate the pinned contract."""


# Приводит путь из Windows-манифеста к безопасной относительной форме.
def _normalized_path(record: dict, root: Path) -> Path:
    # validate_manifest has already constrained this exact relative path.
    return root / record["file"].replace("\\", "/")


# Считает SHA-256 файла для проверки его неизменности.
def _sha256(path: Path) -> str:
    hash_ = hashlib.sha256()
    with path.open("rb") as reader:
        for block in iter(lambda: reader.read(1024 * 1024), b""):
            hash_.update(block)
    return hash_.hexdigest()


# Скачивает во временный файл и принимает только при совпадении SHA-256.
def _download_verified(record: dict, destination: Path, retries: int = 3) -> None:
    """Download into a temporary file; publish only after matching pinned SHA."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(retries):
        temp: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb", prefix=".download-", suffix=".tmp",
                dir=destination.parent, delete=False,
            ) as output:
                temp = Path(output.name)
                with urllib.request.urlopen(record["url"], timeout=120) as response:
                    shutil.copyfileobj(response, output, length=1024 * 1024)
            if _sha256(temp) != record["sha256"]:
                raise IngestionError(f"SHA-256 mismatch for downloaded file: {destination.name}")
            os.replace(temp, destination)
            return
        except IngestionError:
            raise  # Wrong bytes: fail closed, do not repeat or replace known-good files.
        except (OSError, TimeoutError) as exc:
            if attempt == retries - 1:
                raise IngestionError(
                    f"Failed to download {destination.name} after {retries} attempts: {exc}"
                ) from exc
            time.sleep(attempt + 1)
        finally:
            if temp is not None:
                temp.unlink(missing_ok=True)


# Гарантирует наличие всех архивов с корректными хешами; умеет offline.
def ensure_sources(manifest: dict, root: Path, *, offline: bool = False) -> None:
    """Reuse verified files, or fetch missing pinned files; never trust stale cache."""
    validate_manifest(manifest)
    for record in manifest["sources"]:
        path = _normalized_path(record, root)
        if path.exists():
            if not path.is_file() or _sha256(path) != record["sha256"]:
                raise IngestionError(
                    f"SHA-256 mismatch for cached file: {path.name}. "
                    "Investigate and manually remove the invalid file before retrying."
                )
        elif offline:
            raise IngestionError(f"Missing local file in --offline mode: {path.name}")
        else:
            print(f"Downloading verified source: {path.name}", flush=True)
            _download_verified(record, path)
    verify_raw_files(manifest, root)


# Преобразует строку источника к единой схеме учебного корпуса.
def _to_record(item: object, *, site: str, domain: str, url: str, row: int,
               dataset: str) -> dict | None:
    if isinstance(item, list) and len(item) == len(COLUMNS):
        item = dict(zip(COLUMNS, item))
    if not isinstance(item, dict):
        return None
    question = item.get("title_body")
    chosen = item.get("upvoted_answer")
    if not isinstance(question, str) or not question.strip():
        return None
    if not isinstance(chosen, str) or not chosen.strip():
        return None
    rejected = item.get("downvoted_answer")
    return {
        "id": f"{site}:{row}", "source": dataset,
        "source_type": "technical_qa", "source_url": url, "source_row": row,
        "language": "en", "domain": domain, "instruction": INSTRUCTION,
        "context": question, "answer": chosen,
        "rejected": rejected if isinstance(rejected, str) else "",
    }


# Воспроизводимо выбирает ограниченное число записей каждого источника.
def select_records(manifest: dict, root: Path, *, sample_per_domain: int = 60
                   ) -> tuple[list[dict], list[dict]]:
    """Match the original seeded selection and maintain row-level lineage."""
    summary = validate_manifest(manifest)
    if sample_per_domain < 0:
        raise IngestionError("sample_per_domain must be nonnegative")
    domain_by_site = {
        "unix.stackexchange.com": "linux", "dba.stackexchange.com": "databases",
        "serverfault.com": "systems", "codereview.stackexchange.com": "programming",
    }
    selected, sample = [], []
    for record in manifest["sources"]:
        path = _normalized_path(record, root)
        site = path.name.removesuffix(".jsonl.gz")
        site_rows = []
        try:
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle):
                    item = json.loads(line)
                    entry = _to_record(item, site=site, domain=domain_by_site[site],
                                       url=record["url"], row=line_number,
                                       dataset=summary["dataset"])
                    if entry is not None:
                        site_rows.append(entry)
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise IngestionError(f"Cannot parse pinned gzip {path.name}: {exc}") from exc
        if len(site_rows) != record["available_rows"]:
            raise IngestionError(
                f"Row count mismatch {site}: expected {record['available_rows']}, "
                f"found {len(site_rows)}"
            )
        random.Random(manifest["seed"]).shuffle(site_rows)
        chosen = site_rows[:record["selected_rows"]]
        if len(chosen) != record["selected_rows"]:
            raise IngestionError(f"Selected count mismatch for {site}")
        selected.extend(chosen)
        sample.extend(chosen[:sample_per_domain])
        print(f"{site}: valid={len(site_rows)}, selected={len(chosen)}", flush=True)
    if len(selected) != summary["selected_total"]:
        raise IngestionError("Total selected row count differs from pinned manifest")
    return selected, sample


# Сохраняет набор JSONL атомарно, чтобы не оставить частичный результат.
def _atomic_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n", prefix=".subset-",
            suffix=".tmp", dir=path.parent, delete=False,
        ) as writer:
            temp = Path(writer.name)
            for row in rows:
                writer.write(json.dumps(row, ensure_ascii=False) + "\n")
        os.replace(temp, path)
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)


# Оркестрирует чтение манифеста, проверку источников и выборку данных.
def run(root: Path = ROOT, *, offline: bool = False) -> dict:
    root = Path(root)
    manifest_path = root / "data/raw/manifest.json"
    manifest = load_manifest(manifest_path)
    validate_manifest(manifest)
    ensure_sources(manifest, root, offline=offline)
    rows, sample = select_records(manifest, root)
    _atomic_jsonl(root / "data/raw/subset.jsonl", rows)
    _atomic_jsonl(root / "data/sample/technical_qa.jsonl", sample)
    # Preserve a manifest beside the local smoke sample for historical consumers.
    # Never rewrite the pinned reference manifest itself.
    sample_manifest = root / "data/sample/manifest.json"
    sample_manifest.parent.mkdir(parents=True, exist_ok=True)
    sample_manifest.write_bytes(manifest_path.read_bytes())
    print(f"Data ingestion: PASS; selected={len(rows)}; sample={len(sample)}; "
          "raw_sha256=4/4", flush=True)
    return {"selected": len(rows), "sample": len(sample), "sources_verified": 4}


# Обрабатывает параметры командной строки и сообщает итог загрузки.
def main() -> int:
    parser = argparse.ArgumentParser(description="F1.3: verified, pinned dataset ingestion")
    parser.add_argument("--offline", action="store_true",
                        help="Reuse local gzip files only; do not access the network")
    args = parser.parse_args()
    try:
        run(offline=args.offline)
        return 0
    except (OSError, json.JSONDecodeError, ManifestError, IngestionError) as exc:
        print(f"Data ingestion: FAIL: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
