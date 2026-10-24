# Назначение файла: F1.1: проверки конфигурации, JSONL, хеширования и журнала. Используются временные файлы.
# Примечание: комментарии не влияют на работу программы.

"""Автономные тесты foundation-модуля; не требуют датасетов или моделей."""

import hashlib
import json
from pathlib import Path

import pytest

from src.common import io


# Проверяет, что корневая папка проекта определяется независимо от текущего каталога.
def test_root_resolution():
    """Путь к репозиторию должен определяться модулем, а не текущим каталогом."""
    assert (io.ROOT / "src/common/io.py").is_file()


# Проверяет чтение и основные параметры конфигурации CPU.
def test_cpu_config():
    """Базовый профиль существует и содержит необходимые настройки."""
    cfg = io.config()
    assert cfg["seed"] == 42
    assert cfg["data"]["min_words"] == 8
    assert cfg["classical"]["max_features"] == 18000


# Проверяет наследование базовых настроек в GPU-конфигурации.
def test_gpu_config_inheritance():
    """GPU-профиль меняет sft, но наследует остальные базовые настройки."""
    cpu = io.config("configs/local_cpu/lab.yaml")
    gpu = io.config("configs/gpu/lab.yaml")
    assert gpu["seed"] == cpu["seed"]
    assert gpu["data"] == cpu["data"]
    assert gpu["retrieval"] == cpu["retrieval"]
    assert gpu["sft"]["rank"] == 8
    assert cpu["sft"]["rank"] == 4
    assert gpu["sft"]["model"] == "Qwen/Qwen2.5-0.5B-Instruct"


# Проверяет сохранение и обратное чтение записей JSONL.
def test_save_and_read_jsonl(tmp_path):
    rows = [{"id": 1, "text": "Пример"}, {"id": 2, "text": "Next"}]
    path = tmp_path / "nested" / "records.jsonl"
    io.save_jsonl(path, rows)
    assert io.read_jsonl(path) == rows
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    assert io.read_jsonl(path) == rows


# Проверяет запись Unicode и отказ от невалидных JSON-чисел.
def test_save_json_strict_and_unicode(tmp_path):
    path = tmp_path / "reports" / "metrics.json"
    io.save_json(path, {"metric": 0.75, "label": "точность"})
    assert json.loads(path.read_text(encoding="utf-8"))["label"] == "точность"
    with pytest.raises(ValueError):
        io.save_json(path, {"invalid": float("nan")})


# Проверяет, что хеш файла совпадает с вычисленным SHA-256.
def test_digest_sha256(tmp_path):
    path = tmp_path / "payload.bin"
    payload = b"sample-f1-1"
    path.write_bytes(payload)
    assert io.digest(path) == hashlib.sha256(payload).hexdigest()


# Проверяет отметку времени UTC и добавление записей в журнал.
def test_log_event_is_utc_and_appends(tmp_path, monkeypatch):
    monkeypatch.setattr(io, "ROOT", tmp_path)
    io.log_event("F1.1", "PASS", "pytest", "foundation test")
    io.log_event("F1.1", "PASS", "pytest", "second event")
    log_path = tmp_path / "docs/EXECUTION_LOG.md"
    contents = log_path.read_text(encoding="utf-8")
    assert "Журнал выполнения" in contents
    assert contents.count("**F1.1 — PASS**") == 2
    assert "foundation test" in contents
    assert "second event" in contents
