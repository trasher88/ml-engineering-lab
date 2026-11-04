"""F3.2: офлайн-тесты pinned revision, лицензии, snapshot SHA и manifest.

Ни один тест не запрашивает модель у Hugging Face. Сетевые функции Hub
подменяются monkeypatch, а файлы весов — короткими синтетическими bytes.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from transformers_f3.embeddings import ModelSpec  # noqa: E402
from transformers_f3.model_cache import (  # noqa: E402
    MODEL_ASSETS, create_manifest, default_cache_dir, download_snapshot,
    inspect_snapshot, validate_manifest, validate_remote_model, write_manifest,
)


@pytest.fixture
def spec() -> ModelSpec:
    """Версия тестовой модели с фиктивным, но валидным SHA."""
    return ModelSpec("fake/example", "b" * 40, "apache-2.0", 4, 8)


@pytest.fixture
def fake_snapshot(tmp_path: Path) -> Path:
    """Псевдокеш только из безопасных конфигов, токенизатора и весов."""
    folder = tmp_path / "snapshot"
    folder.mkdir()
    for name, content in (
        ("config.json", b'{"hidden_size":4}'),
        ("model.safetensors", b"synthetic-weights-not-a-real-model"),
        ("tokenizer.json", b"{}"),
        ("tokenizer_config.json", b"{}"),
    ):
        (folder / name).write_bytes(content)
    return folder


def test_model_assets_only_safe_formats() -> None:
    """Snapshot не должен скачивать pickle .bin, ONNX и другие тяжёлые форматы."""
    assert "model.safetensors" in MODEL_ASSETS
    assert "pytorch_model.bin" not in MODEL_ASSETS
    assert "model.onnx" not in MODEL_ASSETS
    assert len(MODEL_ASSETS) == len(set(MODEL_ASSETS))


def test_remote_metadata_accepts_exact_sha_license(spec: ModelSpec) -> None:
    """Пропускаем только разрешённую открытую версию с точным SHA."""
    info = SimpleNamespace(sha=spec.revision, card_data={"license": spec.license}, private=False, gated=False)
    assert validate_remote_model(spec, info) is None


@pytest.mark.parametrize("mutation", [
    {"sha": "c" * 40},
    {"card_data": {"license": "mit"}},
    {"card_data": {}},
    {"private": True},
    {"gated": "auto"},
])
def test_remote_metadata_rejects_mismatch(spec: ModelSpec, mutation: dict) -> None:
    """Неверная лицензия, приватность или mutable SHA запрещают download."""
    data = {"sha": spec.revision, "card_data": {"license": spec.license}, "private": False, "gated": False}
    data.update(mutation)
    with pytest.raises(ValueError):
        validate_remote_model(spec, SimpleNamespace(**data))


def test_snapshot_hashes_and_filenames(spec: ModelSpec, fake_snapshot: Path) -> None:
    """Manifest хеширует bytes, а не mtime, путь или имя файла."""
    hashes = inspect_snapshot(spec, fake_snapshot)
    assert set(hashes) == {"config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json"}
    assert all(len(value) == 64 for value in hashes.values())
    assert hashes["model.safetensors"] != hashes["config.json"]


@pytest.mark.parametrize("file_to_remove", ["config.json", "model.safetensors", "tokenizer.json"])
def test_snapshot_requires_essential_files(spec: ModelSpec, fake_snapshot: Path, file_to_remove: str) -> None:
    """Нельзя считать обрезанный кеш полноценной предобученной моделью."""
    (fake_snapshot / file_to_remove).unlink()
    with pytest.raises(FileNotFoundError):
        inspect_snapshot(spec, fake_snapshot)


def test_snapshot_accepts_vocab_tokenizer(spec: ModelSpec, fake_snapshot: Path) -> None:
    """Для BERT-подобного encoder словарь vocab.txt может заменить tokenizer.json."""
    (fake_snapshot / "tokenizer.json").unlink()
    (fake_snapshot / "vocab.txt").write_text("[PAD]\n[UNK]\n", encoding="utf-8")
    assert "vocab.txt" in inspect_snapshot(spec, fake_snapshot)


def test_missing_snapshot_path(spec: ModelSpec, tmp_path: Path) -> None:
    """Не создаём незаметно пустой кеш при отсутствии pinned snapshot."""
    with pytest.raises(FileNotFoundError):
        inspect_snapshot(spec, tmp_path / "not-there")


def test_unknown_auxiliary_files_ignored(spec: ModelSpec, fake_snapshot: Path) -> None:
    """Ограниченный список файлов защищает от хеширования случайного датасета."""
    (fake_snapshot / "private-data.csv").write_text("sensitive", encoding="utf-8")
    assert "private-data.csv" not in inspect_snapshot(spec, fake_snapshot)


def test_manifest_deterministic(spec: ModelSpec, fake_snapshot: Path) -> None:
    """Одинаковые артефакты и версии дают идентичный manifest без timestamp."""
    versions = {"torch": "2.14.1", "transformers": "5.22.1", "huggingface_hub": "1.4.0"}
    one = create_manifest(spec, fake_snapshot, versions=versions)
    two = create_manifest(spec, fake_snapshot, versions=dict(reversed(list(versions.items()))))
    assert one == two
    assert "created_at" not in one
    assert one["schema_version"] == 1
    validate_manifest(one, spec, fake_snapshot)


@pytest.mark.parametrize("versions", [
    {}, {"torch": "2"},
    {"torch": "2", "transformers": "5", "huggingface_hub": ""},
    {"torch": "2", "transformers": "5", "huggingface_hub": "1", "secret": "x"},
])
def test_manifest_rejects_untracked_runtime_versions(spec: ModelSpec, fake_snapshot: Path, versions: dict) -> None:
    """Запись версии зависимости обязательна для повторения вычислений."""
    with pytest.raises(ValueError):
        create_manifest(spec, fake_snapshot, versions=versions)


def test_manifest_detects_byte_tampering(spec: ModelSpec, fake_snapshot: Path) -> None:
    """Повреждение хотя бы одного byte весов делает manifest невалидным."""
    manifest = create_manifest(spec, fake_snapshot, versions={"torch": "2", "transformers": "5", "huggingface_hub": "1"})
    (fake_snapshot / "model.safetensors").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="суммы"):
        validate_manifest(manifest, spec, fake_snapshot)


def test_manifest_detects_config_change(spec: ModelSpec, fake_snapshot: Path) -> None:
    """Разные max_length означают разный encoder pipeline, даже при тех же весах."""
    manifest = create_manifest(spec, fake_snapshot, versions={"torch": "2", "transformers": "5", "huggingface_hub": "1"})
    different = ModelSpec("fake/example", "b" * 40, "apache-2.0", 4, 16)
    with pytest.raises(ValueError, match="другой"):
        validate_manifest(manifest, different, fake_snapshot)


def test_manifest_rejects_unknown_schema(spec: ModelSpec, fake_snapshot: Path) -> None:
    """Несогласованная схема manifest не может считаться стабильным артефактом."""
    manifest = create_manifest(spec, fake_snapshot, versions={"torch": "2", "transformers": "5", "huggingface_hub": "1"})
    manifest["schema_version"] = 999
    with pytest.raises(ValueError):
        validate_manifest(manifest, spec, fake_snapshot)


def test_atomic_manifest_round_trip(spec: ModelSpec, fake_snapshot: Path, tmp_path: Path) -> None:
    """Сохранённый JSON переживает повторную загрузку и checksum-проверку."""
    manifest = create_manifest(spec, fake_snapshot, versions={"torch": "2", "transformers": "5", "huggingface_hub": "1"})
    target = tmp_path / "nested" / "manifest.json"
    write_manifest(manifest, target)
    assert target.is_file()
    assert not target.with_name("manifest.json.tmp").exists()
    assert json.loads(target.read_text(encoding="utf-8")) == manifest
    validate_manifest(json.loads(target.read_text(encoding="utf-8")), spec, fake_snapshot)


def test_atomic_manifest_replacement(spec: ModelSpec, fake_snapshot: Path, tmp_path: Path) -> None:
    """Повторное сохранение обновляет bytes файла без оставления *.tmp."""
    target = tmp_path / "manifest.json"
    write_manifest({"hello": 1}, target)
    write_manifest({"hello": 2}, target)
    assert json.loads(target.read_text(encoding="utf-8")) == {"hello": 2}
    assert not (tmp_path / "manifest.json.tmp").exists()


def test_default_cache_outside_source_tree(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Кеш по умолчанию хранится под домашней папкой, не в src/data/models."""
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    assert default_cache_dir().is_relative_to(tmp_path)
    assert default_cache_dir().name == "hub"


def test_offline_snapshot_does_not_call_network(spec: ModelSpec, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """По умолчанию вызывается только snapshot_download(local_files_only=True)."""
    import types

    calls: list[dict] = []

    def fake_download(**kwargs: object) -> str:
        calls.append(dict(kwargs))
        return str(tmp_path)

    class NoNetwork:
        """Любое обращение к модельным metadata в offline-режиме — ошибка."""

        def model_info(self, **kwargs: object) -> None:
            raise AssertionError("Не должно быть сетевого обращения")

    fake_hub = types.ModuleType("huggingface_hub")
    fake_hub.HfApi = NoNetwork
    fake_hub.snapshot_download = fake_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)
    assert download_snapshot(spec, cache_dir=tmp_path) == tmp_path
    assert len(calls) == 1
    assert calls[0]["local_files_only"] is True
    assert calls[0]["revision"] == spec.revision
    assert calls[0]["allow_patterns"] == list(MODEL_ASSETS)


def test_online_snapshot_checks_license_first(spec: ModelSpec, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Скачивание разрешается только после сверки лицензии и полного SHA."""
    import types

    calls: list[str] = []

    class FakeApi:
        """Подмена Hub: сетевое обращение фактически не производится."""

        def model_info(self, **kwargs: object) -> SimpleNamespace:
            calls.append("metadata")
            assert kwargs["revision"] == spec.revision
            return SimpleNamespace(sha=spec.revision, card_data={"license": spec.license}, private=False, gated=False)

    def fake_download(**kwargs: object) -> str:
        calls.append("download")
        assert kwargs["local_files_only"] is False
        return str(tmp_path)

    fake_hub = types.ModuleType("huggingface_hub")
    fake_hub.HfApi = FakeApi
    fake_hub.snapshot_download = fake_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)
    assert download_snapshot(spec, cache_dir=tmp_path, allow_download=True) == tmp_path
    assert calls == ["metadata", "download"]


def test_online_rejects_license_before_snapshot(spec: ModelSpec, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """При чужой лицензии весовые файлы не скачиваются вообще."""
    import types

    class FakeApi:
        """Подмена metadata с несовпадающей лицензией."""

        def model_info(self, **kwargs: object) -> SimpleNamespace:
            return SimpleNamespace(sha=spec.revision, card_data={"license": "restricted"}, private=False, gated=False)

    def forbid_download(**kwargs: object) -> str:
        raise AssertionError("Нельзя скачивать после license mismatch")

    fake_hub = types.ModuleType("huggingface_hub")
    fake_hub.HfApi = FakeApi
    fake_hub.snapshot_download = forbid_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)
    with pytest.raises(ValueError, match="Лицензия"):
        download_snapshot(spec, cache_dir=tmp_path, allow_download=True)


def test_load_pretrained_encoder_offline_with_stubbed_transformers(
    spec: ModelSpec, fake_snapshot: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Интеграционный путь AutoTokenizer/AutoModel не обращается к Hub при offline."""
    import types
    import torch
    from torch import nn
    from transformers_f3 import model_cache

    interactions: list[tuple[str, dict]] = []

    class Tokenizer:
        """Минимальная замена HF tokenizer."""

        @classmethod
        def from_pretrained(cls, path: str, **kwargs: object) -> "Tokenizer":
            interactions.append(("tokenizer", dict(kwargs)))
            assert Path(path) == fake_snapshot
            return cls()

        def __call__(self, texts: list[str], **kwargs: object) -> dict[str, torch.Tensor]:
            ids = torch.tensor([[1, 2] for _ in texts])
            return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    class Encoder(nn.Module):
        """Упрощённый BERT-совместимый nn.Module для настоящего интерфейса."""

        def __init__(self) -> None:
            super().__init__()
            self.config = SimpleNamespace(hidden_size=4)

        @classmethod
        def from_pretrained(cls, path: str, **kwargs: object) -> "Encoder":
            interactions.append(("model", dict(kwargs)))
            assert Path(path) == fake_snapshot
            return cls()

        def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> SimpleNamespace:
            return SimpleNamespace(last_hidden_state=torch.ones((*input_ids.shape, 4)))

    fake_hub = types.ModuleType("huggingface_hub")
    fake_hub.__version__ = "1.4.0"
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)
    fake_transformers = types.ModuleType("transformers")
    fake_transformers.__version__ = "5.22.1"
    fake_transformers.AutoModel = Encoder
    fake_transformers.AutoTokenizer = Tokenizer
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setattr(model_cache, "download_snapshot", lambda *_args, **_kwargs: fake_snapshot)
    manifest_path = tmp_path / "manifest.json"
    encoder, first = model_cache.load_pretrained_encoder(spec, manifest_path=manifest_path)
    assert encoder.encode(["synthetic text"]).shape == (1, 4)
    assert manifest_path.is_file()
    assert first["model"] == spec.to_dict()
    assert first["library_versions"]["transformers"] == "5.22.1"
    assert interactions[0][1]["local_files_only"] is True
    assert interactions[1][1]["local_files_only"] is True
    assert interactions[1][1]["trust_remote_code"] is False
    assert interactions[1][1]["use_safetensors"] is True

    # Второй вызов обязан подтвердить прошлый fingerprint, а не перезаписать его.
    previous_bytes = manifest_path.read_bytes()
    _, second = model_cache.load_pretrained_encoder(spec, manifest_path=manifest_path)
    assert second == first
    assert manifest_path.read_bytes() == previous_bytes

    # Повреждение весов не должно маскироваться автоматической записью нового SHA.
    (fake_snapshot / "model.safetensors").write_bytes(b"damaged")
    with pytest.raises(ValueError, match="суммы"):
        model_cache.load_pretrained_encoder(spec, manifest_path=manifest_path)
    assert manifest_path.read_bytes() == previous_bytes
