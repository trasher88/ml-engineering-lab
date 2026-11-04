"""F3.2: безопасная загрузка pinned-модели, контроль лицензии и кеша.

В репозитории хранятся только метаданные версии, но не веса и не токенизатор.
Сетевой доступ возможен исключительно при явном allow_download=True; в обычном
режиме snapshot и веса открываются только из локального кеша пользователя.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import torch

from .embeddings import ModelSpec, PretrainedTextEncoder


# Разрешены только файлы для CPU-инференса. Исключаем pickle .bin, ONNX,
# TensorFlow, OpenVINO и случайное скачивание нескольких вариантов модели.
MODEL_ASSETS: tuple[str, ...] = (
    "config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.txt",
    "added_tokens.json",
)


def default_cache_dir() -> Path:
    """Вернуть пользовательский каталог кеша вне исходников и Git-репозитория."""
    return Path.home() / ".cache" / "ml-engineering-lab" / "f3_02" / "hub"


def validate_remote_model(spec: ModelSpec, model_info: Any) -> None:
    """Проверить совпадение лицензии и Git SHA с данными Hugging Face Hub.

    Аргументы:
        spec: Зафиксированный контракт модели.
        model_info: Объект ответа HfApi.model_info для указанной revision.
    Результат:
        None при точном совпадении; ValueError при расхождении.

    Карточка модели — декларация правообладателя, а не отдельная гарантия
    юридической чистоты датасетов, поэтому исходный URL фиксируем в документации.
    """
    sha = getattr(model_info, "sha", None)
    if sha != spec.revision:
        raise ValueError("Hub вернул SHA, отличный от pinned revision")
    if bool(getattr(model_info, "private", False)) or bool(getattr(model_info, "gated", False)):
        raise ValueError("Закрытые или gated-модели не поддерживаются в F3.2")
    card = getattr(model_info, "card_data", None)
    license_value = getattr(card, "license", None)
    if license_value is None and isinstance(card, Mapping):
        license_value = card.get("license")
    if license_value != spec.license:
        raise ValueError("Лицензия Hub не совпадает с зафиксированной лицензией")


def download_snapshot(
    spec: ModelSpec,
    *,
    cache_dir: str | Path | None = None,
    allow_download: bool = False,
) -> Path:
    """Получить локальный путь pinned snapshot, не загружая лишние варианты.

    Аргументы:
        spec: Имя модели, полный SHA и ожидаемая лицензия.
        cache_dir: Кеш Hugging Face; по умолчанию вне рабочего проекта.
        allow_download: По умолчанию False — никаких HTTP-запросов.
    Результат:
        Путь к локальной директории весов, config и tokenizer.
    """
    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError as exc:
        raise RuntimeError("Для pretrained encoder необходим пакет transformers (с huggingface_hub)") from exc

    if allow_download:
        # Запрашиваем metadata строго выбранного коммита до загрузки файлов.
        info = HfApi().model_info(repo_id=spec.model_id, revision=spec.revision)
        validate_remote_model(spec, info)

    location = Path(cache_dir) if cache_dir is not None else default_cache_dir()
    path = snapshot_download(
        repo_id=spec.model_id,
        revision=spec.revision,
        cache_dir=str(location),
        local_files_only=not allow_download,
        allow_patterns=list(MODEL_ASSETS),
    )
    return Path(path)


def _sha256_file(path: Path) -> str:
    """Вычислить SHA-256 файла потоково, не загружая веса в оперативную память."""
    hasher = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def inspect_snapshot(spec: ModelSpec, snapshot: str | Path) -> dict[str, str]:
    """Проверить наличие обязательных файлов и получить их SHA-256.

    Аргументы:
        spec: Модель, для которой собираем manifest.
        snapshot: Папка snapshot от pinned Hugging Face revision.
    Результат:
        Словарь относительных имён и SHA-256 разрешённых артефактов.

    Ссылки Hugging Face snapshot допускаются: хешируются реальные bytes.
    """
    if not isinstance(spec, ModelSpec):
        raise TypeError("spec должен быть ModelSpec")
    root = Path(snapshot)
    if not root.is_dir():
        raise FileNotFoundError(f"Нет snapshot директории: {root}")
    required = ("config.json", "model.safetensors")
    for filename in required:
        if not (root / filename).is_file():
            raise FileNotFoundError(f"Нет обязательного файла модели: {filename}")
    if not (root / "tokenizer.json").is_file() and not (root / "vocab.txt").is_file():
        raise FileNotFoundError("Токенизатор должен содержать tokenizer.json или vocab.txt")

    return {
        filename: _sha256_file(root / filename)
        for filename in MODEL_ASSETS
        if (root / filename).is_file()
    }


def create_manifest(
    spec: ModelSpec,
    snapshot: str | Path,
    *,
    versions: Mapping[str, str],
) -> dict[str, Any]:
    """Сформировать воспроизводимые метаданные без сохранения текста или весов.

    Аргументы:
        spec: Зафиксированный контракт pretrained-модели.
        snapshot: Локальный pinned snapshot.
        versions: Реальные версии torch, transformers, huggingface_hub.
    Результат:
        JSON-совместимый manifest из spec, версий пакетов и SHA-256 файлов.
    """
    expected = {"torch", "transformers", "huggingface_hub"}
    if set(versions) != expected or any(not isinstance(v, str) or not v for v in versions.values()):
        raise ValueError("Необходимо указать действительные версии трёх библиотек")
    return {
        "schema_version": 1,
        "model": spec.to_dict(),
        "library_versions": {name: versions[name] for name in sorted(versions)},
        "sha256": inspect_snapshot(spec, snapshot),
    }


def validate_manifest(
    manifest: Mapping[str, Any], spec: ModelSpec, snapshot: str | Path
) -> None:
    """Сверить fingerprint кеша с manifest, отвергая замену файлов и revision.

    Аргументы:
        manifest: Ранее сформированный объект manifest.
        spec: Ожидаемый pinned контракт модели.
        snapshot: Текущий набор артефактов в локальном кеше.
    Результат:
        None при совпадении, ValueError при несовпадении.
    """
    if set(manifest) != {"schema_version", "model", "library_versions", "sha256"}:
        raise ValueError("Неподдерживаемая структура manifest")
    if manifest["schema_version"] != 1 or manifest["model"] != spec.to_dict():
        raise ValueError("Manifest относится к другой версии модели или конфигурации")
    current = inspect_snapshot(spec, snapshot)
    if manifest["sha256"] != current:
        raise ValueError("Контрольные суммы кеша не совпадают: файлы изменены")


def write_manifest(manifest: Mapping[str, Any], path: str | Path) -> None:
    """Атомарно сохранить только metadata, не содержащие пользовательские тексты.

    Аргументы:
        manifest: Версия модели, версии библиотек и хеши файлов.
        path: Расположение JSON вне Git-репозитория (выбирает вызывающий код).
    Результат:
        None, готовый UTF-8 JSON после успешной атомарной замены файла.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_name(destination.name + ".tmp")
    try:
        tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
        tmp.replace(destination)
    finally:
        tmp.unlink(missing_ok=True)


def load_pretrained_encoder(
    spec: ModelSpec,
    *,
    cache_dir: str | Path | None = None,
    allow_download: bool = False,
    manifest_path: str | Path | None = None,
) -> tuple[PretrainedTextEncoder, dict[str, Any]]:
    """Загрузить pretrained encoder на CPU по SHA и вернуть manifest.

    Аргументы:
        spec: Pinned модель с разрешённой Apache-2.0 лицензией.
        cache_dir: Пользовательский кеш весов, а не папка проекта.
        allow_download: Только явное True допускает сеть и скачивание.
        manifest_path: Если указан, сохранить JSON manifest атомарно.
    Результат:
        (PretrainedTextEncoder, manifest), сеть при False не используется.
    """
    snapshot = download_snapshot(spec, cache_dir=cache_dir, allow_download=allow_download)
    # Проверяем наличие безопасного формата до создания PyTorch-модели.
    inspect_snapshot(spec, snapshot)
    try:
        import huggingface_hub
        import transformers
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Для реального encoder необходимо установить transformers") from exc

    tokenizer = AutoTokenizer.from_pretrained(
        str(snapshot), local_files_only=True, trust_remote_code=False, use_fast=True
    )
    model = AutoModel.from_pretrained(
        str(snapshot), local_files_only=True, trust_remote_code=False, use_safetensors=True
    )
    # Не выполняем train и не скачиваем «альтернативные» веса при несовпадении.
    hidden_size = getattr(getattr(model, "config", None), "hidden_size", None)
    if hidden_size != spec.embedding_dim:
        raise ValueError("hidden_size pretrained encoder не совпадает с embedding_dim")

    manifest = create_manifest(
        spec,
        snapshot,
        versions={
            "torch": str(torch.__version__),
            "transformers": str(transformers.__version__),
            "huggingface_hub": str(huggingface_hub.__version__),
        },
    )
    if manifest_path is not None:
        manifest_file = Path(manifest_path)
        if manifest_file.is_file():
            # Не перезаписываем прежнее доказательство новым хешем:
            # иначе порча кеша осталась бы незамеченной при повторном запуске.
            previous = json.loads(manifest_file.read_text(encoding="utf-8"))
            validate_manifest(previous, spec, snapshot)
            if previous["library_versions"] != manifest["library_versions"]:
                raise ValueError(
                    "Версии библиотек изменились относительно прежнего manifest; "
                    "проверь воспроизводимость перед созданием нового manifest"
                )
        else:
            write_manifest(manifest, manifest_path)
    return PretrainedTextEncoder(spec, tokenizer, model), manifest
