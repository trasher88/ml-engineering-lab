"""F4.2: выбор базовой Causal LM по фиксированному каталогу и бюджету CPU.

Стандартная библиотека Python; импорт torch/transformers возможен только в
явном необязательном probe токенизатора. Загрузка весов здесь отсутствует.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import re
import sys
from typing import Any


class ModelSelectionError(ValueError):
    """Некорректная спецификация модели, окружение или файл отчёта."""


@dataclass(frozen=True, slots=True)
class Requirements:
    """Инженерные ограничения опыта с F4.1 на CPU."""

    task: str
    stage: str
    required_language: str
    allowed_licenses: tuple[str, ...]
    cpu_ram_budget_gib: int
    sequence_length: int
    batch_size: int
    cpu_dtype: str
    minimum_python: str
    minimum_torch: str
    minimum_transformers: str


@dataclass(frozen=True, slots=True)
class ModelCandidate:
    """Проверенные метаданные checkpoint, а не загруженные параметры модели."""

    model_id: str
    revision: str
    tokenizer_id: str
    tokenizer_revision: str
    architecture: str
    model_type: str
    task: str
    stage: str
    license: str
    languages: tuple[str, ...]
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    max_position_embeddings: int
    tie_word_embeddings: bool
    qkv_bias: bool
    eos_token_id: int
    published_parameters_approx: int
    source_model_card: str
    source_config: str


@dataclass(frozen=True, slots=True)
class ModelCatalog:
    """Единственный источник выбора, фиксирующий также внешние ревизии."""

    selected_model_id: str
    requirements: Requirements
    candidates: tuple[ModelCandidate, ...]
    sha256: str


CANDIDATE_KEYS = frozenset(ModelCandidate.__dataclass_fields__)
REQUIREMENT_KEYS = frozenset(Requirements.__dataclass_fields__)
COMMIT_SHA = re.compile(r"[a-f0-9]{40}\Z")
MODEL_ID = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+\Z")
VERSION = re.compile(r"(\d+)\.(\d+)\.(\d+)(?:[+.-].*)?\Z")
SUPPORTED_FAMILIES = {"qwen2": "Qwen2ForCausalLM", "llama": "LlamaForCausalLM"}
DATA_TYPES = {"float32": 4, "float16": 2, "bfloat16": 2}


def _int(raw: Any, name: str, *, minimum: int = 1) -> int:
    if type(raw) is not int or raw < minimum:
        raise ModelSelectionError(f"{name}: expected integer >= {minimum}")
    return raw


def _version_tuple(text: str) -> tuple[int, int, int]:
    match = VERSION.fullmatch(text) if isinstance(text, str) else None
    if match is None:
        raise ModelSelectionError(f"Invalid version format: {text!r}")
    return tuple(map(int, match.groups()))


def _strings(raw: Any, name: str) -> tuple[str, ...]:
    if (not isinstance(raw, list) or not raw
            or any(not isinstance(value, str) or not value or value.strip() != value for value in raw)
            or len(set(raw)) != len(raw)):
        raise ModelSelectionError(f"{name}: expected unique non-empty strings")
    return tuple(raw)


def _validate_requirements(raw: Any) -> Requirements:
    if not isinstance(raw, dict) or set(raw) != REQUIREMENT_KEYS:
        raise ModelSelectionError("requirements: unexpected or missing fields")
    values = dict(raw)
    values["allowed_licenses"] = _strings(raw["allowed_licenses"], "allowed_licenses")
    for key in ("cpu_ram_budget_gib", "sequence_length", "batch_size"):
        values[key] = _int(raw[key], key)
    if raw["cpu_dtype"] != "float32":
        raise ModelSelectionError("F4.2 CPU profile requires float32")
    if raw["task"] != "causal_lm" or raw["stage"] != "base":
        raise ModelSelectionError("F4.2 selects base causal language models only")
    if raw["required_language"] != "ru":
        raise ModelSelectionError("F4.1 instruction dataset is Russian")
    for key in ("minimum_python", "minimum_torch", "minimum_transformers"):
        _version_tuple(raw[key])
    return Requirements(**values)


def _validate_candidate(raw: Any) -> ModelCandidate:
    if not isinstance(raw, dict) or set(raw) != CANDIDATE_KEYS:
        raise ModelSelectionError("candidate: unexpected or missing fields")
    values = dict(raw)
    if (not isinstance(raw["model_id"], str) or not isinstance(raw["tokenizer_id"], str)
            or not MODEL_ID.fullmatch(raw["model_id"]) or not MODEL_ID.fullmatch(raw["tokenizer_id"])):
        raise ModelSelectionError("model_id/tokenizer_id: invalid Hub identifier")
    if raw["tokenizer_id"] != raw["model_id"]:
        raise ModelSelectionError("Tokenizer must match the exact model repository")
    for key in ("revision", "tokenizer_revision"):
        if not isinstance(raw[key], str) or not COMMIT_SHA.fullmatch(raw[key]):
            raise ModelSelectionError(f"{key}: expected complete 40-char immutable SHA")
    if raw["revision"] != raw["tokenizer_revision"]:
        raise ModelSelectionError("Model and tokenizer revisions must match")
    if not isinstance(raw["model_type"], str) or raw["model_type"] not in SUPPORTED_FAMILIES:
        raise ModelSelectionError("Unsupported architecture family")
    if raw["architecture"] != SUPPORTED_FAMILIES[raw["model_type"]]:
        raise ModelSelectionError("Architecture does not match model_type")
    for key in ("vocab_size", "hidden_size", "intermediate_size", "num_hidden_layers",
                "num_attention_heads", "num_key_value_heads", "max_position_embeddings",
                "published_parameters_approx"):
        values[key] = _int(raw[key], key)
    if type(raw["eos_token_id"]) is not int or not 0 <= raw["eos_token_id"] < raw["vocab_size"]:
        raise ModelSelectionError("Invalid eos_token_id")
    if type(raw["tie_word_embeddings"]) is not bool or type(raw["qkv_bias"]) is not bool:
        raise ModelSelectionError("Embedding tying and bias must be boolean")
    values["languages"] = _strings(raw["languages"], "languages")
    if raw["hidden_size"] % raw["num_attention_heads"]:
        raise ModelSelectionError("hidden_size must be divisible by num_attention_heads")
    if (raw["num_key_value_heads"] > raw["num_attention_heads"]
            or raw["num_attention_heads"] % raw["num_key_value_heads"]):
        raise ModelSelectionError("Invalid grouped-query attention dimensions")
    for key in ("source_model_card", "source_config"):
        if not isinstance(raw[key], str) or not raw[key].startswith("https://huggingface.co/"):
            raise ModelSelectionError(f"Invalid model source URL: {key}")
    if not raw["source_model_card"].endswith(raw["model_id"]):
        raise ModelSelectionError("Card URL does not match model ID")
    if f"/{raw['revision']}/config.json" not in raw["source_config"]:
        raise ModelSelectionError("Config URL is not pinned to the selected revision")
    candidate = ModelCandidate(**values)
    computed = count_parameters(candidate)
    if abs(computed - candidate.published_parameters_approx) > computed * 0.05:
        raise ModelSelectionError("Published parameter count inconsistent with architecture")
    return candidate


def load_catalog(path: Path) -> ModelCatalog:
    """Прочитать и строго проверить каталог и SHA-256 его точных байтов."""
    try:
        content = path.read_bytes()
        raw = json.loads(content.decode("utf-8-sig"), parse_constant=lambda x: (_ for _ in ()).throw(ModelSelectionError(f"Invalid JSON constant: {x}")))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ModelSelectionError(f"Cannot read catalog {path}: {exc}") from exc
    if (not isinstance(raw, dict) or set(raw) != {"schema_version", "selected_model_id", "requirements", "candidates"}
            or type(raw["schema_version"]) is not int or raw["schema_version"] != 1):
        raise ModelSelectionError("Invalid catalog schema version or fields")
    requirements = _validate_requirements(raw["requirements"])
    if not isinstance(raw["candidates"], list) or len(raw["candidates"]) < 2:
        raise ModelSelectionError("At least two candidates required for comparison")
    candidates = tuple(_validate_candidate(item) for item in raw["candidates"])
    ids = [item.model_id for item in candidates]
    if len(set(ids)) != len(ids):
        raise ModelSelectionError("Duplicate model IDs")
    if raw["selected_model_id"] not in ids:
        raise ModelSelectionError("Selected model missing from candidates")
    return ModelCatalog(raw["selected_model_id"], requirements, candidates, sha256(content).hexdigest())


def count_parameters(spec: ModelCandidate) -> int:
    """Подсчёт параметров decoder-only Transformer (tied embeddings и SwiGLU).

    Учитываются Q/K/V/O, MLP gate/up/down, RMSNorm, qkv bias и embedding.
    Формула действует для двух утверждённых семейств, но не универсальна.
    """
    hidden = spec.hidden_size
    head_dim = hidden // spec.num_attention_heads
    kv_width = spec.num_key_value_heads * head_dim
    attention = 2 * hidden * hidden + 2 * hidden * kv_width
    attention_bias = hidden + 2 * kv_width if spec.qkv_bias else 0
    mlp = 3 * hidden * spec.intermediate_size
    norms = 2 * hidden
    embeddings = spec.vocab_size * hidden * (1 if spec.tie_word_embeddings else 2)
    return embeddings + spec.num_hidden_layers * (attention + attention_bias + mlp + norms) + hidden


def kv_cache_bytes(spec: ModelCandidate, *, sequence_length: int, batch_size: int, bytes_per_value: int) -> int:
    """Оценка только K/V cache без учёта иных буферов и аллокатора."""
    for name, value in (("sequence_length", sequence_length), ("batch_size", batch_size), ("bytes_per_value", bytes_per_value)):
        _int(value, name)
    if sequence_length > spec.max_position_embeddings:
        raise ModelSelectionError("Sequence exceeds checkpoint context window")
    head_dim = spec.hidden_size // spec.num_attention_heads
    return (2 * spec.num_hidden_layers * spec.num_key_value_heads * head_dim
            * sequence_length * batch_size * bytes_per_value)


def resource_estimate(spec: ModelCandidate, requirements: Requirements) -> dict[str, Any]:
    """Теоретические байты весов/KV; НЕ оценка полной RAM или VRAM."""
    params = count_parameters(spec)
    kv = kv_cache_bytes(spec, sequence_length=min(requirements.sequence_length, spec.max_position_embeddings),
                        batch_size=requirements.batch_size, bytes_per_value=DATA_TYPES[requirements.cpu_dtype])
    weight = params * DATA_TYPES[requirements.cpu_dtype]
    return {
        "estimated_parameters": params,
        "published_parameters_approx": spec.published_parameters_approx,
        "fp32_weights_mib": round(params * 4 / 2**20, 3),
        "fp16_bf16_weights_mib": round(params * 2 / 2**20, 3),
        "ideal_int8_weights_mib": round(params / 2**20, 3),
        "ideal_int4_weights_mib": round(params / 2 / 2**20, 3),
        "kv_cache_fp32_mib": round(kv / 2**20, 3),
        "weights_plus_kv_fp32_mib": round((weight + kv) / 2**20, 3),
        "cpu_budget_gib": requirements.cpu_ram_budget_gib,
        "kv_cache_sequence_used": min(requirements.sequence_length, spec.max_position_embeddings),
        "warning": "Estimates exclude activations, runtime overhead, temporary buffers and optimizer states",
    }


def assess_candidate(spec: ModelCandidate, requirements: Requirements) -> dict[str, Any]:
    """Детерминированный фильтр, а не прогноз качества ответов модели."""
    estimate = resource_estimate(spec, requirements)
    reasons: list[str] = []
    if spec.task != requirements.task:
        reasons.append("task_mismatch")
    if spec.stage != requirements.stage:
        reasons.append("not_base_model")
    if spec.license not in requirements.allowed_licenses:
        reasons.append("license_not_allowed")
    if requirements.required_language not in spec.languages:
        reasons.append("russian_not_declared")
    if requirements.sequence_length > spec.max_position_embeddings:
        reasons.append("context_too_short")
    weight_bytes = count_parameters(spec) * DATA_TYPES[requirements.cpu_dtype]
    kv_bytes = kv_cache_bytes(spec, sequence_length=min(requirements.sequence_length, spec.max_position_embeddings),
                              batch_size=requirements.batch_size, bytes_per_value=DATA_TYPES[requirements.cpu_dtype])
    if weight_bytes + kv_bytes > requirements.cpu_ram_budget_gib * 2**30:
        reasons.append("weight_kv_lower_bound_exceeds_cpu_budget")
    return {"model_id": spec.model_id, "eligible": not reasons, "reasons": reasons, "resources": estimate}


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def runtime_report(requirements: Requirements, *, python_version: tuple[int, int, int] | None = None,
                   packages: dict[str, str | None] | None = None) -> dict[str, Any]:
    """Проверить metadata установленных библиотек, не импортируя сами ML пакеты."""
    py = python_version if python_version is not None else tuple(sys.version_info[:3])
    found = packages if packages is not None else {key: _package_version(key) for key in ("torch", "transformers")}
    issues = []
    if py < _version_tuple(requirements.minimum_python):
        issues.append("python_too_old")
    for package, minimum in (("torch", requirements.minimum_torch), ("transformers", requirements.minimum_transformers)):
        installed = found.get(package)
        if installed is None:
            issues.append(f"{package}_missing")
        elif _version_tuple(installed) < _version_tuple(minimum):
            issues.append(f"{package}_too_old")
    return {"python": ".".join(map(str, py)), "packages": dict(found),
            "ready_for_optional_probe": not issues, "issues": issues}


def create_report(catalog: ModelCatalog, *, runtime: dict[str, Any] | None = None) -> dict[str, Any]:
    """Зафиксировать решение и все альтернативы, включая причины исключения."""
    assessments = [assess_candidate(candidate, catalog.requirements) for candidate in catalog.candidates]
    selected = next(item for item in assessments if item["model_id"] == catalog.selected_model_id)
    if not selected["eligible"]:
        raise ModelSelectionError(f"Selected model fails constraints: {selected['reasons']}")
    model = next(candidate for candidate in catalog.candidates if candidate.model_id == catalog.selected_model_id)
    return {
        "schema_version": 1,
        "stage": "F4.2",
        "status": "METADATA_SELECTED_ONLY",
        "catalog_sha256": catalog.sha256,
        "selection_rule": "Pinned preferred model must meet hard constraints; no quality ranking is claimed",
        "selected_model": {"model_id": model.model_id, "revision": model.revision,
                           "tokenizer_id": model.tokenizer_id, "tokenizer_revision": model.tokenizer_revision,
                           "architecture": model.architecture, "model_type": model.model_type,
                           "license": model.license, "context_window": model.max_position_embeddings,
                           "eos_token_id": model.eos_token_id},
        "requirements": asdict(catalog.requirements),
        "candidate_assessments": assessments,
        "runtime": runtime if runtime is not None else runtime_report(catalog.requirements),
        "weights_loaded": False,
        "tokenizer_loaded": False,
        "inference_run": False,
        "training_run": False,
    }


def report_bytes(report: dict[str, Any]) -> bytes:
    return (json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True) + "\n").encode("utf-8")


def write_report(path: Path, report: dict[str, Any]) -> None:
    """Создать локальный отчёт атомарной заменой (не входит в Git)."""
    data = report_bytes(report)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    try:
        temp.write_bytes(data)
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def verify_report(path: Path, report: dict[str, Any]) -> None:
    """Побайтово сравнить сохранённый план с текущей конфигурацией и окружением."""
    try:
        existing = path.read_bytes()
    except OSError as exc:
        raise ModelSelectionError(f"Cannot read report: {exc}") from exc
    if existing != report_bytes(report):
        raise ModelSelectionError("Report mismatch: catalog, runtime or report was modified")


def probe_tokenizer(spec: ModelCandidate, sample_text: str, *, allow_download: bool = False) -> dict[str, Any]:
    """Необязательная проверка реального токенизатора, БЕЗ весов LLM.

    По умолчанию читает только локальный HF cache. Сеть включается явным
    флагом --allow-download в CLI; trust_remote_code всегда False.
    """
    if not sample_text or not isinstance(sample_text, str):
        raise ModelSelectionError("Sample text required")
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ModelSelectionError("Optional probe requires transformers/tokenizers") from exc
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            spec.tokenizer_id,
            revision=spec.tokenizer_revision,
            local_files_only=not allow_download,
            trust_remote_code=False,
            use_fast=True,
        )
        ids = tokenizer.encode(sample_text, add_special_tokens=False)
        recovered = tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    except Exception as exc:
        raise ModelSelectionError(f"Tokenizer probe could not run: {type(exc).__name__}: {exc}") from exc
    if not ids or not all(type(value) is int and 0 <= value < spec.vocab_size for value in ids):
        raise ModelSelectionError("Invalid token IDs produced")
    if recovered != sample_text:
        raise ModelSelectionError("Tokenizer round-trip did not reproduce input exactly")
    if tokenizer.eos_token_id != spec.eos_token_id:
        raise ModelSelectionError("Tokenizer EOS differs from pinned checkpoint metadata")
    if len(tokenizer) > spec.vocab_size:
        raise ModelSelectionError("Tokenizer has more tokens than model embedding size")
    return {"tokenizer_id": spec.tokenizer_id, "tokenizer_revision": spec.tokenizer_revision,
            "token_count": len(ids), "first_token_ids": ids[:12],
            "roundtrip": True, "weights_loaded": False}
