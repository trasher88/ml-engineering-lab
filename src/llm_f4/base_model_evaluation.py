"""F4.4: воспроизводимая генерация Base Causal LM на sealed F4.3 prompts.

Без явного --run модуль не импортирует torch/transformers и не загружает
веса. В модель передаются только строки prompt; gold/rubric загружаются
отдельно и исключительно по запросу ручного оценивания.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import statistics
import sys
from time import perf_counter
from typing import Any, Callable, Mapping, Protocol, Sequence

from src.llm_f4.evaluation_set import (
    DOMAINS, EvaluationError, load_cases, load_config as load_eval_config,
    score_manual_grades, verify_artifacts,
)
from src.llm_f4.model_selection import (
    ModelSelectionError, assess_candidate, load_catalog, runtime_report,
)


class BaselineError(ValueError):
    """Нарушен контракт, целостность, согласованность или режим исполнения."""


@dataclass(frozen=True, slots=True)
class BaselineConfig:
    """Настройки только для Qwen Base и фиксированного F4.3 holdout."""

    evaluation_config_file: Path
    model_catalog_file: Path
    output_dir: Path
    model_id: str
    model_revision: str
    tokenizer_revision: str
    max_input_tokens: int
    max_new_tokens: int
    seed: int
    cpu_threads: int
    prefix: str
    suffix: str
    smoke_ids: tuple[str, ...]
    sha256: str


@dataclass(frozen=True, slots=True)
class Prompt:
    case_id: str
    domain: str
    prompt: str


@dataclass(frozen=True, slots=True)
class Generation:
    """Выход реальной генерации или подставного backend в unit-тестах."""

    text: str
    prompt_token_count: int
    generated_token_count: int
    stopped_on_eos: bool


class Generator(Protocol):
    def generate(self, prompt: str, *, max_input_tokens: int, max_new_tokens: int) -> Generation: ...


CONFIG_FIELDS = {
    "schema_version", "evaluation_config_file", "model_catalog_file", "output_dir",
    "model_id", "model_revision", "tokenizer_revision", "max_input_tokens",
    "max_new_tokens", "seed", "cpu_threads", "prefix", "suffix", "smoke_ids",
}
PROFILE_NAMES = ("smoke", "full")
CASE_PATTERN = re.compile(r"ev_(py|lx|sq|ml)_[0-9]{2}\Z")
REVISION_PATTERN = re.compile(r"[0-9a-f]{40}\Z")
SHA_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
RESULT_FIELDS = {
    "case_id", "domain", "prompt_sha256", "response", "response_sha256", "prompt_tokens",
    "generated_tokens", "stopped_on_eos", "elapsed_seconds", "model_generated",
}


def _json(raw: bytes, where: str) -> Any:
    def no_duplicate(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        obj: dict[str, Any] = {}
        for key, value in pairs:
            if key in obj:
                raise BaselineError(f"Duplicate JSON key: {key} ({where})")
            obj[key] = value
        return obj

    def no_constant(value: str) -> None:
        raise BaselineError(f"Invalid JSON constant {value} ({where})")

    try:
        return json.loads(raw.decode("utf-8-sig"), object_pairs_hook=no_duplicate, parse_constant=no_constant)
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise BaselineError(f"Invalid JSON ({where}): {exc}") from exc


def _read(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except OSError as exc:
        raise BaselineError(f"Cannot read {path}: {exc}") from exc


def _safe_path(root: Path, name: Any, field: str) -> Path:
    if (not isinstance(name, str) or not name or "\\" in name or ":" in name
            or name.startswith("/") or any(part in ("", ".", "..") for part in name.split("/"))):
        raise BaselineError(f"{field}: relative POSIX path required")
    base = root.resolve()
    target = (base / name).resolve()
    if not target.is_relative_to(base):
        raise BaselineError(f"{field}: path outside repository")
    return target


def _positive(value: Any, field: str, minimum: int = 1, maximum: int = 10_000) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise BaselineError(f"{field}: expected integer between {minimum} and {maximum}")
    return value


def load_config(file: Path, *, root: Path) -> BaselineConfig:
    """Проверить настройки и связать их с ранее утверждённым выбором модели."""
    raw_bytes = _read(file)
    raw = _json(raw_bytes, str(file))
    if not isinstance(raw, dict) or set(raw) != CONFIG_FIELDS or type(raw["schema_version"]) is not int or raw["schema_version"] != 1:
        raise BaselineError("Invalid F4.4 configuration schema")
    for key in ("model_id", "model_revision", "tokenizer_revision", "prefix", "suffix"):
        if not isinstance(raw[key], str) or not raw[key]:
            raise BaselineError(f"{key}: non-empty string required")
    for key in ("model_revision", "tokenizer_revision"):
        if not REVISION_PATTERN.fullmatch(raw[key]):
            raise BaselineError(f"{key}: exact 40-character revision required")
    if raw["model_revision"] != raw["tokenizer_revision"]:
        raise BaselineError("Model and tokenizer revisions must be identical")
    for key in ("prefix", "suffix"):
        if len(raw[key]) > 200 or any(ord(c) < 32 and c != "\n" for c in raw[key]):
            raise BaselineError(f"{key}: invalid prompt formatting")
    if "{prompt}" in raw["prefix"] + raw["suffix"]:
        raise BaselineError("Prompt is concatenated between prefix and suffix, not interpolated")
    smoke = raw["smoke_ids"]
    if (not isinstance(smoke, list) or not 1 <= len(smoke) <= 4
            or any(not isinstance(s, str) or not CASE_PATTERN.fullmatch(s) for s in smoke)
            or len(set(smoke)) != len(smoke)):
        raise BaselineError("smoke_ids: 1-4 unique evaluation IDs required")
    p = BaselineConfig(
        evaluation_config_file=_safe_path(root, raw["evaluation_config_file"], "evaluation_config_file"),
        model_catalog_file=_safe_path(root, raw["model_catalog_file"], "model_catalog_file"),
        output_dir=_safe_path(root, raw["output_dir"], "output_dir"),
        model_id=raw["model_id"],
        model_revision=raw["model_revision"],
        tokenizer_revision=raw["tokenizer_revision"],
        max_input_tokens=_positive(raw["max_input_tokens"], "max_input_tokens", 16, 8192),
        max_new_tokens=_positive(raw["max_new_tokens"], "max_new_tokens", 1, 512),
        seed=_positive(raw["seed"], "seed", 0, 2**31 - 1),
        cpu_threads=_positive(raw["cpu_threads"], "cpu_threads", 1, 64),
        prefix=raw["prefix"], suffix=raw["suffix"], smoke_ids=tuple(smoke),
        sha256=sha256(raw_bytes).hexdigest(),
    )
    cat = load_catalog(p.model_catalog_file)
    if p.model_id != cat.selected_model_id:
        raise BaselineError("F4.4 model differs from selected F4.2 candidate")
    spec = next(c for c in cat.candidates if c.model_id == p.model_id)
    if (spec.revision != p.model_revision or spec.tokenizer_revision != p.tokenizer_revision
            or spec.tokenizer_id != p.model_id):
        raise BaselineError("Pinned F4.2 checkpoint/tokenizer revision mismatch")
    if not assess_candidate(spec, cat.requirements)["eligible"]:
        raise BaselineError("F4.2 selected model is no longer eligible")
    if p.max_input_tokens + p.max_new_tokens > cat.requirements.sequence_length:
        raise BaselineError("Input + generation exceeds frozen F4.2 sequence length")
    if p.max_input_tokens + p.max_new_tokens > spec.max_position_embeddings:
        raise BaselineError("Input + generation exceeds checkpoint context")
    eval_cfg = load_eval_config(p.evaluation_config_file, root=root)
    if eval_cfg.model_catalog_file != p.model_catalog_file:
        raise BaselineError("F4.3 and F4.4 catalogs differ")
    if eval_cfg.model_catalog_sha256 != cat.sha256:
        raise BaselineError("F4.2 catalog changed after F4.3")
    return p


def _line(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def load_prompts(config: BaselineConfig, *, root: Path) -> tuple[list[Prompt], dict[str, Any]]:
    """Читать ТОЛЬКО одобренный экспорт F4.3; ни эталон, ни рубрика не возвращаются.

    verify_artifacts проверяет весь пайплайн F4.3, включая SHA источников.
    Эталоны в модельную функцию никогда не передаются.
    """
    eval_cfg = load_eval_config(config.evaluation_config_file, root=root)
    try:
        outputs = verify_artifacts(eval_cfg)
    except (EvaluationError, ModelSelectionError) as exc:
        raise BaselineError(f"F4.3 export validation failed: {exc}") from exc
    manifest = _json(outputs["manifest.json"], "F4.3 manifest")
    if manifest["model_id"] != config.model_id or manifest["model_revision"] != config.model_revision:
        raise BaselineError("F4.3 model/revision mismatch")
    rows = outputs["prompts.jsonl"].decode("utf-8").splitlines()
    if len(rows) != manifest["total_cases"] or sha256(outputs["prompts.jsonl"]).hexdigest() != manifest["prompts_sha256"]:
        raise BaselineError("Prompt export count or SHA mismatch")
    prompts: list[Prompt] = []
    seen: set[str] = set()
    for line in rows:
        record = _json(line.encode("utf-8"), "prompt JSONL")
        if not isinstance(record, dict) or set(record) != {"case_id", "domain", "prompt"}:
            raise BaselineError("Prompt export contains forbidden fields (reference/rubric?)")
        cid, domain, question = record["case_id"], record["domain"], record["prompt"]
        if (not isinstance(cid, str) or not CASE_PATTERN.fullmatch(cid) or cid in seen
                or domain not in DOMAINS or not isinstance(question, str) or not question.strip()):
            raise BaselineError("Invalid/duplicate case in F4.3 prompt export")
        seen.add(cid)
        prompts.append(Prompt(cid, domain, question))
    if len(prompts) != 32:
        raise BaselineError("Expected full frozen F4.3 set of 32 prompts")
    return prompts, manifest


def selection(prompts: Sequence[Prompt], config: BaselineConfig, profile: str) -> list[Prompt]:
    """Smoke НЕ считается оценкой полного benchmark."""
    if profile not in PROFILE_NAMES:
        raise BaselineError("Unknown profile")
    if profile == "full":
        return list(prompts)
    chosen = {p.case_id: p for p in prompts}
    if not set(config.smoke_ids).issubset(chosen):
        raise BaselineError("Smoke IDs absent from frozen F4.3 export")
    return [chosen[cid] for cid in config.smoke_ids]


def formatted_prompt(item: Prompt, conf: BaselineConfig) -> str:
    return conf.prefix + item.prompt + conf.suffix


def _write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    try:
        temp.write_bytes(payload)
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def _json_pretty(value: Any) -> bytes:
    return (json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def profile_context(config: BaselineConfig, manifest: Mapping[str, Any], profile: str,
                    *, runtime: Mapping[str, Any]) -> dict[str, Any]:
    """Контекст для проверки возобновления и сравнения с F4.7."""
    if profile not in PROFILE_NAMES:
        raise BaselineError("Unsupported profile")
    return {
        "schema_version": 1, "stage": "f4.4", "profile": profile, "model_stage": "base",
        "model_id": config.model_id, "model_revision": config.model_revision,
        "tokenizer_revision": config.tokenizer_revision,
        "evaluation_id": manifest["evaluation_id"], "eval_prompts_sha256": manifest["prompts_sha256"],
        "eval_source_sha256": manifest["source_sha256"],
        "config_sha256": config.sha256, "seed": config.seed, "dtype": "float32", "device": "cpu",
        "decoding": "greedy", "do_sample": False, "num_beams": 1,
        "max_input_tokens": config.max_input_tokens, "max_new_tokens": config.max_new_tokens,
        "prompt_prefix": config.prefix, "prompt_suffix": config.suffix,
        "runtime": dict(runtime),
    }


def _context_sha(context: Mapping[str, Any]) -> str:
    return sha256(_json_pretty(context)).hexdigest()


def _validate_result(row: Any, prompt: Prompt, conf: BaselineConfig) -> None:
    if not isinstance(row, dict) or set(row) != RESULT_FIELDS:
        raise BaselineError("Response: unexpected fields")
    if row["case_id"] != prompt.case_id or row["domain"] != prompt.domain:
        raise BaselineError("Response case_id/domain not aligned")
    expected_hash = sha256(formatted_prompt(prompt, conf).encode("utf-8")).hexdigest()
    if row["prompt_sha256"] != expected_hash:
        raise BaselineError("Prompt fingerprint mismatch")
    if not isinstance(row["response"], str) or len(row["response"]) > 200_000:
        raise BaselineError("Response text is not a bounded string")
    if row["response_sha256"] != sha256(row["response"].encode("utf-8")).hexdigest():
        raise BaselineError("Response checksum mismatch")
    if (type(row["prompt_tokens"]) is not int or not 1 <= row["prompt_tokens"] <= conf.max_input_tokens
            or type(row["generated_tokens"]) is not int or not 1 <= row["generated_tokens"] <= conf.max_new_tokens):
        raise BaselineError("Invalid token counts")
    if type(row["stopped_on_eos"]) is not bool or row["model_generated"] is not True:
        raise BaselineError("Response not marked as real generation")
    seconds = row["elapsed_seconds"]
    if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds < 0:
        raise BaselineError("Invalid elapsed time")


def _validate_rows(rows: Sequence[dict[str, Any]], prompts: Sequence[Prompt], conf: BaselineConfig) -> None:
    if len(rows) > len(prompts):
        raise BaselineError("Too many generated responses")
    for row, p in zip(rows, prompts, strict=False):
        _validate_result(row, p, conf)


def _load_journal(path: Path, prompts: Sequence[Prompt], conf: BaselineConfig) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    raw = _read(path)
    if raw and not raw.endswith(b"\n"):
        raise BaselineError("Interrupted/corrupted progress.jsonl; do not silently repair")
    records = [_json(line, "progress JSONL") for line in raw.splitlines()]
    _validate_rows(records, prompts, conf)
    return records


def _append_journal(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as stream:
        stream.write(_line(record))
        stream.flush()
        os.fsync(stream.fileno())


def _summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    durations = [float(x["elapsed_seconds"]) for x in rows]
    tokens = [int(x["generated_tokens"]) for x in rows]
    total_time = sum(durations)
    return {
        "generated_cases": len(rows), "total_generated_tokens": sum(tokens),
        "total_generation_seconds": round(total_time, 4),
        "tokens_per_generation_second": round(sum(tokens) / total_time, 4) if total_time > 0 else None,
        "mean_tokens_per_case": round(statistics.mean(tokens), 4) if tokens else None,
        "eos_stops": sum(bool(r["stopped_on_eos"]) for r in rows),
        "empty_decoded_responses": sum(not r["response"].strip() for r in rows),
        "quality_score": None, "quality_status": "NOT_HUMAN_GRADED",
        "timing_note": "Generation durations exclude model loading; CPU performance depends on machine/runtime",
    }


def _final_payload(rows: Sequence[dict[str, Any]], context: dict[str, Any], count: int) -> tuple[bytes, bytes]:
    responses = b"".join(_line(r) for r in rows)
    manifest = {
        **context,
        "context_sha256": _context_sha(context),
        "expected_cases": count,
        "completed_cases": len(rows),
        "run_complete": len(rows) == count,
        "responses_sha256": sha256(responses).hexdigest(),
        "generation_performed": True,
        "human_grading_performed": False,
        "metrics": _summary(rows),
    }
    return responses, _json_pretty(manifest)


def execute(config: BaselineConfig, *, root: Path, profile: str, backend_factory: Callable[[], Generator],
            runtime: Mapping[str, Any], resume: bool = False) -> dict[str, Any]:
    """Генерировать последовательно и сохранять каждую завершённую запись.

    backend_factory вызывается ПОСЛЕ валидации prompt-only и контекста;
    unit-тесты используют инъекцию backend без весов и помечены как mock,
    такие результаты никогда не выдаются за выполненную Qwen.
    """
    if runtime.get("kind") != "real_cpu_transformers":
        raise BaselineError("execute requires real_cpu_transformers runtime; mocks use internal test hook")
    return _execute_inner(config, root=root, profile=profile, backend_factory=backend_factory,
                          runtime=runtime, resume=resume)


def _execute_inner(config: BaselineConfig, *, root: Path, profile: str,
                   backend_factory: Callable[[], Generator], runtime: Mapping[str, Any],
                   resume: bool = False) -> dict[str, Any]:
    prompts, source_manifest = load_prompts(config, root=root)
    chosen = selection(prompts, config, profile)
    context = profile_context(config, source_manifest, profile, runtime=runtime)
    folder = config.output_dir / profile
    path_context = folder / "context.json"
    path_progress = folder / "progress.jsonl"
    path_responses = folder / "responses.jsonl"
    path_manifest = folder / "manifest.json"
    if resume:
        if _read(path_context) != _json_pretty(context):
            raise BaselineError("Resume context mismatch (config/revision/versions/profile)")
        saved = _load_journal(path_progress, chosen, config)
        if path_manifest.exists() and path_responses.exists():
            raise BaselineError("Complete run exists; use --verify, not --resume")
        if path_manifest.exists() != path_responses.exists():
            raise BaselineError("Incomplete final files; manual inspection required")
    else:
        if folder.exists() and any(folder.iterdir()):
            raise BaselineError(f"Output {folder} is not empty; use --resume or a clean directory")
        folder.mkdir(parents=True, exist_ok=True)
        _write_atomic(path_context, _json_pretty(context))
        saved = []
    if len(saved) == len(chosen):
        # Например, процесс оборвался между последней записью и final manifest.
        pass
    else:
        generator = backend_factory()
        for p in chosen[len(saved):]:
            prompt = formatted_prompt(p, config)
            start = perf_counter()
            output = generator.generate(prompt, max_input_tokens=config.max_input_tokens,
                                        max_new_tokens=config.max_new_tokens)
            elapsed = round(perf_counter() - start, 4)
            if not isinstance(output, Generation):
                raise BaselineError("Backend must return a Generation")
            row = {
                "case_id": p.case_id, "domain": p.domain,
                "prompt_sha256": sha256(prompt.encode("utf-8")).hexdigest(),
                "response": output.text,
                "response_sha256": sha256(output.text.encode("utf-8")).hexdigest() if isinstance(output.text, str) else "",
                "prompt_tokens": output.prompt_token_count,
                "generated_tokens": output.generated_token_count,
                "stopped_on_eos": output.stopped_on_eos,
                "elapsed_seconds": elapsed,
                "model_generated": True,
            }
            _validate_result(row, p, config)
            _append_journal(path_progress, row)
            saved.append(row)
    output_responses, output_manifest = _final_payload(saved, context, len(chosen))
    _write_atomic(path_responses, output_responses)
    _write_atomic(path_manifest, output_manifest)
    return _json(output_manifest, "F4.4 manifest")


def verify(config: BaselineConfig, *, root: Path, profile: str) -> dict[str, Any]:
    """Сверить журналы, SHA и контракты без повторного inference."""
    prompts, source_manifest = load_prompts(config, root=root)
    chosen = selection(prompts, config, profile)
    folder = config.output_dir / profile
    context = _json(_read(folder / "context.json"), "context")
    manifest = _json(_read(folder / "manifest.json"), "manifest")
    if (not isinstance(context, dict) or not isinstance(manifest, dict)
            or _context_sha(context) != manifest.get("context_sha256")):
        raise BaselineError("Context checksum mismatch")
    if (context.get("model_id") != config.model_id or context.get("model_revision") != config.model_revision
            or context.get("tokenizer_revision") != config.tokenizer_revision
            or context.get("config_sha256") != config.sha256
            or context.get("eval_prompts_sha256") != source_manifest["prompts_sha256"]
            or context.get("eval_source_sha256") != source_manifest["source_sha256"]
            or context.get("profile") != profile):
        raise BaselineError("Result identity does not match frozen F4.2/F4.3/F4.4")
    if (context.get("runtime", {}).get("kind") != "real_cpu_transformers"
            or context.get("model_stage") != "base" or context.get("decoding") != "greedy"):
        raise BaselineError("Not a real Base inference run")
    raw = _read(folder / "responses.jsonl")
    if sha256(raw).hexdigest() != manifest.get("responses_sha256"):
        raise BaselineError("Response SHA256 mismatch")
    if raw and not raw.endswith(b"\n"):
        raise BaselineError("Truncated response JSONL")
    rows = [_json(line, "responses") for line in raw.splitlines()]
    _validate_rows(rows, chosen, config)
    if len(rows) != len(chosen):
        raise BaselineError("Incomplete benchmark; cannot mark full/smoke PASS")
    if _read(folder / "progress.jsonl") != raw:
        raise BaselineError("Progress journal differs from final responses")
    expected_responses, expected_manifest = _final_payload(rows, context, len(chosen))
    if expected_responses != raw or expected_manifest != _read(folder / "manifest.json"):
        raise BaselineError("Manifest does not match computed provenance and metrics")
    if manifest.get("run_complete") is not True or manifest.get("generation_performed") is not True:
        raise BaselineError("Result is not completed inference")
    return manifest


def template_grades(config: BaselineConfig, *, root: Path) -> Path:
    """Создать аннотационную форму только после FULL inference. Gold не в ответах."""
    verify(config, root=root, profile="full")
    cfg = load_eval_config(config.evaluation_config_file, root=root)
    cases = load_cases(cfg.source_file)
    folder = config.output_dir / "full"
    responses = [_json(line, "responses") for line in _read(folder / "responses.jsonl").splitlines()]
    by_id = {x.case_id: x for x in cases}
    entries = []
    for row in responses:
        item = by_id[row["case_id"]]
        entries.append({"case_id": item.case_id, "response_sha256": row["response_sha256"],
                        "fact_scores": {fact.fact_id: None for fact in item.required_facts},
                        "notes": "Оцени каждый факт по шкале 0/1/2 после чтения ответа"})
    path = folder / "grades_template.jsonl"
    if path.exists():
        raise BaselineError("Grades template already exists; do not overwrite expert work")
    _write_atomic(path, b"".join(_line(x) for x in entries))
    return path


def score_grades(config: BaselineConfig, *, root: Path, grades_file: Path) -> dict[str, Any]:
    """Агрегировать только завершённую человеческую оценку F4.3 (не LLM-as-judge)."""
    manifest = verify(config, root=root, profile="full")
    cfg = load_eval_config(config.evaluation_config_file, root=root)
    cases = load_cases(cfg.source_file)
    folder = config.output_dir / "full"
    responses = [_json(line, "responses") for line in _read(folder / "responses.jsonl").splitlines()]
    hashes = {r["case_id"]: r["response_sha256"] for r in responses}
    entries = [_json(line, "grades") for line in _read(grades_file).splitlines()]
    if len(entries) != len(cases):
        raise BaselineError("Expected one human grade row for every evaluation case")
    reduced = []
    for row in entries:
        if not isinstance(row, dict) or set(row) != {"case_id", "response_sha256", "fact_scores", "notes"}:
            raise BaselineError("Grades: invalid fields")
        cid = row["case_id"]
        if cid not in hashes or row["response_sha256"] != hashes[cid]:
            raise BaselineError("Grading stale/foreign response")
        reduced.append({"case_id": cid, "fact_scores": row["fact_scores"], "notes": row["notes"]})
    try:
        scores = score_manual_grades(cases, reduced)
    except EvaluationError as exc:
        raise BaselineError(f"Invalid human annotations: {exc}") from exc
    payload = {
        "schema_version": 1, "model_id": config.model_id, "revision": config.model_revision,
        "evaluation_id": manifest["evaluation_id"], "responses_sha256": manifest["responses_sha256"],
        "grades_sha256": sha256(_read(grades_file)).hexdigest(), "human_scored": True,
        **scores,
    }
    result_path = folder / "manual_scores.json"
    if result_path.exists() and _read(result_path) != _json_pretty(payload):
        raise BaselineError("Manual score report already exists; preserve provenance and review changes")
    _write_atomic(result_path, _json_pretty(payload))
    return payload


class RealCPUGenerator:
    """Qwen Base на CPU, pinned revision, без chat template и без обучения."""

    def __init__(self, config: BaselineConfig, *, allow_download: bool = False) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise BaselineError("torch and transformers are required only for real inference") from exc
        self.torch = torch
        torch.manual_seed(config.seed)
        torch.set_num_threads(config.cpu_threads)
        torch.set_grad_enabled(False)
        kwargs = {"revision": config.model_revision, "local_files_only": not allow_download,
                  "trust_remote_code": False}
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                config.model_id, revision=config.tokenizer_revision, local_files_only=not allow_download,
                trust_remote_code=False, use_fast=True)
            # torch_dtype работает с Transformers 4.x; в 5.x параметр переименован в dtype.
            from importlib.metadata import version
            major = int(version("transformers").split(".")[0])
            dtype_kwarg = "dtype" if major >= 5 else "torch_dtype"
            model = AutoModelForCausalLM.from_pretrained(
                config.model_id, use_safetensors=True, **kwargs, **{dtype_kwarg: torch.float32})
            model = model.to("cpu")
            model.eval()
        except Exception as exc:
            raise BaselineError(f"Cannot load pinned Qwen Base model: {type(exc).__name__}: {exc}") from exc
        if tokenizer.eos_token_id is None or model.config.eos_token_id != tokenizer.eos_token_id:
            raise BaselineError("EOS token mismatch between model and tokenizer")
        if tokenizer.vocab_size > model.get_input_embeddings().num_embeddings:
            raise BaselineError("Tokenizer vocabulary exceeds model embedding")
        if model.config.model_type != "qwen2":
            raise BaselineError("Unexpected loaded architecture")
        self.tokenizer, self.model = tokenizer, model

    def generate(self, prompt: str, *, max_input_tokens: int, max_new_tokens: int) -> Generation:
        torch = self.torch
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        ids = inputs["input_ids"].to("cpu")
        attention_mask = inputs["attention_mask"].to("cpu")
        n = ids.shape[1]
        if n == 0 or n > max_input_tokens:
            raise BaselineError(f"Prompt length {n} outside 1..{max_input_tokens}; truncation forbidden")
        with torch.inference_mode():
            outputs = self.model.generate(
                input_ids=ids, attention_mask=attention_mask,
                max_new_tokens=max_new_tokens, do_sample=False, num_beams=1,
                use_cache=True, eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new_ids = outputs[0, n:].tolist()
        if not new_ids or len(new_ids) > max_new_tokens:
            raise BaselineError("Invalid generated token count")
        # Срез после prompt: модель не получает эталон и ответ не содержит текст вопроса.
        decoded = self.tokenizer.decode(new_ids, skip_special_tokens=True,
                                        clean_up_tokenization_spaces=False)
        return Generation(decoded, n, len(new_ids), new_ids[-1] == self.tokenizer.eos_token_id)


def real_runtime(config: BaselineConfig) -> dict[str, Any]:
    """Библиотечные версии фиксируются перед реальным запуском; импорта весов нет."""
    from importlib.metadata import PackageNotFoundError, version
    cat = load_catalog(config.model_catalog_file)
    report = runtime_report(cat.requirements)
    if report["issues"]:
        raise BaselineError(f"Runtime is incompatible: {report['issues']}")
    try:
        safetensors_version = version("safetensors")
    except PackageNotFoundError as exc:
        raise BaselineError("safetensors required for pinned checkpoint") from exc
    return {"kind": "real_cpu_transformers", "python": report["python"],
            "torch": report["packages"]["torch"], "transformers": report["packages"]["transformers"],
            "safetensors": safetensors_version, "platform": sys.platform}
