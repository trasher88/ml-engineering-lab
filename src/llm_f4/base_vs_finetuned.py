"""F4.7: честное парное сравнение сохранённого Qwen Base и Qwen+LoRA.

Без --run никаких весов и ML-импортов. Разрешена только подлинная генерация
для публичного --run; test doubles доступны внутри unit-тестов. Рубрика
подключается ПОСЛЕ генерации, а не в prompt.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Mapping, Sequence
import json
import math
import os
import re
import sys

from src.llm_f4.base_model_evaluation import (
    BaselineConfig, BaselineError, Generation, Prompt, RealCPUGenerator,
    _context_sha, _final_payload, _json, _json_pretty, _line, _read, _summary,
    _validate_result, _write_atomic, formatted_prompt, load_config as load_base,
    load_prompts, real_runtime as base_real_runtime, selection, verify as verify_base,
)
from src.llm_f4.evaluation_set import (
    EvaluationError, load_cases, load_config as load_eval, score_manual_grades,
)
from src.llm_f4.sft_lora import (
    Config as SFTConfig, SFTError, load_config as load_sft, verify_run,
)


class ComparisonError(ValueError):
    """Ошибка контракта, проверки целостности или процедуры сравнения."""


@dataclass(frozen=True, slots=True)
class Config:
    baseline_config_file: Path
    sft_config_file: Path
    output_dir: Path
    sft_profile: str
    sha256: str


CONFIG_KEYS = {"schema_version", "baseline_config_file", "sft_config_file", "sft_profile", "output_dir"}
RESPONSE_KEYS = {"case_id", "domain", "prompt_sha256", "base_response", "finetuned_response",
                 "base_response_sha256", "finetuned_response_sha256", "exact_text_match",
                 "base_generated_tokens", "finetuned_generated_tokens"}
GRADE_KEYS = {"case_id", "base_response_sha256", "finetuned_response_sha256",
              "base_fact_scores", "finetuned_fact_scores", "notes"}
HASH_RE = re.compile(r"[a-f0-9]{64}\Z")


def _decode(raw: bytes, label: str) -> Any:
    def distinct(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for k, v in pairs:
            if k in output:
                raise ComparisonError(f"Повторный ключ JSON {k}: {label}")
            output[k] = v
        return output

    def invalid_constant(v: str) -> None:
        raise ComparisonError(f"Недопустимая числовая константа {v}: {label}")

    try:
        return json.loads(raw.decode("utf-8-sig"), object_pairs_hook=distinct,
                          parse_constant=invalid_constant)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ComparisonError(f"Неверный JSON {label}: {exc}") from exc


def _rel_path(root: Path, value: Any, name: str) -> Path:
    if (not isinstance(value, str) or not value or ":" in value or "\\" in value
            or value.startswith("/") or any(part in ("", ".", "..") for part in value.split("/"))):
        raise ComparisonError(f"{name}: нужен относительный POSIX-путь без выхода из проекта")
    base = root.resolve()
    candidate = (base / value).resolve()
    if not candidate.is_relative_to(base):
        raise ComparisonError(f"{name}: путь выходит за пределы проекта")
    return candidate


def load_config(path: Path, *, root: Path) -> Config:
    raw = _read_safe(path)
    obj = _decode(raw, "F4.7 config")
    if not isinstance(obj, dict) or set(obj) != CONFIG_KEYS or type(obj["schema_version"]) is not int or obj["schema_version"] != 1:
        raise ComparisonError("Неверная структура F4.7 config")
    if obj["sft_profile"] not in ("cpu", "gpu"):
        raise ComparisonError("Только проверенные SFT профили cpu/gpu")
    conf = Config(
        _rel_path(root, obj["baseline_config_file"], "baseline_config_file"),
        _rel_path(root, obj["sft_config_file"], "sft_config_file"),
        _rel_path(root, obj["output_dir"], "output_dir"),
        obj["sft_profile"], sha256(raw).hexdigest(),
    )
    base = load_base(conf.baseline_config_file, root=root)
    sft = load_sft(conf.sft_config_file, root=root)
    if (sft.baseline_config_file != conf.baseline_config_file or
            (base.model_id, base.model_revision, base.tokenizer_revision) !=
            (sft.model_id, sft.model_revision, sft.tokenizer_revision)):
        raise ComparisonError("Несогласованная модель/ревизия F4.4 и F4.5")
    if conf.output_dir == base.output_dir or conf.output_dir == sft.output_dir:
        raise ComparisonError("F4.7 не должен перезаписывать F4.4/F4.5")
    return conf


def _read_safe(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except OSError as exc:
        raise ComparisonError(f"Нет файла {path}: {exc}") from exc


def _require_sources(conf: Config, *, root: Path, profile: str) -> tuple[BaselineConfig, SFTConfig, list[Prompt], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Строгая связка: frozen F4.3, действительный Base и реальный LoRA."""
    if profile not in ("smoke", "full"):
        raise ComparisonError("Неизвестный профиль")
    base = load_base(conf.baseline_config_file, root=root)
    sft = load_sft(conf.sft_config_file, root=root)
    prompts, eval_manifest = load_prompts(base, root=root)
    try:
        base_manifest = verify_base(base, root=root, profile=profile)
    except (BaselineError, OSError) as exc:
        raise ComparisonError(f"Нет проверенного F4.4 {profile}: запусти F4.4 --run {profile}, затем --verify {profile}. {exc}") from exc
    try:
        sft_manifest = verify_run(sft, root=root, profile=conf.sft_profile)
    except (SFTError, OSError) as exc:
        raise ComparisonError(f"Нет проверенного адаптера F4.5 {conf.sft_profile}: {exc}") from exc
    if (base_manifest["eval_prompts_sha256"] != eval_manifest["prompts_sha256"] or
            base_manifest["model_revision"] != sft_manifest["model_revision"] or
            base_manifest["model_id"] != sft_manifest["model_id"] or
            sft_manifest["holdout_sha256"] != eval_manifest["source_sha256"]):
        raise ComparisonError("Несовместимые provenance F4.3/F4.4/F4.5")
    return base, sft, selection(prompts, base, profile), eval_manifest, base_manifest, sft_manifest


def plan(conf: Config, *, root: Path) -> dict[str, Any]:
    """Подготовка без требований к наличию model weights и результатов inference."""
    base = load_base(conf.baseline_config_file, root=root)
    sft = load_sft(conf.sft_config_file, root=root)
    prompts, ev = load_prompts(base, root=root)
    return {"model": base.model_id, "revision": base.model_revision,
            "evaluation_id": ev["evaluation_id"], "evaluation_cases": len(prompts),
            "smoke_cases": len(selection(prompts, base, "smoke")),
            "sft_profile": conf.sft_profile, "config_sha256": conf.sha256,
            "adapter_directory": str(sft.output_dir / conf.sft_profile / "adapter"),
            "baseline_directory": str(base.output_dir),
            "generation_performed": False, "human_scored": False}


def _context(conf: Config, base: BaselineConfig, profile: str, ev: Mapping[str, Any],
             base_manifest: Mapping[str, Any], sft_manifest: Mapping[str, Any], runtime: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1, "stage": "f4.7", "profile": profile,
        "model_stage": "base_plus_lora", "model_id": base.model_id,
        "model_revision": base.model_revision, "tokenizer_revision": base.tokenizer_revision,
        "evaluation_id": ev["evaluation_id"], "eval_prompts_sha256": ev["prompts_sha256"],
        "eval_source_sha256": ev["source_sha256"],
        "config_sha256": conf.sha256, "baseline_config_sha256": base.sha256,
        "base_responses_sha256": base_manifest["responses_sha256"],
        "adapter_profile": conf.sft_profile,
        "adapter_config_sha256": sft_manifest["config_sha256"],
        "adapter_files_sha256": sft_manifest["adapter_files_sha256"],
        "adapter_metrics_sha256": sft_manifest["metrics_sha256"],
        "adapter_optimizer_steps": sft_manifest["optimizer_steps"],
        "device": "cpu", "dtype": "float32", "seed": base.seed,
        "decoding": "greedy", "do_sample": False, "num_beams": 1,
        "max_input_tokens": base.max_input_tokens, "max_new_tokens": base.max_new_tokens,
        "prompt_prefix": base.prefix, "prompt_suffix": base.suffix,
        "runtime": dict(runtime),
    }


def real_runtime(conf: Config, *, root: Path) -> dict[str, Any]:
    base = load_base(conf.baseline_config_file, root=root)
    info = real_runtime_base(base)
    try:
        peft_ver = version("peft")
    except PackageNotFoundError as exc:
        raise ComparisonError("Для реального сравнения требуется peft") from exc
    return {**info, "kind": "real_cpu_peft", "peft": peft_ver}


def real_runtime_base(base: BaselineConfig) -> dict[str, Any]:
    try:
        return base_real_runtime(base)
    except BaselineError as exc:
        raise ComparisonError(str(exc)) from exc


class RealAdapterGenerator(RealCPUGenerator):
    """Переиспользует идентичные encode/generate/decode параметры F4.4."""

    def __init__(self, base: BaselineConfig, adapter_dir: Path, *, allow_download: bool) -> None:
        super().__init__(base, allow_download=allow_download)
        try:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, str(adapter_dir),
                                                   is_trainable=False).to("cpu")
            self.model.eval()
            if not any("lora_" in name for name, _ in self.model.named_parameters()):
                raise ComparisonError("LoRA-модули не обнаружены")
        except Exception as exc:
            raise ComparisonError(f"Не удалось загрузить сохранённый PEFT адаптер: {type(exc).__name__}: {exc}") from exc


def _read_rows(path: Path, expected: Sequence[Prompt], base: BaselineConfig) -> list[dict[str, Any]]:
    raw = _read_safe(path)
    if raw and not raw.endswith(b"\n"):
        raise ComparisonError("Оборванный файл ответов JSONL")
    rows = [_decode(line, str(path)) for line in raw.splitlines()]
    if len(rows) != len(expected):
        raise ComparisonError("Неполный набор ответов")
    for record, item in zip(rows, expected, strict=True):
        _validate_result(record, item, base)
    return rows


def _progress(path: Path, expected: Sequence[Prompt], base: BaselineConfig) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    raw = _read_safe(path)
    if raw and not raw.endswith(b"\n"):
        raise ComparisonError("Повреждён незавершённый JSONL")
    rows = [_decode(x, "progress") for x in raw.splitlines()]
    if len(rows) > len(expected):
        raise ComparisonError("Слишком много ответов")
    for record, item in zip(rows, expected, strict=False):
        _validate_result(record, item, base)
    return rows


def _append(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as stream:
        stream.write(_line(row))
        stream.flush()
        os.fsync(stream.fileno())


def _runtime_is_real(runtime: Mapping[str, Any]) -> bool:
    return runtime.get("kind") == "real_cpu_peft" and isinstance(runtime.get("peft"), str)


def run(conf: Config, *, root: Path, profile: str, allow_download: bool = False,
        resume: bool = False) -> dict[str, Any]:
    runtime = real_runtime(conf, root=root)
    if not _runtime_is_real(runtime):
        raise ComparisonError("Нет подтверждённого runtime PEFT")
    return _execute(conf, root=root, profile=profile, runtime=runtime,
                    factory=lambda base, sft: RealAdapterGenerator(
                        base, sft.output_dir / conf.sft_profile / "adapter",
                        allow_download=allow_download), resume=resume)


def _execute(conf: Config, *, root: Path, profile: str, runtime: Mapping[str, Any],
             factory: Callable[[BaselineConfig, SFTConfig], Any], resume: bool = False) -> dict[str, Any]:
    """Внутренний тестируемый runner; реальные вызовы идут через run()."""
    base, sft, prompts, ev, base_m, sft_m = _require_sources(conf, root=root, profile=profile)
    context = _context(conf, base, profile, ev, base_m, sft_m, runtime)
    folder = conf.output_dir / profile
    context_file, journal = folder / "context.json", folder / "progress.jsonl"
    if resume:
        if _read_safe(context_file) != _json_pretty(context):
            raise ComparisonError("Resume заблокирован: изменены модель/конфигурация/адаптер/библиотеки")
        if (folder / "manifest.json").exists() or (folder / "responses.jsonl").exists():
            raise ComparisonError("Завершённый или неоднозначный запуск: проверь результаты, не перезаписывай")
        saved = _progress(journal, prompts, base)
    else:
        if folder.exists() and any(folder.iterdir()):
            raise ComparisonError("Каталог результата уже содержит файлы: перезапись запрещена")
        folder.mkdir(parents=True, exist_ok=True)
        _write_atomic(context_file, _json_pretty(context))
        saved = []
    if len(saved) < len(prompts):
        generator = factory(base, sft)
        for item in prompts[len(saved):]:
            prompt = formatted_prompt(item, base)
            start = perf_counter()
            output = generator.generate(prompt, max_input_tokens=base.max_input_tokens,
                                        max_new_tokens=base.max_new_tokens)
            if not isinstance(output, Generation):
                raise ComparisonError("Backend не вернул Generation")
            row = {"case_id": item.case_id, "domain": item.domain,
                   "prompt_sha256": sha256(prompt.encode("utf-8")).hexdigest(),
                   "response": output.text,
                   "response_sha256": sha256(output.text.encode("utf-8")).hexdigest() if isinstance(output.text, str) else "",
                   "prompt_tokens": output.prompt_token_count,
                   "generated_tokens": output.generated_token_count,
                   "stopped_on_eos": output.stopped_on_eos,
                   "elapsed_seconds": round(perf_counter() - start, 4),
                   "model_generated": True}
            _validate_result(row, item, base)
            _append(journal, row)
            saved.append(row)
    response_bytes, manifest_bytes = _final_payload(saved, context, len(prompts))
    _write_atomic(folder / "responses.jsonl", response_bytes)
    _write_atomic(folder / "manifest.json", manifest_bytes)
    return _decode(manifest_bytes, "F4.7 manifest")


def verify(conf: Config, *, root: Path, profile: str) -> dict[str, Any]:
    base, _, prompts, ev, base_m, sft_m = _require_sources(conf, root=root, profile=profile)
    folder = conf.output_dir / profile
    context = _decode(_read_safe(folder / "context.json"), "F4.7 context")
    manifest = _decode(_read_safe(folder / "manifest.json"), "F4.7 manifest")
    if not isinstance(context, dict) or not isinstance(manifest, dict) or not _runtime_is_real(context.get("runtime", {})):
        raise ComparisonError("Нет действительного real_cpu_peft контекста")
    expected_context = _context(conf, base, profile, ev, base_m, sft_m, context["runtime"])
    if context != expected_context or _context_sha(context) != manifest.get("context_sha256"):
        raise ComparisonError("Контекст inference не соответствует F4.2/F4.3/F4.4/F4.5")
    rows = _read_rows(folder / "responses.jsonl", prompts, base)
    raw_response, raw_manifest = _final_payload(rows, context, len(prompts))
    if (_read_safe(folder / "progress.jsonl") != raw_response or
            _read_safe(folder / "responses.jsonl") != raw_response or
            _read_safe(folder / "manifest.json") != raw_manifest):
        raise ComparisonError("Ответы, progress или manifest изменены")
    if manifest.get("run_complete") is not True or manifest.get("generation_performed") is not True:
        raise ComparisonError("Генерация не завершена")
    return manifest


def paired_rows(conf: Config, *, root: Path, profile: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Ни эталоны, ни рубрики не читаются для сравнительной таблицы."""
    base, _, prompts, ev, base_m, sft_m = _require_sources(conf, root=root, profile=profile)
    adapted = verify(conf, root=root, profile=profile)
    base_rows = _read_rows(base.output_dir / profile / "responses.jsonl", prompts, base)
    ft_rows = _read_rows(conf.output_dir / profile / "responses.jsonl", prompts, base)
    paired = []
    for item, original, tuned in zip(prompts, base_rows, ft_rows, strict=True):
        if original["prompt_sha256"] != tuned["prompt_sha256"]:
            raise ComparisonError(f"Промпты отличаются: {item.case_id}")
        row = {"case_id": item.case_id, "domain": item.domain,
               "prompt_sha256": original["prompt_sha256"],
               "base_response": original["response"], "finetuned_response": tuned["response"],
               "base_response_sha256": original["response_sha256"],
               "finetuned_response_sha256": tuned["response_sha256"],
               "exact_text_match": original["response"] == tuned["response"],
               "base_generated_tokens": original["generated_tokens"],
               "finetuned_generated_tokens": tuned["generated_tokens"]}
        paired.append(row)
    source = {"schema_version": 1, "stage": "F4.7", "profile": profile,
              "evaluation_id": ev["evaluation_id"], "cases": len(paired),
              "base_responses_sha256": base_m["responses_sha256"],
              "finetuned_responses_sha256": adapted["responses_sha256"],
              "adapter_optimizer_steps": sft_m["optimizer_steps"],
              "unchanged_text_count": sum(r["exact_text_match"] for r in paired),
              "changed_text_count": sum(not r["exact_text_match"] for r in paired),
              "human_scoring_performed": False,
              "quality_conclusion": "NOT_GRADED; text changes are not quality gains"}
    return paired, source


def compare(conf: Config, *, root: Path, profile: str) -> dict[str, Any]:
    rows, summary = paired_rows(conf, root=root, profile=profile)
    folder = conf.output_dir / profile
    dest_rows, dest_summary = folder / "paired_responses.jsonl", folder / "comparison.json"
    if dest_rows.exists() or dest_summary.exists():
        raise ComparisonError("Парное сравнение уже существует; применяй --verify")
    data = b"".join(_line(r) for r in rows)
    summary["paired_sha256"] = sha256(data).hexdigest()
    _write_atomic(dest_rows, data)
    _write_atomic(dest_summary, _json_pretty(summary))
    return summary


def verify_comparison(conf: Config, *, root: Path, profile: str) -> dict[str, Any]:
    rows, expected = paired_rows(conf, root=root, profile=profile)
    data = b"".join(_line(r) for r in rows)
    expected["paired_sha256"] = sha256(data).hexdigest()
    folder = conf.output_dir / profile
    if (_read_safe(folder / "paired_responses.jsonl") != data or
            _read_safe(folder / "comparison.json") != _json_pretty(expected)):
        raise ComparisonError("Парная таблица и SHA не соответствуют исходным ответам")
    return expected


def grades_template(conf: Config, *, root: Path) -> tuple[Path, Path]:
    """Создать форму оценивания ПОСЛЕ inference на полном frozen benchmark."""
    verify_comparison(conf, root=root, profile="full")
    cfg = load_eval(load_base(conf.baseline_config_file, root=root).evaluation_config_file, root=root)
    cases = load_cases(cfg.source_file)
    pairs, _ = paired_rows(conf, root=root, profile="full")
    lookup = {c.case_id: c for c in cases}
    forms = []
    review = ["# F4.7 — экспертная оценка 32 ответов Base и LoRA", "", "Оценка фактов: 0 — неверно, 1 — частично, 2 — полностью. Редактируй только *_fact_scores и notes в JSONL.", ""]
    for p in pairs:
        c = lookup[p["case_id"]]
        fact_ids = {f.fact_id: None for f in c.required_facts}
        forms.append({"case_id": p["case_id"], "base_response_sha256": p["base_response_sha256"],
                      "finetuned_response_sha256": p["finetuned_response_sha256"],
                      "base_fact_scores": fact_ids, "finetuned_fact_scores": dict(fact_ids),
                      "notes": "Оценить оба ответа по одной рубрике"})
        review.extend([f"## {c.case_id} — {c.domain}", "", "**Вопрос:**", c.instruction,
                       "", "**Данные:**", c.input or "(нет)", "", "**Эталон:**", c.reference_answer,
                       "", "**Факты для проверки:**"])
        review.extend([f"- `{f.fact_id}`: {f.description}" for f in c.required_facts])
        review.extend(["", "**Base:**", "", p["base_response"] or "(пусто)", "", "**LoRA:**", "",
                       p["finetuned_response"] or "(пусто)", "", "---", ""])
    folder = conf.output_dir / "full"
    grade_path, review_path = folder / "grades_template.jsonl", folder / "human_review.md"
    if grade_path.exists() or review_path.exists():
        raise ComparisonError("Форма экспертной оценки уже существует; не перезаписываем")
    _write_atomic(grade_path, b"".join(_line(f) for f in forms))
    _write_atomic(review_path, ("\n".join(review).rstrip() + "\n").encode("utf-8"))
    return grade_path, review_path


def _validated_grades(rows: Sequence[Mapping[str, Any]], pairs: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if len(rows) != len(pairs):
        raise ComparisonError("Нужны экспертные оценки всех 32 случаев")
    by_id = {r["case_id"]: r for r in pairs}
    seen: set[str] = set()
    base_grades, ft_grades = [], []
    for g in rows:
        if not isinstance(g, dict) or set(g) != GRADE_KEYS:
            raise ComparisonError("Неверная форма оценки")
        cid = g["case_id"]
        if not isinstance(cid, str) or cid not in by_id or cid in seen:
            raise ComparisonError("Повтор или неизвестный case_id")
        seen.add(cid)
        p = by_id[cid]
        if g["base_response_sha256"] != p["base_response_sha256"] or g["finetuned_response_sha256"] != p["finetuned_response_sha256"]:
            raise ComparisonError("Оценки привязаны к другим ответам")
        for key in ("base_fact_scores", "finetuned_fact_scores"):
            scores = g[key]
            if not isinstance(scores, dict) or any(type(x) is not int or x not in (0, 1, 2) for x in scores.values()):
                raise ComparisonError(f"Не выставлены оценки 0/1/2: {cid}")
        if not isinstance(g["notes"], str) or not g["notes"].strip() or len(g["notes"]) > 500:
            raise ComparisonError("Необходим текстовый комментарий эксперта")
        base_grades.append({"case_id": cid, "fact_scores": g["base_fact_scores"], "notes": g["notes"]})
        ft_grades.append({"case_id": cid, "fact_scores": g["finetuned_fact_scores"], "notes": g["notes"]})
    return base_grades, ft_grades


def score_grades(conf: Config, *, root: Path, grades_file: Path) -> dict[str, Any]:
    source = verify_comparison(conf, root=root, profile="full")
    base = load_base(conf.baseline_config_file, root=root)
    cfg = load_eval(base.evaluation_config_file, root=root)
    cases = load_cases(cfg.source_file)
    pairs, _ = paired_rows(conf, root=root, profile="full")
    raw = _read_safe(grades_file)
    if not raw or not raw.endswith(b"\n"):
        raise ComparisonError("Экспертный JSONL пуст или оборван")
    entries = [_decode(x, "expert grades") for x in raw.splitlines()]
    bg, fg = _validated_grades(entries, pairs)
    try:
        bs = score_manual_grades(cases, bg)
        fs = score_manual_grades(cases, fg)
    except EvaluationError as exc:
        raise ComparisonError(f"Ошибка рубрики: {exc}") from exc
    diffs = {cid: fs["per_case"][cid]["score"] - bs["per_case"][cid]["score"] for cid in bs["per_case"]}
    wins = sum(x > 0 for x in diffs.values())
    losses = sum(x < 0 for x in diffs.values())
    ties = sum(x == 0 for x in diffs.values())
    result = {"schema_version": 1, "stage": "F4.7", "evaluation_id": source["evaluation_id"],
              "cases": len(cases), "base_macro": bs["macro_over_cases"],
              "finetuned_macro": fs["macro_over_cases"],
              "delta_macro": fs["macro_over_cases"] - bs["macro_over_cases"],
              "base_by_domain": bs["macro_by_domain"],
              "finetuned_by_domain": fs["macro_by_domain"],
              "delta_by_domain": {d: fs["macro_by_domain"][d] - bs["macro_by_domain"][d] for d in bs["macro_by_domain"]},
              "paired_wins": wins, "paired_losses": losses, "paired_ties": ties,
              "per_case_delta": dict(sorted(diffs.items())),
              "human_scoring_performed": True, "quality_status": "HUMAN_GRADED_SMALL_BENCHMARK",
              "adapter_optimizer_steps": source["adapter_optimizer_steps"],
              "base_responses_sha256": source["base_responses_sha256"],
              "finetuned_responses_sha256": source["finetuned_responses_sha256"],
              "grades_sha256": sha256(raw).hexdigest()}
    path = conf.output_dir / "full" / "manual_comparison.json"
    if path.exists() and _read_safe(path) != _json_pretty(result):
        raise ComparisonError("Уже существует другой результат ручной оценки; не перезаписываем")
    _write_atomic(path, _json_pretty(result))
    return result
