"""F4.9 — воспроизводимый итоговый отчёт по *проверенным* F4.1–F4.8.

Команды --plan, --build, --verify обходятся без ML-весов и imports torch/peft.
Счётчики тестов и GitHub CI не берутся из локальных артефактов и не
выдаются за машинно подтверждённые. Ручное качество F4.7 — опционально.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping

from src.llm_f4.instruction_dataset import (load_config as load_f41, verify_artifacts as verify_f41,
                                            prepare_artifacts as prepare_f41)
from src.llm_f4.model_selection import load_catalog, assess_candidate
from src.llm_f4.evaluation_set import (load_config as load_f43, load_cases,
                                        prepare_artifacts as prepare_f43, verify_artifacts as verify_f43,
                                        score_manual_grades)
from src.llm_f4.base_model_evaluation import load_config as load_f44, verify as verify_f44
from src.llm_f4.sft_lora import load_config as load_f45, verify_run as verify_f45
from src.llm_f4.training_visualization import load_config as load_f46, verify_report as verify_f46
from src.llm_f4.base_vs_finetuned import (load_config as load_f47,
                                         verify_comparison as verify_f47,
                                         paired_rows, _validated_grades)
from src.llm_f4.dpo_experiment import load_config as load_f48, verify_run as verify_f48


class ReportError(ValueError):
    """Некорректные исходники, недостающие доказательства или повреждённый отчёт."""


PATH_KEYS = (
    "instruction_config_file", "model_catalog_file", "evaluation_config_file",
    "baseline_config_file", "sft_config_file", "visualization_config_file",
    "comparison_config_file", "dpo_config_file",
)
EXPECTED_PATHS = (
    "configs/f4_01_instruction_dataset.json", "configs/f4_02_model_selection.json",
    "configs/f4_03_evaluation_set.json", "configs/f4_04_base_model_evaluation.json",
    "configs/f4_05_sft_lora.json", "configs/f4_06_training_visualization.json",
    "configs/f4_07_base_vs_finetuned.json", "configs/f4_08_dpo_experiment.json",
)
OUTPUT = "data/processed/f4_09"
GENERATED = ("final_report.md", "evidence_summary.json")


@dataclass(frozen=True, slots=True)
class Config:
    root: Path
    output_dir: Path
    profile: str
    dependencies: tuple[Path, ...]
    config_sha256: str


def _no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, value in pairs:
        if name in out:
            raise ReportError(f"Повторяющийся ключ JSON: {name}")
        out[name] = value
    return out


def _bad_constant(value: str) -> None:
    raise ReportError(f"Неконечная JSON-константа: {value}")


def _parse(payload: bytes, description: str) -> Any:
    try:
        return json.loads(payload.decode("utf-8"), object_pairs_hook=_no_duplicates,
                          parse_constant=_bad_constant)
    except (ValueError, UnicodeError) as exc:
        raise ReportError(f"Повреждён JSON {description}: {exc}") from exc


def _read(path: Path) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ReportError(f"Нет обычного файла: {path}")
    return path.read_bytes()


def _digest(path: Path) -> str:
    return sha256(_read(path)).hexdigest()


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                       allow_nan=False) + "\n").encode("utf-8")


def _pretty(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                       allow_nan=False) + "\n").encode("utf-8")


def load_config(path: Path, *, root: Path) -> Config:
    """Строгий immutable-контракт путей, запрет traversal и выхода за корень."""
    root = root.resolve()
    data = _parse(_read(path), "F4.9 config")
    keys = {"schema_version", "output_dir", "profile", *PATH_KEYS}
    if not isinstance(data, dict) or set(data) != keys or type(data["schema_version"]) is not int or data["schema_version"] != 1:
        raise ReportError("F4.9: неизвестные поля или schema_version")
    if data["profile"] != "cpu" or data["output_dir"] != OUTPUT:
        raise ReportError("F4.9 поддерживает только зафиксированный CPU-профиль и каталог")
    if any(data[key] != expected for key, expected in zip(PATH_KEYS, EXPECTED_PATHS, strict=True)):
        raise ReportError("Путь зависимостей F4.1–F4.8 изменён")
    deps = tuple(root / value for value in EXPECTED_PATHS)
    for dep in deps:
        if not dep.is_relative_to(root) or dep.is_symlink() or dep.resolve() != dep:
            raise ReportError("Некорректный путь зависимости")
    output = root / OUTPUT
    if output.is_symlink() or output.resolve() != output:
        raise ReportError("Каталог результата выходит за корень")
    return Config(root, output, "cpu", deps, sha256(_read(path)).hexdigest())


def _sources(conf: Config) -> dict[str, str]:
    """Фингерпринты всех контролирующих конфигураций (без артефактов моделей)."""
    return {str(path.relative_to(conf.root).as_posix()): _digest(path) for path in conf.dependencies}


def plan(conf: Config) -> dict[str, Any]:
    """Проверить базовые метаданные и узнать наличие артефактов; без весов."""
    split = load_f41(conf.dependencies[0], root=conf.root)
    source_files = prepare_f41(split)
    catalog = load_catalog(conf.dependencies[1])
    candidates = [x for x in catalog.candidates if x.model_id == catalog.selected_model_id]
    if len(candidates) != 1:
        raise ReportError("Выбранная модель не найдена в каталоге")
    evaluation = load_f43(conf.dependencies[2], root=conf.root)
    eval_files = prepare_f43(evaluation)
    cases = load_cases(evaluation.source_file)
    paths = {
        "f4.4_full": conf.root / "data/processed/f4_04/full/manifest.json",
        "f4.5_cpu": conf.root / "data/processed/f4_05/cpu/run_manifest.json",
        "f4.6_cpu": conf.root / "data/processed/f4_06/cpu/visualization_manifest.json",
        "f4.7_full": conf.root / "data/processed/f4_07/full/comparison.json",
        "f4.8_cpu": conf.root / "data/processed/f4_08/cpu/run_manifest.json",
        "manual_f4.7": conf.root / "data/processed/f4_07/full/manual_comparison.json",
    }
    return {
        "model_id": catalog.selected_model_id,
        "model_revision": candidates[0].revision,
        "instruction_records": _parse(source_files["manifest.json"], "F4.1 manifest")["total_records"],
        "evaluation_cases": len(cases),
        "rubric_facts": sum(len(x.required_facts) for x in cases),
        "verified_evaluation_export": bool(eval_files["manifest.json"]),
        "artifacts_present": {key: path.is_file() for key, path in paths.items()},
        "model_training_or_inference_performed_by_plan": False,
    }


def _validate_number(value: Any, name: str, minimum: float | None = None) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or (minimum is not None and value < minimum):
        raise ReportError(f"Недопустимое число {name}")
    return float(value)


def _finite_score(value: Any, name: str) -> float:
    return _validate_number(value, name, 0.0)


def _read_optional_human_score(conf: Config, cfg47: Any, compare: Mapping[str, Any]) -> dict[str, Any] | None:
    """Наличие шаблона с null не является экспертной оценкой.

    Оцениваем заполненный grades_filled.jsonl НЕ изменяя F4.7 файлы,
    перепроверяя вычисления по рубрике и соответствие manual_comparison.
    """
    folder = cfg47.output_dir / "full"
    scored = folder / "manual_comparison.json"
    grades = folder / "grades_filled.jsonl"
    if not scored.exists() and not grades.exists():
        return None
    if not scored.exists():
        return None  # шаблон может быть скопирован, но ещё не оценён
    if not scored.is_file() or not grades.is_file():
        raise ReportError("Не хватает manual_comparison.json или grades_filled.jsonl: оценка не доказана")
    record = _parse(_read(scored), "F4.7 manual comparison")
    raw = _read(grades)
    if not raw.endswith(b"\n"):
        raise ReportError("Оборванный файл оценок")
    rows = [_parse(line, "grades_filled.jsonl") for line in raw.splitlines()]
    pairs, _ = paired_rows(cfg47, root=conf.root, profile="full")
    base_grade, tuned_grade = _validated_grades(rows, pairs)
    eval_cfg = load_f43(conf.dependencies[2], root=conf.root)
    cases = load_cases(eval_cfg.source_file)
    base_score = score_manual_grades(cases, base_grade)
    fine_score = score_manual_grades(cases, tuned_grade)
    diff = fine_score["macro_over_cases"] - base_score["macro_over_cases"]
    vals = [fine_score["per_case"][c.case_id]["score"] - base_score["per_case"][c.case_id]["score"] for c in cases]
    fields = {
        "human_scoring_performed": True,
        "quality_status": "HUMAN_GRADED_SMALL_BENCHMARK",
        "cases": len(cases), "base_macro": base_score["macro_over_cases"],
        "finetuned_macro": fine_score["macro_over_cases"], "delta_macro": diff,
        "paired_wins": sum(v > 0 for v in vals), "paired_losses": sum(v < 0 for v in vals),
        "paired_ties": sum(v == 0 for v in vals),
        "grades_sha256": sha256(raw).hexdigest(),
        "base_responses_sha256": compare["base_responses_sha256"],
        "finetuned_responses_sha256": compare["finetuned_responses_sha256"],
    }
    if not isinstance(record, dict) or any(record.get(k) != v for k, v in fields.items()):
        raise ReportError("Итог экспертной оценки не совпадает с сырыми баллами")
    return {"base_macro": fields["base_macro"], "finetuned_macro": fields["finetuned_macro"],
            "delta_macro": fields["delta_macro"], "paired_wins": fields["paired_wins"],
            "paired_losses": fields["paired_losses"], "paired_ties": fields["paired_ties"],
            "grades_sha256": fields["grades_sha256"]}


def collect_verified(conf: Config) -> dict[str, Any]:
    """Собрать только подлинные verify-результаты всех зависимых фаз."""
    p = plan(conf)
    f41 = load_f41(conf.dependencies[0], root=conf.root)
    split_files = verify_f41(f41)
    split_manifest = _parse(split_files["manifest.json"], "F4.1 manifest")
    f43 = load_f43(conf.dependencies[2], root=conf.root)
    f43_files = verify_f43(f43)
    eval_manifest = _parse(f43_files["manifest.json"], "F4.3 manifest")
    f44 = load_f44(conf.dependencies[3], root=conf.root)
    base = verify_f44(f44, root=conf.root, profile="full")
    f45 = load_f45(conf.dependencies[4], root=conf.root)
    sft = verify_f45(f45, root=conf.root, profile=conf.profile)
    f46 = load_f46(conf.dependencies[5], root=conf.root)
    viz = verify_f46(f46, root=conf.root, profile=conf.profile)
    f47 = load_f47(conf.dependencies[6], root=conf.root)
    comparison = verify_f47(f47, root=conf.root, profile="full")
    f48 = load_f48(conf.dependencies[7], root=conf.root)
    dpo = verify_f48(f48, root=conf.root, profile=conf.profile)
    catalog = load_catalog(conf.dependencies[1])
    spec = next(x for x in catalog.candidates if x.model_id == catalog.selected_model_id)
    candidate = assess_candidate(spec, catalog.requirements)
    if not candidate.get("eligible") or base["model_id"] != spec.model_id or base["model_revision"] != spec.revision:
        raise ReportError("Модель/ревизия Base не согласована с F4.2")
    if sft.get("real_training_performed") is not True or sft.get("optimizer_steps") != viz.get("optimizer_steps"):
        raise ReportError("Не подтверждено реальное обучение F4.5")
    if comparison.get("cases") != len(load_cases(f43.source_file)) or comparison.get("base_responses_sha256") != base["responses_sha256"]:
        raise ReportError("F4.7 не совпадает с исходным F4.4/F4.3")
    if comparison.get("adapter_optimizer_steps") != sft["optimizer_steps"]:
        raise ReportError("F4.7 использует другой SFT-адаптер")
    if dpo.get("real_dpo_performed") is not True or dpo.get("quality_improvement_proven") is not False:
        raise ReportError("Непроверенный DPO или необоснованное заявление об улучшении")
    if dpo.get("source_sft_manifest_sha256") != _digest(f45.output_dir / conf.profile / "run_manifest.json"):
        raise ReportError("DPO обучался не на зафиксированном SFT")
    if dpo.get("f43_used_for_training") is not False or dpo.get("f47_human_grades_used_for_training") is not False:
        raise ReportError("Обнаружено нарушение holdout в DPO")
    if viz.get("quality_improvement_measured") is not False or viz.get("benchmark_f4_03_used_in_training") is not False:
        raise ReportError("F4.6 недопустимо заявляет улучшение или использует holdout")
    human = _read_optional_human_score(conf, f47, comparison)
    summary: dict[str, Any] = {
        "schema_version": 1, "stage": "F4.9", "profile": conf.profile,
        "model": {"id": spec.model_id, "revision": spec.revision,
                  "tokenizer_revision": spec.tokenizer_revision,
                  "architecture": spec.architecture, "license": spec.license,
                  "estimated_parameters": candidate["resources"]["estimated_parameters"]},
        "instruction_dataset": {
            "records": split_manifest["total_records"], "groups": split_manifest["total_groups"],
            "splits": {k: {"records": v["records"], "groups": v["groups"]} for k, v in split_manifest["splits"].items()}},
        "evaluation": {"cases": eval_manifest["total_cases"],
                       "fact_count": p["rubric_facts"], "evaluation_id": eval_manifest["evaluation_id"]},
        "base": {"responses": base["completed_cases"], "generation_performed": base["generation_performed"],
                 "generation_metrics": base["metrics"]},
        "sft": {"optimizer_steps": sft["optimizer_steps"],
                "train_records_used": sft["train_records_used"],
                "validation_records_used": sft["validation_records_used"],
                "trainable_parameters": sft["trainable_parameters"],
                "all_parameters_with_adapters": sft["all_parameters_with_adapters"],
                "learning_rate_configured": viz["learning_rate_configured"],
                "train_loss": viz["train_loss_last"], "grad_norm_preclipping": viz["grad_norm_max_pre_clipping"],
                "validation_loss_before": viz["validation_loss_before"],
                "validation_loss_after": viz["validation_loss_after"],
                "validation_loss_delta": viz["validation_loss_delta"],
                "validation_loss_relative_pct": viz["validation_loss_relative_pct"]},
        "comparison": {"cases": comparison["cases"],
                       "changed_text_count": comparison["changed_text_count"],
                       "unchanged_text_count": comparison["unchanged_text_count"],
                       "human_grades": human},
        "dpo": {"optimizer_steps": dpo["optimizer_steps"],
                "train_pairs_used": dpo["train_pairs_used"],
                "validation_pairs_used": dpo["validation_pairs_used"],
                "beta": dpo["beta"],
                "initial_validation_margin": dpo["initial_validation_margin"],
                "final_validation_margin": dpo["final_validation_margin"],
                "quality_improvement_proven": False},
        "limitations": {"small_manual_benchmark": True,
                        "sft_single_step": sft["optimizer_steps"] == 1,
                        "dpo_single_step": dpo["optimizer_steps"] == 1,
                        "no_human_grades": human is None,
                        "gpu_training_not_verified_by_this_report": True,
                        "no_quality_claim_without_grades": human is None},
    }
    # Отдельные source fingerprint позволяют обнаружить изменения внешних
    # артефактов даже если визуальные числа остались прежними.
    manifest_files = {
        "f4.1": f41.output_dir / "manifest.json", "f4.3": f43.output_dir / "manifest.json",
        "f4.4": f44.output_dir / "full/manifest.json",
        "f4.5": f45.output_dir / "cpu/run_manifest.json",
        "f4.6": f46.output_dir / "cpu/visualization_manifest.json",
        "f4.7": f47.output_dir / "full/comparison.json",
        "f4.8": f48.output_dir / "cpu/run_manifest.json",
    }
    fingerprints = _sources(conf)
    for label, path in manifest_files.items():
        fingerprints[str(path.relative_to(conf.root).as_posix())] = _digest(path)
    if human is not None:
        fingerprints[str((f47.output_dir / "full/manual_comparison.json").relative_to(conf.root))] = _digest(f47.output_dir / "full/manual_comparison.json")
        fingerprints[str((f47.output_dir / "full/grades_filled.jsonl").relative_to(conf.root))] = _digest(f47.output_dir / "full/grades_filled.jsonl")
    return {"summary": summary, "source_fingerprints": dict(sorted(fingerprints.items()))}


def render_markdown(data: Mapping[str, Any]) -> bytes:
    """Только факты из проверенных источников, без оценок из шаблона с null."""
    m = data["model"]; ds = data["instruction_dataset"]; ev = data["evaluation"]
    base = data["base"]; sft = data["sft"]; cp = data["comparison"]; dpo = data["dpo"]
    for key in ("train_loss", "validation_loss_before", "validation_loss_after", "validation_loss_delta", "grad_norm_preclipping"):
        _validate_number(sft[key], key)
    for key in ("initial_validation_margin", "final_validation_margin"):
        _validate_number(dpo[key], key)
    for name in ("optimizer_steps", "train_records_used", "validation_records_used", "trainable_parameters", "all_parameters_with_adapters"):
        if type(sft[name]) is not int or sft[name] < 1:
            raise ReportError(f"Неверное целое SFT: {name}")
    if cp["cases"] != ev["cases"] or cp["changed_text_count"] + cp["unchanged_text_count"] != cp["cases"]:
        raise ReportError("Несогласованы результаты парного сравнения")
    if ds["records"] != sum(row["records"] for row in ds["splits"].values()):
        raise ReportError("Несогласованы размеры instruction split")
    if not all(isinstance(m[key], str) and re.fullmatch(r"[a-zA-Z0-9_.\-/]+", m[key]) for key in ("id", "revision", "tokenizer_revision", "architecture", "license")):
        raise ReportError("Недопустимые метаданные модели")
    human = cp["human_grades"]
    if human is None:
        quality = ("**Экспертная оценка качества пока не выполнена.** Изменение текста не равно "
                   "улучшению: баллы Base и Fine-Tuned не приведены, поскольку заполненных оценок нет.")
    else:
        for k in ("base_macro", "finetuned_macro", "delta_macro"):
            _validate_number(human[k], k)
        quality = (f"Экспертная оценка по 32 заданиям: Base **{human['base_macro']:.4f}**, "
                   f"LoRA **{human['finetuned_macro']:.4f}**, разница "
                   f"**{human['delta_macro']:+.4f}**. На отдельных заданиях: "
                   f"улучшений {human['paired_wins']}, ухудшений {human['paired_losses']}, "
                   f"совпадений {human['paired_ties']}. Малый benchmark не даёт широких выводов.")
    frac = sft["trainable_parameters"] / sft["all_parameters_with_adapters"] * 100
    lines = [
        "# ML Engineering Lab — Фаза 4: итоговый отчёт (F4.9)", "",
        "> Основан только на проверенных локальных артефактах F4.1–F4.8. "
        "Модель и обучение не запускаются при формировании этого файла.", "",
        "## 1. Резюме", "",
        f"На зафиксированной **{m['id']}** построен учебный цикл instruction dataset → "
        "baseline → SFT + LoRA → визуализация → парное сравнение → DPO. "
        f"Выполнено **{sft['optimizer_steps']}** SFT-шаг(ов) и **{dpo['optimizer_steps']}** "
        "DPO-шаг(ов) на CPU. Это подтверждает инженерную работоспособность, "
        "но само по себе не доказывает улучшение ответов.", "",
        "## 2. Карта реализованных этапов", "",
        "| Этап | Проверяемый результат |", "|---|---|",
        f"| F4.1 | {ds['records']} инструкций, {ds['groups']} тематических групп; train/validation/test |",
        f"| F4.2 | {m['id']}, точная ревизия модели и токенизатора |",
        f"| F4.3 | {ev['cases']} контрольных заданий, {ev['fact_count']} факта рубрики |",
        f"| F4.4 | {base['responses']} сохранённых Base-ответов (real inference) |",
        f"| F4.5 | {sft['optimizer_steps']} real SFT/LoRA optimizer step(ов) |",
        "| F4.6 | Проверенные метрики + офлайн HTML/SVG |",
        f"| F4.7 | {cp['cases']} пар Base/LoRA, "
        f"{cp['changed_text_count']} изменённых текстов |",
        f"| F4.8 | {dpo['optimizer_steps']} real DPO optimizer step(ов), frozen reference |", "",
        "## 3. Базовая модель и ресурсы", "",
        f"- Checkpoint: `{m['id']}`; revision: `{m['revision']}`.",
        f"- Tokenizer revision: `{m['tokenizer_revision']}`.",
        f"- Архитектура: `{m['architecture']}`; лицензия: `{m['license']}`.",
        f"- Оценка общего числа исходных параметров по архитектуре: {m['estimated_parameters']:,}.",
        "- Реальные эксперименты в этом отчёте: CPU, не много-GPU fine-tuning.", "",
        "## 4. Данные, разделение и отсутствие известных пересечений", "",
        f"Датасет F4.1: {ds['records']} записей в {ds['groups']} группах; "
        + ", ".join(f"{name} {row['records']} ({row['groups']} групп)" for name, row in ds["splits"].items()) + ".",
        f"Независимый F4.3: {ev['cases']} задания, {ev['fact_count']} проверяемых факта. "
        "Верификаторы проверяют целостность, формальные дубликаты и "
        "заявленные инварианты разделения; это не абсолютная гарантия отсутствия "
        "любого семантического пересечения.", "",
        "## 5. Исходный baseline F4.4", "",
        f"Base-модель сгенерировала **{base['responses']}** ответов на зафиксированном F4.3. "
        "Декодирование использовало режим, зафиксированный в F4.4; ответов "
        "с человеческими оценками качества у baseline в этом отчёте не предполагается.", "",
        "## 6. Настоящий SFT + LoRA (F4.5–F4.6)", "",
        f"- Шаги оптимизатора: **{sft['optimizer_steps']}**; обучающих записей: "
        f"**{sft['train_records_used']}**; validation-записей: **{sft['validation_records_used']}**.",
        f"- Обучаемых LoRA-параметров: **{sft['trainable_parameters']:,}** из "
        f"{sft['all_parameters_with_adapters']:,} (**{frac:.4f}%**).",
        f"- Training loss сохранённого шага: **{sft['train_loss']:.6f}**; "
        f"norm градиентов до clipping: **{sft['grad_norm_preclipping']:.6f}**.",
        f"- Validation loss до: **{sft['validation_loss_before']:.6f}**, "
        f"после: **{sft['validation_loss_after']:.6f}**, "
        f"Δ: **{sft['validation_loss_delta']:+.6f}** "
        f"({sft['validation_loss_relative_pct']:+.4f}%).",
        f"- Learning rate **{sft['learning_rate_configured']:.6g}** указан в конфигурации, "
        "но поминутный/пошаговый LR не измерялся.",
        "- При одном шаге уменьшение validation loss на одной записи нельзя "
        "интерпретировать как устойчивый тренд или улучшение качества ответов.", "",
        "## 7. Base vs Fine-Tuned (F4.7)", "",
        f"Пары проверены для **{cp['cases']}** идентичных вопросов: "
        f"текст изменился в **{cp['changed_text_count']}** ответах и совпал "
        f"в **{cp['unchanged_text_count']}**.", "",
        quality, "",
        "## 8. DPO Experiment (F4.8)", "",
        f"DPO после SFT: **{dpo['optimizer_steps']}** реальных optimizer step(ов), "
        f"{dpo['train_pairs_used']} train-пар, {dpo['validation_pairs_used']} validation-пар; "
        f"β = **{dpo['beta']}**. Frozen reference — состояние SFT до DPO.",
        f"Validation preference margin: **{dpo['initial_validation_margin']:.6f} → "
        f"{dpo['final_validation_margin']:.6f}**. Изменение margin на "
        "малом объёме не доказывает улучшение технических ответов. "
        "DPO-адаптер не оценивался здесь по 32 контрольным вопросам.", "",
        "## 9. Доказано и не доказано", "",
        "**Подтверждено:** корректная структура и проверки исходных данных; "
        "фиксированная модель; реальный inference Base/LoRA; фактический "
        "LoRA optimizer step; реальные визуализированные метрики; "
        "проверенное попарное сопоставление; реальный DPO и неизменная reference.", "",
        "**Не доказано:** статистически значимое улучшение качества, устойчивые "
        "кривые обучения по одному шагу, эффект на новых доменах, "
        "преимущество DPO относительно SFT, эффективность GPU/QLoRA. "
        "Тесты и подтверждения CI являются отдельным свидетельством: данный "
        "отчёт не выводит их статусы из хешей артефактов.", "",
        "## 10. Воспроизводимость и дальнейшая работа", "",
        "- Проверенные числовые свидетельства: `evidence_summary.json`.",
        "- Хеши исходников и файлов отчёта: `report_manifest.json`.",
        "- Воспроизводимая проверка: "
        "`python scripts/f4_final_phase_report.py --verify cpu`.",
        "- Для содержательного качества завершить слепую экспертную оценку "
        "F4.7 и провести более продолжительное SFT, удерживая F4.3 вне обучения.",
        "- При появлении новых экспериментов формировать **новую** версию отчёта "
        "в отдельном каталоге; не перезаписывать старый отчёт незаметно.", "",
    ]
    return ("\n".join(lines) + "\n").encode("utf-8")


def render_files(collected: Mapping[str, Any]) -> dict[str, bytes]:
    summary = collected["summary"]
    return {"final_report.md": render_markdown(summary),
            "evidence_summary.json": _pretty(summary)}


def build_report(conf: Config) -> dict[str, Any]:
    """Атомарная неразрушающая запись двух отчётов и проверочного manifest."""
    result = collect_verified(conf)
    content = render_files(result)
    output = conf.output_dir / conf.profile
    if output.exists() or output.is_symlink():
        raise ReportError(f"Каталог уже существует; не перезаписываем: {output}")
    manifest = {"schema_version": 1, "stage": "F4.9", "profile": conf.profile,
                "config_sha256": conf.config_sha256,
                "source_fingerprints": result["source_fingerprints"],
                "generated_files_sha256": {name: sha256(body).hexdigest() for name, body in content.items()}}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".f49_", dir=output.parent) as tmp:
        directory = Path(tmp)
        for name, body in content.items():
            (directory / name).write_bytes(body)
        (directory / "report_manifest.json").write_bytes(_pretty(manifest))
        if output.exists() or output.is_symlink():
            raise ReportError("Каталог появился во время построения")
        directory.rename(output)
    return result["summary"]


def verify_report(conf: Config) -> dict[str, Any]:
    """Полностью пересчитать отчёт из действительных артефактов и сравнить побайтово."""
    result = collect_verified(conf)
    directory = conf.output_dir / conf.profile
    manifest = _parse(_read(directory / "report_manifest.json"), "F4.9 manifest")
    expected = render_files(result)
    if (not isinstance(manifest, dict)
            or set(manifest) != {"schema_version", "stage", "profile", "config_sha256", "source_fingerprints", "generated_files_sha256"}
            or manifest["schema_version"] != 1 or type(manifest["schema_version"]) is not int
            or manifest["stage"] != "F4.9" or manifest["profile"] != conf.profile
            or manifest["config_sha256"] != conf.config_sha256
            or manifest["source_fingerprints"] != result["source_fingerprints"]
            or manifest["generated_files_sha256"] != {name: sha256(body).hexdigest() for name, body in expected.items()}):
        raise ReportError("Manifest F4.9 не соответствует текущим источникам")
    if {p.name for p in directory.iterdir()} != set(GENERATED) | {"report_manifest.json"}:
        raise ReportError("Файлы отчёта добавлены или отсутствуют")
    for name, body in expected.items():
        if _read(directory / name) != body:
            raise ReportError(f"Отчёт изменён или повреждён: {name}")
    if _read(directory / "report_manifest.json") != _pretty(manifest):
        raise ReportError("Изменён формат manifest F4.9")
    return result["summary"]
