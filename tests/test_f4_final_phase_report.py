"""F4.9: детерминизм, доверенные источники и отказ от выдуманных улучшений.

Числа фикстуры — ИСКУССТВЕННЫЕ для тестирования отчёта, не Qwen run.
"""
from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from src.llm_f4 import final_phase_report as mod

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "configs/f4_09_final_phase_report.json"
CLI = ROOT / "scripts/f4_final_phase_report.py"


@pytest.fixture(scope="module")
def conf():
    return mod.load_config(CFG, root=ROOT)


@pytest.fixture
def summary():
    # Только тестовый образец структуры проверенных метрик; цифры не являются
    # результатом реальной Qwen и никогда не попадают в поставляемый отчёт.
    return {
        "schema_version": 1, "stage": "F4.9", "profile": "cpu",
        "model": {"id": "Qwen/Qwen2.5-0.5B", "revision": "a" * 40,
                  "tokenizer_revision": "a" * 40, "architecture": "Qwen2ForCausalLM",
                  "license": "apache-2.0", "estimated_parameters": 494032768},
        "instruction_dataset": {"records": 80, "groups": 40,
                                "splits": {"train": {"records": 56, "groups": 28},
                                           "validation": {"records": 12, "groups": 6},
                                           "test": {"records": 12, "groups": 6}}},
        "evaluation": {"cases": 32, "fact_count": 64, "evaluation_id": "fixture_v1"},
        "base": {"responses": 32, "generation_performed": True, "generation_metrics": {}},
        "sft": {"optimizer_steps": 1, "train_records_used": 1,
                "validation_records_used": 1, "trainable_parameters": 540672,
                "all_parameters_with_adapters": 494573440,
                "learning_rate_configured": 0.0002, "train_loss": 2.4,
                "grad_norm_preclipping": 3.5,
                "validation_loss_before": 2.28, "validation_loss_after": 2.27,
                "validation_loss_delta": -0.01, "validation_loss_relative_pct": -0.4385964912},
        "comparison": {"cases": 32, "changed_text_count": 3,
                       "unchanged_text_count": 29, "human_grades": None},
        "dpo": {"optimizer_steps": 1, "train_pairs_used": 1,
                "validation_pairs_used": 1, "beta": 0.1,
                "initial_validation_margin": -0.15,
                "final_validation_margin": -0.10,
                "quality_improvement_proven": False},
        "limitations": {"small_manual_benchmark": True, "sft_single_step": True,
                        "dpo_single_step": True, "no_human_grades": True,
                        "gpu_training_not_verified_by_this_report": True,
                        "no_quality_claim_without_grades": True},
    }


@pytest.fixture
def fake_evidence(summary):
    return {"summary": summary, "source_fingerprints": {"config": "b" * 64}}


def test_frozen_config(conf):
    assert conf.profile == "cpu"
    assert len(conf.dependencies) == 8
    assert conf.output_dir == ROOT / "data/processed/f4_09"
    assert len(conf.config_sha256) == 64


def test_actual_metadata_plan_without_real_weights(conf):
    p = mod.plan(conf)
    assert p["instruction_records"] == 80
    assert p["evaluation_cases"] == 32
    assert p["rubric_facts"] == 64
    assert p["model_training_or_inference_performed_by_plan"] is False
    assert set(p["artifacts_present"]) == {"f4.4_full", "f4.5_cpu", "f4.6_cpu", "f4.7_full", "f4.8_cpu", "manual_f4.7"}


@pytest.mark.parametrize("bad", [
    {"schema_version": 2}, {"schema_version": True},
    {"profile": "gpu"}, {"profile": "CPU"},
    {"output_dir": "data/processed/f4_05"},
    {"output_dir": "../outside"},
    {"instruction_config_file": "../escape"},
    {"dpo_config_file": "configs/f4_07_base_vs_finetuned.json"},
    {"baseline_config_file": "/tmp/secret"},
    {"comparison_config_file": "configs/../f4_07_base_vs_finetuned.json"},
])
def test_config_rejects_changes(tmp_path, bad):
    data = json.loads(CFG.read_text(encoding="utf-8"))
    data.update(bad)
    name = tmp_path / "config.json"
    name.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(mod.ReportError):
        mod.load_config(name, root=ROOT)


def test_config_rejects_unknown_key(tmp_path):
    raw = json.loads(CFG.read_text())
    raw["grades_are_optional"] = False
    target = tmp_path / "config.json"
    target.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(mod.ReportError):
        mod.load_config(target, root=ROOT)


@pytest.mark.parametrize("bad", [b'{}{}', b'{"a":1,"a":2}', b'{"v":NaN}',
                                  b'{"v":Infinity}', b'\xff', b'{ "missing":'])
def test_strict_json(bad):
    with pytest.raises(mod.ReportError):
        mod._parse(bad, "fixture")


def test_canonical_uses_utf8_no_nan():
    assert mod._canonical({"а": 2}) == '{"а":2}\n'.encode("utf-8")
    with pytest.raises(ValueError):
        mod._canonical({"a": float("nan")})


def test_read_rejects_missing(tmp_path):
    with pytest.raises(mod.ReportError):
        mod._read(tmp_path / "missing")


def test_read_rejects_symlink(tmp_path, monkeypatch):
    dst = tmp_path / "a"
    dst.write_text("abc", encoding="utf-8")
    link = tmp_path / "symlink"
    try:
        link.symlink_to(dst)
    except (OSError, NotImplementedError):
        # На Windows без Developer Mode моделируем именно is_symlink(),
        # чтобы не добавлять очередной skipped к регрессии проекта.
        link.write_text("abc", encoding="utf-8")
        original = Path.is_symlink
        monkeypatch.setattr(Path, "is_symlink", lambda self: self == link or original(self))
    with pytest.raises(mod.ReportError):
        mod._read(link)


def test_finite_numbers():
    assert mod._validate_number(-1.2, "a") == -1.2
    assert mod._validate_number(0.0, "a", 0) == 0


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf"),
                                  True, "3", None, -1.0])
def test_bad_nonnegative(bad):
    with pytest.raises(mod.ReportError):
        mod._finite_score(bad, "score")


def test_render_is_deterministic(summary):
    assert mod.render_markdown(summary) == mod.render_markdown(summary)
    assert mod.render_markdown(summary).startswith(b"# ML Engineering Lab")


def test_render_contains_eight_prior_stages(summary):
    text = mod.render_markdown(summary).decode("utf-8")
    for i in range(1, 9):
        assert f"F4.{i}" in text


def test_render_does_not_claim_ungraded_quality(summary):
    text = mod.render_markdown(summary).decode("utf-8")
    assert "Экспертная оценка качества пока не выполнена" in text
    assert "**0.0000**" not in text
    assert "статистически значимое улучшение" in text
    assert "изменённых текстов" not in text or "улучшение:" not in text


def test_render_validates_negative_dpo_margin(summary):
    text = mod.render_markdown(summary).decode("utf-8")
    assert "-0.150000 → -0.100000" in text


def test_render_does_not_invent_training_curve(summary):
    text = mod.render_markdown(summary).decode("utf-8")
    assert "При одном шаге" in text
    assert "устойчивый тренд" in text


def test_render_discloses_configured_lr(summary):
    text = mod.render_markdown(summary).decode("utf-8")
    assert "Learning rate" in text
    assert "не измерялся" in text


def test_grade_scores_only_if_supplied(summary):
    summary["comparison"]["human_grades"] = {
        "base_macro": 0.4, "finetuned_macro": 0.45, "delta_macro": 0.05,
        "paired_wins": 4, "paired_losses": 2, "paired_ties": 26,
        "grades_sha256": "c" * 64,
    }
    text = mod.render_markdown(summary).decode("utf-8")
    assert "Экспертная оценка по 32 заданиям" in text
    assert "+0.0500" in text
    assert "улучшений 4" in text


@pytest.mark.parametrize("field,value", [
    ("train_loss", float("nan")), ("grad_norm_preclipping", float("inf")),
    ("validation_loss_delta", "none"), ("optimizer_steps", 0),
    ("train_records_used", True), ("trainable_parameters", -1),
])
def test_render_rejects_invalid_sft(summary, field, value):
    summary["sft"][field] = value
    with pytest.raises(mod.ReportError):
        mod.render_markdown(summary)


def test_render_rejects_count_mismatch(summary):
    summary["comparison"]["changed_text_count"] = 100
    with pytest.raises(mod.ReportError):
        mod.render_markdown(summary)


def test_render_rejects_inconsistent_splits(summary):
    summary["instruction_dataset"]["splits"]["test"]["records"] += 1
    with pytest.raises(mod.ReportError):
        mod.render_markdown(summary)


def test_render_rejects_model_metadata_html(summary):
    summary["model"]["id"] = "</script>"
    with pytest.raises(mod.ReportError):
        mod.render_markdown(summary)


def test_report_files_are_exact(fake_evidence):
    files = mod.render_files(fake_evidence)
    assert set(files) == {"final_report.md", "evidence_summary.json"}
    assert mod._parse(files["evidence_summary.json"], "fixture") == fake_evidence["summary"]


def test_build_verify_deterministic(monkeypatch, fake_evidence, conf, tmp_path):
    monkeypatch.setattr(mod, "collect_verified", lambda _: fake_evidence)
    c = replace(conf, output_dir=tmp_path / "f49")
    assert mod.build_report(c) == fake_evidence["summary"]
    assert mod.verify_report(c) == fake_evidence["summary"]
    assert {p.name for p in (c.output_dir / "cpu").iterdir()} == {
        "final_report.md", "evidence_summary.json", "report_manifest.json"}
    assert (c.output_dir / "cpu" / "report_manifest.json").is_file()


def test_build_never_overwrites(monkeypatch, fake_evidence, conf, tmp_path):
    monkeypatch.setattr(mod, "collect_verified", lambda _: fake_evidence)
    c = replace(conf, output_dir=tmp_path / "f49")
    mod.build_report(c)
    with pytest.raises(mod.ReportError, match="существует"):
        mod.build_report(c)


@pytest.mark.parametrize("file", ["final_report.md", "evidence_summary.json", "report_manifest.json"])
def test_verify_detects_changed_output(monkeypatch, fake_evidence, conf, tmp_path, file):
    monkeypatch.setattr(mod, "collect_verified", lambda _: fake_evidence)
    c = replace(conf, output_dir=tmp_path / "f49")
    mod.build_report(c)
    (c.output_dir / "cpu" / file).write_bytes(b"CORRUPTED\n")
    with pytest.raises(mod.ReportError):
        mod.verify_report(c)


def test_verify_detects_new_file(monkeypatch, fake_evidence, conf, tmp_path):
    monkeypatch.setattr(mod, "collect_verified", lambda _: fake_evidence)
    c = replace(conf, output_dir=tmp_path / "f49")
    mod.build_report(c)
    (c.output_dir / "cpu" / "extra.txt").write_text("surprise")
    with pytest.raises(mod.ReportError):
        mod.verify_report(c)


def test_verify_detects_config_change(monkeypatch, fake_evidence, conf, tmp_path):
    monkeypatch.setattr(mod, "collect_verified", lambda _: fake_evidence)
    c = replace(conf, output_dir=tmp_path / "f49")
    mod.build_report(c)
    with pytest.raises(mod.ReportError):
        mod.verify_report(replace(c, config_sha256="0" * 64))


def test_verify_detects_changed_upstream_hash(monkeypatch, fake_evidence, conf, tmp_path):
    monkeypatch.setattr(mod, "collect_verified", lambda _: fake_evidence)
    c = replace(conf, output_dir=tmp_path / "f49")
    mod.build_report(c)
    fake_evidence["source_fingerprints"]["config"] = "0" * 64
    with pytest.raises(mod.ReportError):
        mod.verify_report(c)


def test_grade_template_alone_is_not_quality(monkeypatch, conf, tmp_path):
    fake = type("Stub", (), {"output_dir": tmp_path})()
    (tmp_path / "full").mkdir()
    (tmp_path / "full" / "grades_filled.jsonl").write_text('{"fact_scores":null}\n')
    assert mod._read_optional_human_score(conf, fake, {}) is None


def test_fails_explicitly_on_verified_missing_run(monkeypatch, conf):
    # Подставляем именно отказ старого верификатора, без обращения к настоящим
    # результатам пользователя. Отчёт не должен скрывать ошибку upstream.
    monkeypatch.setattr(mod, "verify_f41", lambda c: mod.prepare_f41(c))
    monkeypatch.setattr(mod, "verify_f43", lambda c: mod.prepare_f43(c))
    monkeypatch.setattr(mod, "verify_f44", lambda *args, **kw: (_ for _ in ()).throw(
        mod.ReportError("missing verified baseline")))
    with pytest.raises(mod.ReportError, match="missing verified baseline"):
        mod.collect_verified(conf)


def test_collect_verified_true_contract_with_test_doubles(monkeypatch, conf):
    # Все backend-подтверждения здесь тестовые заглушки, НЕ реальные запуски.
    # Их задача — обнаружить ошибки связывания ключей между восемью этапами.
    monkeypatch.setattr(mod, "verify_f41", lambda c: mod.prepare_f41(c))
    monkeypatch.setattr(mod, "verify_f43", lambda c: mod.prepare_f43(c))
    monkeypatch.setattr(mod, "_digest", lambda path: "f" * 64)
    monkeypatch.setattr(mod, "_sources", lambda conf: {"fixture": "f" * 64})
    catalog = mod.load_catalog(conf.dependencies[1])
    model = next(m for m in catalog.candidates if m.model_id == catalog.selected_model_id)
    base = {"model_id": model.model_id, "model_revision": model.revision,
            "completed_cases": 32, "generation_performed": True,
            "responses_sha256": "a" * 64, "metrics": {}}
    sft = {"real_training_performed": True, "optimizer_steps": 1,
           "train_records_used": 1, "validation_records_used": 1,
           "trainable_parameters": 540672, "all_parameters_with_adapters": 494573440}
    viz = {"optimizer_steps": 1, "learning_rate_configured": 0.0002,
           "train_loss_last": 2.4, "grad_norm_max_pre_clipping": 3.5,
           "validation_loss_before": 2.28, "validation_loss_after": 2.27,
           "validation_loss_delta": -0.01, "validation_loss_relative_pct": -0.4386,
           "quality_improvement_measured": False,
           "benchmark_f4_03_used_in_training": False}
    comparison = {"cases": 32, "base_responses_sha256": "a" * 64,
                  "adapter_optimizer_steps": 1, "changed_text_count": 2,
                  "unchanged_text_count": 30}
    dpo = {"real_dpo_performed": True, "optimizer_steps": 1,
           "source_sft_manifest_sha256": "f" * 64,
           "train_pairs_used": 1, "validation_pairs_used": 1, "beta": 0.1,
           "initial_validation_margin": -0.1, "final_validation_margin": -0.08,
           "quality_improvement_proven": False, "f43_used_for_training": False,
           "f47_human_grades_used_for_training": False}
    monkeypatch.setattr(mod, "verify_f44", lambda *args, **kw: base)
    monkeypatch.setattr(mod, "verify_f45", lambda *args, **kw: sft)
    monkeypatch.setattr(mod, "verify_f46", lambda *args, **kw: viz)
    monkeypatch.setattr(mod, "verify_f47", lambda *args, **kw: comparison)
    monkeypatch.setattr(mod, "verify_f48", lambda *args, **kw: dpo)
    monkeypatch.setattr(mod, "_read_optional_human_score", lambda *args: None)
    result = mod.collect_verified(conf)
    assert result["summary"]["base"]["responses"] == 32
    assert result["summary"]["sft"]["optimizer_steps"] == 1
    assert result["summary"]["dpo"]["optimizer_steps"] == 1
    assert result["summary"]["comparison"]["human_grades"] is None
    assert b"DPO" in mod.render_markdown(result["summary"])




@pytest.mark.parametrize("encoding", ["cp1251", "cp866", "cp1252"])
def test_cli_help_windows_codepages(encoding):
    env = dict(os.environ, PYTHONIOENCODING=encoding)
    result = subprocess.run([sys.executable, str(CLI), "--help"], cwd=ROOT,
                            capture_output=True, text=True, encoding="ascii", errors="replace",
                            env=env, check=False)
    assert result.returncode == 0, result.stderr
    assert "--build" in result.stdout
    assert "--verify" in result.stdout
