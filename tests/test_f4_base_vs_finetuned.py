"""F4.7: unit/integration с изолированными TEST-DOUBLE ответами; веса не скачиваются."""
from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from src.llm_f4 import base_vs_finetuned as m
from src.llm_f4.base_model_evaluation import (
    Generation, _execute_inner as fake_base_run, load_config as load_base,
)
from src.llm_f4.sft_lora import load_config as load_sft, make_plan
from src.llm_f4.instruction_dataset import export_artifacts as export_split, load_config as load_split
from src.llm_f4.evaluation_set import export_artifacts as export_eval, load_config as load_eval, load_cases

ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "scripts/f4_base_vs_finetuned.py"
CONFIG = ROOT / "configs/f4_07_base_vs_finetuned.json"
COPY = [
    "configs/f4_01_instruction_dataset.json", "configs/f4_02_model_selection.json",
    "configs/f4_03_evaluation_set.json", "configs/f4_04_base_model_evaluation.json",
    "configs/f4_05_sft_lora.json", "configs/f4_07_base_vs_finetuned.json",
    "data/synthetic/f4_01_instruction_seed.jsonl", "data/evaluation/f4_03_ru_technical_v1.jsonl",
]
BASE_RUNTIME = {"kind": "real_cpu_transformers", "python": "TEST ONLY", "torch": "mock", "transformers": "mock", "safetensors": "mock"}
ADAPTER_RUNTIME = {**BASE_RUNTIME, "kind": "real_cpu_peft", "peft": "mock-test-only"}


class FakeGenerator:
    def __init__(self, tag: str = "Base", stop: int | None = None):
        self.prompts: list[str] = []
        self.tag = tag
        self.stop = stop

    def generate(self, prompt: str, *, max_input_tokens: int, max_new_tokens: int) -> Generation:
        if self.stop is not None and len(self.prompts) >= self.stop:
            raise RuntimeError("Simulated interruption")
        assert "reference_answer" not in prompt and "required_facts" not in prompt
        assert prompt.startswith("Вопрос:\n") and prompt.endswith("\nОтвет:\n")
        self.prompts.append(prompt)
        return Generation(f"MOCK {self.tag}: {len(self.prompts)}", 10, 5, True)


@pytest.fixture
def repo(tmp_path: Path):
    for path in COPY:
        dest = tmp_path / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / path, dest)
    export_split(load_split(tmp_path / "configs/f4_01_instruction_dataset.json", root=tmp_path))
    export_eval(load_eval(tmp_path / "configs/f4_03_evaluation_set.json", root=tmp_path))
    base = load_base(tmp_path / "configs/f4_04_base_model_evaluation.json", root=tmp_path)
    sft = load_sft(tmp_path / "configs/f4_05_sft_lora.json", root=tmp_path)
    plan, _ = make_plan(sft, root=tmp_path)
    # Только тестовый адаптер для проверки SHA и provenance, НЕ веса обученной Qwen.
    adapter = sft.output_dir / "cpu" / "adapter"
    adapter.mkdir(parents=True)
    (adapter / "adapter_config.json").write_bytes(b'{"mock":true}')
    (adapter / "adapter_model.safetensors").write_bytes(b'TEST-ONLY-NOT-A-MODEL')
    metric_bytes = b'{"step":1,"train_loss":2.4,"grad_norm":3.5,"micro_batches":1}\n'
    (adapter.parent / "metrics.jsonl").write_bytes(metric_bytes)
    manifest = {
        "real_training_performed": True, "mode": "cpu", "model_id": sft.model_id,
        "model_revision": sft.model_revision, "config_sha256": sft.config_sha256,
        "train_sha256": plan["train_sha256"], "validation_sha256": plan["validation_sha256"],
        "holdout_sha256": plan["holdout_sha256"], "optimizer_steps": 1,
        "adapter_files_sha256": {p.name: sha256(p.read_bytes()).hexdigest() for p in adapter.iterdir()},
        "metrics_sha256": sha256(metric_bytes).hexdigest(),
    }
    (adapter.parent / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return tmp_path, m.load_config(tmp_path / "configs/f4_07_base_vs_finetuned.json", root=tmp_path)


def make_base(repo, profile: str):
    root, conf = repo
    base = load_base(conf.baseline_config_file, root=root)
    fake = FakeGenerator("BASE")
    fake_base_run(base, root=root, profile=profile, runtime=BASE_RUNTIME,
                  backend_factory=lambda: fake)
    return fake


def make_finetuned(repo, profile: str, stop: int | None = None, resume=False):
    root, conf = repo
    fake = FakeGenerator("LORA", stop=stop)
    result = m._execute(conf, root=root, profile=profile, runtime=ADAPTER_RUNTIME,
                        factory=lambda b, s: fake, resume=resume)
    return fake, result


def make_paired(repo, profile="smoke"):
    make_base(repo, profile)
    make_finetuned(repo, profile)
    root, conf = repo
    return m.compare(conf, root=root, profile=profile)


def test_plan(repo):
    root, conf = repo
    result = m.plan(conf, root=root)
    assert result["model"] == "Qwen/Qwen2.5-0.5B"
    assert (result["evaluation_cases"], result["smoke_cases"]) == (32, 2)
    assert result["human_scored"] is False


def test_plan_source_in_project_environment():
    assert len(m.plan(m.load_config(CONFIG, root=ROOT), root=ROOT)["config_sha256"]) == 64


def test_config_paths_and_revisions(repo):
    root, conf = repo
    assert conf.sft_profile == "cpu"
    assert conf.baseline_config_file == root / "configs/f4_04_base_model_evaluation.json"
    assert conf.output_dir != conf.baseline_config_file.parent


@pytest.mark.parametrize("edit", [
    lambda d: d.update(schema_version=2), lambda d: d.update(schema_version=True),
    lambda d: d.pop("schema_version"), lambda d: d.update(extra=True),
    lambda d: d.update(sft_profile="foobar"), lambda d: d.update(sft_profile=[]),
    lambda d: d.update(baseline_config_file="../../evil"),
    lambda d: d.update(sft_config_file="/tmp/model"),
    lambda d: d.update(output_dir="C:/secret"),
    lambda d: d.update(output_dir="data\\secret"),
    lambda d: d.update(output_dir="data//processed"),
    lambda d: d.update(output_dir="data/./processed"),
    lambda d: d.update(output_dir="data/processed/../bad"),
    lambda d: d.update(output_dir="data/processed/f4_04"),
    lambda d: d.update(output_dir="data/processed/f4_05"),
])
def test_bad_config(repo, edit):
    root, conf = repo
    obj = json.loads((root / "configs/f4_07_base_vs_finetuned.json").read_text(encoding="utf8"))
    edit(obj)
    file = root / "configs/test_mutation.json"
    file.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(m.ComparisonError):
        m.load_config(file, root=root)


@pytest.mark.parametrize("bad", ["", "../secret", "/var/a", "C:/a", "a\\b", "a//b", "a/./b", "a/../b", None])
def test_path_rejected(bad):
    with pytest.raises(m.ComparisonError):
        m._rel_path(ROOT, bad, "input")


def test_duplicate_key():
    with pytest.raises(m.ComparisonError, match="Повторный"):
        m._decode(b'{"a":1,"a":2}', "test")


def test_nan_json():
    with pytest.raises(m.ComparisonError, match="NaN"):
        m._decode(b'{"a": NaN}', "test")


def test_utf8_invalid():
    with pytest.raises(m.ComparisonError):
        m._decode(b'\xff', "bad")


def test_real_runtime_check():
    assert m._runtime_is_real(ADAPTER_RUNTIME)
    assert not m._runtime_is_real(BASE_RUNTIME)
    assert not m._runtime_is_real({"kind": "real_cpu_peft", "peft": None})


def test_real_runtime_routes_to_f44_without_recursive_shadowing(repo, monkeypatch):
    """F4.7 обязан вызывать именно runtime F4.4, не одноимённую функцию F4.7."""
    root, conf = repo
    captured = []

    def fake_f44_runtime(base):
        captured.append(base)
        return dict(BASE_RUNTIME)

    monkeypatch.setattr(m, "base_real_runtime", fake_f44_runtime)
    monkeypatch.setattr(m, "version", lambda package: "TEST-PEFT" if package == "peft" else "unexpected")
    info = m.real_runtime(conf, root=root)
    assert len(captured) == 1
    assert captured[0].model_id == "Qwen/Qwen2.5-0.5B"
    assert info["kind"] == "real_cpu_peft"
    assert info["peft"] == "TEST-PEFT"
    assert info["torch"] == "mock"


def test_f44_runtime_error_translated_to_comparison_error(repo, monkeypatch):
    """Ошибки проверки окружения Base не должны выходить как необработанный traceback."""
    from src.llm_f4.base_model_evaluation import BaselineError

    def unavailable(base):
        raise BaselineError("Test-only F4.4 runtime incompatible")

    monkeypatch.setattr(m, "base_real_runtime", unavailable)
    with pytest.raises(m.ComparisonError, match="Test-only F4.4 runtime incompatible"):
        m.real_runtime_base(load_base(repo[1].baseline_config_file, root=repo[0]))


def test_missing_baseline_rejected(repo):
    root, conf = repo
    with pytest.raises(m.ComparisonError, match="F4.4 smoke"):
        m._require_sources(conf, root=root, profile="smoke")


def test_full_is_not_smoke(repo):
    make_base(repo, "smoke")
    root, conf = repo
    with pytest.raises(m.ComparisonError, match="F4.4 full"):
        m._require_sources(conf, root=root, profile="full")


def test_mock_smoke_generation_and_integrity(repo):
    make_base(repo, "smoke")
    fake, result = make_finetuned(repo, "smoke")
    root, conf = repo
    assert result["expected_cases"] == 2
    assert len(fake.prompts) == 2
    assert result["metrics"]["quality_status"] == "NOT_HUMAN_GRADED"
    assert m.verify(conf, root=root, profile="smoke")["completed_cases"] == 2


def test_mock_full_generation(repo):
    make_base(repo, "full")
    fake, result = make_finetuned(repo, "full")
    assert len(fake.prompts) == result["completed_cases"] == 32


def test_compare_smoke_is_not_quality(repo):
    compared = make_paired(repo)
    assert compared["cases"] == 2
    assert compared["changed_text_count"] == 2
    assert compared["human_scoring_performed"] is False
    assert "NOT_GRADED" in compared["quality_conclusion"]


def test_paired_case_id_and_hashes(repo):
    make_paired(repo)
    root, conf = repo
    pairs, context = m.paired_rows(conf, root=root, profile="smoke")
    assert [p["case_id"] for p in pairs] == ["ev_py_01", "ev_ml_01"]
    assert len({p["prompt_sha256"] for p in pairs}) == 2
    assert context["adapter_optimizer_steps"] == 1
    assert all(set(p) == m.RESPONSE_KEYS for p in pairs)
    assert all("reference_answer" not in str(p) and "rubric" not in str(p) for p in pairs)


def test_compare_not_rewrite(repo):
    make_paired(repo)
    root, conf = repo
    with pytest.raises(m.ComparisonError, match="существует"):
        m.compare(conf, root=root, profile="smoke")


def test_verify_comparison(repo):
    summary = make_paired(repo)
    root, conf = repo
    assert m.verify_comparison(conf, root=root, profile="smoke") == summary


def test_tampered_paired_rejected(repo):
    make_paired(repo)
    root, conf = repo
    path = conf.output_dir / "smoke" / "paired_responses.jsonl"
    path.write_bytes(path.read_bytes().replace(b"MOCK", b"CHEAT"))
    with pytest.raises(m.ComparisonError):
        m.verify_comparison(conf, root=root, profile="smoke")


def test_tampered_context_rejected(repo):
    make_base(repo, "smoke")
    make_finetuned(repo, "smoke")
    root, conf = repo
    path = conf.output_dir / "smoke" / "context.json"
    path.write_bytes(path.read_bytes().replace(b"greedy", b"sample"))
    with pytest.raises(m.ComparisonError):
        m.verify(conf, root=root, profile="smoke")


def test_tampered_response_rejected(repo):
    make_base(repo, "smoke")
    make_finetuned(repo, "smoke")
    root, conf = repo
    path = conf.output_dir / "smoke" / "responses.jsonl"
    path.write_bytes(path.read_bytes().replace(b"MOCK", b"INVALID"))
    with pytest.raises((m.ComparisonError, ValueError)):
        m.verify(conf, root=root, profile="smoke")


def test_tampered_adapter_rejected(repo):
    make_base(repo, "smoke")
    root, conf = repo
    sft = load_sft(conf.sft_config_file, root=root)
    (sft.output_dir / "cpu" / "adapter" / "adapter_model.safetensors").write_bytes(b'CHANGED')
    with pytest.raises(m.ComparisonError, match="F4.5"):
        m._require_sources(conf, root=root, profile="smoke")


def test_tampered_base_rejected(repo):
    make_base(repo, "smoke")
    root, conf = repo
    base = load_base(conf.baseline_config_file, root=root)
    path = base.output_dir / "smoke" / "responses.jsonl"
    path.write_bytes(path.read_bytes().replace(b"MOCK", b"CHANGED"))
    with pytest.raises(m.ComparisonError, match="F4.4"):
        m._require_sources(conf, root=root, profile="smoke")


def test_resume_after_interruption(repo):
    make_base(repo, "smoke")
    root, conf = repo
    with pytest.raises(RuntimeError, match="Simulated"):
        make_finetuned(repo, "smoke", stop=1)
    progress = conf.output_dir / "smoke" / "progress.jsonl"
    assert len(progress.read_bytes().splitlines()) == 1
    resumed = FakeGenerator("LORA-RESUMED")
    m._execute(conf, root=root, profile="smoke", runtime=ADAPTER_RUNTIME,
               factory=lambda b, s: resumed, resume=True)
    assert len(resumed.prompts) == 1
    assert m.verify(conf, root=root, profile="smoke")["completed_cases"] == 2


def test_resume_context_change_blocked(repo):
    make_base(repo, "smoke")
    root, conf = repo
    with pytest.raises(RuntimeError):
        make_finetuned(repo, "smoke", stop=1)
    changed = replace(conf, sha256="f" * 64)
    with pytest.raises(m.ComparisonError, match="Resume"):
        m._execute(changed, root=root, profile="smoke", runtime=ADAPTER_RUNTIME,
                   factory=lambda b, s: FakeGenerator(), resume=True)


def test_refuse_to_overwrite_run(repo):
    make_base(repo, "smoke")
    make_finetuned(repo, "smoke")
    root, conf = repo
    with pytest.raises(m.ComparisonError, match="перезапись"):
        make_finetuned(repo, "smoke")


def test_incomplete_progress_rejected(repo):
    make_base(repo, "smoke")
    root, conf = repo
    with pytest.raises(RuntimeError):
        make_finetuned(repo, "smoke", stop=1)
    path = conf.output_dir / "smoke" / "progress.jsonl"
    path.write_bytes(path.read_bytes()[:-1])
    with pytest.raises(m.ComparisonError):
        make_finetuned(repo, "smoke", resume=True)


def test_grades_template_after_full(repo):
    make_paired(repo, "full")
    root, conf = repo
    grade_path, review_path = m.grades_template(conf, root=root)
    assert grade_path.exists() and review_path.exists()
    lines = grade_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 32
    review = review_path.read_text(encoding="utf-8")
    assert "Эталон" in review and "MOCK BASE" in review and "MOCK LORA" in review
    assert json.loads(lines[0])["base_fact_scores"]


def test_grades_template_not_rewrite(repo):
    make_paired(repo, "full")
    root, conf = repo
    m.grades_template(conf, root=root)
    with pytest.raises(m.ComparisonError):
        m.grades_template(conf, root=root)


def _grade_rows(repo):
    root, conf = repo
    pairs, _ = m.paired_rows(conf, root=root, profile="full")
    cases = load_cases(load_eval(load_base(conf.baseline_config_file, root=root).evaluation_config_file, root=root).source_file)
    lookup = {c.case_id: c for c in cases}
    rows = []
    for i, row in enumerate(pairs):
        scores = {f.fact_id: 1 for f in lookup[row["case_id"]].required_facts}
        better = {f.fact_id: (2 if i % 4 == 0 else 1) for f in lookup[row["case_id"]].required_facts}
        rows.append({"case_id": row["case_id"], "base_response_sha256": row["base_response_sha256"],
                     "finetuned_response_sha256": row["finetuned_response_sha256"],
                     "base_fact_scores": scores, "finetuned_fact_scores": better,
                     "notes": "TEST ONLY - mock grades"})
    path = root / "fake_grades.jsonl"
    path.write_bytes(b"".join(m._line(r) for r in rows))
    return path, rows


def test_manual_scores_numeric_reference(repo):
    make_paired(repo, "full")
    root, conf = repo
    grades, rows = _grade_rows(repo)
    result = m.score_grades(conf, root=root, grades_file=grades)
    assert result["base_macro"] == pytest.approx(.5)
    assert result["finetuned_macro"] == pytest.approx(.625)
    assert result["delta_macro"] == pytest.approx(.125)
    assert (result["paired_wins"], result["paired_losses"], result["paired_ties"]) == (8, 0, 24)
    assert result["human_scoring_performed"] is True
    assert len(result["per_case_delta"]) == 32


def test_unfilled_grades_rejected(repo):
    make_paired(repo, "full")
    root, conf = repo
    path, review = m.grades_template(conf, root=root)
    with pytest.raises(m.ComparisonError, match="Не выставлены"):
        m.score_grades(conf, root=root, grades_file=path)


def test_stale_grade_hash_rejected(repo):
    make_paired(repo, "full")
    root, conf = repo
    path, rows = _grade_rows(repo)
    rows[0]["base_response_sha256"] = "a" * 64
    path.write_bytes(b"".join(m._line(r) for r in rows))
    with pytest.raises(m.ComparisonError, match="другим ответам"):
        m.score_grades(conf, root=root, grades_file=path)


def test_missing_grade_case_rejected(repo):
    make_paired(repo, "full")
    root, conf = repo
    path, rows = _grade_rows(repo)
    path.write_bytes(b"".join(m._line(r) for r in rows[:-1]))
    with pytest.raises(m.ComparisonError, match="32"):
        m.score_grades(conf, root=root, grades_file=path)


def test_wrong_fact_ids_rejected(repo):
    make_paired(repo, "full")
    root, conf = repo
    path, rows = _grade_rows(repo)
    rows[0]["base_fact_scores"] = {"wrong_id": 2}
    path.write_bytes(b"".join(m._line(r) for r in rows))
    with pytest.raises(m.ComparisonError):
        m.score_grades(conf, root=root, grades_file=path)


def test_grading_no_overwrite(repo):
    make_paired(repo, "full")
    root, conf = repo
    path, rows = _grade_rows(repo)
    m.score_grades(conf, root=root, grades_file=path)
    rows[0]["finetuned_fact_scores"][next(iter(rows[0]["finetuned_fact_scores"]))] = 0
    path.write_bytes(b"".join(m._line(r) for r in rows))
    with pytest.raises(m.ComparisonError, match="не перезаписываем"):
        m.score_grades(conf, root=root, grades_file=path)


@pytest.mark.parametrize("codepage", ["cp1251", "cp866", "cp1252"])
def test_windows_help(codepage):
    call = subprocess.run([sys.executable, str(CLI), "--help"], cwd=ROOT, capture_output=True,
                          text=True, encoding="utf-8", env={**__import__("os").environ, "PYTHONIOENCODING": codepage})
    assert call.returncode == 0
    assert "--score-grades" in call.stdout


def test_cli_plan():
    call = subprocess.run([sys.executable, str(CLI), "--plan"], cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
    assert call.returncode == 0
    assert "F4.7 PLAN: PASS" in call.stdout
    assert "NOT RUN" in call.stdout


def test_cli_refuses_unwanted_download():
    call = subprocess.run([sys.executable, str(CLI), "--allow-download"], cwd=ROOT,
                          capture_output=True, text=True, encoding="utf-8")
    assert call.returncode != 0
    assert "--run" in call.stderr


def test_cli_refuses_unwanted_resume():
    call = subprocess.run([sys.executable, str(CLI), "--resume"], cwd=ROOT,
                          capture_output=True, text=True, encoding="utf-8")
    assert call.returncode != 0
    assert "--run" in call.stderr
