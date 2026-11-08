"""F4.4: CPU-free контрактные тесты и test double; реальные веса НЕ используются."""

from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from src.llm_f4.base_model_evaluation import (
    BaselineError, Generation, Prompt, RealCPUGenerator, _context_sha, _execute_inner,
    _final_payload, _json, _json_pretty, _line, _load_journal, _read, _safe_path,
    _summary, _validate_result, _validate_rows, _write_atomic, execute,
    formatted_prompt, load_config, load_prompts, profile_context,
    score_grades, selection, template_grades, verify,
)
from src.llm_f4.evaluation_set import export_artifacts, load_config as eval_load_config

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/f4_base_model_evaluation.py"
SOURCE_CONFIG = ROOT / "configs/f4_04_base_model_evaluation.json"
NEEDED = (
    "configs/f4_02_model_selection.json", "configs/f4_03_evaluation_set.json",
    "configs/f4_04_base_model_evaluation.json", "data/evaluation/f4_03_ru_technical_v1.jsonl",
    "data/synthetic/f4_01_instruction_seed.jsonl",
)
RUNTIME = {"kind": "real_cpu_transformers", "python": "test", "torch": "mock",
           "transformers": "mock", "safetensors": "mock", "platform": "test"}


class FakeGenerator:
    """Исключительно подставной backend для проверки конвейера, НЕ модель."""

    def __init__(self, *, limit: int | None = None) -> None:
        self.received: list[str] = []
        self.limit = limit

    def generate(self, prompt: str, *, max_input_tokens: int, max_new_tokens: int) -> Generation:
        if self.limit is not None and len(self.received) >= self.limit:
            raise RuntimeError("simulate interrupted CPU run")
        self.received.append(prompt)
        assert "required_facts" not in prompt and "reference_answer" not in prompt
        assert "Вопрос:" in prompt and prompt.endswith("Ответ:\n")
        return Generation("Synthetic unit-test answer", 10, 5, True)


@pytest.fixture
def repo(tmp_path: Path):
    for rel in NEEDED:
        dest = tmp_path / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / rel, dest)
    eval_cfg = eval_load_config(tmp_path / "configs/f4_03_evaluation_set.json", root=tmp_path)
    export_artifacts(eval_cfg)
    conf = load_config(tmp_path / "configs/f4_04_base_model_evaluation.json", root=tmp_path)
    return tmp_path, conf


def write_conf(root: Path, modifier) -> Path:
    doc = json.loads((root / "configs/f4_04_base_model_evaluation.json").read_text(encoding="utf8"))
    modifier(doc)
    path = root / "configs/f4_04_custom.json"
    path.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    return path


def update_json(path: Path, modifier):
    obj = json.loads(path.read_text(encoding="utf8"))
    modifier(obj)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf8")


def prepared_run(repo, profile="smoke"):
    root, conf = repo
    fake = FakeGenerator()
    manifest = _execute_inner(conf, root=root, profile=profile, backend_factory=lambda: fake,
                              runtime=RUNTIME)
    return root, conf, fake, manifest


def test_config_loads_pinned_base(repo):
    _, conf = repo
    assert conf.model_id == "Qwen/Qwen2.5-0.5B"
    assert conf.model_revision == "060db6499f32faf8b98477b0a26969ef7d8b9987"
    assert conf.tokenizer_revision == conf.model_revision


def test_cpu_context_window_bound(repo):
    _, conf = repo
    assert conf.max_input_tokens + conf.max_new_tokens <= 512
    assert (conf.max_input_tokens, conf.max_new_tokens) == (384, 96)


def test_source_config_is_pinned_immutable_sha(repo):
    _, conf = repo
    assert len(conf.sha256) == 64 and conf.sha256 == sha256(SOURCE_CONFIG.read_bytes()).hexdigest()


@pytest.mark.parametrize("edit", [
    lambda d: d.update(schema_version=2),
    lambda d: d.pop("seed"),
    lambda d: d.update(unknown=7),
    lambda d: d.update(schema_version=True),
    lambda d: d.update(model_id="Qwen/Qwen2.5-0.5B-Instruct"),
    lambda d: d.update(model_revision="main"),
    lambda d: d.update(tokenizer_revision="a" * 40),
    lambda d: d.update(max_input_tokens=510),
    lambda d: d.update(max_new_tokens=0),
    lambda d: d.update(seed=-1),
    lambda d: d.update(cpu_threads=0),
    lambda d: d.update(smoke_ids=[]),
    lambda d: d.update(smoke_ids=["ev_py_01", "ev_py_01"]),
    lambda d: d.update(smoke_ids=["invalid"]),
    lambda d: d.update(prefix="{prompt}"),
    lambda d: d.update(suffix="\t"),
    lambda d: d.update(evaluation_config_file="../../etc/passwd"),
    lambda d: d.update(output_dir="C:/notallowed"),
    lambda d: d.update(output_dir="data\\escape"),
    lambda d: d.update(output_dir="data/processed/../bad"),
])
def test_bad_configuration_rejected(repo, edit):
    root, _ = repo
    with pytest.raises(BaselineError):
        load_config(write_conf(root, edit), root=root)


def test_non_integer_bool_rejected(repo):
    root, _ = repo
    with pytest.raises(BaselineError):
        load_config(write_conf(root, lambda d: d.update(max_new_tokens=True)), root=root)


def test_nonfinite_constants_and_duplicate_keys_rejected():
    with pytest.raises(BaselineError):
        _json(b'{"a": NaN}', "test")
    with pytest.raises(BaselineError):
        _json(b'{"a":1,"a":2}', "test")


def test_safe_path_blocks_symlink_escape(tmp_path):
    outside = tmp_path.parent / "outside"
    outside.mkdir(exist_ok=True)
    target = tmp_path / "escape"
    try:
        target.symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("Symlinks unavailable on this platform")
    with pytest.raises(BaselineError):
        _safe_path(tmp_path, "escape/answers.jsonl", "escape")


def test_exact_prompt_only_input_and_manifest(repo):
    root, conf = repo
    prompts, meta = load_prompts(conf, root=root)
    assert len(prompts) == 32 and len({p.case_id for p in prompts}) == 32
    assert all(p.domain in ("python", "linux", "sql", "ml") for p in prompts)
    assert meta["total_cases"] == 32
    assert all("rubric" not in p.prompt and "reference_answer" not in p.prompt for p in prompts)


def test_prompt_export_must_exist(repo):
    root, conf = repo
    (root / "data/processed/f4_03/prompts.jsonl").unlink()
    with pytest.raises(BaselineError, match="F4.3 export validation failed"):
        load_prompts(conf, root=root)


def test_prompt_export_tampering_detected(repo):
    root, conf = repo
    file = root / "data/processed/f4_03/prompts.jsonl"
    file.write_bytes(_read(file).replace(b'"domain":"python"', b'"domain":"sql"', 1))
    with pytest.raises(BaselineError):
        load_prompts(conf, root=root)


def test_eval_source_tampering_detected(repo):
    root, conf = repo
    file = root / "data/evaluation/f4_03_ru_technical_v1.jsonl"
    file.write_bytes(_read(file) + b" ")
    with pytest.raises(BaselineError):
        load_prompts(conf, root=root)


def test_fixed_smoke_and_full_selection(repo):
    root, conf = repo
    prompts, _ = load_prompts(conf, root=root)
    assert [p.case_id for p in selection(prompts, conf, "smoke")] == ["ev_py_01", "ev_ml_01"]
    assert len(selection(prompts, conf, "full")) == 32
    with pytest.raises(BaselineError):
        selection(prompts, conf, "partial")


def test_smoke_absent_case_fails(repo):
    root, conf = repo
    prompts, _ = load_prompts(conf, root=root)
    with pytest.raises(BaselineError):
        selection(prompts, replace(conf, smoke_ids=("ev_py_88",)), "smoke")


def test_formatted_prompt_retains_data_without_gold(repo):
    root, conf = repo
    prompts, _ = load_prompts(conf, root=root)
    p = prompts[0]
    assert formatted_prompt(p, conf) == "Вопрос:\n" + p.prompt + "\nОтвет:\n"
    assert "[2, 4, 6]" not in formatted_prompt(p, conf)


def test_generation_result_writes_only_required_fields(repo):
    root, conf, fake, manifest = prepared_run(repo)
    raw = _read(conf.output_dir / "smoke/responses.jsonl")
    rows = [_json(line, "response") for line in raw.splitlines()]
    assert len(rows) == 2
    assert all(set(r) == {
        "case_id", "domain", "prompt_sha256", "response", "response_sha256", "prompt_tokens",
        "generated_tokens", "stopped_on_eos", "elapsed_seconds", "model_generated"
    } for r in rows)
    assert len(fake.received) == 2
    assert all("reference_answer" not in value and "required_facts" not in value for value in fake.received)
    assert manifest["metrics"]["quality_status"] == "NOT_HUMAN_GRADED"
    assert manifest["human_grading_performed"] is False


def test_verify_detects_corrupt_response(repo):
    root, conf, fake, manifest = prepared_run(repo)
    file = conf.output_dir / "smoke/responses.jsonl"
    file.write_bytes(_read(file) + b"!corrupted")
    with pytest.raises(BaselineError, match="SHA256 mismatch"):
        verify(conf, root=root, profile="smoke")


def test_verify_checks_journal_matches_response(repo):
    root, conf, fake, manifest = prepared_run(repo)
    file = conf.output_dir / "smoke/progress.jsonl"
    file.write_bytes(b"wrong")
    with pytest.raises(BaselineError, match="Progress journal differs"):
        verify(conf, root=root, profile="smoke")


def test_verify_checks_pinned_model_identity(repo):
    root, conf, fake, manifest = prepared_run(repo)
    with pytest.raises(BaselineError):
        verify(replace(conf, model_revision="0" * 40), root=root, profile="smoke")


def test_verify_checks_manifest_content(repo):
    root, conf, fake, manifest = prepared_run(repo)
    path = conf.output_dir / "smoke/manifest.json"
    update_json(path, lambda d: d["metrics"].update(total_generated_tokens=999))
    with pytest.raises(BaselineError, match="Manifest does not match"):
        verify(conf, root=root, profile="smoke")


def test_verify_rejects_pseudoruntime(repo):
    root, conf = repo
    _execute_inner(conf, root=root, profile="smoke", backend_factory=FakeGenerator,
                   runtime={"kind": "test_double"})
    with pytest.raises(BaselineError, match="Not a real Base"):
        verify(conf, root=root, profile="smoke")


def test_execute_public_api_refuses_mocks(repo):
    root, conf = repo
    with pytest.raises(BaselineError, match="requires real_cpu_transformers"):
        execute(conf, root=root, profile="smoke", runtime={"kind": "test_double"}, backend_factory=FakeGenerator)


def test_run_refuses_to_overwrite_existing_outputs(repo):
    root, conf, fake, manifest = prepared_run(repo)
    with pytest.raises(BaselineError, match="not empty"):
        _execute_inner(conf, root=root, profile="smoke", backend_factory=FakeGenerator, runtime=RUNTIME)


def test_resume_after_crash_without_duplicate_generations(repo):
    root, conf = repo
    first = FakeGenerator(limit=1)
    with pytest.raises(RuntimeError, match="interrupted"):
        _execute_inner(conf, root=root, profile="smoke", backend_factory=lambda: first, runtime=RUNTIME)
    progress = conf.output_dir / "smoke/progress.jsonl"
    assert len(_read(progress).splitlines()) == 1
    second = FakeGenerator()
    _execute_inner(conf, root=root, profile="smoke", backend_factory=lambda: second, runtime=RUNTIME, resume=True)
    assert len(second.received) == 1
    assert verify(conf, root=root, profile="smoke")["completed_cases"] == 2


def test_resume_with_changed_runtime_rejected(repo):
    root, conf = repo
    with pytest.raises(RuntimeError):
        _execute_inner(conf, root=root, profile="smoke", backend_factory=lambda: FakeGenerator(limit=0), runtime=RUNTIME)
    with pytest.raises(BaselineError, match="context mismatch"):
        _execute_inner(conf, root=root, profile="smoke", backend_factory=FakeGenerator,
                       runtime={**RUNTIME, "torch": "updated"}, resume=True)


def test_resume_completed_run_rejected(repo):
    root, conf, fake, manifest = prepared_run(repo)
    with pytest.raises(BaselineError, match="Complete run"):
        _execute_inner(conf, root=root, profile="smoke", backend_factory=FakeGenerator, runtime=RUNTIME, resume=True)


def test_resume_does_not_silently_repair_truncated_journal(repo):
    root, conf = repo
    with pytest.raises(RuntimeError):
        _execute_inner(conf, root=root, profile="smoke", backend_factory=lambda: FakeGenerator(limit=1), runtime=RUNTIME)
    path = conf.output_dir / "smoke/progress.jsonl"
    path.write_bytes(_read(path).rstrip(b"\n"))
    with pytest.raises(BaselineError, match="Interrupted/corrupted"):
        _execute_inner(conf, root=root, profile="smoke", backend_factory=FakeGenerator, runtime=RUNTIME, resume=True)


def test_resume_does_not_silently_accept_reordered_journal(repo):
    root, conf, fake, manifest = prepared_run(repo)
    path = conf.output_dir / "smoke/progress.jsonl"
    first, second = _read(path).splitlines()
    path.write_bytes(second + b"\n" + first + b"\n")
    (conf.output_dir / "smoke/responses.jsonl").unlink()
    (conf.output_dir / "smoke/manifest.json").unlink()
    with pytest.raises(BaselineError):
        _execute_inner(conf, root=root, profile="smoke", backend_factory=FakeGenerator, runtime=RUNTIME, resume=True)


def test_negative_generation_counts_rejected(repo):
    root, conf = repo
    prompts, _ = load_prompts(conf, root=root)
    p = prompts[0]
    row = {"case_id": p.case_id, "domain": p.domain,
           "prompt_sha256": sha256(formatted_prompt(p, conf).encode()).hexdigest(),
           "response": "hi", "response_sha256": sha256(b"hi").hexdigest(),
           "prompt_tokens": 2, "generated_tokens": 2, "stopped_on_eos": True,
           "elapsed_seconds": 0.1, "model_generated": True}
    for key, value in (("prompt_tokens", 0), ("generated_tokens", 97), ("stopped_on_eos", 1),
                       ("model_generated", False), ("elapsed_seconds", float("nan")),
                       ("response", 777)):
        mutated = {**row, key: value}
        with pytest.raises(BaselineError):
            _validate_result(mutated, p, conf)


def test_text_sha_required_for_each_response(repo):
    root, conf = repo
    prompts, _ = load_prompts(conf, root=root)
    p = prompts[0]
    row = {"case_id": p.case_id, "domain": p.domain,
           "prompt_sha256": sha256(formatted_prompt(p, conf).encode()).hexdigest(),
           "response": "one", "response_sha256": sha256(b"two").hexdigest(),
           "prompt_tokens": 2, "generated_tokens": 2, "stopped_on_eos": False,
           "elapsed_seconds": 1.0, "model_generated": True}
    with pytest.raises(BaselineError):
        _validate_result(row, p, conf)


def test_summary_uses_token_rates_not_quality_metrics():
    rows = [{"elapsed_seconds": 2.0, "generated_tokens": 10, "response": "x", "stopped_on_eos": True},
            {"elapsed_seconds": 3.0, "generated_tokens": 20, "response": "", "stopped_on_eos": False}]
    score = _summary(rows)
    assert score["tokens_per_generation_second"] == 6.0
    assert score["mean_tokens_per_case"] == 15
    assert score["empty_decoded_responses"] == 1
    assert score["quality_score"] is None


def test_context_is_pinned_to_f4_3_export(repo):
    root, conf = repo
    _, meta = load_prompts(conf, root=root)
    context = profile_context(conf, meta, "full", runtime=RUNTIME)
    assert context["eval_prompts_sha256"] == meta["prompts_sha256"]
    assert context["eval_source_sha256"] == meta["source_sha256"]
    assert context["do_sample"] is False
    assert context["num_beams"] == 1
    assert context["device"] == "cpu"
    assert len(_context_sha(context)) == 64


def test_scoring_requires_full_32_not_smoke(repo):
    root, conf, fake, manifest = prepared_run(repo)
    with pytest.raises(BaselineError):
        template_grades(conf, root=root)


def test_grades_template_has_no_answers_or_gold(repo):
    root, conf, fake, manifest = prepared_run(repo, profile="full")
    path = template_grades(conf, root=root)
    rows = [_json(line, "grades") for line in _read(path).splitlines()]
    assert len(rows) == 32
    assert all(set(r) == {"case_id", "response_sha256", "fact_scores", "notes"} for r in rows)
    assert all(set(r["fact_scores"].values()) == {None} for r in rows)
    assert b"reference_answer" not in _read(path)
    with pytest.raises(BaselineError, match="already exists"):
        template_grades(conf, root=root)


def test_unscored_template_not_gradeable(repo):
    root, conf, fake, manifest = prepared_run(repo, profile="full")
    path = template_grades(conf, root=root)
    with pytest.raises(BaselineError):
        score_grades(conf, root=root, grades_file=path)


def test_human_grade_macro_normalizes_correctly(repo):
    root, conf, fake, manifest = prepared_run(repo, profile="full")
    path = template_grades(conf, root=root)
    rows = [_json(line, "grade") for line in _read(path).splitlines()]
    for row in rows:
        row["fact_scores"] = dict.fromkeys(row["fact_scores"], 2)
        row["notes"] = "Expert verified response; synthetic unit test only"
    path.write_bytes(b"".join(_line(r) for r in rows))
    score = score_grades(conf, root=root, grades_file=path)
    assert score["macro_over_cases"] == 1.0
    assert len(score["per_case"]) == 32
    assert (conf.output_dir / "full/manual_scores.json").exists()


def test_grades_reject_foreign_response_hash(repo):
    root, conf, fake, manifest = prepared_run(repo, profile="full")
    path = template_grades(conf, root=root)
    rows = [_json(line, "grade") for line in _read(path).splitlines()]
    rows[0]["response_sha256"] = "a" * 64
    path.write_bytes(b"".join(_line(r) for r in rows))
    with pytest.raises(BaselineError, match="stale/foreign"):
        score_grades(conf, root=root, grades_file=path)


def test_grades_reject_duplicate_case(repo):
    root, conf, fake, manifest = prepared_run(repo, profile="full")
    path = template_grades(conf, root=root)
    rows = [_json(line, "grade") for line in _read(path).splitlines()]
    rows[0] = rows[1]
    for row in rows:
        row["fact_scores"] = dict.fromkeys(row["fact_scores"], 0)
    path.write_bytes(b"".join(_line(r) for r in rows))
    with pytest.raises(BaselineError):
        score_grades(conf, root=root, grades_file=path)


@pytest.mark.parametrize("codec", ["cp1251", "cp866", "cp1252"])
def test_help_works_on_windows_codepages(codec):
    env = {**os.environ, "PYTHONIOENCODING": codec}
    proc = subprocess.run([sys.executable, str(SCRIPT), "--help"], cwd=ROOT, env=env,
                          capture_output=True, text=True, encoding="ascii", errors="replace")
    assert proc.returncode == 0, proc.stderr
    assert "--run {smoke,full}" in proc.stdout
    assert "--verify {smoke,full}" in proc.stdout


def test_cli_rejects_download_without_run():
    process = subprocess.run([sys.executable, str(SCRIPT), "--allow-download"], cwd=ROOT,
                             capture_output=True, text=True, encoding="utf-8")
    assert process.returncode == 2
    assert "--allow-download requires --run" in process.stderr


def test_source_compiles():
    import py_compile
    py_compile.compile(str(ROOT / "src/llm_f4/base_model_evaluation.py"), doraise=True)
    py_compile.compile(str(SCRIPT), doraise=True)


def test_docs_have_all_nine_sections():
    path = ROOT / "docs/steps/F4_04_base_model_evaluation.md"
    content = path.read_text(encoding="utf8")
    for heading in (
        "## 1. Цель этапа", "## 2. Что реализовано", "## 3. Последовательность реализации",
        "## 4. Алгоритм и математический пример", "## 5. Схема работы",
        "## 6. Локальная проверка PowerShell",
        "## 7. Что доказано тестами и какие ограничения остаются",
        "## 8. Основная мысль этапа", "## 9. Файлы этапа",
    ):
        assert heading in content


def test_real_generation_adapter_slices_only_new_token_ids_without_any_ml_import():
    """Проверяем именно тело RealCPUGenerator.generate с test doubles для tensors."""
    from contextlib import nullcontext

    class Tokens:
        def __init__(self, ids):
            self.ids = list(ids)
            self.shape = (1, len(ids))

        def to(self, device):
            assert device == "cpu"
            return self

        def __getitem__(self, index):
            batch, part = index
            assert batch == 0
            return SliceTokens(self.ids[part])

    class SliceTokens:
        def __init__(self, ids):
            self.ids = ids

        def tolist(self):
            return self.ids

    class Tokenizer:
        eos_token_id = 9

        def __call__(self, text, **kwargs):
            assert kwargs == {"return_tensors": "pt", "add_special_tokens": False}
            return {"input_ids": Tokens([11, 12]), "attention_mask": Tokens([1, 1])}

        def decode(self, new_ids, **kwargs):
            assert new_ids == [4, 9]  # Prefix 11,12 already stripped.
            assert kwargs["skip_special_tokens"] is True
            return "Ответ модели"

    class Model:
        def generate(self, **kwargs):
            assert kwargs["do_sample"] is False
            assert kwargs["num_beams"] == 1
            assert kwargs["max_new_tokens"] == 16
            return Tokens([11, 12, 4, 9])

    gen = RealCPUGenerator.__new__(RealCPUGenerator)
    gen.torch = type("StubTorch", (), {"inference_mode": staticmethod(nullcontext)})()
    gen.tokenizer = Tokenizer()
    gen.model = Model()
    result = gen.generate("Вопрос: x", max_input_tokens=8, max_new_tokens=16)
    assert result == Generation("Ответ модели", 2, 2, True)


def test_real_generation_adapter_fails_on_oversized_prompt():
    from contextlib import nullcontext

    class Tokens:
        shape = (1, 500)
        def to(self, _: str):
            return self

    class Tokenizer:
        def __call__(self, text, **kwargs):
            return {"input_ids": Tokens(), "attention_mask": Tokens()}

    class NoCall:
        def generate(self, **kwargs):
            raise AssertionError("Must reject before generating")

    gen = RealCPUGenerator.__new__(RealCPUGenerator)
    gen.torch = type("StubTorch", (), {"inference_mode": staticmethod(nullcontext)})()
    gen.tokenizer = Tokenizer()
    gen.model = NoCall()
    with pytest.raises(BaselineError, match="truncation forbidden"):
        gen.generate("too long", max_input_tokens=384, max_new_tokens=96)
