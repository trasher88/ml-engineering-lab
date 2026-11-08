"""F4.3: invariants, leakage controls, grading arithmetic, Windows CLI regression."""

from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from src.llm_f4.evaluation_set import (
    DOMAINS,
    EvaluationError,
    _ngram_similarity,
    _prompt_key,
    _safe_path,
    _load_case,
    export_artifacts,
    lexical_coverage,
    load_cases,
    load_config,
    model_prompt,
    prepare_artifacts,
    score_manual_grades,
    validate_cases,
    verify_artifacts,
)
from src.llm_f4.instruction_dataset import load_examples, normalized_prompt

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/f4_03_evaluation_set.json"
SOURCE = ROOT / "data/evaluation/f4_03_ru_technical_v1.jsonl"
SCRIPT = ROOT / "scripts/f4_evaluation_set.py"


@pytest.fixture
def conf():
    return load_config(CONFIG, root=ROOT)


@pytest.fixture
def cases(conf):
    return load_cases(conf.source_file)


@pytest.fixture
def instruction_rows(conf):
    return load_examples(conf.instruction_source_file)


def temp_config(tmp_path: Path, modifier):
    obj = json.loads(CONFIG.read_text(encoding="utf-8"))
    modifier(obj)
    file = tmp_path / "config.json"
    file.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
    return file


def temp_source(tmp_path: Path, modifier):
    rows = [json.loads(line) for line in SOURCE.read_text(encoding="utf8").splitlines()]
    modifier(rows)
    path = tmp_path / "eval.jsonl"
    path.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows), encoding="utf8")
    return path


def test_dataset_has_32_balanced_cases_and_64_facts(cases):
    assert len(cases) == 32
    assert {d: sum(c.domain == d for c in cases) for d in DOMAINS} == {d: 8 for d in DOMAINS}
    assert sum(len(c.required_facts) for c in cases) == 64


def test_difficulties_and_unique_topics(cases):
    assert {c.difficulty for c in cases} == {"basic", "intermediate"}
    assert len({(c.domain, c.topic_id) for c in cases}) == 32
    assert len({c.case_id for c in cases}) == 32


def test_frozen_gold_sha256(conf):
    assert sha256(conf.source_file.read_bytes()).hexdigest() == conf.evaluation_source_sha256
    assert conf.evaluation_source_sha256 == "50537bbdbd4c0a1598e4a2cf504f9385d0c07555f4bb75d08667f975488aaf4d"


def test_pinned_preceding_phases_sha256(conf):
    assert sha256(conf.instruction_source_file.read_bytes()).hexdigest() == conf.instruction_source_sha256
    assert sha256(conf.model_catalog_file.read_bytes()).hexdigest() == conf.model_catalog_sha256


def test_no_literal_overlap_with_any_f4_1_split(cases, instruction_rows):
    older = {normalized_prompt(r) for r in instruction_rows}
    assert len(older) == 80
    assert all(_prompt_key(c.instruction, c.input) not in older for c in cases)


def test_all_gold_references_cover_formal_rubric(cases):
    assert all(lexical_coverage(case, case.reference_answer)["matched_count"] == 2 for case in cases)


def test_all_cases_with_validated_contract(cases, instruction_rows, conf):
    validate_cases(cases, conf, instruction_rows)


def test_actual_model_revision_in_manifest(conf):
    manifest = json.loads(prepare_artifacts(conf)["manifest.json"])
    assert manifest["model_id"] == "Qwen/Qwen2.5-0.5B"
    assert manifest["model_revision"] == "060db6499f32faf8b98477b0a26969ef7d8b9987"
    assert manifest["model_evaluated"] is False and manifest["generation_performed"] is False


def test_model_prompts_have_only_three_fields(cases):
    assert all(set(model_prompt(case)) == {"case_id", "domain", "prompt"} for case in cases)


def test_prompts_never_include_gold_or_rubric_fields(conf, cases):
    payload = prepare_artifacts(conf)["prompts.jsonl"]
    records = [json.loads(line) for line in payload.decode("utf-8").splitlines()]
    assert len(records) == 32
    assert all(set(item) == {"case_id", "domain", "prompt"} for item in records)
    assert b"reference_answer" not in payload and b"required_facts" not in payload
    for case, exported in zip(cases, records, strict=True):
        assert exported == model_prompt(case)
        assert case.reference_answer not in exported["prompt"]


def test_context_is_explicitly_delimited(cases):
    case = next(c for c in cases if c.input)
    text = model_prompt(case)["prompt"]
    assert text.endswith(case.input)
    assert "\n\nДанные:\n" in text


def test_no_context_no_empty_separator(cases):
    case = next(c for c in cases if not c.input)
    assert model_prompt(case)["prompt"] == case.instruction


def test_manifest_counts_and_integrity(conf):
    payload = prepare_artifacts(conf)
    meta = json.loads(payload["manifest.json"])
    assert meta["total_cases"] == 32
    assert meta["domain_counts"] == {d: 8 for d in DOMAINS}
    assert meta["difficulty_counts"] == {"basic": 22, "intermediate": 10}
    assert meta["required_facts_total"] == 64
    assert sha256(payload["prompts.jsonl"]).hexdigest() == meta["prompts_sha256"]
    assert sha256(SOURCE.read_bytes()).hexdigest() == meta["source_sha256"]


def test_build_bitwise_deterministic(conf):
    assert prepare_artifacts(conf) == prepare_artifacts(conf)


def test_no_external_ml_imports_during_prepare(conf):
    before = set(sys.modules)
    prepare_artifacts(conf)
    assert not any(name in sys.modules and name not in before for name in ("torch", "transformers", "huggingface_hub"))


def test_export_then_verify_and_modify(tmp_path, conf):
    conf = replace(conf, output_dir=tmp_path)
    content = export_artifacts(conf)
    assert verify_artifacts(conf) == content
    file = tmp_path / "prompts.jsonl"
    file.write_bytes(file.read_bytes() + b"{}\n")
    with pytest.raises(EvaluationError, match="изменён"):
        verify_artifacts(conf)


def test_verify_without_export_fails(tmp_path, conf):
    with pytest.raises(EvaluationError, match="прочитать"):
        verify_artifacts(replace(conf, output_dir=tmp_path))


def test_export_does_not_create_gold_file(tmp_path, conf):
    export_artifacts(replace(conf, output_dir=tmp_path))
    assert sorted(p.name for p in tmp_path.iterdir()) == ["manifest.json", "prompts.jsonl"]


@pytest.mark.parametrize("bad", [
    "../outside", "/tmp/external", "C:\\secret.txt", "x/../../outside", "a//b", "x:y", "./foo", "", "\\\\server\\share",
])
def test_path_traversal_rejected(bad):
    with pytest.raises(EvaluationError):
        _safe_path(ROOT, bad, "example")


@pytest.mark.parametrize("edit", [
    lambda x: x.__setitem__("schema_version", 2),
    lambda x: x.__setitem__("schema_version", True),
    lambda x: x.__setitem__("evaluation_id", "other_name"),
    lambda x: x.__setitem__("expected_cases", 31),
    lambda x: x["domain_quotas"].__setitem__("python", 9),
    lambda x: x.__setitem__("instruction_source_sha256", "ab"),
    lambda x: x.__setitem__("model_catalog_sha256", "ab"),
    lambda x: x.__setitem__("evaluation_source_sha256", "ab"),
    lambda x: x.__setitem__("unsupported", 1),
])
def test_bad_config_rejected(tmp_path, edit):
    path = temp_config(tmp_path, edit)
    with pytest.raises(EvaluationError):
        load_config(path, root=ROOT)


@pytest.mark.parametrize("alter", [
    lambda rows: rows[0].__setitem__("case_id", rows[1]["case_id"]),
    lambda rows: rows[0].__setitem__("topic_id", rows[1]["topic_id"]),
    lambda rows: rows[0].__setitem__("domain", "unknown"),
    lambda rows: rows[0].__setitem__("origin", "copied_from_web"),
    lambda rows: rows[0].__setitem__("difficulty", "extreme"),
    lambda rows: rows[0].__setitem__("instruction", "short"),
    lambda rows: rows[0].__setitem__("reference_answer", ""),
    lambda rows: rows[0].__setitem__("extra", "field"),
    lambda rows: rows[0].__setitem__("rubric", {}),
    lambda rows: rows[0]["rubric"].__setitem__("required_facts", []),
    lambda rows: rows[0]["rubric"]["required_facts"][0].__setitem__("aliases", []),
    lambda rows: rows[0]["rubric"]["required_facts"][0].__setitem__("fact_id", "a!"),
    lambda rows: rows[0]["rubric"]["required_facts"][0].__setitem__("aliases", ["same", "same"]),
    lambda rows: rows[0]["rubric"]["required_facts"][1].__setitem__("fact_id", rows[0]["rubric"]["required_facts"][0]["fact_id"]),
])
def test_malformed_gold_rejected(tmp_path, alter):
    path = temp_source(tmp_path, alter)
    with pytest.raises(EvaluationError):
        load_cases(path)


def test_duplicate_json_key_rejected(tmp_path):
    original = SOURCE.read_text(encoding="utf-8").splitlines()[0]
    bad = original[:-1] + ',"case_id":"ev_py_01"}'
    file = tmp_path / "duplicate_key.jsonl"
    file.write_text(bad + "\n", encoding="utf-8")
    with pytest.raises(EvaluationError, match="Повторный ключ"):
        load_cases(file)


@pytest.mark.parametrize("payload", ["", "{}\n\n", "{invalid}\n", '{"value": NaN}\n'])
def test_invalid_jsonl_rejected(tmp_path, payload):
    file = tmp_path / "bad.jsonl"
    file.write_text(payload, encoding="utf-8")
    with pytest.raises(EvaluationError):
        load_cases(file)


def test_duplicate_case_detection_with_modified_objects(cases, instruction_rows, conf):
    altered = list(cases)
    altered[0] = replace(altered[0], case_id=altered[1].case_id)
    with pytest.raises(EvaluationError, match="Повторные"):
        validate_cases(altered, conf, instruction_rows)


def test_cross_corpus_exact_leak_detected(cases, instruction_rows, conf):
    altered = list(cases)
    altered[0] = replace(altered[0], instruction=instruction_rows[0].instruction, input=instruction_rows[0].input)
    with pytest.raises(EvaluationError, match="Повторная инструкция"):
        validate_cases(altered, conf, instruction_rows)


def test_cross_corpus_near_literal_leak_detected(cases, instruction_rows, conf):
    altered = list(cases)
    old = instruction_rows[0].instruction
    # Slightly changed word; 3-gram similarity well above threshold for long input.
    long = old + " Обязательно перечисли причины и объясни механизм подробно."
    assert _ngram_similarity(old, long) < 0.85  # Дополнительный критерий не ловит всё.
    with pytest.raises(EvaluationError, match="совпадение"):
        validate_cases([replace(altered[0], instruction=old + " ?")]+altered[1:], conf, instruction_rows)


def test_duplicate_prompt_inside_eval_detected(cases, instruction_rows, conf):
    altered = list(cases)
    altered[1] = replace(altered[1], instruction=altered[0].instruction, input=altered[0].input)
    with pytest.raises(EvaluationError, match="Повторная инструкция"):
        validate_cases(altered, conf, instruction_rows)


def test_quota_change_detected(cases, instruction_rows, conf):
    altered = list(cases)
    altered[0] = replace(altered[0], domain="linux", case_id="ev_lx_99")
    with pytest.raises(EvaluationError, match="квота"):
        validate_cases(altered, conf, instruction_rows)


def test_gold_reference_missing_required_term_fails(cases, instruction_rows, conf):
    altered = list(cases)
    altered[0] = replace(altered[0], reference_answer="Этот текст технически пуст и специально не содержит обязательные понятия.")
    with pytest.raises(EvaluationError, match="Эталон"):
        validate_cases(altered, conf, instruction_rows)


def test_lexical_terms_are_not_correctness(cases):
    case = cases[0]
    bad = "Неверно: 2, 4, 6; утверждение 'не включает правую границу' здесь якобы неверно."
    probe = lexical_coverage(case, bad)
    assert probe["matched_count"] == 2
    assert probe["lexical_only"] is True


def test_term_boundary_prevents_substring_false_positive(cases):
    case = cases[0]
    probe = lexical_coverage(case, "Я вижу числа 12, 4, 60, но не утверждаю результат")
    assert probe["matched_count"] < 2


def test_lexical_diagnostic_rejects_non_text(cases):
    with pytest.raises(EvaluationError, match="answer"):
        lexical_coverage(cases[0], None)


def _grades(cases, values=(2, 2)):
    return [{"case_id": c.case_id, "fact_scores": {f.fact_id: values[i] for i, f in enumerate(c.required_facts)},
             "notes": "Экспертная оценка для теста"} for c in cases]


def test_manual_all_2_is_one(cases):
    summary = score_manual_grades(cases, _grades(cases))
    assert summary["macro_over_cases"] == 1
    assert summary["macro_by_domain"] == {d: 1 for d in DOMAINS}
    assert all(v["max_points"] == 4 for v in summary["per_case"].values())


def test_manual_all_zero_is_zero(cases):
    summary = score_manual_grades(cases, _grades(cases, (0, 0)))
    assert summary["macro_over_cases"] == 0
    assert all(v == 0 for v in summary["macro_by_domain"].values())


def test_manual_half_is_half(cases):
    summary = score_manual_grades(cases, _grades(cases, (2, 0)))
    assert summary["macro_over_cases"] == .5
    assert all(v == .5 for v in summary["macro_by_domain"].values())


def test_manual_independent_arithmetic(cases):
    grades = _grades(cases, (1, 2))
    # One case 0, remaining 31 cases 3/4; independent mathematical reference.
    grades[0]["fact_scores"] = {f.fact_id: 0 for f in cases[0].required_facts}
    summary = score_manual_grades(cases, grades)
    assert summary["macro_over_cases"] == (31 * 0.75) / 32
    assert summary["macro_by_domain"]["python"] == 7 * 0.75 / 8
    assert summary["macro_by_domain"]["linux"] == 0.75


@pytest.mark.parametrize("tamper", [
    lambda grades: grades.pop(),
    lambda grades: grades[0].__setitem__("case_id", "unknown"),
    lambda grades: grades[0].__setitem__("case_id", grades[1]["case_id"]),
    lambda grades: grades[0].__setitem__("notes", ""),
    lambda grades: grades[0].__setitem__("fact_scores", {}),
    lambda grades: grades[0]["fact_scores"].__setitem__("extra", 2),
    lambda grades: grades[0]["fact_scores"].__setitem__("items", 3),
    lambda grades: grades[0]["fact_scores"].__setitem__("items", True),
])
def test_bad_manual_grades_rejected(cases, tamper):
    grades = _grades(cases)
    tamper(grades)
    with pytest.raises(EvaluationError):
        score_manual_grades(cases, grades)


@pytest.mark.parametrize("encoding", ["cp1251", "cp866", "cp1252"])
def test_windows_legacy_cli_help(encoding):
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = encoding
    result = subprocess.run([sys.executable, str(SCRIPT), "--help"], cwd=ROOT,
                            capture_output=True, text=True, encoding=encoding, env=env, timeout=30)
    assert result.returncode == 0, result.stderr
    assert "--verify" in result.stdout
    assert "--show-case" in result.stdout


@pytest.mark.parametrize("encoding", ["cp1251", "cp866", "cp1252"])
def test_windows_legacy_cli_build_output(encoding):
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = encoding
    result = subprocess.run([sys.executable, str(SCRIPT)], cwd=ROOT,
                            capture_output=True, text=True, encoding=encoding, env=env, timeout=30)
    assert result.returncode == 0, result.stderr
    assert "F4.3 BUILD: PASS" in result.stdout
    assert "NOT RUN" in result.stdout


def test_cli_verify_and_expert_case():
    result = subprocess.run([sys.executable, str(SCRIPT), "--verify", "--show-case", "ev_py_01"], cwd=ROOT,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert "F4.3 VERIFY: PASS" in result.stdout
    assert "EXPERT-ONLY CASE PREVIEW" in result.stdout
    assert "reference_answer" in result.stdout


def test_cli_unknown_case_error_clean():
    result = subprocess.run([sys.executable, str(SCRIPT), "--show-case", "wrong"], cwd=ROOT,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 1
    assert "F4.3: FAIL" in result.stderr
    assert "Traceback" not in result.stderr


def test_docs_have_nine_sections_and_learning_files():
    contents = (ROOT / "docs/steps/F4_03_evaluation_set.md").read_text(encoding="utf-8")
    for n in range(1, 10):
        assert f"## {n}." in contents
    assert "Файлы этапа" in contents
    assert "score_manual_grades" in contents
    assert "PowerShell" in contents
