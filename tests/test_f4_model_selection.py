"""F4.2: независимые числовые эталоны и регрессия спецификации выбора."""

from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from src.llm_f4.model_selection import (
    ModelSelectionError,
    assess_candidate,
    count_parameters,
    create_report,
    kv_cache_bytes,
    load_catalog,
    probe_tokenizer,
    report_bytes,
    resource_estimate,
    runtime_report,
    verify_report,
    write_report,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/f4_02_model_selection.json"
SCRIPT = ROOT / "scripts/f4_model_selection.py"


@pytest.fixture
def catalog():
    return load_catalog(CONFIG)


def _edit_config(tmp_path: Path, edit) -> Path:
    data = json.loads(CONFIG.read_text(encoding="utf-8"))
    edit(data)
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return path


def test_catalog_contains_two_pinned_distinct_checkpoints(catalog):
    assert len(catalog.candidates) == 2
    assert len({x.model_id for x in catalog.candidates}) == 2
    assert all(len(x.revision) == 40 and x.revision == x.tokenizer_revision for x in catalog.candidates)


def test_selected_base_model_and_license(catalog):
    selected = next(x for x in catalog.candidates if x.model_id == catalog.selected_model_id)
    assert selected.model_id == "Qwen/Qwen2.5-0.5B"
    assert selected.model_type == "qwen2"
    assert selected.architecture == "Qwen2ForCausalLM"
    assert selected.license == "apache-2.0"
    assert selected.stage == "base" and selected.task == "causal_lm"
    assert "ru" in selected.languages


def test_optional_smollm_is_not_russian_reference(catalog):
    other = next(x for x in catalog.candidates if x.model_id != catalog.selected_model_id)
    result = assess_candidate(other, catalog.requirements)
    assert not result["eligible"]
    assert result["reasons"] == ["russian_not_declared"]


def test_parameter_count_is_exact_for_selected_architecture(catalog):
    selected = catalog.candidates[0]
    # Embeddings + 24 * (Q/K/V/O + biases + 3 SwiGLU matrices + 2 RMSNorm) + final RMSNorm.
    h, i, v, l = 896, 4864, 151936, 24
    kv_width = 2 * (h // 14)
    attention = 2 * h * h + 2 * h * kv_width
    biases = h + 2 * kv_width
    expected = v * h + l * (attention + biases + 3 * h * i + 2 * h) + h
    assert expected == 494_032_768
    assert count_parameters(selected) == expected


def test_smollm_parameter_count_matches_reported(catalog):
    other = catalog.candidates[1]
    assert count_parameters(other) == 134_515_008
    assert other.published_parameters_approx == 134_515_008


def test_kv_cache_independent_numerical_reference(catalog):
    candidate = catalog.candidates[0]
    assert kv_cache_bytes(candidate, sequence_length=512, batch_size=1, bytes_per_value=4) == 12_582_912
    assert kv_cache_bytes(candidate, sequence_length=256, batch_size=2, bytes_per_value=4) == 12_582_912
    assert kv_cache_bytes(candidate, sequence_length=512, batch_size=1, bytes_per_value=2) == 6_291_456


def test_weights_memory_reference(catalog):
    estimate = resource_estimate(catalog.candidates[0], catalog.requirements)
    assert estimate["fp32_weights_mib"] == round(494_032_768 * 4 / 2**20, 3)
    assert estimate["kv_cache_fp32_mib"] == 12.0
    assert estimate["ideal_int4_weights_mib"] == round(494_032_768 / 2 / 2**20, 3)


def test_memory_is_lower_bound_not_ram_measurement(catalog):
    estimate = resource_estimate(catalog.candidates[0], catalog.requirements)
    assert "exclude activations" in estimate["warning"]
    assert estimate["weights_plus_kv_fp32_mib"] > estimate["fp32_weights_mib"]


def test_selected_meets_hard_conditions(catalog):
    assessments = [assess_candidate(item, catalog.requirements) for item in catalog.candidates]
    assert [item["eligible"] for item in assessments] == [True, False]


def test_context_window_guard_raises(catalog):
    candidate = catalog.candidates[0]
    with pytest.raises(ModelSelectionError, match="context"):
        kv_cache_bytes(candidate, sequence_length=32769, batch_size=1, bytes_per_value=4)


@pytest.mark.parametrize("values", [
    {"sequence_length": 0, "batch_size": 1, "bytes_per_value": 4},
    {"sequence_length": 1, "batch_size": 0, "bytes_per_value": 4},
    {"sequence_length": 1, "batch_size": True, "bytes_per_value": 4},
    {"sequence_length": 1, "batch_size": 1, "bytes_per_value": 0},
])
def test_kv_cache_rejects_invalid_dimensions(catalog, values):
    with pytest.raises(ModelSelectionError):
        kv_cache_bytes(catalog.candidates[0], **values)


def test_context_beyond_model_is_rejected_not_crashed(catalog):
    requirements = replace(catalog.requirements, sequence_length=40000)
    result = assess_candidate(catalog.candidates[0], requirements)
    assert not result["eligible"]
    assert "context_too_short" in result["reasons"]


def test_tiny_cpu_budget_rejects_large_checkpoint(catalog):
    requirements = replace(catalog.requirements, cpu_ram_budget_gib=1)
    result = assess_candidate(catalog.candidates[0], requirements)
    assert "weight_kv_lower_bound_exceeds_cpu_budget" in result["reasons"]


def test_report_contains_explanations_and_is_metadata_only(catalog):
    report = create_report(catalog, runtime={"ready_for_optional_probe": False, "issues": ["transformers_missing"]})
    assert report["status"] == "METADATA_SELECTED_ONLY"
    assert report["selected_model"]["tokenizer_revision"] == catalog.candidates[0].revision
    assert report["catalog_sha256"] == catalog.sha256
    assert report["weights_loaded"] is False
    assert report["inference_run"] is False and report["training_run"] is False
    assert all(item["resources"] for item in report["candidate_assessments"])


def test_report_rejects_ineligible_preference(catalog):
    bad = replace(catalog, selected_model_id="HuggingFaceTB/SmolLM2-135M")
    with pytest.raises(ModelSelectionError, match="fails constraints"):
        create_report(bad)


def test_report_roundtrip_and_tamper_detection(catalog, tmp_path):
    report = create_report(catalog, runtime={"ready_for_optional_probe": True, "issues": []})
    path = tmp_path / "model_selection_report.json"
    write_report(path, report)
    verify_report(path, report)
    assert path.read_bytes() == report_bytes(report)
    path.write_bytes(path.read_bytes().replace(b"METADATA_SELECTED_ONLY", b"FAKE_TRAINING_RUN___"))
    with pytest.raises(ModelSelectionError, match="Report mismatch"):
        verify_report(path, report)


def test_report_missing_file_is_rejected(catalog, tmp_path):
    with pytest.raises(ModelSelectionError, match="Cannot read"):
        verify_report(tmp_path / "missing.json", create_report(catalog))


def test_runtime_python_313_supported(catalog):
    state = runtime_report(catalog.requirements, python_version=(3, 13, 2),
                           packages={"torch": "2.10.0+cpu", "transformers": "4.57.2"})
    assert state["ready_for_optional_probe"] is True
    assert state["issues"] == []


def test_runtime_missing_libraries_is_nonfatal(catalog):
    state = runtime_report(catalog.requirements, python_version=(3, 13, 2),
                           packages={"torch": None, "transformers": None})
    assert state["issues"] == ["torch_missing", "transformers_missing"]


def test_runtime_old_library_is_reported(catalog):
    state = runtime_report(catalog.requirements, python_version=(3, 12, 1),
                           packages={"torch": "2.5.1", "transformers": "4.36.2"})
    assert set(state["issues"]) == {"python_too_old", "torch_too_old", "transformers_too_old"}


@pytest.mark.parametrize("issue,change", [
    ("schema", lambda d: d.update(schema_version=2)),
    ("unknown_field", lambda d: d.update(unexpected="value")),
    ("no_candidates", lambda d: d.update(candidates=[])),
    ("selected_missing", lambda d: d.update(selected_model_id="Unknown/model")),
    ("duplicate_ids", lambda d: d["candidates"].append(dict(d["candidates"][0]))),
    ("revision_not_pinned", lambda d: d["candidates"][0].update(revision="main")),
    ("tokenizer_revision_drift", lambda d: d["candidates"][0].update(tokenizer_revision="a" * 40)),
    ("wrong_tokenizer", lambda d: d["candidates"][0].update(tokenizer_id="Fake/tokenizer")),
    ("wrong_architecture", lambda d: d["candidates"][0].update(architecture="GPT2LMHeadModel")),
    ("wrong_heads", lambda d: d["candidates"][0].update(num_attention_heads=15)),
    ("wrong_kv", lambda d: d["candidates"][0].update(num_key_value_heads=5)),
    ("bad_eos", lambda d: d["candidates"][0].update(eos_token_id=999999)),
    ("false_integer", lambda d: d["candidates"][0].update(num_hidden_layers=True)),
    ("fake_params", lambda d: d["candidates"][0].update(published_parameters_approx=5)),
    ("wrong_config_source", lambda d: d["candidates"][0].update(source_config="https://huggingface.co/example")),
    ("float32_mandatory", lambda d: d["requirements"].update(cpu_dtype="float16")),
    ("bad_ratio", lambda d: d["requirements"].update(sequence_length=0)),
    ("bad_version", lambda d: d["requirements"].update(minimum_torch="broken")),
    ("wrong_language", lambda d: d["requirements"].update(required_language="en")),
    ("unknown_license", lambda d: d["requirements"].update(allowed_licenses=[])),
    ("model_id_type", lambda d: d["candidates"][0].update(model_id=123)),
    ("model_type_type", lambda d: d["candidates"][0].update(model_type=[])),
    ("version_type", lambda d: d["requirements"].update(minimum_python=None)),
])
def test_invalid_catalog_rejected(tmp_path, issue, change):
    path = _edit_config(tmp_path, change)
    with pytest.raises(ModelSelectionError):
        load_catalog(path)


def test_malformed_json_is_rejected(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{", encoding="utf-8")
    with pytest.raises(ModelSelectionError):
        load_catalog(path)


def test_tokenizer_probe_is_never_implicit(catalog):
    report = create_report(catalog)
    assert report["tokenizer_loaded"] is False


def test_optional_tokenizer_fake_hf_dependency(catalog, monkeypatch):
    candidate = catalog.candidates[0]
    args = {}

    class FakeTokenizer:
        eos_token_id = candidate.eos_token_id

        def __len__(self):
            return candidate.vocab_size - 5

        def encode(self, text, add_special_tokens):
            assert add_special_tokens is False
            return [11, 12, 13]

        def decode(self, ids, *, skip_special_tokens, clean_up_tokenization_spaces):
            assert ids == [11, 12, 13]
            return "Тест tokenizer"

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            args.update(kwargs)
            assert model_id == candidate.tokenizer_id
            return FakeTokenizer()

    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoTokenizer=FakeAutoTokenizer))
    result = probe_tokenizer(candidate, "Тест tokenizer")
    assert result["token_count"] == 3 and result["roundtrip"] is True
    assert args == {"revision": candidate.revision, "local_files_only": True,
                    "trust_remote_code": False, "use_fast": True}
    probe_tokenizer(candidate, "Тест tokenizer", allow_download=True)
    assert args["local_files_only"] is False


def test_optional_tokenizer_fails_on_wrong_roundtrip(catalog, monkeypatch):
    candidate = catalog.candidates[0]

    class BadTokenizer:
        eos_token_id = candidate.eos_token_id

        def __len__(self):
            return candidate.vocab_size

        def encode(self, *_args, **_kwargs):
            return [9]

        def decode(self, *_args, **_kwargs):
            return "different"

    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **kw: BadTokenizer())
    ))
    with pytest.raises(ModelSelectionError, match="round-trip"):
        probe_tokenizer(candidate, "original")


@pytest.mark.parametrize("encoding", ["utf-8", "cp1251", "cp866", "cp1252"])
def test_cli_help_in_windows_codepages(encoding):
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = encoding
    call = subprocess.run([sys.executable, str(SCRIPT), "--help"], cwd=ROOT,
                          env=env, text=True, capture_output=True, check=True)
    assert "--probe-tokenizer" in call.stdout
    assert "--allow-download" in call.stdout


def test_cli_default_does_not_load_model():
    call = subprocess.run([sys.executable, str(SCRIPT)], cwd=ROOT,
                          text=True, encoding="utf-8", capture_output=True, check=True)
    assert "F4.2 MODEL SELECTION: PASS" in call.stdout
    assert "Inference / model training: NOT RUN" in call.stdout
    assert "weights_loaded" not in call.stdout


def test_cli_rejects_download_without_probe():
    call = subprocess.run([sys.executable, str(SCRIPT), "--allow-download"], cwd=ROOT,
                          text=True, capture_output=True)
    assert call.returncode != 0
    assert b"requires --probe-tokenizer" in call.stderr.encode("utf-8")


def test_docs_cover_nine_sections():
    content = (ROOT / "docs/steps/F4_02_model_selection.md").read_text(encoding="utf-8")
    for heading in (
        "## 1. Цель этапа", "## 2. Что реализовано", "## 3. Последовательность реализации",
        "## 4. Алгоритм и математический пример", "## 5. Схема работы",
        "## 6. Локальная проверка PowerShell", "## 7. Что доказано тестами",
        "## 8. Основная мысль этапа", "## 9. Файлы, с которыми стоит познакомиться",
    ):
        assert heading in content
