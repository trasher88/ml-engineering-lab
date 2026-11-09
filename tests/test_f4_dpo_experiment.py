"""F4.8: числовые эталоны DPO, leakage, provenance, Windows CLI и артефакты."""
from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
from pathlib import Path
import json
import math
import os
import subprocess
import sys

import pytest

from src.llm_f4.dpo_experiment import (
    Config, DPOError, Preference, TokenizedPair, _atomic, _canonical,
    _json, _path, dpo_loss, load_config, load_preferences,
    make_plan, tokenize_pair, verify_run,
)

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "configs/f4_08_dpo_experiment.json"
DATA = ROOT / "data/preferences/f4_08_dpo_pairs_v1.jsonl"


class TinyTokenizer:
    eos_token_id = 500
    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        return [10 + (ord(c) % 300) for c in text]


@pytest.fixture(scope="module")
def conf() -> Config:
    return load_config(CFG, root=ROOT)


@pytest.fixture(scope="module")
def checked_plan(conf):
    return make_plan(conf, root=ROOT)


def write_config(tmp_path: Path, patch: dict) -> Path:
    value = json.loads(CFG.read_text(encoding="utf-8"))
    value.update(patch)
    dest = tmp_path / "conf.json"
    dest.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    return dest


def rows() -> list[dict]:
    return [json.loads(t) for t in DATA.read_text(encoding="utf-8").splitlines()]


def write_rows(tmp_path: Path, entries: list[dict]) -> Path:
    dest = tmp_path / "pairs.jsonl"
    dest.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in entries), encoding="utf-8")
    return dest


def test_plan_contract(checked_plan):
    plan, entries = checked_plan
    assert plan["split_counts"] == {"train": 16, "validation": 4}
    assert len(entries["train"]) == 16 and len(entries["validation"]) == 4
    assert not plan["training_performed"]
    assert plan["f43_used_for_training"] is False
    assert plan["f47_human_grades_used_for_training"] is False


def test_paired_domains_balanced(checked_plan):
    plan, _ = checked_plan
    assert plan["split_domains"]["train"] == {"python": 4, "linux": 4, "sql": 4, "ml": 4}
    assert plan["split_domains"]["validation"] == {"python": 1, "linux": 1, "sql": 1, "ml": 1}


def test_reference_is_frozen_sft(checked_plan):
    plan, _ = checked_plan
    assert plan["reference"] == "frozen_verified_f45_sft_adapter_before_dpo"
    assert plan["model_id"] == "Qwen/Qwen2.5-0.5B"
    assert plan["revision"] == "060db6499f32faf8b98477b0a26969ef7d8b9987"


def test_no_group_leak(checked_plan):
    _, splits = checked_plan
    assert {p.group_id for p in splits["train"]}.isdisjoint({p.group_id for p in splits["validation"]})


def test_source_ids_are_unique(checked_plan):
    _, splits = checked_plan
    all_rows = splits["train"] + splits["validation"]
    assert len({row.source_id for row in all_rows}) == len(all_rows)


def test_hashes_recorded(conf, checked_plan):
    plan, _ = checked_plan
    assert plan["preference_sha256"] == sha256(DATA.read_bytes()).hexdigest()
    assert plan["config_sha256"] == sha256(CFG.read_bytes()).hexdigest()
    assert len(plan["f43_holdout_sha256"]) == 64


@pytest.mark.parametrize("invalid", ["../secret", "a/../b", "C:/project", "a\\b", "/etc/passwd", "a//b", "", "./config", "a/./b"])
def test_reject_unsafe_paths(invalid):
    with pytest.raises(DPOError):
        _path(ROOT, invalid, "test")


@pytest.mark.parametrize("patch", [
    {"schema_version": 2}, {"schema_version": True}, {"seed": -1}, {"seed": "42"},
    {"sft_profile": "other"}, {"beta": 0}, {"beta": -1}, {"beta": 11},
    {"max_total_tokens": 5}, {"max_prompt_tokens": 600}, {"cpu_threads": 0},
    {"output_dir": "../../bad"}, {"dataset_file": "C:/secret"}, {"profiles": {}},
    {"profiles": {"cpu": {}, "gpu": {}}},
    {"profiles": {"cpu": {"max_steps": 0, "train_pairs": 1, "validation_pairs": 1, "learning_rate": 1e-5},
                   "gpu": {"max_steps": 12, "train_pairs": 16, "validation_pairs": 4, "learning_rate": 1e-5}}},
])
def test_invalid_config(tmp_path, patch):
    with pytest.raises((DPOError, ValueError)):
        load_config(write_config(tmp_path, patch), root=ROOT)


@pytest.mark.parametrize("bad", [b"{", b'{"a":1,"a":2}', b'{"x":NaN}', b"\xff"])
def test_invalid_json_rejected(bad):
    with pytest.raises(DPOError):
        _json(bad, "case")


def test_empty_dataset_rejected(tmp_path):
    p = tmp_path / "empty.jsonl"; p.write_bytes(b"")
    with pytest.raises(DPOError, match="пуст"):
        load_preferences(p)


def test_empty_line_rejected(tmp_path):
    p = tmp_path / "bad.jsonl"; p.write_text("\n", encoding="utf-8")
    with pytest.raises(DPOError, match="Пустая"):
        load_preferences(p)


@pytest.mark.parametrize("patch", [
    {"origin": "generated_llm"}, {"split": "test"}, {"chosen": "неверный ответ"},
    {"id": "bad space"}, {"domain": 123}, {"prompt": "x"},
])
def test_invalid_preference_rows(tmp_path, patch):
    data = rows(); data[0] = {**data[0], **patch}
    if patch == {"chosen": "неверный ответ"}:
        # Структурно другой ответ допустим, но несоответствие chosen источнику ловит make_plan.
        assert load_preferences(write_rows(tmp_path, data))
    else:
        with pytest.raises(DPOError):
            load_preferences(write_rows(tmp_path, data))


def test_duplicate_preference_id(tmp_path):
    data = rows(); data[1]["id"] = data[0]["id"]
    with pytest.raises(DPOError, match="Повтор"):
        load_preferences(write_rows(tmp_path, data))


def test_duplicate_source_id(tmp_path):
    data = rows(); data[1]["source_id"] = data[0]["source_id"]
    with pytest.raises(DPOError, match="Повтор"):
        load_preferences(write_rows(tmp_path, data))


def test_duplicate_prompt(tmp_path):
    data = rows(); data[1]["prompt"] = data[0]["prompt"]
    with pytest.raises(DPOError, match="Повтор"):
        load_preferences(write_rows(tmp_path, data))


def test_chosen_equals_rejected(tmp_path):
    data = rows(); data[0]["rejected"] = data[0]["chosen"]
    with pytest.raises(DPOError, match="совпада"):
        load_preferences(write_rows(tmp_path, data))


def test_group_split_mismatch(tmp_path):
    data = rows(); data[-1]["group_id"] = data[0]["group_id"]
    with pytest.raises(DPOError, match="группа"):
        load_preferences(write_rows(tmp_path, data))


@pytest.mark.parametrize("patch", [
    {"chosen": "Фиктивная подмена ответа F4.1, которая должна быть обнаружена планом."},
    {"prompt": "Вопрос:\nПодмена.\nОтвет:\n"},
    {"group_id": "f4g999"},
    {"source_id": "f4q999_a"},
    {"split": "validation"},
])
def test_plan_rejects_tampered_pair(tmp_path, conf, patch):
    data = rows(); data[0] = {**data[0], **patch}
    target = write_rows(tmp_path, data)
    with pytest.raises(DPOError):
        make_plan(replace(conf, dataset_file=target), root=ROOT)


def test_case_insensitive_equal_answers_rejected(tmp_path):
    data = rows(); data[0]["rejected"] = data[0]["chosen"].upper()
    with pytest.raises(DPOError):
        load_preferences(write_rows(tmp_path, data))


@pytest.mark.parametrize("beta", [0.01, 0.1, 0.5, 1.0])
def test_dpo_equal_policy_reference_ln2(beta):
    loss, margin, prob = dpo_loss(-3.0, -5.0, -3.0, -5.0, beta)
    assert loss == pytest.approx(math.log(2), abs=1e-12)
    assert margin == pytest.approx(0.0)
    assert prob == pytest.approx(0.5)


def test_dpo_positive_margin_reduces_loss():
    before, _, _ = dpo_loss(-2.0, -3.0, -2.5, -3.0, 0.1)
    after, margin, p = dpo_loss(-1.0, -3.0, -2.5, -3.0, 0.1)
    assert after < before and margin > 0 and p > 0.5


def test_dpo_negative_margin_increases_loss():
    _, margin, p = dpo_loss(-4.0, -3.0, -2.5, -3.0, 0.1)
    assert margin < 0 and p < 0.5


def test_dpo_extreme_margins_finite():
    for v in (10000, -10000):
        loss, margin, prob = dpo_loss(v, 0.0, 0.0, 0.0, 1.0)
        assert math.isfinite(loss) and math.isfinite(prob) and margin == v


@pytest.mark.parametrize("values", [
    (math.nan, 0, 0, 0, .1), (0, math.inf, 0, 0, .1),
    (0, 0, 0, 0, 0), (0, 0, 0, 0, -1),
    (0, 0, 0, 0, math.nan), (0, 0, 0, 0, "beta"),
])
def test_dpo_rejects_bad_inputs(values):
    with pytest.raises(DPOError):
        dpo_loss(*values)


def test_tokenizer_masks_prompt_and_includes_eos(conf, checked_plan):
    _, split = checked_plan
    out = tokenize_pair(split["train"][0], TinyTokenizer(), conf)
    assert out.prompt_tokens > 0
    assert all(n == -100 for n in out.chosen_labels[:out.prompt_tokens])
    assert all(n == -100 for n in out.rejected_labels[:out.prompt_tokens])
    assert out.chosen_ids[-1] == out.rejected_ids[-1] == 500
    assert out.chosen_labels[-1] == out.rejected_labels[-1] == 500
    assert out.chosen_ids[:out.prompt_tokens] == out.rejected_ids[:out.prompt_tokens]
    assert len(out.chosen_ids) == len(out.chosen_labels)
    assert len(out.rejected_ids) == len(out.rejected_labels)


def test_tokenizer_rejects_truncation(conf, checked_plan):
    _, split = checked_plan
    with pytest.raises(DPOError, match="бюджет"):
        tokenize_pair(split["train"][0], TinyTokenizer(), replace(conf, max_prompt_tokens=5))


def test_tokenizer_rejects_no_eos(conf, checked_plan):
    _, split = checked_plan
    t = TinyTokenizer(); t.eos_token_id = None
    with pytest.raises(DPOError):
        tokenize_pair(split["train"][0], t, conf)


def test_torch_dpo_matches_scalar():
    torch = pytest.importorskip("torch")
    from src.llm_f4.dpo_experiment import torch_dpo_loss
    a = [-1.0, -2.0, -1.5, -2.5]
    ref_loss, _, _ = dpo_loss(*a, 0.1)
    loss = torch_dpo_loss(*[torch.tensor(x) for x in a], beta=.1)
    assert loss.item() == pytest.approx(ref_loss, abs=1e-6)


def test_torch_dpo_gradients_correct_direction():
    torch = pytest.importorskip("torch")
    from src.llm_f4.dpo_experiment import torch_dpo_loss
    pc = torch.tensor(-2.0, requires_grad=True)
    pr = torch.tensor(-3.0, requires_grad=True)
    loss = torch_dpo_loss(pc, pr, torch.tensor(-2.5), torch.tensor(-3.0), beta=.1)
    loss.backward()
    assert pc.grad.item() < 0 and pr.grad.item() > 0


def test_torch_completion_logprob_respects_mask():
    torch = pytest.importorskip("torch")
    from src.llm_f4.dpo_experiment import completion_logprob
    class Mock:
        def __call__(self, *, input_ids, attention_mask):
            vocab = 9
            logits = torch.zeros(1, input_ids.shape[1], vocab)
            return type("Result", (), {"logits": logits})()
    # Одинаковые logits: log(1/9) за каждый токен ответа, включая EOS.
    outcome = completion_logprob(Mock(), [1, 2, 3, 4], [-100, -100, 3, 4], device="cpu", torch=torch)
    assert outcome.item() == pytest.approx(-2 * math.log(9), abs=1e-6)


def test_torch_completion_logprob_bad_mask():
    torch = pytest.importorskip("torch")
    from src.llm_f4.dpo_experiment import completion_logprob
    with pytest.raises(DPOError):
        completion_logprob(None, [1, 2, 3], [-100, -100, -100], device="cpu", torch=torch)


def test_atomic_refuses_overwrite(tmp_path):
    p = tmp_path / "x.json"
    _atomic(p, b"original")
    with pytest.raises(DPOError):
        _atomic(p, b"changed")
    assert p.read_bytes() == b"original"


def test_manifest_refuse_when_missing(conf):
    with pytest.raises(DPOError):
        verify_run(conf, root=ROOT, profile="unknown")


def test_plan_has_no_training_side_effects(conf, tmp_path):
    alternate = replace(conf, output_dir=tmp_path / "planned")
    make_plan(alternate, root=ROOT)
    assert not alternate.output_dir.exists()


def test_cli_help_without_ml_dependencies():
    result = subprocess.run([sys.executable, str(ROOT / "scripts/f4_dpo_experiment.py"), "--help"],
                            cwd=ROOT, capture_output=True, text=True, check=True)
    assert "--plan" in result.stdout and "--run" in result.stdout


@pytest.mark.parametrize("encoding", ["cp1251", "cp866", "cp1252"])
def test_cli_help_windows_encodings(encoding):
    env = dict(os.environ); env["PYTHONIOENCODING"] = encoding
    result = subprocess.run([sys.executable, str(ROOT / "scripts/f4_dpo_experiment.py"), "--help"],
                            cwd=ROOT, env=env, capture_output=True, check=True)
    assert b"--math-demo" in result.stdout


def test_cli_plan_without_ml_imports():
    result = subprocess.run([sys.executable, str(ROOT / "scripts/f4_dpo_experiment.py"), "--plan"],
                            cwd=ROOT, capture_output=True, text=True, check=True)
    assert "F4.8 PLAN: PASS" in result.stdout
    assert "NOT RUN" in result.stdout


def test_cli_math_demo():
    result = subprocess.run([sys.executable, str(ROOT / "scripts/f4_dpo_experiment.py"), "--math-demo"],
                            cwd=ROOT, capture_output=True, text=True, check=True)
    assert "F4.8 MATH DEMO: PASS" in result.stdout


def test_cli_torch_smoke():
    pytest.importorskip("torch")
    result = subprocess.run([sys.executable, str(ROOT / "scripts/f4_dpo_experiment.py"), "--torch-smoke"],
                            cwd=ROOT, capture_output=True, text=True, check=True)
    assert "SYNTHETIC, NO LLM TRAINING" in result.stdout

@pytest.fixture
def fake_real_artifacts(tmp_path, conf, checked_plan, monkeypatch):
    """Искусственный манифест для проверки VERIFY, не результат реального DPO."""
    from types import SimpleNamespace
    from src.llm_f4 import dpo_experiment as module
    plan, parts = checked_plan
    conf = replace(conf, output_dir=tmp_path / "f48")
    sft_root = tmp_path / "f45"
    sft_cpu = sft_root / "cpu"
    sft_cpu.mkdir(parents=True)
    sft_manifest_bytes = b'{"unit_test_only":true}\n'
    (sft_cpu / "run_manifest.json").write_bytes(sft_manifest_bytes)
    fake_hash = {"adapter_config.json": "a" * 64, "adapter_model.safetensors": "b" * 64}
    monkeypatch.setattr(module, "make_plan", lambda *_a, **_kw: (plan, parts))
    monkeypatch.setattr(module, "load_sft_config", lambda *_a, **_kw: SimpleNamespace(output_dir=sft_root))
    monkeypatch.setattr(module, "verify_sft", lambda *_a, **_kw: {"adapter_files_sha256": fake_hash})
    directory = conf.output_dir / "cpu"
    adapter = directory / "adapter"
    adapter.mkdir(parents=True)
    payloads = {"adapter_config.json": b"fake config test", "adapter_model.safetensors": b"fake weights test"}
    for name, data in payloads.items():
        (adapter / name).write_bytes(data)
    metrics = _canonical({"step": 1, "pair_id": "dpf4q001_a", "loss": math.log(2),
                          "preclip_gradient_norm": 0.1, "margin_pre_step": 0.0,
                          "chosen_win_probability_pre_step": 0.5})
    (directory / "metrics.jsonl").write_bytes(metrics)
    manifest = {
        "schema_version": 1, "stage": "F4.8", "profile": "cpu",
        "real_dpo_performed": True, "optimizer_steps": 1,
        "config_sha256": conf.config_sha256,
        "preference_sha256": plan["preference_sha256"],
        "f41_train_sha256": plan["f41_train_sha256"],
        "f41_validation_sha256": plan["f41_validation_sha256"],
        "f43_holdout_sha256": plan["f43_holdout_sha256"],
        "model_revision": plan["revision"],
        "model_id": plan["model_id"],
        "beta": conf.beta,
        "train_pairs_used": conf.profiles["cpu"].train_pairs,
        "validation_pairs_used": conf.profiles["cpu"].validation_pairs,
        "source_sft_adapter_sha256": fake_hash,
        "source_sft_manifest_sha256": sha256(sft_manifest_bytes).hexdigest(),
        "reference": plan["reference"], "reference_frozen": True,
        "reference_logprobs_precomputed_before_updates": True,
        "metrics_sha256": sha256(metrics).hexdigest(),
        "adapter_files_sha256": {name: sha256(data).hexdigest() for name, data in payloads.items()},
        "manual_quality_score": None, "quality_improvement_proven": False,
    }
    (directory / "run_manifest.json").write_bytes(_canonical(manifest))
    return conf, directory, manifest


def test_artifact_verifier_accepts_consistent_test_double(fake_real_artifacts):
    conf, _, _ = fake_real_artifacts
    # Ожидается PASS только для намеренно созданной test-double структуры.
    assert verify_run(conf, root=ROOT, profile="cpu")["optimizer_steps"] == 1


@pytest.mark.parametrize("change", [
    {"real_dpo_performed": False},
    {"optimizer_steps": 2},
    {"reference_frozen": False},
    {"reference_logprobs_precomputed_before_updates": False},
    {"preference_sha256": "0" * 64},
    {"quality_improvement_proven": True},
    {"manual_quality_score": 0.9},
    {"source_sft_manifest_sha256": "0" * 64},
])
def test_artifact_verifier_rejects_tampered_manifest(fake_real_artifacts, change):
    conf, directory, manifest = fake_real_artifacts
    manifest.update(change)
    (directory / "run_manifest.json").write_bytes(_canonical(manifest))
    with pytest.raises(DPOError):
        verify_run(conf, root=ROOT, profile="cpu")


def test_artifact_verifier_rejects_changed_metrics(fake_real_artifacts):
    conf, directory, _ = fake_real_artifacts
    (directory / "metrics.jsonl").write_bytes(b'{"step":1,"loss":999}\n')
    with pytest.raises(DPOError, match="метрик"):
        verify_run(conf, root=ROOT, profile="cpu")


def test_artifact_verifier_rejects_changed_weights(fake_real_artifacts):
    conf, directory, _ = fake_real_artifacts
    (directory / "adapter" / "adapter_model.safetensors").write_bytes(b"tampered")
    with pytest.raises(DPOError, match="Повреждён"):
        verify_run(conf, root=ROOT, profile="cpu")


def test_real_run_control_flow_with_tiny_torch_test_double(tmp_path, conf, monkeypatch):
    """Синтетическая модель проверяет весь runtime, но НЕ является Qwen/DPO evidence."""
    torch = pytest.importorskip("torch")
    from types import SimpleNamespace
    from src.llm_f4 import dpo_experiment as module

    torch.set_num_threads(1)

    class ToyPeftModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lora_A = torch.nn.Parameter(torch.tensor(0.1))
            self.register_buffer("vocab_bias", torch.linspace(-1.0, 1.0, 600))
            self.config = SimpleNamespace(use_cache=True)

        def forward(self, *, input_ids, attention_mask):
            vocab_logits = self.vocab_bias * self.lora_A
            logits = vocab_logits[None, None, :].expand(1, input_ids.shape[1], 600)
            return SimpleNamespace(logits=logits)

        def save_pretrained(self, where, *, safe_serialization):
            assert safe_serialization is True
            target = Path(where)
            (target / "adapter_config.json").write_bytes(b'{"unit_test_only":true}')
            (target / "adapter_model.safetensors").write_bytes(b"tiny fake tensor bytes")

    sft_root = tmp_path / "f45"
    (sft_root / "cpu").mkdir(parents=True)
    (sft_root / "cpu" / "run_manifest.json").write_bytes(b'{"mock_sft":true}\n')
    fake_sft_hashes = {"adapter_config.json": "a" * 64, "adapter_model.safetensors": "b" * 64}
    initial_plan = module.make_plan(conf, root=ROOT)
    monkeypatch.setattr(module, "make_plan", lambda *_a, **_kw: initial_plan)
    monkeypatch.setattr(module, "load_sft_config", lambda *_a, **_kw: SimpleNamespace(output_dir=sft_root))
    monkeypatch.setattr(module, "verify_sft", lambda *_a, **_kw: {"adapter_files_sha256": fake_sft_hashes})
    monkeypatch.setattr(module, "_model_and_tokenizer", lambda *_a, **_kw: (
        ToyPeftModel(), TinyTokenizer(), torch,
        {"trainable_parameters": 1, "adapter_files_sha256": fake_sft_hashes},
        {"torch": torch.__version__, "transformers": "TEST_DOUBLE", "peft": "TEST_DOUBLE"}))

    alternate = replace(conf, output_dir=tmp_path / "f48")
    manifest = module.run_real(alternate, root=ROOT, profile="cpu")
    assert manifest["real_dpo_performed"] is True
    assert manifest["optimizer_steps"] == 1
    assert manifest["reference_frozen"] is True
    assert manifest["initial_validation_margin"] == pytest.approx(0.0, abs=1e-6)
    assert manifest["quality_improvement_proven"] is False
    assert module.verify_run(alternate, root=ROOT, profile="cpu")["optimizer_steps"] == 1
    assert (alternate.output_dir / "cpu" / "adapter" / "adapter_model.safetensors").is_file()
    # Повторный run не должен перезаписывать зафиксированный эксперимент.
    with pytest.raises(DPOError, match="перезапись"):
        module.run_real(alternate, root=ROOT, profile="cpu")
