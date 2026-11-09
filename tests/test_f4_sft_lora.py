"""F4.5: числовые эталоны, data contracts, CLI и сохранение real-training отчётов."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import os
import subprocess
import sys

import pytest

from src.llm_f4.instruction_dataset import InstructionExample
from src.llm_f4.sft_lora import (
    Config, Encoded, SFTError, collate_rows, encode_sample, format_prompt,
    load_config, lora_delta, make_plan, supervised_token_count, torch_smoke,
    verify_run, _path, _parse,
)

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "configs" / "f4_05_sft_lora.json"


class FakeTokenizer:
    eos_token_id = 999

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert not add_special_tokens
        return [1 + ord(char) % 997 for char in text]


@pytest.fixture(scope="module")
def conf() -> Config:
    return load_config(CFG, root=ROOT)


@pytest.fixture(scope="module")
def plan_rows(conf):
    return make_plan(conf, root=ROOT)


@pytest.fixture
def example():
    return InstructionExample("f4q999_a", "f4g999", "python", "Что такое генератор?", "", "Генератор вычисляет значения лениво.", "authored_synthetic")


def write_config(tmp_path: Path, changes: dict) -> Path:
    data = json.loads(CFG.read_text(encoding="utf-8"))
    for key, val in changes.items():
        data[key] = val
    p = tmp_path / "mutated.json"
    p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return p


def test_model_selected(conf):
    assert conf.model_id == "Qwen/Qwen2.5-0.5B"
    assert conf.model_revision == conf.tokenizer_revision
    assert conf.rank == 8 and conf.alpha == 16
    assert conf.target_modules == ("q_proj", "v_proj")


def test_frozen_contracts(plan_rows):
    plan, splits = plan_rows
    assert plan["dataset_counts"] == {"train": 56, "validation": 12, "test": 12}
    assert len(set(x.id for name in splits for x in splits[name])) == 80
    assert {x.group_id for x in splits["train"]}.isdisjoint({x.group_id for x in splits["validation"]})
    assert {x.group_id for x in splits["train"]}.isdisjoint({x.group_id for x in splits["test"]})
    assert plan["training_performed"] is False


def test_holdout_is_not_in_training(plan_rows):
    plan, _ = plan_rows
    assert len(plan["holdout_sha256"]) == 64
    assert set(plan["dataset_counts"]) == {"train", "validation", "test"}


def test_split_tampering_detected(conf, monkeypatch):
    from src.llm_f4 import sft_lora
    monkeypatch.setattr(sft_lora, "verify_instruction", lambda _: (_ for _ in ()).throw(SFTError("tamper")))
    with pytest.raises(SFTError, match="tamper"):
        make_plan(conf, root=ROOT)


def test_holdout_tampering_detected(conf, monkeypatch):
    from src.llm_f4 import sft_lora
    monkeypatch.setattr(sft_lora, "prepare_eval", lambda _: (_ for _ in ()).throw(SFTError("holdout")))
    with pytest.raises(SFTError, match="holdout"):
        make_plan(conf, root=ROOT)


@pytest.mark.parametrize("bad", ["../secret", "/etc/passwd", "C:/users/a", "foo\\bar", "foo//bar", "foo/./bar", "foo/../bar", ""])
def test_unsafe_paths(bad):
    with pytest.raises(SFTError):
        _path(ROOT, bad, "source")


@pytest.mark.parametrize("patch", [
    {"schema_version": 2},
    {"schema_version": True},
    {"seed": "42"},
    {"seed": -1},
    {"model_id": "Other/Model"},
    {"model_revision": "main"},
    {"tokenizer_revision": "a" * 40},
    {"max_total_tokens": 0},
    {"max_total_tokens": 513},
    {"max_prompt_tokens": 300},
    {"cpu_threads": 0},
    {"instruction_config_file": "../../x"},
    {"output_dir": "../../x"},
    {"profiles": {}},
    {"lora": {"rank": 0, "alpha": 16, "dropout": 0.0, "target_modules": ["q_proj", "v_proj"]}},
    {"lora": {"rank": 8, "alpha": 16, "dropout": 0.0, "target_modules": ["unknown"]}},
    {"lora": {"rank": 8, "alpha": 16, "dropout": float("nan"), "target_modules": ["q_proj"]}},
    {"lora": {"rank": 8, "alpha": 16, "dropout": 0.0, "target_modules": ["v_proj", "v_proj"]}},
])
def test_invalid_config(tmp_path, patch):
    p = write_config(tmp_path, patch)
    with pytest.raises(SFTError):
        load_config(p, root=ROOT)


def test_duplicate_json_key():
    with pytest.raises(SFTError, match="Повторный"):
        _parse(b'{"a":1,"a":2}', "sample")


def test_nan_constant_json():
    with pytest.raises(SFTError):
        _parse(b'{"a":NaN}', "sample")


def test_prompt_format_same_as_f44(example, conf):
    from src.llm_f4.base_model_evaluation import load_config as baseline_conf
    base = baseline_conf(conf.baseline_config_file, root=ROOT)
    text = format_prompt(example, prefix=base.prefix, suffix=base.suffix)
    assert text == "Вопрос:\nЧто такое генератор?\nОтвет:\n"


def test_input_context(example):
    row = replace(example, input="Python 3.13")
    text = format_prompt(row, prefix="Вопрос:\n", suffix="\nОтвет:\n")
    assert "\n\nДанные:\nPython 3.13\nОтвет:\n" in text


def test_mask_labels_and_eos(example, conf):
    tokenized = encode_sample(example, FakeTokenizer(), conf, prefix="Вопрос:\n", suffix="\nОтвет:\n")
    assert all(x == -100 for x in tokenized.labels[:tokenized.prompt_tokens])
    assert tokenized.labels[-1] == tokenized.input_ids[-1] == 999
    assert tokenized.answer_tokens == len(tokenized.labels) - tokenized.prompt_tokens
    assert supervised_token_count([tokenized]) == tokenized.answer_tokens


def test_right_padding(example, conf):
    tok = FakeTokenizer()
    s1 = encode_sample(example, tok, conf, prefix="Вопрос:\n", suffix="\nОтвет:\n")
    s2 = encode_sample(replace(example, response="Короткий ответ."), tok, conf, prefix="Вопрос:\n", suffix="\nОтвет:\n")
    batch = collate_rows([s1, s2], pad_id=tok.eos_token_id)
    assert len(batch["input_ids"][0]) == len(batch["input_ids"][1])
    for i, s in enumerate((s1, s2)):
        assert batch["labels"][i][:len(s.labels)] == list(s.labels)
        assert batch["attention_mask"][i] == [1] * len(s.input_ids) + [0] * (len(batch["input_ids"][i])-len(s.input_ids))
        assert batch["labels"][i][len(s.labels):] == [-100] * (len(batch["labels"][i])-len(s.labels))


def test_collate_bad(example):
    with pytest.raises(SFTError):
        collate_rows([], pad_id=1)
    with pytest.raises(SFTError):
        collate_rows([Encoded("x", (1,), (1, 2), 0, 0)], pad_id=1)


@pytest.mark.parametrize("n", [1, 5, 20])
def test_token_budget_error(example, conf, n):
    with pytest.raises(SFTError, match="budget"):
        encode_sample(example, FakeTokenizer(), replace(conf, max_prompt_tokens=n), prefix="Вопрос:\n", suffix="\nОтвет:\n")


def test_answer_budget_error(example, conf):
    with pytest.raises(SFTError, match="budget"):
        encode_sample(example, FakeTokenizer(), replace(conf, max_answer_tokens=5), prefix="Вопрос:\n", suffix="\nОтвет:\n")


def test_wrong_eos(example, conf):
    tok = FakeTokenizer()
    tok.eos_token_id = None
    with pytest.raises(SFTError, match="eos"):
        encode_sample(example, tok, conf, prefix="Вопрос:\n", suffix="\nОтвет:\n")


def test_toy_lora_matrix():
    w = [[1.0, 0.0], [0.0, 1.0]]
    a = [[2.0, 3.0]]
    b = [[4.0], [5.0]]
    actual = lora_delta(w, a, b, alpha=2.0)
    assert actual == [[17.0, 24.0], [20.0, 31.0]]


def test_toy_lora_two_ranks():
    assert lora_delta([[0.0]], [[2.0], [3.0]], [[5.0, 7.0]], alpha=4.0) == [[62.0]]


@pytest.mark.parametrize("w,a,b", [
    ([[1., 2.]], [[1.]], [[1.]]),
    ([[1., 2.]], [[1., 2.]], [[1.], [2.]]),
    ([], [[1.]], [[1.]]),
    ([[1.]], [[1.]], []),
])
def test_lora_bad_shapes(w, a, b):
    with pytest.raises(SFTError):
        lora_delta(w, a, b, alpha=1.0)


def test_torch_smoke_real_autograd():
    pytest.importorskip("torch")
    r = torch_smoke()
    assert r["initial_loss"] == pytest.approx(.5)
    assert r["adapter_changed"] is True and r["llm_training_performed"] is False


def test_verify_without_real_run(conf, tmp_path):
    with pytest.raises(SFTError, match="Нет обязательного файла"):
        verify_run(replace(conf, output_dir=tmp_path), root=ROOT, profile="cpu")


def test_verify_synthetic_manifest_rejected(conf, tmp_path):
    folder = tmp_path / "cpu"
    folder.mkdir()
    (folder / "run_manifest.json").write_text('{"real_training_performed":false,"mode":"cpu"}', encoding="utf-8")
    with pytest.raises(SFTError, match="Not a confirmed"):
        verify_run(replace(conf, output_dir=tmp_path), root=ROOT, profile="cpu")


def test_cli_help_no_ml_imports():
    call = subprocess.run([sys.executable, str(ROOT / "scripts/f4_sft_lora.py"), "--help"],
                          cwd=ROOT, capture_output=True, text=True, check=True)
    assert "--torch-smoke" in call.stdout and "--run" in call.stdout


def test_cli_plan_no_model_download():
    call = subprocess.run([sys.executable, str(ROOT / "scripts/f4_sft_lora.py"), "--plan"],
                          cwd=ROOT, capture_output=True, text=True, check=True)
    assert "F4.5 PLAN: PASS" in call.stdout
    assert "training: NOT RUN" in call.stdout


@pytest.mark.parametrize("encoding", ["cp1251", "cp866", "cp1252"])
def test_cli_help_windows_encodings(encoding):
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = encoding
    call = subprocess.run([sys.executable, str(ROOT / "scripts/f4_sft_lora.py"), "--help"],
                          cwd=ROOT, capture_output=True, check=True, env=env)
    assert b"--run" in call.stdout


def test_cli_download_flag_requires_run():
    call = subprocess.run([sys.executable, str(ROOT / "scripts/f4_sft_lora.py"), "--allow-download"],
                          cwd=ROOT, capture_output=True)
    assert call.returncode != 0


def test_no_model_import_at_module_level():
    import src.llm_f4.sft_lora as mod
    assert "transformers" not in mod.__dict__
    assert "peft" not in mod.__dict__


def test_source_contains_no_gold_in_optimizer():
    import inspect
    from src.llm_f4 import sft_lora
    source = inspect.getsource(sft_lora.run_real)
    assert 'data["train"]' in source
    assert 'data["validation"]' in source
    assert 'data["test"]' not in source
    assert 'reference_answer' not in source


def _fixture_run_files(conf: Config, tmp_path: Path) -> Config:
    """Искусственный fixture для теста verifier; НЕ результат обучения LLM."""
    from hashlib import sha256
    plan, _ = make_plan(conf, root=ROOT)
    folder = tmp_path / "cpu"
    adapter = folder / "adapter"
    adapter.mkdir(parents=True)
    (adapter / "adapter_config.json").write_bytes(b'{"test_fixture":true}')
    (adapter / "adapter_model.safetensors").write_bytes(b"synthetic file, not real model")
    metrics = b'{"step":1,"train_loss":1.0,"grad_norm":0.2,"micro_batches":1}\n'
    (folder / "metrics.jsonl").write_bytes(metrics)
    report = {"real_training_performed": True, "mode": "cpu", "model_id": conf.model_id,
              "model_revision": conf.model_revision, "config_sha256": conf.config_sha256,
              "train_sha256": plan["train_sha256"], "validation_sha256": plan["validation_sha256"],
              "holdout_sha256": plan["holdout_sha256"], "optimizer_steps": 1,
              "metrics_sha256": sha256(metrics).hexdigest(),
              "adapter_files_sha256": {p.name: sha256(p.read_bytes()).hexdigest() for p in adapter.iterdir()}}
    (folder / "run_manifest.json").write_text(json.dumps(report), encoding="utf-8")
    return replace(conf, output_dir=tmp_path)


def test_verify_fixture_happy_path(conf, tmp_path):
    fake = _fixture_run_files(conf, tmp_path)
    result = verify_run(fake, root=ROOT, profile="cpu")
    assert result["optimizer_steps"] == 1


def test_verify_fixture_detects_tampered_metrics(conf, tmp_path):
    fake = _fixture_run_files(conf, tmp_path)
    (tmp_path / "cpu" / "metrics.jsonl").write_text("tampered", encoding="utf-8")
    with pytest.raises(SFTError, match="Modified training metrics"):
        verify_run(fake, root=ROOT, profile="cpu")


def test_verify_fixture_detects_tampered_adapter(conf, tmp_path):
    fake = _fixture_run_files(conf, tmp_path)
    (tmp_path / "cpu" / "adapter" / "adapter_model.safetensors").write_bytes(b"changed")
    with pytest.raises(SFTError, match="Modified adapter artifact"):
        verify_run(fake, root=ROOT, profile="cpu")


def test_prompt_with_input_identical_to_f43_f44(example, conf):
    from types import SimpleNamespace
    from src.llm_f4.evaluation_set import model_prompt
    from src.llm_f4.base_model_evaluation import formatted_prompt, load_config as baseline_conf
    base = baseline_conf(conf.baseline_config_file, root=ROOT)
    row = replace(example, input="Python 3.13")
    benchmark_prompt = model_prompt(SimpleNamespace(instruction=row.instruction, input=row.input, case_id="ev_py_01", domain="python"))["prompt"]
    expected = formatted_prompt(SimpleNamespace(prompt=benchmark_prompt), base)
    assert format_prompt(row, prefix=base.prefix, suffix=base.suffix) == expected
