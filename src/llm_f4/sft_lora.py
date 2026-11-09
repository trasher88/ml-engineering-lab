"""F4.5: проверяемый SFT с маскированием prompt и LoRA без QLoRA/DPO.

Обычные unit-тесты используют только stdlib. Реальные веса и зависимости
transformers/peft импортируются лишь внутри run_real().
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
import json
import math
from pathlib import Path
import random
import re
from typing import Any, Mapping, Sequence

from src.llm_f4.instruction_dataset import (
    DatasetError, InstructionExample, load_config as load_instruction_config,
    load_examples, validate_splits, verify_artifacts as verify_instruction,
)
from src.llm_f4.evaluation_set import (
    EvaluationError, load_config as load_eval_config,
    prepare_artifacts as prepare_eval,
)
from src.llm_f4.base_model_evaluation import (
    BaselineError, load_config as load_baseline_config,
)
from src.llm_f4.model_selection import load_catalog, ModelSelectionError


class SFTError(ValueError):
    """Нарушение проверяемого контракта или требований real-training."""


@dataclass(frozen=True, slots=True)
class Profile:
    max_steps: int
    batch_size: int
    gradient_accumulation: int
    learning_rate: float
    train_limit: int
    validation_limit: int


@dataclass(frozen=True, slots=True)
class Config:
    instruction_config_file: Path
    evaluation_config_file: Path
    baseline_config_file: Path
    model_catalog_file: Path
    output_dir: Path
    model_id: str
    model_revision: str
    tokenizer_revision: str
    seed: int
    max_total_tokens: int
    max_prompt_tokens: int
    max_answer_tokens: int
    cpu_threads: int
    rank: int
    alpha: int
    dropout: float
    target_modules: tuple[str, ...]
    profiles: Mapping[str, Profile]
    config_sha256: str


@dataclass(frozen=True, slots=True)
class Encoded:
    row_id: str
    input_ids: tuple[int, ...]
    labels: tuple[int, ...]
    prompt_tokens: int
    answer_tokens: int


FIELDS = {"schema_version", "instruction_config_file", "evaluation_config_file", "baseline_config_file",
          "model_catalog_file", "output_dir", "model_id", "model_revision", "tokenizer_revision",
          "seed", "max_total_tokens", "max_prompt_tokens", "max_answer_tokens", "cpu_threads",
          "lora", "profiles"}
PROFILE_FIELDS = {"max_steps", "batch_size", "gradient_accumulation", "learning_rate", "train_limit", "validation_limit"}
SHA40 = re.compile(r"[0-9a-f]{40}\Z")


def _duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for k, v in pairs:
        if k in result:
            raise SFTError(f"Повторный JSON-ключ: {k}")
        result[k] = v
    return result


def _bad_constant(name: str) -> None:
    raise SFTError(f"Недопустимая числовая константа: {name}")


def _parse(raw: bytes, where: str) -> Any:
    try:
        return json.loads(raw.decode("utf-8-sig"), object_pairs_hook=_duplicate_keys, parse_constant=_bad_constant)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise SFTError(f"Не удалось прочитать JSON {where}: {exc}") from exc


def _bytes(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except OSError as exc:
        raise SFTError(f"Нет обязательного файла {path}: {exc}") from exc


def _path(root: Path, value: Any, name: str) -> Path:
    if (not isinstance(value, str) or not value or ":" in value or "\\" in value
            or Path(value).is_absolute() or any(part in ("", ".", "..") for part in value.split("/"))):
        raise SFTError(f"{name}: нужен относительный путь POSIX без ..")
    base = root.resolve()
    resolved = (base / value).resolve()
    if not resolved.is_relative_to(base):
        raise SFTError(f"{name}: запрещён выход из репозитория")
    return resolved


def _int(value: Any, name: str, lower: int, upper: int) -> int:
    if type(value) is not int or not lower <= value <= upper:
        raise SFTError(f"{name}: целое {lower}..{upper}")
    return value


def _float(value: Any, name: str, lower: float, upper: float) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or not lower <= value <= upper:
        raise SFTError(f"{name}: число {lower}..{upper}")
    return float(value)


def load_config(path: Path, *, root: Path) -> Config:
    """Привязать LoRA к checkpoint, split и frozen F4.3/F4.4 контрактам."""
    payload = _bytes(path)
    obj = _parse(payload, str(path))
    if not isinstance(obj, dict) or set(obj) != FIELDS or type(obj["schema_version"]) is not int or obj["schema_version"] != 1:
        raise SFTError("Неверная структура конфигурации F4.5")
    rev = obj["model_revision"]
    if (not isinstance(obj["model_id"], str) or not obj["model_id"] or
            not isinstance(rev, str) or not SHA40.fullmatch(rev) or obj["tokenizer_revision"] != rev):
        raise SFTError("Модель и tokenizer должны иметь одинаковый pinned SHA")
    lora = obj["lora"]
    if not isinstance(lora, dict) or set(lora) != {"rank", "alpha", "dropout", "target_modules"}:
        raise SFTError("Неверная структура LoRA")
    modules = lora["target_modules"]
    if not isinstance(modules, list) or not modules or len(modules) != len(set(str(m) for m in modules)) or any(
            not isinstance(m, str) or not re.fullmatch(r"[a-z]+_proj", m) for m in modules):
        raise SFTError("Некорректные LoRA target_modules")
    if set(modules) - {"q_proj", "k_proj", "v_proj", "o_proj"}:
        raise SFTError("Для Qwen разрешены только attention projection")
    profiles = obj["profiles"]
    if not isinstance(profiles, dict) or set(profiles) != {"cpu", "gpu"}:
        raise SFTError("Требуются профили cpu и gpu")
    profile_map = {}
    for name, p in profiles.items():
        if not isinstance(p, dict) or set(p) != PROFILE_FIELDS:
            raise SFTError(f"Неверный профиль {name}")
        profile_map[name] = Profile(
            _int(p["max_steps"], "max_steps", 1, 10000),
            _int(p["batch_size"], "batch_size", 1, 64),
            _int(p["gradient_accumulation"], "gradient_accumulation", 1, 128),
            _float(p["learning_rate"], "learning_rate", 1e-7, 0.1),
            _int(p["train_limit"], "train_limit", 1, 100000),
            _int(p["validation_limit"], "validation_limit", 1, 100000),
        )
    conf = Config(
        instruction_config_file=_path(root, obj["instruction_config_file"], "instruction_config_file"),
        evaluation_config_file=_path(root, obj["evaluation_config_file"], "evaluation_config_file"),
        baseline_config_file=_path(root, obj["baseline_config_file"], "baseline_config_file"),
        model_catalog_file=_path(root, obj["model_catalog_file"], "model_catalog_file"),
        output_dir=_path(root, obj["output_dir"], "output_dir"),
        model_id=obj["model_id"], model_revision=rev, tokenizer_revision=obj["tokenizer_revision"],
        seed=_int(obj["seed"], "seed", 0, 2**31-1),
        max_total_tokens=_int(obj["max_total_tokens"], "max_total_tokens", 16, 32768),
        max_prompt_tokens=_int(obj["max_prompt_tokens"], "max_prompt_tokens", 4, 32768),
        max_answer_tokens=_int(obj["max_answer_tokens"], "max_answer_tokens", 2, 32768),
        cpu_threads=_int(obj["cpu_threads"], "cpu_threads", 1, 64),
        rank=_int(lora["rank"], "rank", 1, 256),
        alpha=_int(lora["alpha"], "alpha", 1, 1024),
        dropout=_float(lora["dropout"], "dropout", 0.0, 0.9),
        target_modules=tuple(modules), profiles=profile_map,
        config_sha256=sha256(payload).hexdigest(),
    )
    if conf.max_prompt_tokens + conf.max_answer_tokens > conf.max_total_tokens:
        raise SFTError("sum of prompt+answer budgets exceeds max_total_tokens")
    cat = load_catalog(conf.model_catalog_file)
    if cat.selected_model_id != conf.model_id:
        raise SFTError("F4.5 model must equal F4.2 selected model")
    model = next(c for c in cat.candidates if c.model_id == conf.model_id)
    if model.revision != rev or model.tokenizer_revision != rev or model.model_type != "qwen2":
        raise SFTError("F4.5 Qwen revision/architecture mismatch")
    if conf.max_total_tokens > cat.requirements.sequence_length:
        raise SFTError("Token budget exceeds F4.2 frozen context")
    base = load_baseline_config(conf.baseline_config_file, root=root)
    if (base.model_id != conf.model_id or base.model_revision != rev or
            base.tokenizer_revision != rev or base.model_catalog_file != conf.model_catalog_file):
        raise SFTError("F4.4 baseline mismatch")
    ev = load_eval_config(conf.evaluation_config_file, root=root)
    if ev.model_catalog_file != conf.model_catalog_file:
        raise SFTError("F4.3 benchmark mismatch")
    return conf


def make_plan(conf: Config, *, root: Path) -> tuple[dict[str, Any], dict[str, list[InstructionExample]]]:
    """Проверить split побайтово и holdout, не открывая ни torch, ни веса модели."""
    split_conf = load_instruction_config(conf.instruction_config_file, root=root)
    verify_instruction(split_conf)
    rows = {name: load_examples(split_conf.output_dir / f"{name}.jsonl")
            for name in ("train", "validation", "test")}
    validate_splits(rows, source=load_examples(split_conf.source_file))
    ev = load_eval_config(conf.evaluation_config_file, root=root)
    prepare_eval(ev)  # проверяем фиксированный внешний holdout; его ответы в обучение не входят
    for name, profile in conf.profiles.items():
        if profile.train_limit > len(rows["train"]) or profile.validation_limit > len(rows["validation"]):
            raise SFTError(f"{name}: requested sample larger than split")
    plan = {
        "schema_version": 1, "stage": "F4.5", "kind": "sft_lora_plan_only",
        "model_id": conf.model_id, "revision": conf.model_revision,
        "dataset_counts": {name: len(data) for name, data in rows.items()},
        "train_sha256": sha256(_bytes(split_conf.output_dir / "train.jsonl")).hexdigest(),
        "validation_sha256": sha256(_bytes(split_conf.output_dir / "validation.jsonl")).hexdigest(),
        "holdout_sha256": ev.evaluation_source_sha256,
        "config_sha256": conf.config_sha256,
        "lora": {"rank": conf.rank, "alpha": conf.alpha, "dropout": conf.dropout,
                 "target_modules": list(conf.target_modules)},
        "profiles": {name: asdict(p) for name, p in conf.profiles.items()},
        "training_performed": False,
    }
    return plan, rows


def format_prompt(row: InstructionExample, *, prefix: str, suffix: str) -> str:
    """Тот же prompt-only шаблон F4.3/F4.4, включая блок Данные при input."""
    if not prefix or not suffix:
        raise SFTError("Empty prompt template")
    question = row.instruction
    if row.input:
        question += f"\n\nДанные:\n{row.input}"
    return prefix + question + suffix


def encode_sample(row: InstructionExample, tokenizer: Any, conf: Config,
                  *, prefix: str, suffix: str) -> Encoded:
    """Раздельная токенизация prompt/answer: labels prompt=-100, EOS под loss.

    Токенизация без спец. токенов и без усечения. Таким образом prompt на
    inference токенизируется так же, как префикс обучающего примера.
    """
    prompt = format_prompt(row, prefix=prefix, suffix=suffix)
    p = tokenizer.encode(prompt, add_special_tokens=False)
    a = tokenizer.encode(row.response, add_special_tokens=False)
    eos = tokenizer.eos_token_id
    if not isinstance(eos, int) or eos < 0:
        raise SFTError("Tokenizer must provide eos_token_id")
    if (not isinstance(p, (list, tuple)) or not isinstance(a, (list, tuple)) or
            not p or not a or any(type(n) is not int or n < 0 for n in [*p, *a])):
        raise SFTError(f"{row.id}: malformed token sequence")
    # Сохраняем EOS как отдельный целевой токен, предотвращаем дубликат EOS.
    if a[-1] == eos:
        raise SFTError(f"{row.id}: explicit EOS already inside response")
    if (len(p) > conf.max_prompt_tokens or len(a)+1 > conf.max_answer_tokens or
            len(p)+len(a)+1 > conf.max_total_tokens):
        raise SFTError(f"{row.id}: token budget exceeded (no silent truncation)")
    ids = tuple(p) + tuple(a) + (eos,)
    labels = (-100,) * len(p) + tuple(a) + (eos,)
    assert len(ids) == len(labels)
    return Encoded(row.id, ids, labels, len(p), len(a)+1)


def collate_rows(items: Sequence[Encoded], *, pad_id: int) -> dict[str, list[list[int]]]:
    """Правое выравнивание: input_ids/padding, attention_mask и labels=-100."""
    if not items or type(pad_id) is not int or pad_id < 0:
        raise SFTError("Cannot collate empty batch or invalid padding token")
    longest = max(len(x.input_ids) for x in items)
    inputs, masks, labels = [], [], []
    for item in items:
        if not item.input_ids or len(item.input_ids) != len(item.labels):
            raise SFTError("Invalid Encoded record")
        gap = longest - len(item.input_ids)
        inputs.append(list(item.input_ids) + [pad_id] * gap)
        masks.append([1] * len(item.input_ids) + [0] * gap)
        labels.append(list(item.labels) + [-100] * gap)
    return {"input_ids": inputs, "attention_mask": masks, "labels": labels}


def supervised_token_count(items: Sequence[Encoded]) -> int:
    return sum(sum(x != -100 for x in e.labels[1:]) for e in items)


def lora_delta(weight_out_in: Sequence[Sequence[float]], a: Sequence[Sequence[float]],
               b: Sequence[Sequence[float]], *, alpha: float) -> list[list[float]]:
    """Для маленьких числовых примеров: W' = W + (alpha/r) B@A."""
    if not weight_out_in or not a or not b or not a[0]:
        raise SFTError("LoRA matrices must be non-empty")
    n_out, n_in, rank = len(weight_out_in), len(weight_out_in[0]), len(a)
    if (not all(len(row) == n_in for row in weight_out_in) or
            not all(len(row) == n_in for row in a) or
            len(b) != n_out or not all(len(row) == rank for row in b) or
            not math.isfinite(alpha) or alpha <= 0):
        raise SFTError("LoRA matrix shape mismatch")
    return [[weight_out_in[i][j] + (alpha/rank) * sum(b[i][k] * a[k][j] for k in range(rank))
             for j in range(n_in)] for i in range(n_out)]


def torch_smoke() -> dict[str, Any]:
    """Честный синтетический autograd-smoke LoRA без LLM, PEFT и сети."""
    try:
        import torch
    except ImportError as exc:
        raise SFTError("torch is required for numerical torch smoke") from exc
    torch.manual_seed(42)
    w = torch.eye(2)
    a = torch.nn.Parameter(torch.tensor([[1.0, 0.0]], dtype=torch.float32))
    b = torch.nn.Parameter(torch.tensor([[0.0], [0.0]], dtype=torch.float32))
    target = torch.tensor([[1.0, 1.0]], dtype=torch.float32)
    x = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    y = x @ (w + 2.0 * b @ a).T
    loss = (y - target).pow(2).mean()
    loss.backward()
    if w.requires_grad or a.grad is None or b.grad is None or b.grad.abs().sum().item() <= 0:
        raise SFTError("LoRA smoke gradient invariant failed")
    optimizer = torch.optim.SGD([a, b], lr=0.1)
    before = b.detach().clone()
    optimizer.step()
    if torch.equal(before, b.detach()):
        raise SFTError("LoRA smoke weights did not update")
    return {"kind": "synthetic_torch_autograd_only", "initial_loss": float(loss.item()),
            "adapter_changed": True, "llm_training_performed": False}


def _line(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _atomic(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_bytes(content)
    temp.replace(path)


def _check_runtime(profile: str) -> dict[str, str]:
    required = ("torch", "transformers", "peft", "safetensors")
    detected = {}
    for pkg in required:
        try:
            detected[pkg] = version(pkg)
        except PackageNotFoundError as exc:
            raise SFTError(f"Для реального SFT отсутствует библиотека {pkg}") from exc
    if int(detected["transformers"].split(".")[0]) < 4:
        raise SFTError("Unsupported Transformers major version")
    if profile not in ("cpu", "gpu"):
        raise SFTError("Only cpu and gpu modes supported")
    return detected


def run_real(conf: Config, *, root: Path, profile: str, allow_download: bool = False) -> dict[str, Any]:
    """Настоящий SFT: Qwen Base, замороженные веса, LoRA update, val loss, адаптер."""
    if profile not in conf.profiles:
        raise SFTError("Unknown training profile")
    plan, data = make_plan(conf, root=root)
    deps = _check_runtime(profile)
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError as exc:
        raise SFTError(f"Real SFT runtime import failed: {exc}") from exc
    if profile == "gpu" and not torch.cuda.is_available():
        raise SFTError("GPU profile requires NVIDIA/CUDA; CPU mode is separate")
    torch.manual_seed(conf.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(conf.seed)
    torch.set_num_threads(conf.cpu_threads)
    device = "cpu" if profile == "cpu" else "cuda"
    dtype = torch.float32 if profile == "cpu" else torch.bfloat16
    if profile == "gpu" and not torch.cuda.is_bf16_supported():
        raise SFTError("GPU profile requires BF16 support; do not silently change precision")
    destination = conf.output_dir / profile
    if destination.exists() and any(destination.iterdir()):
        raise SFTError(f"Output folder already contains files: {destination}; refusing overwrite")
    kw = {"revision": conf.model_revision, "local_files_only": not allow_download, "trust_remote_code": False}
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            conf.model_id, revision=conf.tokenizer_revision,
            local_files_only=not allow_download, trust_remote_code=False, use_fast=True,
        )
        major = int(deps["transformers"].split(".")[0])
        dtype_kw = "dtype" if major >= 5 else "torch_dtype"
        base = AutoModelForCausalLM.from_pretrained(conf.model_id, use_safetensors=True,
                                                    **kw, **{dtype_kw: dtype})
        if base.config.model_type != "qwen2" or base.config.eos_token_id != tokenizer.eos_token_id:
            raise SFTError("Loaded tokenizer/model architecture or EOS mismatch")
        if tokenizer.vocab_size > base.get_input_embeddings().num_embeddings:
            raise SFTError("Loaded tokenizer vocabulary exceeds embedding size")
    except Exception as exc:
        raise SFTError(f"Failed to load pinned checkpoint: {type(exc).__name__}: {exc}") from exc
    baseline = load_baseline_config(conf.baseline_config_file, root=root)
    tr = [encode_sample(row, tokenizer, conf, prefix=baseline.prefix, suffix=baseline.suffix)
          for row in data["train"]]
    va = [encode_sample(row, tokenizer, conf, prefix=baseline.prefix, suffix=baseline.suffix)
          for row in data["validation"]]
    cfg = conf.profiles[profile]
    tr = tr[:cfg.train_limit]
    va = va[:cfg.validation_limit]
    if not tr or not va:
        raise SFTError("Train/validation dataset empty")
    adapter_cfg = LoraConfig(task_type=TaskType.CAUSAL_LM, r=conf.rank, lora_alpha=conf.alpha,
                             lora_dropout=conf.dropout, bias="none", target_modules=list(conf.target_modules))
    base.config.use_cache = False
    if profile == "gpu":
        base.gradient_checkpointing_enable()
        base.enable_input_require_grads()
    model = get_peft_model(base, adapter_cfg).to(device)
    trainable = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
    if (not trainable or any("lora_" not in name for name, _ in trainable)):
        raise SFTError("Frozen base parameter invariant broken")
    n_train = sum(p.numel() for _, p in trainable)
    n_all = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW([p for _, p in trainable], lr=cfg.learning_rate, weight_decay=0.0)
    pad_id = tokenizer.eos_token_id

    def batch(items: Sequence[Encoded]) -> dict[str, Any]:
        return {k: torch.tensor(v, dtype=torch.long, device=device)
                for k, v in collate_rows(items, pad_id=pad_id).items()}

    def validation_loss() -> float:
        model.eval()
        weighted, count = 0.0, 0
        with torch.no_grad():
            for start in range(0, len(va), cfg.batch_size):
                selected = va[start:start+cfg.batch_size]
                loss = model(**batch(selected)).loss
                n = supervised_token_count(selected)
                weighted += float(loss.item()) * n
                count += n
        model.train()
        if count <= 0 or not math.isfinite(weighted):
            raise SFTError("Nonfinite/empty validation loss")
        return weighted / count

    model.train()
    initial_val = validation_loss()
    # Сначала строим полный список микробатчей; повторяем train set при недостатке шагов.
    rng = random.Random(conf.seed)
    order = list(range(len(tr)))
    cursor = len(order)
    records = []
    for step in range(1, cfg.max_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for _ in range(cfg.gradient_accumulation):
            if cursor >= len(order):
                rng.shuffle(order)
                cursor = 0
            indexes = order[cursor:cursor+cfg.batch_size]
            cursor += len(indexes)
            selected = [tr[i] for i in indexes]
            if not selected:
                raise SFTError("Empty minibatch")
            loss = model(**batch(selected)).loss
            if not torch.isfinite(loss):
                raise SFTError("Nonfinite SFT loss")
            (loss / cfg.gradient_accumulation).backward()
            losses.append(float(loss.item()))
        norm = torch.nn.utils.clip_grad_norm_([p for _, p in trainable], 1.0)
        if not torch.isfinite(norm):
            raise SFTError("Nonfinite gradient norm")
        optimizer.step()
        records.append({"step": step, "train_loss": sum(losses)/len(losses),
                        "grad_norm": float(norm.item()), "micro_batches": len(losses)})
    final_val = validation_loss()
    adapter_dir = destination / "adapter"
    if destination.exists() and any(destination.iterdir()):
        raise SFTError("Output folder unexpectedly populated")
    adapter_dir.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(str(adapter_dir), safe_serialization=True)
    artifacts = {p.name: sha256(_bytes(p)).hexdigest() for p in adapter_dir.iterdir() if p.is_file()}
    if "adapter_config.json" not in artifacts or "adapter_model.safetensors" not in artifacts:
        raise SFTError("PEFT did not save required adapter artifacts")
    summary = {"schema_version": 1, "mode": profile, "real_training_performed": True,
               "model_id": conf.model_id, "model_revision": conf.model_revision,
               "config_sha256": conf.config_sha256, "train_sha256": plan["train_sha256"],
               "validation_sha256": plan["validation_sha256"],
               "holdout_sha256": plan["holdout_sha256"], "versions": deps,
               "precision": str(dtype), "device": device,
               "train_records_used": len(tr), "validation_records_used": len(va),
               "trainable_parameters": n_train, "all_parameters_with_adapters": n_all,
               "trainable_fraction": n_train/n_all, "initial_validation_loss": initial_val,
               "final_validation_loss": final_val, "optimizer_steps": len(records),
               "adapter_files_sha256": artifacts, "manual_quality_score": None,
               "evaluation_holdout_used_in_training": False}
    metric_bytes = b"".join(_line(row) for row in records)
    summary["metrics_sha256"] = sha256(metric_bytes).hexdigest()
    _atomic(destination / "metrics.jsonl", metric_bytes)
    _atomic(destination / "run_manifest.json", json.dumps(summary, ensure_ascii=False, sort_keys=True,
                                                           indent=2, allow_nan=False).encode("utf-8") + b"\n")
    return summary


def verify_run(conf: Config, *, root: Path, profile: str) -> dict[str, Any]:
    """Проверка хешей сохранённого адаптера и входных данных, без импорта ML библиотек."""
    if profile not in conf.profiles:
        raise SFTError("Unknown profile")
    plan, _ = make_plan(conf, root=root)
    folder = conf.output_dir / profile
    result = _parse(_bytes(folder / "run_manifest.json"), "run_manifest.json")
    if not isinstance(result, dict) or not result.get("real_training_performed") or result.get("mode") != profile:
        raise SFTError("Not a confirmed real training manifest")
    for key, value in (("config_sha256", conf.config_sha256), ("train_sha256", plan["train_sha256"]),
                       ("validation_sha256", plan["validation_sha256"]), ("holdout_sha256", plan["holdout_sha256"]),
                       ("model_revision", conf.model_revision), ("model_id", conf.model_id)):
        if result.get(key) != value:
            raise SFTError(f"Modified run context: {key}")
    if not isinstance(result.get("adapter_files_sha256"), dict):
        raise SFTError("Invalid adapter hash manifest")
    for name, expected in result["adapter_files_sha256"].items():
        if name not in ("adapter_model.safetensors", "adapter_config.json", "README.md") or not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise SFTError("Unexpected adapter artifact in manifest")
        if sha256(_bytes(folder / "adapter" / name)).hexdigest() != expected:
            raise SFTError(f"Modified adapter artifact: {name}")
    if not {"adapter_model.safetensors", "adapter_config.json"} <= set(result.get("adapter_files_sha256", {})):
        raise SFTError("Missing adapter files")
    metrics_bytes = _bytes(folder / "metrics.jsonl")
    if sha256(metrics_bytes).hexdigest() != result.get("metrics_sha256"):
        raise SFTError("Modified training metrics artifact")
    raw_metrics = metrics_bytes.decode("utf-8").splitlines()
    if len(raw_metrics) != result.get("optimizer_steps") or not raw_metrics:
        raise SFTError("Training metrics count mismatch")
    for idx, line in enumerate(raw_metrics, 1):
        metric = _parse(line.encode("utf-8"), f"metrics line {idx}")
        if (not isinstance(metric, dict) or metric.get("step") != idx or
                any(not isinstance(metric.get(k), (float, int)) or not math.isfinite(metric[k])
                    for k in ("train_loss", "grad_norm"))):
            raise SFTError("Invalid training metric")
    return result
