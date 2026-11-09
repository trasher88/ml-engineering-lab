"""F4.8: DPO на F4.5 SFT-LoRA с неизменной reference policy.

По умолчанию доступны только проверяемые stdlib-функции. Тяжёлые импорты и
реальное обучение выполняются исключительно при явном вызове --run.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping, Sequence
import json
import math
import re
import unicodedata

from src.llm_f4.instruction_dataset import load_config as load_instruction_config, load_examples, verify_artifacts as verify_f41
from src.llm_f4.sft_lora import load_config as load_sft_config, make_plan as make_sft_plan, verify_run as verify_sft, format_prompt
from src.llm_f4.base_model_evaluation import load_config as load_base_config


class DPOError(ValueError):
    """Дефект данных, происхождения, вычислений или артефактов эксперимента."""


@dataclass(frozen=True, slots=True)
class Profile:
    max_steps: int
    train_pairs: int
    validation_pairs: int
    learning_rate: float


@dataclass(frozen=True, slots=True)
class Config:
    dataset_file: Path
    instruction_config_file: Path
    sft_config_file: Path
    baseline_config_file: Path
    output_dir: Path
    sft_profile: str
    seed: int
    beta: float
    max_prompt_tokens: int
    max_answer_tokens: int
    max_total_tokens: int
    cpu_threads: int
    profiles: Mapping[str, Profile]
    config_sha256: str


@dataclass(frozen=True, slots=True)
class Preference:
    id: str
    split: str
    source_id: str
    group_id: str
    domain: str
    prompt: str
    chosen: str
    rejected: str
    rationale: str
    origin: str


@dataclass(frozen=True, slots=True)
class TokenizedPair:
    id: str
    prompt_tokens: int
    chosen_ids: tuple[int, ...]
    chosen_labels: tuple[int, ...]
    rejected_ids: tuple[int, ...]
    rejected_labels: tuple[int, ...]


CFG_KEYS = {"schema_version", "dataset_file", "instruction_config_file", "sft_config_file",
            "baseline_config_file", "output_dir", "sft_profile", "seed", "beta",
            "max_prompt_tokens", "max_answer_tokens", "max_total_tokens", "cpu_threads", "profiles"}
ROW_KEYS = set(Preference.__dataclass_fields__)
ID_PATTERN = re.compile(r"[a-z][a-z0-9_-]{2,63}\Z")
DOMAIN_PATTERN = re.compile(r"[a-z][a-z0-9_-]{1,63}\Z")
HEX64 = re.compile(r"[a-f0-9]{64}\Z")
ORIGIN = "authored_synthetic_preference"


def _object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DPOError(f"Повторный JSON-ключ: {key}")
        result[key] = value
    return result


def _bad_constant(name: str) -> None:
    raise DPOError(f"Некорректная JSON-константа: {name}")


def _read(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except OSError as exc:
        raise DPOError(f"Не удалось прочитать {path}: {exc}") from exc


def _json(raw: bytes, where: str) -> Any:
    try:
        return json.loads(raw.decode("utf-8-sig"), object_pairs_hook=_object_pairs, parse_constant=_bad_constant)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise DPOError(f"Некорректный JSON {where}: {exc}") from exc


def _canonical(data: Any) -> bytes:
    return (json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def _pretty(data: Any) -> bytes:
    return (json.dumps(data, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def _int(value: Any, field: str, low: int, high: int) -> int:
    if type(value) is not int or not low <= value <= high:
        raise DPOError(f"{field}: нужно целое от {low} до {high}")
    return value


def _number(value: Any, field: str, low: float, high: float) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
        raise DPOError(f"{field}: число {low}..{high}")
    return float(value)


def _path(root: Path, value: Any, field: str) -> Path:
    if (not isinstance(value, str) or not value or value.startswith(("/", "~"))
            or ":" in value or "\\" in value or any(s in ("", ".", "..") for s in value.split("/"))):
        raise DPOError(f"{field}: нужен безопасный относительный POSIX-путь")
    root = root.resolve()
    resolved = (root / value).resolve()
    if not resolved.is_relative_to(root):
        raise DPOError(f"{field}: выход за пределы репозитория запрещён")
    return resolved


def load_config(path: Path, *, root: Path) -> Config:
    raw = _read(path)
    data = _json(raw, str(path))
    if not isinstance(data, dict) or set(data) != CFG_KEYS or type(data["schema_version"]) is not int or data["schema_version"] != 1:
        raise DPOError("Неверная структура конфигурации F4.8")
    profiles = data["profiles"]
    if not isinstance(profiles, dict) or set(profiles) != {"cpu", "gpu"}:
        raise DPOError("Требуются профили cpu и gpu")
    pmap: dict[str, Profile] = {}
    for name, raw_profile in profiles.items():
        if not isinstance(raw_profile, dict) or set(raw_profile) != {"max_steps", "train_pairs", "validation_pairs", "learning_rate"}:
            raise DPOError(f"Неверный профиль: {name}")
        pmap[name] = Profile(_int(raw_profile["max_steps"], "max_steps", 1, 1000),
                             _int(raw_profile["train_pairs"], "train_pairs", 1, 1000),
                             _int(raw_profile["validation_pairs"], "validation_pairs", 1, 1000),
                             _number(raw_profile["learning_rate"], "learning_rate", 1e-8, 1e-2))
    conf = Config(
        dataset_file=_path(root, data["dataset_file"], "dataset_file"),
        instruction_config_file=_path(root, data["instruction_config_file"], "instruction_config_file"),
        sft_config_file=_path(root, data["sft_config_file"], "sft_config_file"),
        baseline_config_file=_path(root, data["baseline_config_file"], "baseline_config_file"),
        output_dir=_path(root, data["output_dir"], "output_dir"),
        sft_profile=data["sft_profile"],
        seed=_int(data["seed"], "seed", 0, 2**31 - 1),
        beta=_number(data["beta"], "beta", 1e-5, 10.0),
        max_prompt_tokens=_int(data["max_prompt_tokens"], "max_prompt_tokens", 4, 8192),
        max_answer_tokens=_int(data["max_answer_tokens"], "max_answer_tokens", 4, 8192),
        max_total_tokens=_int(data["max_total_tokens"], "max_total_tokens", 16, 16384),
        cpu_threads=_int(data["cpu_threads"], "cpu_threads", 1, 64),
        profiles=pmap, config_sha256=sha256(raw).hexdigest(),
    )
    if conf.sft_profile not in ("cpu", "gpu"):
        raise DPOError("Неизвестный профиль SFT")
    if conf.max_prompt_tokens + conf.max_answer_tokens > conf.max_total_tokens:
        raise DPOError("Несогласованные token budgets")
    sft = load_sft_config(conf.sft_config_file, root=root)
    base = load_base_config(conf.baseline_config_file, root=root)
    if (sft.model_id != base.model_id or sft.model_revision != base.model_revision
            or sft.tokenizer_revision != base.tokenizer_revision):
        raise DPOError("Модель F4.5 не совпадает с F4.4")
    if sft.instruction_config_file != conf.instruction_config_file:
        raise DPOError("F4.1 пути F4.5/F4.8 не совпадают")
    if conf.max_total_tokens > sft.max_total_tokens:
        raise DPOError("Бюджет F4.8 превышает F4.5")
    return conf


def _norm(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def load_preferences(path: Path) -> list[Preference]:
    data = _read(path)
    result: list[Preference] = []
    for lineno, line in enumerate(data.splitlines(), 1):
        if not line.strip():
            raise DPOError(f"Пустая JSONL-строка {lineno}")
        item = _json(line, f"{path}:{lineno}")
        if not isinstance(item, dict) or set(item) != ROW_KEYS:
            raise DPOError(f"Строка {lineno}: неверные поля preference")
        if any(not isinstance(v, str) for v in item.values()):
            raise DPOError(f"Строка {lineno}: все поля должны быть строками")
        for key in ("id", "source_id", "group_id", "domain"):
            if not (DOMAIN_PATTERN if key == "domain" else ID_PATTERN).fullmatch(item[key]):
                raise DPOError(f"Строка {lineno}: неверный идентификатор {key}")
        if item["split"] not in ("train", "validation") or item["origin"] != ORIGIN:
            raise DPOError(f"Строка {lineno}: неразрешённый split/origin")
        for key in ("chosen", "rejected", "rationale"):
            value = item[key]
            if value != value.strip() or not 12 <= len(value) <= 4000 or "\x00" in value:
                raise DPOError(f"Строка {lineno}: неверный текст {key}")
        if not 12 <= len(item["prompt"]) <= 4000 or "\x00" in item["prompt"]:
            raise DPOError(f"Строка {lineno}: неверный текст prompt")
        if _norm(item["chosen"]) == _norm(item["rejected"]):
            raise DPOError(f"Строка {lineno}: chosen и rejected совпадают")
        result.append(Preference(**item))
    if not result:
        raise DPOError("Preference dataset пуст")
    ids: set[str] = set()
    source_ids: set[str] = set()
    group_split: dict[str, str] = {}
    prompts: set[str] = set()
    for row in result:
        if row.id in ids or row.source_id in source_ids or _norm(row.prompt) in prompts:
            raise DPOError("Повтор preference id/source_id/prompt")
        ids.add(row.id); source_ids.add(row.source_id); prompts.add(_norm(row.prompt))
        previous = group_split.setdefault(row.group_id, row.split)
        if previous != row.split:
            raise DPOError("Смысловая группа попала в разные split")
    return result


def make_plan(conf: Config, *, root: Path) -> tuple[dict[str, Any], dict[str, list[Preference]]]:
    """Проверить происхождение предпочтений относительно F4.1 и F4.3 без LLM."""
    split_cfg = load_instruction_config(conf.instruction_config_file, root=root)
    verify_f41(split_cfg)
    source = {name: load_examples(split_cfg.output_dir / f"{name}.jsonl")
              for name in ("train", "validation", "test")}
    sft = load_sft_config(conf.sft_config_file, root=root)
    sft_plan, _ = make_sft_plan(sft, root=root)
    base = load_base_config(conf.baseline_config_file, root=root)
    preferences = load_preferences(conf.dataset_file)
    by_split = {name: {row.id: row for row in rows} for name, rows in source.items()}
    sorted_pairs: dict[str, list[Preference]] = {"train": [], "validation": []}
    for item in preferences:
        row = by_split[item.split].get(item.source_id)
        if row is None or row.group_id != item.group_id or row.domain != item.domain:
            raise DPOError(f"{item.id}: источник не принадлежит объявленному F4.1 split")
        expected = format_prompt(row, prefix=base.prefix, suffix=base.suffix)
        if item.prompt != expected or item.chosen != row.response:
            raise DPOError(f"{item.id}: prompt/chosen отличаются от верифицированного F4.1")
        sorted_pairs[item.split].append(item)
    if not sorted_pairs["train"] or not sorted_pairs["validation"]:
        raise DPOError("Пустой train/validation для DPO")
    for name, profile in conf.profiles.items():
        if profile.train_pairs > len(sorted_pairs["train"]) or profile.validation_pairs > len(sorted_pairs["validation"]):
            raise DPOError(f"{name}: profile требует больше preference rows, чем доступно")
    plan = {
        "schema_version": 1, "stage": "F4.8", "kind": "preference_plan_only",
        "model_id": sft.model_id, "revision": sft.model_revision,
        "reference": "frozen_verified_f45_sft_adapter_before_dpo",
        "preference_sha256": sha256(_read(conf.dataset_file)).hexdigest(),
        "f41_train_sha256": sft_plan["train_sha256"],
        "f41_validation_sha256": sft_plan["validation_sha256"],
        "f43_holdout_sha256": sft_plan["holdout_sha256"],
        "f43_used_for_training": False,
        "f47_human_grades_used_for_training": False,
        "split_counts": {name: len(items) for name, items in sorted_pairs.items()},
        "split_domains": {name: {domain: sum(p.domain == domain for p in items)
                                 for domain in sorted(set(p.domain for p in items))}
                          for name, items in sorted_pairs.items()},
        "beta": conf.beta, "profiles": {name: asdict(p) for name, p in conf.profiles.items()},
        "config_sha256": conf.config_sha256, "training_performed": False,
    }
    return plan, sorted_pairs


def dpo_loss(policy_chosen: float, policy_rejected: float, reference_chosen: float,
             reference_rejected: float, beta: float) -> tuple[float, float, float]:
    """Независимый scalar-эталон DPO: (loss, margin, chosen-win probability)."""
    values = (policy_chosen, policy_rejected, reference_chosen, reference_rejected, beta)
    if any(type(x) not in (int, float) or not math.isfinite(x) for x in values) or beta <= 0:
        raise DPOError("Для DPO нужны конечные log-probabilities и beta > 0")
    margin = beta * ((policy_chosen - policy_rejected) - (reference_chosen - reference_rejected))
    loss = math.log1p(math.exp(-abs(margin))) + max(-margin, 0.0)
    probability = 1 / (1 + math.exp(-margin)) if margin >= 0 else math.exp(margin) / (1 + math.exp(margin))
    return loss, margin, probability


def tokenize_pair(row: Preference, tokenizer: Any, conf: Config) -> TokenizedPair:
    """EOS участвует в logπ, prompt и padding — нет; усечения не допускаются."""
    if not isinstance(tokenizer.eos_token_id, int) or tokenizer.eos_token_id < 0:
        raise DPOError("Нет корректного EOS")
    prompt_ids = tokenizer.encode(row.prompt, add_special_tokens=False)
    if not prompt_ids or any(type(t) is not int for t in prompt_ids):
        raise DPOError("Некорректная токенизация prompt")
    if len(prompt_ids) > conf.max_prompt_tokens:
        raise DPOError(f"{row.id}: превышен бюджет prompt")
    encoded: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    for answer in (row.chosen, row.rejected):
        ids = tokenizer.encode(answer, add_special_tokens=False)
        if not ids or any(type(t) is not int for t in ids) or ids[-1] == tokenizer.eos_token_id:
            raise DPOError(f"{row.id}: некорректный ответ/EOS")
        if len(ids) + 1 > conf.max_answer_tokens or len(ids) + 1 + len(prompt_ids) > conf.max_total_tokens:
            raise DPOError(f"{row.id}: token budget exceeded; no truncation")
        full = tuple(prompt_ids + ids + [tokenizer.eos_token_id])
        labels = tuple([-100] * len(prompt_ids) + ids + [tokenizer.eos_token_id])
        encoded.append((full, labels))
    return TokenizedPair(row.id, len(prompt_ids), *encoded[0], *encoded[1])


def torch_dpo_loss(policy_chosen: Any, policy_rejected: Any, reference_chosen: Any,
                   reference_rejected: Any, *, beta: float) -> Any:
    """Differentiable direct DPO с фиксированными ref logπ."""
    import torch.nn.functional as F
    margin = beta * ((policy_chosen - policy_rejected) - (reference_chosen - reference_rejected))
    return F.softplus(-margin).mean()


def completion_logprob(model: Any, item_ids: Sequence[int], labels: Sequence[int], *, device: str, torch: Any) -> Any:
    """Сумма сдвинутых logp только по tokens ответа, включая EOS."""
    if len(item_ids) != len(labels) or len(item_ids) < 2 or not any(x != -100 for x in labels[1:]):
        raise DPOError("Некорректные sequence/labels для DPO")
    input_ids = torch.tensor([list(item_ids)], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    prediction = logits[:, :-1, :].float()
    truth = torch.tensor([list(labels[1:])], dtype=torch.long, device=device)
    valid = truth != -100
    if not bool(valid.any()):
        raise DPOError("Нет supervised tokens")
    log_probs = prediction.log_softmax(dim=-1)
    gathered = log_probs.gather(-1, truth.clamp_min(0).unsqueeze(-1)).squeeze(-1)
    return (gathered * valid).sum(dim=-1).squeeze(0)


def _atomic(path: Path, content: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise DPOError(f"Файл уже существует: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists() or temporary.is_symlink():
        raise DPOError(f"Временный файл уже существует: {temporary}")
    temporary.write_bytes(content)
    temporary.replace(path)


def _runtime_versions() -> dict[str, str]:
    from importlib.metadata import version, PackageNotFoundError
    out = {}
    for key in ("torch", "transformers", "peft", "safetensors"):
        try:
            out[key] = version(key)
        except PackageNotFoundError as exc:
            raise DPOError(f"Для real DPO отсутствует библиотека {key}") from exc
    return out


def _model_and_tokenizer(conf: Config, *, root: Path, profile: str, allow_download: bool) -> tuple[Any, Any, Any, dict[str, Any], dict[str, str]]:
    """Поднять SFT policy с trainable LoRA; reference — предобученные logp до updates."""
    deps = _runtime_versions()
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from peft import PeftModel
    except ImportError as exc:
        raise DPOError(f"Ошибка импорта ML-библиотек: {exc}") from exc
    sft = load_sft_config(conf.sft_config_file, root=root)
    sft_manifest = verify_sft(sft, root=root, profile=conf.sft_profile)
    if profile == "gpu" and not torch.cuda.is_available():
        raise DPOError("GPU профиль требует CUDA; используйте cpu для smoke")
    if profile == "gpu" and not torch.cuda.is_bf16_supported():
        raise DPOError("GPU профиль требует BF16 (не меняем dtype скрытно)")
    torch.manual_seed(conf.seed)
    torch.set_num_threads(conf.cpu_threads)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(conf.seed)
    dtype = torch.float32 if profile == "cpu" else torch.bfloat16
    device = "cpu" if profile == "cpu" else "cuda"
    kw = {"revision": sft.model_revision, "local_files_only": not allow_download, "trust_remote_code": False}
    try:
        tokenizer = AutoTokenizer.from_pretrained(sft.model_id, use_fast=True,
                                                   revision=sft.tokenizer_revision,
                                                   local_files_only=not allow_download, trust_remote_code=False)
        major = int(deps["transformers"].split(".")[0])
        dtype_arg = "dtype" if major >= 5 else "torch_dtype"
        base = AutoModelForCausalLM.from_pretrained(sft.model_id, use_safetensors=True, **kw, **{dtype_arg: dtype})
        if base.config.model_type != "qwen2" or base.config.eos_token_id != tokenizer.eos_token_id:
            raise DPOError("Qwen2/EOS несовместимы")
        model = PeftModel.from_pretrained(base, str(sft.output_dir / conf.sft_profile / "adapter"), is_trainable=True)
        model.to(device)
    except Exception as exc:
        raise DPOError(f"Не удалось загрузить модель/адаптер: {type(exc).__name__}: {exc}") from exc
    trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    if not trainable or any("lora_" not in name for name, _ in trainable):
        raise DPOError("В обучении должны участвовать только LoRA-параметры")
    if sum(p.numel() for _, p in trainable) != sft_manifest["trainable_parameters"]:
        raise DPOError("Число trainable parameters не совпадает с F4.5")
    model.config.use_cache = False
    return model, tokenizer, torch, sft_manifest, deps


def run_real(conf: Config, *, root: Path, profile: str, allow_download: bool = False) -> dict[str, Any]:
    """Опциональное реальное DPO, никакого синтетического SUCCESS под видом SFT/DPO."""
    if profile not in conf.profiles:
        raise DPOError("Неизвестный профиль")
    plan, pairs = make_plan(conf, root=root)
    destination = conf.output_dir / profile
    if destination.exists() and (destination.is_symlink() or any(destination.iterdir())):
        raise DPOError(f"Каталог уже содержит файлы: {destination}; перезапись запрещена")
    model, tokenizer, torch, sft_manifest, deps = _model_and_tokenizer(conf, root=root, profile=profile, allow_download=allow_download)
    p = conf.profiles[profile]
    train = [tokenize_pair(row, tokenizer, conf) for row in pairs["train"][:p.train_pairs]]
    validation = [tokenize_pair(row, tokenizer, conf) for row in pairs["validation"][:p.validation_pairs]]
    device = "cpu" if profile == "cpu" else "cuda"
    if profile == "gpu":
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    # Reference = СТРОГО первоначальный SFT-LoRA, до любого шага DPO.
    # Кэшируем именно суммарные logπ ответа; их значения далее не пересчитываем.
    all_pairs = train + validation
    ref_values: dict[str, tuple[float, float]] = {}
    model.eval()
    with torch.no_grad():
        for pair in all_pairs:
            chosen = completion_logprob(model, pair.chosen_ids, pair.chosen_labels, device=device, torch=torch)
            rejected = completion_logprob(model, pair.rejected_ids, pair.rejected_labels, device=device, torch=torch)
            ref_values[pair.id] = (float(chosen.item()), float(rejected.item()))
    if len(ref_values) != len(all_pairs) or any(not math.isfinite(t) for v in ref_values.values() for t in v):
        raise DPOError("Некорректные frozen reference logπ")

    def evaluate_margin() -> float:
        model.eval()
        margins: list[float] = []
        with torch.no_grad():
            for item in validation:
                pc = float(completion_logprob(model, item.chosen_ids, item.chosen_labels, device=device, torch=torch).item())
                pr = float(completion_logprob(model, item.rejected_ids, item.rejected_labels, device=device, torch=torch).item())
                rc, rr = ref_values[item.id]
                _, margin, _ = dpo_loss(pc, pr, rc, rr, conf.beta)
                margins.append(margin)
        return sum(margins) / len(margins)

    initial_validation_margin = evaluate_margin()
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=p.learning_rate, weight_decay=0.0)
    records = []
    for step in range(1, p.max_steps + 1):
        model.train()
        item = train[(step - 1) % len(train)]
        optimizer.zero_grad(set_to_none=True)
        pc = completion_logprob(model, item.chosen_ids, item.chosen_labels, device=device, torch=torch)
        pr = completion_logprob(model, item.rejected_ids, item.rejected_labels, device=device, torch=torch)
        rc, rr = ref_values[item.id]
        loss = torch_dpo_loss(pc, pr, pc.new_tensor(rc), pc.new_tensor(rr), beta=conf.beta)
        if not bool(torch.isfinite(loss)):
            raise DPOError("DPO loss не конечен")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        if not bool(torch.isfinite(norm)):
            raise DPOError("DPO norm не конечен")
        optimizer.step()
        _, margin, prob = dpo_loss(float(pc.detach().item()), float(pr.detach().item()), rc, rr, conf.beta)
        records.append({"step": step, "pair_id": item.id, "loss": float(loss.item()),
                        "preclip_gradient_norm": float(norm.item()), "margin_pre_step": margin,
                        "chosen_win_probability_pre_step": prob})
    final_validation_margin = evaluate_margin()
    adapter_dir = destination / "adapter"
    if destination.exists() and any(destination.iterdir()):
        raise DPOError("Каталог результата неожиданно уже заполнен")
    adapter_dir.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(str(adapter_dir), safe_serialization=True)
    files = {f.name: sha256(_read(f)).hexdigest() for f in adapter_dir.iterdir() if f.is_file()}
    if not {"adapter_config.json", "adapter_model.safetensors"} <= set(files):
        raise DPOError("PEFT не сохранил обязательные артефакты")
    metrics_bytes = b"".join(_canonical(row) for row in records)
    manifest = {
        "schema_version": 1, "stage": "F4.8", "profile": profile,
        "real_dpo_performed": True, "optimizer_steps": len(records),
        "model_id": plan["model_id"], "model_revision": plan["revision"],
        "reference": plan["reference"], "reference_frozen": True,
        "reference_logprobs_precomputed_before_updates": True,
        "config_sha256": conf.config_sha256,
        "preference_sha256": plan["preference_sha256"],
        "f41_train_sha256": plan["f41_train_sha256"],
        "f41_validation_sha256": plan["f41_validation_sha256"],
        "f43_holdout_sha256": plan["f43_holdout_sha256"],
        "source_sft_manifest_sha256": sha256(_read(load_sft_config(conf.sft_config_file, root=root).output_dir / conf.sft_profile / "run_manifest.json")).hexdigest(),
        "source_sft_adapter_sha256": sft_manifest["adapter_files_sha256"],
        "f43_used_for_training": False, "f47_human_grades_used_for_training": False,
        "train_pairs_used": len(train), "validation_pairs_used": len(validation),
        "trainable_parameters": sum(p.numel() for p in trainable),
        "beta": conf.beta, "learning_rate_configured": p.learning_rate,
        "initial_validation_margin": initial_validation_margin,
        "final_validation_margin": final_validation_margin,
        "manual_quality_score": None, "quality_improvement_proven": False,
        "versions": deps, "adapter_files_sha256": files,
        "metrics_sha256": sha256(metrics_bytes).hexdigest(),
    }
    _atomic(destination / "metrics.jsonl", metrics_bytes)
    _atomic(destination / "run_manifest.json", _pretty(manifest))
    return manifest


def verify_run(conf: Config, *, root: Path, profile: str) -> dict[str, Any]:
    """Безопасно перепроверить артефакты и исходные F4.1/F4.5; без ML-импортов."""
    if profile not in conf.profiles:
        raise DPOError("Неизвестный профиль")
    plan, _ = make_plan(conf, root=root)
    sft = load_sft_config(conf.sft_config_file, root=root)
    sft_manifest = verify_sft(sft, root=root, profile=conf.sft_profile)
    directory = conf.output_dir / profile
    manifest = _json(_read(directory / "run_manifest.json"), "run_manifest.json")
    if not isinstance(manifest, dict) or manifest.get("stage") != "F4.8" or manifest.get("profile") != profile or not manifest.get("real_dpo_performed"):
        raise DPOError("Не является проверенным настоящим DPO run")
    for key, value in (("config_sha256", conf.config_sha256), ("preference_sha256", plan["preference_sha256"]),
                       ("f41_train_sha256", plan["f41_train_sha256"]),
                       ("f41_validation_sha256", plan["f41_validation_sha256"]),
                       ("f43_holdout_sha256", plan["f43_holdout_sha256"]),
                       ("model_revision", plan["revision"]),
                       ("model_id", plan["model_id"]),
                       ("beta", conf.beta),
                       ("train_pairs_used", conf.profiles[profile].train_pairs),
                       ("validation_pairs_used", conf.profiles[profile].validation_pairs),
                       ("source_sft_adapter_sha256", sft_manifest["adapter_files_sha256"]),
                       ("source_sft_manifest_sha256", sha256(_read(sft.output_dir / conf.sft_profile / "run_manifest.json")).hexdigest())):
        if manifest.get(key) != value:
            raise DPOError(f"Изменён обязательный контекст: {key}")
    if (manifest.get("reference") != plan["reference"] or manifest.get("reference_frozen") is not True
            or manifest.get("reference_logprobs_precomputed_before_updates") is not True):
        raise DPOError("Неверная reference policy")
    n = manifest.get("optimizer_steps")
    if type(n) is not int or n != conf.profiles[profile].max_steps or n <= 0:
        raise DPOError("Неверное число реальных шагов")
    data = _read(directory / "metrics.jsonl")
    if sha256(data).hexdigest() != manifest.get("metrics_sha256"):
        raise DPOError("Изменён журнал метрик")
    lines = data.splitlines()
    if len(lines) != n:
        raise DPOError("Число строк метрик не совпадает")
    for index, line in enumerate(lines, 1):
        row = _json(line, f"metric {index}")
        if not isinstance(row, dict) or row.get("step") != index:
            raise DPOError("Повреждён журнал шагов")
        for name in ("loss", "preclip_gradient_norm", "margin_pre_step", "chosen_win_probability_pre_step"):
            value = row.get(name)
            if type(value) not in (float, int) or not math.isfinite(value):
                raise DPOError("Некорректная метрика")
    files = manifest.get("adapter_files_sha256")
    if not isinstance(files, dict) or not {"adapter_model.safetensors", "adapter_config.json"} <= set(files):
        raise DPOError("Отсутствуют веса адаптера")
    for name, expected in files.items():
        if name not in {"adapter_model.safetensors", "adapter_config.json", "README.md"} or not isinstance(expected, str) or not HEX64.fullmatch(expected):
            raise DPOError("Недопустимое имя/хеш адаптера")
        if sha256(_read(directory / "adapter" / name)).hexdigest() != expected:
            raise DPOError(f"Повреждён файл адаптера: {name}")
    if manifest.get("manual_quality_score") is not None or manifest.get("quality_improvement_proven") is not False:
        raise DPOError("Нельзя объявлять качество доказанным")
    return manifest
