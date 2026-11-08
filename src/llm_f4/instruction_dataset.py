"""F4.1: валидация, разделение и воспроизводимый экспорт instruction dataset.

В этом модуле нет токенизации, вызовов LLM или обучения: единица
разделения — группа тематически связанных примеров (group_id).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any, Mapping
import unicodedata

SPLIT_NAMES = ("train", "validation", "test")
FIELDS = frozenset({"id", "group_id", "domain", "instruction", "input", "response", "origin"})
ID_PATTERN = re.compile(r"[a-z][a-z0-9_-]{2,63}\Z")
DOMAIN_PATTERN = re.compile(r"[a-z][a-z0-9_-]{1,63}\Z")
ORIGIN = "authored_synthetic"


class DatasetError(ValueError):
    """Ошибка формата данных, конфигурации или целостности split."""


@dataclass(frozen=True, slots=True)
class InstructionExample:
    """Один ответ на инструкцию с явно указанной группой утечки."""

    id: str
    group_id: str
    domain: str
    instruction: str
    input: str
    response: str
    origin: str


@dataclass(frozen=True, slots=True)
class SplitConfig:
    """Фиксированный seed и целочисленные доли в процентах."""

    seed: int
    ratios: Mapping[str, int]
    source_file: Path
    output_dir: Path


def _reject_constant(value: str) -> None:
    raise DatasetError(f"Некорректная JSON-константа: {value}")


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"), parse_constant=_reject_constant)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DatasetError(f"Не удалось прочитать JSON {path}: {exc}") from exc


def _check_text(name: str, value: Any, *, min_length: int, max_length: int) -> str:
    if not isinstance(value, str):
        raise DatasetError(f"{name}: ожидалась строка")
    if value != value.strip():
        raise DatasetError(f"{name}: пробелы по краям запрещены")
    if not min_length <= len(value) <= max_length or (min_length and not value.strip()):
        raise DatasetError(f"{name}: допустимая длина {min_length}..{max_length}")
    if any(unicodedata.category(char) == "Cc" and char != "\n" for char in value):
        raise DatasetError(f"{name}: недопустимые управляющие символы")
    return value


def _validate_record(raw: Any, *, row_number: int) -> InstructionExample:
    if not isinstance(raw, dict) or set(raw) != FIELDS:
        raise DatasetError(f"Строка {row_number}: требуются поля {sorted(FIELDS)} без лишних")
    for field in ("id", "group_id", "domain"):
        pattern = DOMAIN_PATTERN if field == "domain" else ID_PATTERN
        if not isinstance(raw[field], str) or not pattern.fullmatch(raw[field]):
            raise DatasetError(f"Строка {row_number}: некорректный {field}")
    if raw["origin"] != ORIGIN:
        raise DatasetError(f"Строка {row_number}: origin должен быть {ORIGIN}")
    for field, lo, hi in (("instruction", 12, 1000), ("input", 0, 2000), ("response", 20, 4000)):
        _check_text(f"Строка {row_number}, {field}", raw[field], min_length=lo, max_length=hi)
    return InstructionExample(**raw)


def normalized_prompt(example: InstructionExample) -> str:
    """Ключ буквального совпадения после Unicode NFKC/casefold/пунктуации.

    Это НЕ семантический детектор: смысловые парафразы связывает group_id.
    """
    combined = f"{example.instruction} \u241f {example.input}"
    combined = unicodedata.normalize("NFKC", combined).casefold()
    return " ".join(re.findall(r"\w+", combined, flags=re.UNICODE))


def validate_examples(examples: list[InstructionExample]) -> None:
    """Проверить уникальность ID/запроса и согласованность метаданных группы."""
    if not examples:
        raise DatasetError("Dataset пуст")
    ids: set[str] = set()
    prompts: dict[str, str] = {}
    group_domains: dict[str, str] = {}
    for idx, example in enumerate(examples, start=1):
        # Проверяем и записи, переданные напрямую из Python, а не только JSONL.
        _validate_record(asdict(example), row_number=idx)
        if example.id in ids:
            raise DatasetError(f"Повторный id: {example.id}")
        ids.add(example.id)
        key = normalized_prompt(example)
        if key in prompts:
            raise DatasetError(f"Повторный instruction+input: {prompts[key]} и {example.id}")
        prompts[key] = example.id
        previous_domain = group_domains.setdefault(example.group_id, example.domain)
        if previous_domain != example.domain:
            raise DatasetError(f"У группы {example.group_id} разные domain")


def load_examples(path: Path) -> list[InstructionExample]:
    """Прочитать JSONL с указанием номера испорченной строки."""
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except (OSError, UnicodeError) as exc:
        raise DatasetError(f"Не удалось прочитать JSONL {path}: {exc}") from exc
    result: list[InstructionExample] = []
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            raise DatasetError(f"Строка {number}: пустая строка в JSONL")
        try:
            payload = json.loads(line, parse_constant=_reject_constant)
        except json.JSONDecodeError as exc:
            raise DatasetError(f"Строка {number}: ошибка JSON: {exc}") from exc
        result.append(_validate_record(payload, row_number=number))
    validate_examples(result)
    return result


def load_config(path: Path, *, root: Path) -> SplitConfig:
    """Загрузить декларативные параметры и разрешить пути внутри репозитория."""
    raw = _read_json(path)
    fields = {"schema_version", "seed", "ratios", "source_file", "output_dir"}
    if not isinstance(raw, dict) or set(raw) != fields or raw["schema_version"] != 1:
        raise DatasetError("Конфигурация: неправильная версия или список полей")
    seed = raw["seed"]
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise DatasetError("seed: требуется целое число 0..2^32-1")
    ratios = raw["ratios"]
    if (not isinstance(ratios, dict) or set(ratios) != set(SPLIT_NAMES)
            or any(type(ratios[name]) is not int or ratios[name] <= 0 for name in SPLIT_NAMES)
            or sum(ratios.values()) != 100):
        raise DatasetError("ratios: три положительных целых процента с суммой 100")

    def resolve_path(key: str) -> Path:
        value = raw[key]
        if not isinstance(value, str) or not value or "\\" in value:
            raise DatasetError(f"{key}: ожидается POSIX-путь относительно корня")
        rel = Path(value)
        if rel.is_absolute() or any(part in (".", "..") for part in rel.parts):
            raise DatasetError(f"{key}: запрещён абсолютный путь или ..")
        root_path = root.resolve()
        resolved = (root_path / rel).resolve()
        if not resolved.is_relative_to(root_path):
            raise DatasetError(f"{key}: выход за пределы репозитория")
        return resolved

    return SplitConfig(
        seed=seed,
        ratios={name: ratios[name] for name in SPLIT_NAMES},
        source_file=resolve_path("source_file"),
        output_dir=resolve_path("output_dir"),
    )


def allocate_group_counts(total: int, ratios: Mapping[str, int]) -> dict[str, int]:
    """Выделить целые числа групп, оставляя хотя бы одну группу на split."""
    if total < len(SPLIT_NAMES):
        raise DatasetError("Для трёх непустых split нужно минимум три group_id")
    counts = {name: 1 for name in SPLIT_NAMES}
    # Последовательная выдача оставшихся групп по максимальному дефициту
    # относительно идеальной квоты (целочисленная арифметика).
    for _ in range(total - len(SPLIT_NAMES)):
        candidate = max(SPLIT_NAMES, key=lambda name: (total * ratios[name] - 100 * counts[name], -SPLIT_NAMES.index(name)))
        counts[candidate] += 1
    return counts


def split_examples(examples: list[InstructionExample], *, seed: int,
                   ratios: Mapping[str, int]) -> dict[str, list[InstructionExample]]:
    """Разделить группы целиком, независимо от порядка строк JSONL."""
    validate_examples(examples)
    groups: dict[str, list[InstructionExample]] = {}
    for example in examples:
        groups.setdefault(example.group_id, []).append(example)
    # hashlib стабилен между Windows/Linux и не зависит от hash randomization.
    ordered = sorted(groups, key=lambda group_id: (sha256(f"{seed}:{group_id}".encode("utf-8")).digest(), group_id))
    counts = allocate_group_counts(len(ordered), ratios)
    splits: dict[str, list[InstructionExample]] = {}
    start = 0
    for name in SPLIT_NAMES:
        selected = ordered[start:start + counts[name]]
        splits[name] = sorted((record for gid in selected for record in groups[gid]), key=lambda row: row.id)
        start += counts[name]
    validate_splits(splits, source=examples)
    return splits


def validate_splits(splits: Mapping[str, list[InstructionExample]],
                    *, source: list[InstructionExample] | None = None) -> None:
    """Убедиться в отсутствии пересечений ID, групп и буквальных запросов."""
    if set(splits) != set(SPLIT_NAMES):
        raise DatasetError("Нужны ровно train, validation, test")
    ids: set[str] = set()
    group_to_split: dict[str, str] = {}
    prompts: dict[str, str] = {}
    for name in SPLIT_NAMES:
        if not splits[name]:
            raise DatasetError(f"Пустая выборка {name}")
        validate_examples(splits[name])
        for example in splits[name]:
            if example.id in ids:
                raise DatasetError(f"Повторный id между выборками: {example.id}")
            ids.add(example.id)
            previous = group_to_split.setdefault(example.group_id, name)
            if previous != name:
                raise DatasetError(f"Утечка группы {example.group_id}: {previous} и {name}")
            key = normalized_prompt(example)
            if key in prompts:
                raise DatasetError(f"Совпадающий запрос между выборками: {example.id} и {prompts[key]}")
            prompts[key] = example.id
    if source is not None:
        if ids != {example.id for example in source}:
            raise DatasetError("Split не покрывает исходные записи ровно по одному разу")
        originals = {example.id: example for example in source}
        for name in SPLIT_NAMES:
            for example in splits[name]:
                if example != originals[example.id]:
                    raise DatasetError(f"Split изменил содержимое записи {example.id}")


def _json_bytes(obj: Any) -> bytes:
    return (json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def create_artifacts(examples: list[InstructionExample], config: SplitConfig,
                     *, source_bytes: bytes) -> dict[str, bytes]:
    """Сформировать детерминированные файлы split и text-free manifest в памяти."""
    splits = split_examples(examples, seed=config.seed, ratios=config.ratios)
    payloads: dict[str, bytes] = {}
    for name in SPLIT_NAMES:
        payloads[f"{name}.jsonl"] = b"".join(_json_bytes(asdict(row)) for row in splits[name])
    manifest = {
        "schema_version": 1,
        "algorithm": "sha256_group_order_v1",
        "origin": ORIGIN,
        "seed": config.seed,
        "ratios_percent": dict(config.ratios),
        "source_sha256": sha256(source_bytes).hexdigest(),
        "total_records": len(examples),
        "total_groups": len({example.group_id for example in examples}),
        "splits": {
            name: {
                "records": len(splits[name]),
                "groups": len({e.group_id for e in splits[name]}),
                "sha256": sha256(payloads[f"{name}.jsonl"]).hexdigest(),
            } for name in SPLIT_NAMES
        },
    }
    payloads["manifest.json"] = json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8") + b"\n"
    return payloads


def prepare_artifacts(config: SplitConfig) -> dict[str, bytes]:
    """Проверить содержимое файла и вернуть готовые к записи результаты."""
    examples = load_examples(config.source_file)
    try:
        source_bytes = config.source_file.read_bytes()
    except OSError as exc:
        raise DatasetError(f"Не удалось прочитать источник: {exc}") from exc
    return create_artifacts(examples, config, source_bytes=source_bytes)


def export_artifacts(config: SplitConfig) -> dict[str, bytes]:
    """Сохранить производные данные только после валидации всего корпуса."""
    payloads = prepare_artifacts(config)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    for name, payload in payloads.items():
        (config.output_dir / name).write_bytes(payload)
    return payloads


def verify_artifacts(config: SplitConfig) -> dict[str, bytes]:
    """Проверить split побайтово против текущего source, config и manifest."""
    expected = prepare_artifacts(config)
    for name, data in expected.items():
        path = config.output_dir / name
        try:
            actual = path.read_bytes()
        except OSError as exc:
            raise DatasetError(f"Нет файла результата {path}: {exc}") from exc
        if actual != data:
            raise DatasetError(f"Результат {name} не соответствует исходным данным и конфигурации")
    return expected
