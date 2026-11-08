"""F4.3: фиксированный независимый evaluation set и прозрачная ручная рубрика.

Только стандартная библиотека и компоненты F4.1/F4.2. Веса модели и
эталонные ответы никогда не включаются в экспорт заданий для inference.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence
import unicodedata

from src.llm_f4.instruction_dataset import load_examples, normalized_prompt
from src.llm_f4.model_selection import load_catalog

DOMAINS = ("python", "linux", "sql", "ml")
DIFFICULTIES = ("basic", "intermediate")
ORIGIN = "authored_synthetic"
CASE_ID = re.compile(r"ev_(?:py|lx|sq|ml)_[0-9]{2}\Z")
TOPIC_ID = re.compile(r"[a-z][a-z0-9_]{3,63}\Z")
FACT_ID = re.compile(r"[a-z][a-z0-9_]{1,47}\Z")
SCHEMA_FIELDS = {"case_id", "domain", "topic_id", "difficulty", "instruction", "input", "reference_answer", "rubric", "origin"}
RUBRIC_FIELDS = {"required_facts", "review_checks"}
FACT_FIELDS = {"fact_id", "description", "aliases"}
SOURCE_KEYS = {"schema_version", "evaluation_id", "source_file", "instruction_source_file", "model_catalog_file", "output_dir", "expected_cases", "domain_quotas", "instruction_source_sha256", "model_catalog_sha256", "evaluation_source_sha256"}


class EvaluationError(ValueError):
    """Некорректный benchmark, обнаруженная утечка или изменённый артефакт."""


@dataclass(frozen=True, slots=True)
class Fact:
    """Обязательный смысловой пункт для экспертного оценивания."""

    fact_id: str
    description: str
    aliases: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class EvaluationCase:
    """Задание, его эталон и отдельная рубрика ручной оценки."""

    case_id: str
    domain: str
    topic_id: str
    difficulty: str
    instruction: str
    input: str
    reference_answer: str
    required_facts: tuple[Fact, ...]
    review_checks: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    """Один утверждённый evaluation set, связанный с F4.1 и F4.2."""

    evaluation_id: str
    source_file: Path
    instruction_source_file: Path
    model_catalog_file: Path
    output_dir: Path
    expected_cases: int
    domain_quotas: Mapping[str, int]
    instruction_source_sha256: str
    model_catalog_sha256: str
    evaluation_source_sha256: str


def _fail_constant(value: str) -> None:
    raise EvaluationError(f"Некорректная JSON-константа: {value}")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise EvaluationError(f"Повторный ключ JSON: {key}")
        result[key] = value
    return result


def _parse_json(line: str, where: str) -> Any:
    try:
        return json.loads(line, parse_constant=_fail_constant, object_pairs_hook=_reject_duplicate_keys)
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise EvaluationError(f"Некорректный JSON {where}: {exc}") from exc


def _read_bytes(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except OSError as exc:
        raise EvaluationError(f"Не удалось прочитать {path}: {exc}") from exc


def _read_utf8(path: Path) -> str:
    try:
        return _read_bytes(path).decode("utf-8-sig")
    except UnicodeError as exc:
        raise EvaluationError(f"Некорректный UTF-8 в {path}: {exc}") from exc


def _safe_path(root: Path, value: Any, name: str) -> Path:
    if (not isinstance(value, str) or not value or "\\" in value or ":" in value
            or Path(value).is_absolute() or any(part in (".", "..", "") for part in value.split("/"))):
        raise EvaluationError(f"{name}: требуется относительный POSIX-путь внутри репозитория")
    base = root.resolve()
    path = (base / value).resolve()
    if not path.is_relative_to(base):
        raise EvaluationError(f"{name}: путь выходит за пределы репозитория")
    return path


def load_config(path: Path, *, root: Path) -> EvaluationConfig:
    """Прочитать строгую конфигурацию и проверить версии входных контрактов."""
    raw = _parse_json(_read_utf8(path), str(path))
    if not isinstance(raw, dict) or set(raw) != SOURCE_KEYS or raw["schema_version"] != 1 or type(raw["schema_version"]) is not int:
        raise EvaluationError("Неверная схема конфигурации F4.3")
    if not isinstance(raw["evaluation_id"], str) or not re.fullmatch(r"f4_03_[a-z0-9_]+_v1", raw["evaluation_id"]):
        raise EvaluationError("Неверный evaluation_id")
    quota = raw["domain_quotas"]
    if (not isinstance(quota, dict) or set(quota) != set(DOMAINS)
            or any(type(quota[d]) is not int or quota[d] < 1 for d in DOMAINS)):
        raise EvaluationError("domain_quotas: требуются четыре положительные целые квоты")
    n = raw["expected_cases"]
    if type(n) is not int or n != sum(quota.values()) or n < 8:
        raise EvaluationError("expected_cases не соответствует квотам")
    for field in ("instruction_source_sha256", "model_catalog_sha256", "evaluation_source_sha256"):
        if not isinstance(raw[field], str) or not re.fullmatch(r"[0-9a-f]{64}", raw[field]):
            raise EvaluationError(f"{field}: ожидается полный SHA-256")
    return EvaluationConfig(
        evaluation_id=raw["evaluation_id"],
        source_file=_safe_path(root, raw["source_file"], "source_file"),
        instruction_source_file=_safe_path(root, raw["instruction_source_file"], "instruction_source_file"),
        model_catalog_file=_safe_path(root, raw["model_catalog_file"], "model_catalog_file"),
        output_dir=_safe_path(root, raw["output_dir"], "output_dir"),
        expected_cases=n,
        domain_quotas={d: quota[d] for d in DOMAINS},
        instruction_source_sha256=raw["instruction_source_sha256"],
        model_catalog_sha256=raw["model_catalog_sha256"],
        evaluation_source_sha256=raw["evaluation_source_sha256"],
    )


def _text(value: Any, where: str, *, min_len: int = 1, max_len: int = 3000) -> str:
    if (not isinstance(value, str) or value != value.strip() or not min_len <= len(value) <= max_len
            or any(unicodedata.category(c) == "Cc" and c != "\n" for c in value)):
        raise EvaluationError(f"{where}: некорректная строка")
    return value


def _strings(raw: Any, where: str, *, lower: int = 1, upper: int = 5) -> tuple[str, ...]:
    if not isinstance(raw, list) or not lower <= len(raw) <= upper:
        raise EvaluationError(f"{where}: неверное число строк")
    values = tuple(_text(s, where, min_len=3, max_len=300) for s in raw)
    if len(set(s.casefold() for s in values)) != len(values):
        raise EvaluationError(f"{where}: повторяющиеся значения")
    return values


def _load_case(raw: Any, line: int) -> EvaluationCase:
    if not isinstance(raw, dict) or set(raw) != SCHEMA_FIELDS:
        raise EvaluationError(f"Строка {line}: неизвестные/пропущенные поля задания")
    cid = raw["case_id"]
    domain = raw["domain"]
    if not isinstance(cid, str) or not CASE_ID.fullmatch(cid):
        raise EvaluationError(f"Строка {line}: неверный case_id")
    domain_prefix = {"python": "py", "linux": "lx", "sql": "sq", "ml": "ml"}
    if domain not in DOMAINS or not cid.startswith(f"ev_{domain_prefix[domain]}_"):
        raise EvaluationError(f"Строка {line}: неверный domain/case_id")
    topic = raw["topic_id"]
    if not isinstance(topic, str) or not TOPIC_ID.fullmatch(topic):
        raise EvaluationError(f"Строка {line}: неверный topic_id")
    if raw["difficulty"] not in DIFFICULTIES or raw["origin"] != ORIGIN:
        raise EvaluationError(f"Строка {line}: неверная сложность/origin")
    instruction = _text(raw["instruction"], "instruction", min_len=20, max_len=600)
    context = _text(raw["input"], "input", min_len=0, max_len=1000)
    answer = _text(raw["reference_answer"], "reference_answer", min_len=40, max_len=1500)
    rubric = raw["rubric"]
    if not isinstance(rubric, dict) or set(rubric) != RUBRIC_FIELDS:
        raise EvaluationError(f"Строка {line}: неверная рубрика")
    req = rubric["required_facts"]
    if not isinstance(req, list) or not 2 <= len(req) <= 4:
        raise EvaluationError(f"Строка {line}: требуется 2-4 смысловых пункта")
    facts: list[Fact] = []
    for item in req:
        if not isinstance(item, dict) or set(item) != FACT_FIELDS:
            raise EvaluationError(f"Строка {line}: неверный обязательный пункт")
        fid = item["fact_id"]
        if not isinstance(fid, str) or not FACT_ID.fullmatch(fid):
            raise EvaluationError(f"Строка {line}: неверный fact_id")
        facts.append(Fact(fid, _text(item["description"], "description", min_len=10, max_len=250),
                          _strings(item["aliases"], "aliases", lower=1, upper=5)))
    if len({x.fact_id for x in facts}) != len(facts):
        raise EvaluationError(f"Строка {line}: повторный fact_id")
    checks = _strings(rubric["review_checks"], "review_checks", lower=1, upper=3)
    return EvaluationCase(cid, domain, topic, raw["difficulty"], instruction, context, answer,
                          tuple(facts), checks)


def load_cases(path: Path) -> list[EvaluationCase]:
    """Прочитать gold-данные (не передавать эту функцию inference-конвейеру)."""
    lines = _read_utf8(path).splitlines()
    if not lines or any(not line.strip() for line in lines):
        raise EvaluationError("Пустой evaluation JSONL или пустая строка")
    cases = [_load_case(_parse_json(line, f"строка {i}"), i) for i, line in enumerate(lines, 1)]
    ids = [x.case_id for x in cases]
    topics = [(x.domain, x.topic_id) for x in cases]
    if len(set(ids)) != len(ids) or len(set(topics)) != len(topics):
        raise EvaluationError("Повторный case_id или topic_id")
    return cases


def _tokens(text: str) -> tuple[str, ...]:
    return tuple(re.findall(r"\w+", unicodedata.normalize("NFKC", text).casefold(), re.UNICODE))


def _ngram_similarity(left: str, right: str, n: int = 3) -> float:
    a, b = _tokens(left), _tokens(right)
    # Множества 3-грамм слов; только дополнительный эвристический сигнал.
    sa = {a[i:i+n] for i in range(max(0, len(a) - n + 1))}
    sb = {b[i:i+n] for i in range(max(0, len(b) - n + 1))}
    return len(sa & sb) / len(sa | sb) if sa or sb else 0.0


def _prompt_key(instruction: str, context: str) -> str:
    # Идентично F4.1 normalized_prompt: instruction + input, NFKC + casefold.
    combined = f"{instruction} \u241f {context}"
    return " ".join(_tokens(combined))


def validate_cases(cases: Sequence[EvaluationCase], config: EvaluationConfig,
                   instruction_rows: Sequence[Any]) -> None:
    """Проверить квоты, уникальность, буквальные/почти-буквальные пересечения."""
    if len(cases) != config.expected_cases:
        raise EvaluationError(f"Требуется ровно {config.expected_cases} кейсов")
    for domain in DOMAINS:
        if sum(x.domain == domain for x in cases) != config.domain_quotas[domain]:
            raise EvaluationError(f"Нарушена квота {domain}")
    # Повторная валидация нужна и при вызове API со сконструированными объектами.
    json_rows = []
    for c in cases:
        json_rows.append({"case_id": c.case_id, "domain": c.domain, "topic_id": c.topic_id,
                          "difficulty": c.difficulty, "instruction": c.instruction, "input": c.input,
                          "reference_answer": c.reference_answer, "origin": ORIGIN,
                          "rubric": {"required_facts": [asdict(f) | {"aliases": list(f.aliases)} for f in c.required_facts],
                                     "review_checks": list(c.review_checks)}})
    reloaded = [_load_case(row, n) for n, row in enumerate(json_rows, 1)]
    if len({x.case_id for x in reloaded}) != len(reloaded) or len({(x.domain, x.topic_id) for x in reloaded}) != len(reloaded):
        raise EvaluationError("Повторные ID/темы в evaluation")
    old_keys = {normalized_prompt(x) for x in instruction_rows}
    seen: set[str] = set()
    for case in cases:
        key = _prompt_key(case.instruction, case.input)
        if key in old_keys or key in seen:
            raise EvaluationError(f"Повторная инструкция в F4.1/evaluation: {case.case_id}")
        seen.add(key)
    for i, case in enumerate(cases):
        for other in cases[i+1:]:
            if _ngram_similarity(case.instruction, other.instruction) >= 0.85:
                raise EvaluationError(f"Похожие формулировки двух evaluation-кейсов: {case.case_id}/{other.case_id}")
        for row in instruction_rows:
            if _ngram_similarity(case.instruction, row.instruction) >= 0.85:
                raise EvaluationError(f"Почти буквальное совпадение с F4.1: {case.case_id}/{row.id}")
    # Не допускаем неработающих эталонных алиасов: проверка только текстовая.
    for case in cases:
        matched = lexical_coverage(case, case.reference_answer)
        if matched["matched_count"] != matched["total_facts"]:
            raise EvaluationError(f"Эталон не покрывает формальную рубрику: {case.case_id}")


def lexical_coverage(case: EvaluationCase, answer: str) -> dict[str, Any]:
    """Только диагностическое совпадение терминов; НЕ оценка правильности LLM.

    Из-за отрицаний/синонимов текстовая проверка может ложно срабатывать и
    ложно отклонять ответ. Результат не используется для quality benchmark.
    """
    if not isinstance(answer, str):
        raise EvaluationError("answer: ожидается строка")
    normalized = " ".join(_tokens(answer))
    matched: list[str] = []
    for fact in case.required_facts:
        if any(re.search(r"(?<!\w)" + re.escape(" ".join(_tokens(alias))) + r"(?!\w)", normalized)
               for alias in fact.aliases):
            matched.append(fact.fact_id)
    return {"matched_fact_ids": matched, "matched_count": len(matched),
            "total_facts": len(case.required_facts), "lexical_only": True}


def model_prompt(case: EvaluationCase) -> dict[str, str]:
    """Строго ограниченный набор полей: НИ reference, НИ rubric, НИ topic_id."""
    prompt = case.instruction
    if case.input:
        prompt = f"{prompt}\n\nДанные:\n{case.input}"
    return {"case_id": case.case_id, "domain": case.domain, "prompt": prompt}


def _json_line(obj: Any) -> bytes:
    return (json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def prepare_artifacts(config: EvaluationConfig) -> dict[str, bytes]:
    """Проверить версии входов и построить prompts.jsonl и text-free manifest."""
    source = _read_bytes(config.source_file)
    if sha256(source).hexdigest() != config.evaluation_source_sha256:
        raise EvaluationError("F4.3 evaluation source SHA-256 не совпадает с зафиксированным")
    instruction_source = _read_bytes(config.instruction_source_file)
    model_catalog = _read_bytes(config.model_catalog_file)
    if sha256(instruction_source).hexdigest() != config.instruction_source_sha256:
        raise EvaluationError("F4.1 source SHA-256 не совпадает с зафиксированным")
    if sha256(model_catalog).hexdigest() != config.model_catalog_sha256:
        raise EvaluationError("F4.2 catalog SHA-256 не совпадает с зафиксированным")
    rows = load_examples(config.instruction_source_file)
    catalog = load_catalog(config.model_catalog_file)
    cases = load_cases(config.source_file)
    validate_cases(cases, config, rows)
    prompt_bytes = b"".join(_json_line(model_prompt(c)) for c in cases)
    difficulty_counts = {d: sum(c.difficulty == d for c in cases) for d in DIFFICULTIES}
    manifest = {
        "schema_version": 1, "evaluation_id": config.evaluation_id,
        "purpose": "frozen_holdout_for_base_vs_finetuned",
        "origin": ORIGIN, "model_id": catalog.selected_model_id,
        "model_revision": next(c.revision for c in catalog.candidates if c.model_id == catalog.selected_model_id),
        "source_sha256": sha256(source).hexdigest(),
        "instruction_source_sha256": sha256(instruction_source).hexdigest(),
        "model_catalog_sha256": sha256(model_catalog).hexdigest(),
        "prompts_sha256": sha256(prompt_bytes).hexdigest(),
        "total_cases": len(cases),
        "domain_counts": {d: sum(c.domain == d for c in cases) for d in DOMAINS},
        "difficulty_counts": difficulty_counts,
        "required_facts_total": sum(len(c.required_facts) for c in cases),
        "grading": "manual_fact_scores_0_1_2_macro_per_case",
        "generation_performed": False,
        "model_evaluated": False,
    }
    return {"prompts.jsonl": prompt_bytes, "manifest.json": json.dumps(manifest, ensure_ascii=False,
            sort_keys=True, indent=2).encode("utf-8") + b"\n"}


def export_artifacts(config: EvaluationConfig) -> dict[str, bytes]:
    """Записать prompts только после успешной проверки всех входов."""
    payloads = prepare_artifacts(config)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    for name, content in payloads.items():
        (config.output_dir / name).write_bytes(content)
    return payloads


def verify_artifacts(config: EvaluationConfig) -> dict[str, bytes]:
    """Проверить текущий экспорт побайтово против исходных данных."""
    expected = prepare_artifacts(config)
    for name, content in expected.items():
        if _read_bytes(config.output_dir / name) != content:
            raise EvaluationError(f"Файл {name} изменён после экспорта")
    return expected


def score_manual_grades(cases: Sequence[EvaluationCase], grades: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Агрегировать оценки эксперта 0/1/2 за каждый обязательный факт.

    Данные с моделью и экспертными оценками поступят только в F4.4/F4.7;
    здесь тестируем исключительно арифметику на искусственных оценках.
    """
    by_id = {c.case_id: c for c in cases}
    if len(by_id) != len(cases) or len(grades) != len(cases):
        raise EvaluationError("Каждый case_id должен получить ровно одну оценку")
    seen: set[str] = set()
    per_case: dict[str, dict[str, Any]] = {}
    by_domain: dict[str, list[float]] = {d: [] for d in DOMAINS}
    for grade in grades:
        if not isinstance(grade, Mapping) or set(grade) != {"case_id", "fact_scores", "notes"}:
            raise EvaluationError("Неверная форма экспертной оценки")
        case_id = grade["case_id"]
        if not isinstance(case_id, str) or case_id not in by_id or case_id in seen:
            raise EvaluationError("Неизвестный или повторный case_id в экспертных оценках")
        seen.add(case_id)
        case = by_id[case_id]
        fs = grade["fact_scores"]
        expected = {f.fact_id for f in case.required_facts}
        if not isinstance(fs, dict) or set(fs) != expected or any(type(x) is not int or x not in (0, 1, 2) for x in fs.values()):
            raise EvaluationError(f"Неверные баллы обязательных фактов {case_id}")
        _text(grade["notes"], "notes", min_len=1, max_len=500)
        score = sum(fs.values()) / (2 * len(expected))
        per_case[case_id] = {"domain": case.domain, "score": score,
                             "points": sum(fs.values()), "max_points": 2 * len(expected)}
        by_domain[case.domain].append(score)
    if set(per_case) != set(by_id):
        raise EvaluationError("Неполное покрытие экспертной оценкой")
    return {
        "per_case": dict(sorted(per_case.items())),
        "macro_over_cases": sum(x["score"] for x in per_case.values()) / len(per_case),
        "macro_by_domain": {domain: sum(scores) / len(scores) if scores else None
                            for domain, scores in by_domain.items()},
        "scoring_method": "manual_fact_scores_0_1_2",
    }
