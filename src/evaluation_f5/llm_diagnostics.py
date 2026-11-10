"""F5.2: воспроизводимая диагностика сохранённых Base/LoRA ответов.

Token overlap и диагностические флаги являются proxy, не экспертной оценкой.
В отчёты не копируются тексты заданий, эталонов или генераций.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import re
from typing import Any


class DiagnosticsError(ValueError):
    """Ошибка схемы, происхождения данных или запроса на публикацию."""


_TOKEN = re.compile(r"\w+|[^\w\s]", re.UNICODE)
_CASE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{1,79}$")
_LABEL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$")
_HEADER = re.compile(r"^##\s+(\S+)\s+[—-]\s+(\S+)\s*$", re.MULTILINE)
_FACT = re.compile(r"^-\s+`([A-Za-z0-9_-]+)`:\s*(.+)$")
_SECTION = re.compile(r"^\*\*(Вопрос|Данные|Эталон|Факты для проверки|Base|LoRA):\*\*\s*$", re.MULTILINE)
_MARKERS = re.compile(r"(?im)^\s*(?:вопрос|данные|ответ)\s*:")
_EXPECTED_KEYS = ("Вопрос", "Данные", "Эталон", "Факты для проверки", "Base", "LoRA")


def _path(root: Path, name: str, purpose: str) -> Path:
    """Принять только путь относительно корня, не позволяя traversal/symlink escape."""
    if not isinstance(name, str) or not name or "\\" in name or name.startswith("/"):
        raise DiagnosticsError(f"{purpose}: необходим относительный POSIX-путь")
    parts = name.split("/")
    if any(part in {"", ".", ".."} or ":" in part or "\x00" in part for part in parts):
        raise DiagnosticsError(f"{purpose}: небезопасный путь")
    root = root.resolve()
    target = root.joinpath(*parts).resolve()
    if not target.is_relative_to(root):
        raise DiagnosticsError(f"{purpose}: выход за корень репозитория")
    return target


def _json_load(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"), parse_constant=lambda v: (_ for _ in ()).throw(DiagnosticsError(f"Недопустимая константа JSON: {v}")))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DiagnosticsError(f"Невозможно прочитать JSON {path.name}: {exc}") from exc


def load_config(root: Path, name: str = "configs/f5_02_llm_diagnostics.json") -> dict[str, Any]:
    cfg = _json_load(_path(root, name, "config"))
    if not isinstance(cfg, dict) or type(cfg.get("schema_version")) is not int or cfg["schema_version"] != 1:
        raise DiagnosticsError("config: ожидается schema_version=1")
    paths = cfg.get("input_candidates")
    if not isinstance(paths, list) or not paths or len(set(paths)) != len(paths):
        raise DiagnosticsError("config: необходим уникальный непустой input_candidates")
    for candidate in paths:
        if not candidate.endswith((".jsonl", ".json", ".md")):
            raise DiagnosticsError("input_candidates: поддерживаются json/jsonl/md")
        _path(root, candidate, "input_candidate")
    for key, dirname, ext in (("json", "reports/public/f5_02/", ".json"), ("markdown", "docs/reports/F5_02_", ".md")):
        outs = cfg.get("outputs")
        if not isinstance(outs, dict) or not isinstance(outs.get(key), str):
            raise DiagnosticsError(f"config.outputs.{key}: отсутствует")
        dest = outs[key]
        if not dest.startswith(dirname) or not dest.endswith(ext):
            raise DiagnosticsError(f"config.outputs.{key}: путь вне разрешённой области")
        _path(root, dest, f"outputs.{key}")
    if set(cfg["outputs"]) != {"json", "markdown"} or cfg["outputs"]["json"] == cfg["outputs"]["markdown"]:
        raise DiagnosticsError("config.outputs: ожидаются только json и markdown")
    expected = cfg.get("expected_cases")
    facts = cfg.get("expected_facts_per_case")
    threshold = cfg.get("low_overlap_f1_threshold")
    if type(expected) is not int or expected < 1 or type(facts) is not int or facts < 1:
        raise DiagnosticsError("config: expected_cases/expected_facts_per_case должны быть положительными целыми")
    if isinstance(threshold, bool) or not isinstance(threshold, (float, int)) or not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise DiagnosticsError("config.low_overlap_f1_threshold: число от 0 до 1")
    return cfg


def _choose_input(root: Path, cfg: dict[str, Any], explicit: str | None) -> tuple[Path, str]:
    if explicit is not None:
        p = _path(root, explicit, "input")
        if not p.is_file() or p.suffix not in {".json", ".jsonl", ".md"}:
            raise DiagnosticsError(f"Явный input не найден или формат не поддерживается: {explicit}")
        return p, explicit
    found = [(c, _path(root, c, "input")) for c in cfg["input_candidates"] if _path(root, c, "input").is_file()]
    if not found:
        # В старых checkout имя подкаталога CPU могло отличаться. Искать
        # только явно именованные review-экспорты внутри каталогов F4.7.
        for folder in ("data/processed", "reports/public", "outputs", "artifacts"):
            parent = _path(root, folder, "discovery")
            if not parent.is_dir():
                continue
            for discovered in parent.rglob("human_review.*"):
                if not discovered.is_file() or discovered.suffix not in {".json", ".jsonl", ".md"}:
                    continue
                relative = discovered.relative_to(root).as_posix()
                if not any(part.lower().startswith("f4_07") for part in discovered.relative_to(parent).parts):
                    continue
                if _path(root, relative, "discovered").is_file():
                    found.append((relative, discovered))
    if not found:
        raise DiagnosticsError("Не найден F4.7 input: передай --input или положи экспорт в один из input_candidates")
    # Не выбирать молча один из разных результатов, полученных в разных запусках.
    if len(found) > 1:
        raise DiagnosticsError("Найдено несколько входов F4.7: " + ", ".join(c for c, _ in found) + "; укажи --input")
    return found[0][1], found[0][0]


def _field(item: dict[str, Any], names: tuple[str, ...], name: str, *, required: bool = True) -> Any:
    present = [(n, item[n]) for n in names if n in item]
    if not present:
        if required:
            raise DiagnosticsError(f"Отсутствует {name}: варианты {', '.join(names)}")
        return None
    if len(present) > 1 and any(value != present[0][1] for _, value in present[1:]):
        raise DiagnosticsError(f"Несогласованные альтернативы {name}")
    return present[0][1]


@dataclass(frozen=True)
class Case:
    case_id: str
    domain: str
    question: str
    reference: str
    base: str
    lora: str
    fact_ids: tuple[str, ...]
    raw_scores: tuple[Any, Any]
    reviewer: str | None


def _normalize(item: Any, index: int) -> Case:
    if not isinstance(item, dict):
        raise DiagnosticsError(f"Строка {index}: должен быть объект")
    case_id = _field(item, ("case_id", "id", "eval_id", "task_id"), "case_id")
    if not isinstance(case_id, str) or not _CASE_ID.fullmatch(case_id):
        raise DiagnosticsError(f"Строка {index}: некорректный case_id")
    domain = _field(item, ("domain", "category"), "domain")
    if not isinstance(domain, str) or not _LABEL.fullmatch(domain):
        raise DiagnosticsError(f"{case_id}: некорректный domain")
    question = _field(item, ("question", "prompt", "instruction"), "question")
    reference = _field(item, ("reference", "reference_answer", "expected_answer", "ground_truth"), "reference")
    base = _field(item, ("base_response", "base_answer", "base_text", "base_output"), "base")
    lora = _field(item, ("lora_response", "lora_answer", "lora_text", "lora_output", "sft_response"), "lora")
    for name, value in (("question", question), ("reference", reference), ("base", base), ("lora", lora)):
        if not isinstance(value, str) or (name in ("question", "reference") and not value.strip()):
            raise DiagnosticsError(f"{case_id}: некорректное текстовое поле {name}")
    facts = _field(item, ("facts", "criteria", "rubric_facts", "checkpoints"), "facts")
    if not isinstance(facts, list):
        raise DiagnosticsError(f"{case_id}: facts/criteria должен быть массивом")
    ids = []
    for fact in facts:
        fid = (fact.get("id") or fact.get("key")) if isinstance(fact, dict) else fact
        if not isinstance(fid, str) or not _LABEL.fullmatch(fid):
            raise DiagnosticsError(f"{case_id}: некорректный fact id")
        ids.append(fid)
    if len(set(ids)) != len(ids):
        raise DiagnosticsError(f"{case_id}: повторяются критерии")
    reviewer = _field(item, ("reviewer_id", "reviewer"), "reviewer", required=False)
    if reviewer is not None and (not isinstance(reviewer, str) or not reviewer.strip()):
        raise DiagnosticsError(f"{case_id}: reviewer_id должен быть непустой строкой")
    return Case(case_id, domain, question, reference, base, lora, tuple(ids),
                (_field(item, ("base_fact_scores",), "base_fact_scores", required=False),
                 _field(item, ("lora_fact_scores",), "lora_fact_scores", required=False)), reviewer)


def _parse_md(text: str) -> list[dict[str, Any]]:
    """Разобрать экспорт human_review.md из F4.7; не оценивать факты автоматически."""
    headers = list(_HEADER.finditer(text))
    if not headers:
        raise DiagnosticsError("Markdown F4.7: не найдены разделы ## ev_* — domain")
    items = []
    for idx, match in enumerate(headers):
        chunk = text[match.end():headers[idx + 1].start() if idx + 1 < len(headers) else len(text)]
        sections = list(_SECTION.finditer(chunk))
        found: dict[str, str] = {}
        for j, section in enumerate(sections):
            name = section.group(1)
            if name in found:
                raise DiagnosticsError(f"{match.group(1)}: дублирован раздел {name}")
            value = chunk[section.end():sections[j + 1].start() if j + 1 < len(sections) else len(chunk)]
            # Разделитель между случаями не является содержимым ответа.
            found[name] = re.sub(r"\n\s*---\s*$", "", value).strip()
        if not all(k in found for k in _EXPECTED_KEYS):
            raise DiagnosticsError(f"{match.group(1)}: отсутствуют секции human_review.md")
        facts = []
        for line in found["Факты для проверки"].splitlines():
            f = _FACT.match(line.strip())
            if not f:
                raise DiagnosticsError(f"{match.group(1)}: неправильная строка факта")
            facts.append({"id": f.group(1), "description": f.group(2)})
        items.append({"case_id": match.group(1), "domain": match.group(2), "question": found["Вопрос"],
                      "reference": found["Эталон"], "base_response": found["Base"],
                      "lora_response": found["LoRA"], "facts": facts})
    return items


def load_cases(path: Path) -> list[Case]:
    """Поддержать JSON/JSONL и штатный Markdown-экспорт F4.7."""
    try:
        raw = path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeError) as exc:
        raise DiagnosticsError(f"Невозможно прочитать F4.7: {exc}") from exc
    if path.suffix == ".md":
        items = _parse_md(raw)
    elif path.suffix == ".json":
        loaded = _json_load(path)
        items = loaded if isinstance(loaded, list) else loaded.get("cases") if isinstance(loaded, dict) else None
        if not isinstance(items, list):
            raise DiagnosticsError("F4.7 JSON: требуется массив или объект с cases[]")
    elif path.suffix == ".jsonl":
        items = []
        for line_num, line in enumerate(raw.splitlines(), 1):
            if not line.strip():
                continue
            try:
                items.append(json.loads(line, parse_constant=lambda v: (_ for _ in ()).throw(DiagnosticsError(f"Недопустимое JSONL число: {v}"))))
            except json.JSONDecodeError as exc:
                raise DiagnosticsError(f"F4.7 JSONL строка {line_num}: {exc}") from exc
    else:
        raise DiagnosticsError("Неизвестный формат F4.7")
    cases = [_normalize(item, i) for i, item in enumerate(items, 1)]
    ids = [case.case_id for case in cases]
    if len(set(ids)) != len(ids):
        raise DiagnosticsError("Дублирован case_id в F4.7")
    return cases


def tokenize(text: str) -> tuple[str, ...]:
    """Unicode токены: слова и одиночные знаки пунктуации, casefold."""
    return tuple(_TOKEN.findall(text.casefold()))


def overlap(reference: str, candidate: str) -> dict[str, Any]:
    """Multiset overlap: слова с повторами учитываются через Counter."""
    a, b = tokenize(reference), tokenize(candidate)
    matches = sum((Counter(a) & Counter(b)).values())
    precision = matches / len(b) if b else 0.0
    recall = matches / len(a) if a else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "exact_match": a == b,
            "reference_token_count": len(a), "candidate_token_count": len(b)}


def _repetition(tokens: tuple[str, ...]) -> bool:
    if len(tokens) < 12:
        return False
    grams = Counter(tuple(tokens[i:i + 4]) for i in range(len(tokens) - 3))
    return max(grams.values(), default=0) >= 3


def _flags(case: Case, answer: str, score: dict[str, Any], threshold: float) -> list[str]:
    flags: list[str] = []
    if not answer.strip():
        flags.append("empty_response")
    if score["f1"] < threshold:
        flags.append("low_reference_overlap")
    if _MARKERS.search(answer):
        flags.append("prompt_template_marker")
    question_tokens = tokenize(case.question)
    answer_tokens = tokenize(answer)
    if len(question_tokens) >= 5:
        q = tuple(question_tokens[: min(8, len(question_tokens))])
        if any(answer_tokens[i:i + len(q)] == q for i in range(len(answer_tokens) - len(q) + 1)):
            flags.append("possible_prompt_echo")
    if _repetition(answer_tokens):
        flags.append("repeated_fourgram")
    return flags


def _fact_scores(raw: Any, facts: tuple[str, ...], case_id: str) -> list[int] | None:
    if raw is None:
        return None
    if isinstance(raw, dict):
        if set(raw) != set(facts):
            raise DiagnosticsError(f"{case_id}: неверные id в fact_scores")
        raw = [raw[fid] for fid in facts]
    if not isinstance(raw, list) or len(raw) != len(facts):
        raise DiagnosticsError(f"{case_id}: fact_scores должен содержать значения для всех критериев")
    for value in raw:
        if value is not None and (type(value) is not int or value not in {0, 1, 2}):
            raise DiagnosticsError(f"{case_id}: оценки допустимы только 0/1/2/null")
    return raw


def evaluate(cases: list[Case], source: str, fingerprint: str, cfg: dict[str, Any]) -> dict[str, Any]:
    """Возвратить только агрегаты и case_id: тексты в публичный отчёт не попадают."""
    expected = cfg["expected_cases"]
    required_facts = cfg["expected_facts_per_case"]
    if len(cases) != expected or any(len(c.fact_ids) != required_facts for c in cases):
        raise DiagnosticsError(f"F4.7: ожидается {expected} заданий по {required_facts} критерия, получено {len(cases)}")
    threshold = float(cfg["low_overlap_f1_threshold"])
    rows: list[dict[str, Any]] = []
    score_arrays: dict[str, list[float]] = {"base": [], "lora": []}
    human_values: dict[str, list[int]] = {"base": [], "lora": []}
    unreviewed = 0
    unattributed = 0
    changed = 0
    for c in cases:
        b, l = overlap(c.reference, c.base), overlap(c.reference, c.lora)
        bf, lf = _flags(c, c.base, b, threshold), _flags(c, c.lora, l, threshold)
        changed_here = c.base != c.lora
        changed += changed_here
        human: dict[str, Any] = {}
        for tag, raw in (("base", c.raw_scores[0]), ("lora", c.raw_scores[1])):
            scores = _fact_scores(raw, c.fact_ids, c.case_id)
            if scores is not None and not c.reviewer:
                unattributed += sum(value is not None for value in scores)
            validated = scores if c.reviewer is not None else None
            non_null = [x for x in validated or [] if x is not None]
            human[tag] = {"scored": len(non_null), "total": len(c.fact_ids),
                          "mean_score": sum(non_null) / len(non_null) if non_null else None}
            human_values[tag].extend(non_null)
            score_arrays[tag].append((b if tag == "base" else l)["f1"])
        if not (human["base"]["scored"] == required_facts and human["lora"]["scored"] == required_facts):
            unreviewed += 1
        rows.append({"case_id": c.case_id, "domain": c.domain, "answers_changed": changed_here,
                     "token_overlap": {"base": b, "lora": l}, "diagnostic_flags": {"base": bf, "lora": lf},
                     "human_review": human})
    def mean(values: list[float | int]) -> float | None:
        return sum(values) / len(values) if values else None
    return {
        "schema_version": 1, "stage": "F5.2", "status": "PASS" if unreviewed == 0 else "PARTIAL",
        "source": {"path": source, "sha256": fingerprint, "format": Path(source).suffix.lstrip(".")},
        "protocol": {"expected_cases": expected, "facts_per_case": required_facts,
                     "tokenizer": "unicode_word_or_punctuation_casefold_v1", "overlap": "multiset_precision_recall_f1",
                     "low_overlap_f1_threshold": threshold, "expert_scale": "0_incorrect_1_partial_2_correct",
                     "human_provenance_rule": "requires_explicit_reviewer_id"},
        "summary": {"cases": len(cases), "criteria_total": len(cases) * required_facts,
                    "changed_pairs": changed, "unchanged_pairs": len(cases) - changed,
                    "average_token_f1_base": mean(score_arrays["base"]),
                    "average_token_f1_lora": mean(score_arrays["lora"]),
                    "delta_token_f1_lora_minus_base": mean(score_arrays["lora"]) - mean(score_arrays["base"]),
                    "flag_counts": {k: dict(sorted(Counter(flag for r in rows for flag in r["diagnostic_flags"][k]).items())) for k in ("base", "lora")},
                    "human_review": {"complete_cases": len(cases) - unreviewed,
                                     "unreviewed_cases": unreviewed,
                                     "unattributed_scores_ignored": unattributed,
                                     "base_criteria_scored": len(human_values["base"]),
                                     "lora_criteria_scored": len(human_values["lora"]),
                                     "base_average_score": mean(human_values["base"]),
                                     "lora_average_score": mean(human_values["lora"])}},
        "cases": rows,
        "limitations": [
            "Token overlap измеряет сходство токенов, а не фактическую корректность ответа.",
            "Flags — эвристические сигналы для разбора, не диагноз hallucination и не подтверждённые ошибки.",
            "Критерии рубрики перечислены для учёта покрытия; оценивание фактов требует человека.",
            "Оценки без reviewer_id игнорируются; указанное reviewer_id — декларация источника, не доказательство квалификации эксперта.",
            "Сравнение Base/LoRA по одному frozen набору без экспертного анализа не доказывает улучшение качества.",
            "Результаты не являются LLM-as-a-judge и не предназначены для вывода о безопасности/production readiness.",
        ],
    }


def to_json(report: dict[str, Any]) -> str:
    return json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"


def _display(value: float | None) -> str:
    return "NOT RATED" if value is None else f"{value:.6f}"


def _markdown_safe(value: str) -> str:
    return value.replace("|", "\\|").replace("`", "\\`").replace("\n", " ")


def to_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    human = summary["human_review"]
    lines = ["# F5.2 — LLM Evaluation & Diagnostics", "",
             f"**Статус доказательств:** `{report['status']}`. Автоматические метрики — proxy; они не устанавливают factual correctness.", "",
             "## Источник и протокол", "",
             f"- Исходный файл: `{_markdown_safe(report['source']['path'])}`.",
             f"- SHA-256: `{report['source']['sha256']}`.",
             f"- Заданий: {summary['cases']}; проверяемых фактов: {summary['criteria_total']}.",
             "- Токенизация: Unicode-слова/пунктуация, casefold, multiset overlap.", "",
             "## Автоматические proxy-метрики", "",
             "| Показатель | Base | LoRA |", "|---|---:|---:|",
             f"| Средний token F1 | {_display(summary['average_token_f1_base'])} | {_display(summary['average_token_f1_lora'])} |",
             f"| Изменённых ответов (из {summary['cases']}) | — | {summary['changed_pairs']} |", "",
             f"Разница token F1 (LoRA − Base): **{summary['delta_token_f1_lora_minus_base']:+.6f}**. Положительная разница не означает улучшения correctness.", "",
             "## Диагностические сигналы", "",
             "| Flag | Base | LoRA |", "|---|---:|---:|" ]
    flags = sorted(set(summary["flag_counts"]["base"]) | set(summary["flag_counts"]["lora"]))
    for flag in flags:
        lines.append(f"| `{flag}` | {summary['flag_counts']['base'].get(flag, 0)} | {summary['flag_counts']['lora'].get(flag, 0)} |")
    if not flags:
        lines.append("| Сигналов не выявлено | 0 | 0 |")
    lines.extend(["", "## Экспертная оценка по рубрике", "",
                  f"- Полностью оценённых случаев: **{human['complete_cases']}/{summary['cases']}**.",
                  f"- Неоценённых/частично оценённых случаев: **{human['unreviewed_cases']}**.",
                  f"- Оценок без reviewer_id, исключённых из подсчёта: **{human['unattributed_scores_ignored']}**.",
                  f"- Средняя оценка Base (0–2): **{_display(human['base_average_score'])}**.",
                  f"- Средняя оценка LoRA (0–2): **{_display(human['lora_average_score'])}**.", "",
                  "Отсутствующие экспертные баллы остаются `null`; система не выставляет их по сходству текста с эталоном.", "",
                  "## Покейсный аудит (без публикации текста)", "",
                  "| Case | Домен | Изменился текст | F1 Base | F1 LoRA | Флаги Base | Флаги LoRA |", 
                  "|---|---|---|---:|---:|---|---|" ])
    for r in report["cases"]:
        lines.append(f"| `{r['case_id']}` | `{r['domain']}` | {'да' if r['answers_changed'] else 'нет'} | "
                     f"{r['token_overlap']['base']['f1']:.4f} | {r['token_overlap']['lora']['f1']:.4f} | "
                     f"{', '.join(r['diagnostic_flags']['base']) or '—'} | {', '.join(r['diagnostic_flags']['lora']) or '—'} |")
    lines.extend(["", "## Ограничения", ""])
    lines.extend("- " + x for x in report["limitations"])
    lines.append("")
    return "\n".join(lines)


def run(root: Path, *, config: str = "configs/f5_02_llm_diagnostics.json", input_path: str | None = None,
        check: bool = False, require_expert: bool = False) -> dict[str, Any]:
    root = root.resolve()
    cfg = load_config(root, config)
    source, rel = _choose_input(root, cfg, input_path)
    cases = load_cases(source)
    report = evaluate(cases, rel, sha256(source.read_bytes()).hexdigest(), cfg)
    if require_expert and report["status"] != "PASS":
        raise DiagnosticsError("Экспертные оценки не покрывают все случаи; статус PARTIAL")
    artifacts = [(cfg["outputs"]["json"], to_json(report)), (cfg["outputs"]["markdown"], to_markdown(report))]
    if check:
        for relative, content in artifacts:
            target = _path(root, relative, "output")
            if not target.is_file() or target.read_bytes() != content.encode("utf-8"):
                raise DiagnosticsError(f"Устаревший или отсутствующий отчёт: {relative}")
    else:
        # Валидация каждого выходного пути перед любой записью.
        targets = [(_path(root, relative, "output"), content) for relative, content in artifacts]
        for target, content in targets:
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + ".tmp")
            try:
                temporary.write_bytes(content.encode("utf-8"))
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)
    return report
