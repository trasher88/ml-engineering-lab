"""F5.2: только маленькие искусственные тексты; ни одной загрузки весов."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

import pytest

from src.evaluation_f5.llm_diagnostics import (
    Case, DiagnosticsError, _fact_scores, _parse_md, _path, _repetition,
    evaluate, load_cases, load_config, overlap, run, to_json, tokenize,
)
from scripts.f5_llm_diagnostics import main


CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "f5_02_llm_diagnostics.json"


def setup_root(root: Path, *, reviewed: bool = False, use_markdown: bool = False) -> Path:
    config = root / "configs/f5_02_llm_diagnostics.json"
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_bytes(CONFIG_PATH.read_bytes())
    file = root / ("data/processed/f4_07/cpu/human_review.md" if use_markdown else "data/processed/f4_07/cpu/human_review.jsonl")
    file.parent.mkdir(parents=True, exist_ok=True)
    if use_markdown:
        sections = []
        for i in range(32):
            sections.append(f"## ev_py_{i:02d} — python\n\n**Вопрос:**\nКак проверить значение {i}?\n\n**Данные:**\n(нет)\n\n**Эталон:**\nВерное значение {i}\n\n**Факты для проверки:**\n- `value`: Верное значение\n- `context`: Контекст значения\n\n**Base:**\n\nЗначение {i}\n\n**LoRA:**\n\nЗначение {i} и ещё контекст\n\n---\n")
        file.write_text("\n".join(sections), encoding="utf-8")
    else:
        data = []
        for i in range(32):
            row = {"case_id": f"ev_py_{i:02d}", "domain": "python", "question": f"Как проверить значение {i}?",
                   "reference": f"Верное значение {i}", "facts": ["value", "context"],
                   "base_response": f"Значение {i}", "lora_response": f"Значение {i} и ещё контекст"}
            if reviewed:
                row.update({"reviewer_id": "reviewer_01", "base_fact_scores": {"value": 1, "context": 0},
                            "lora_fact_scores": {"value": 2, "context": 1}})
            data.append(json.dumps(row, ensure_ascii=False))
        file.write_text("\n".join(data) + "\n", encoding="utf-8")
    return file


def test_tokenization_unicode_punctuation() -> None:
    assert tokenize("ПрИвЕт, Python 3.13!") == ("привет", ",", "python", "3", ".", "13", "!")


def test_multiset_overlap_and_exact() -> None:
    score = overlap("a a b", "a b b")
    assert score["precision"] == pytest.approx(2/3)
    assert score["recall"] == pytest.approx(2/3)
    assert score["f1"] == pytest.approx(2/3)
    assert not score["exact_match"]
    assert overlap("А Б", "а б")["exact_match"] is True


def test_empty_overlap_no_nan() -> None:
    for ref, answer in (("abc", ""), ("", "abc"), ("", "")):
        score = overlap(ref, answer)
        assert score["f1"] == 0


def test_repetition_only_4grams() -> None:
    assert _repetition(tuple("a b c d a b c d a b c d".split()))
    assert not _repetition(tuple("a b c d".split()))


def test_source_build_and_check(tmp_path: Path) -> None:
    source = setup_root(tmp_path)
    r = run(tmp_path)
    assert r["status"] == "PARTIAL"
    assert r["source"]["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert r["summary"]["cases"] == 32
    assert r["summary"]["criteria_total"] == 64
    assert r["summary"]["changed_pairs"] == 32
    assert r["summary"]["human_review"]["unreviewed_cases"] == 32
    assert r["summary"]["human_review"]["base_average_score"] is None
    assert r["summary"]["human_review"]["base_criteria_scored"] == 0
    assert run(tmp_path, check=True) == r
    report = (tmp_path / "reports/public/f5_02/F5_02_llm_diagnostics.json").read_text(encoding="utf-8")
    assert "Как проверить значение" not in report
    assert "Верное значение" not in report
    assert "Значение 0 и" not in report


def test_human_review_complete(tmp_path: Path) -> None:
    setup_root(tmp_path, reviewed=True)
    r = run(tmp_path, require_expert=True)
    assert r["status"] == "PASS"
    assert r["summary"]["human_review"]["complete_cases"] == 32
    assert r["summary"]["human_review"]["base_criteria_scored"] == 64
    assert r["summary"]["human_review"]["lora_average_score"] == pytest.approx(1.5)


def test_human_review_without_reviewer_disregarded(tmp_path: Path) -> None:
    p = setup_root(tmp_path)
    rows = [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines()]
    rows[0]["base_fact_scores"] = [2, 2]
    rows[0]["lora_fact_scores"] = [2, 2]
    p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    r = run(tmp_path)
    assert r["status"] == "PARTIAL"
    assert r["summary"]["human_review"]["unattributed_scores_ignored"] == 4
    assert r["summary"]["human_review"]["base_criteria_scored"] == 0


def test_markdown_f47_export(tmp_path: Path) -> None:
    setup_root(tmp_path, use_markdown=True)
    r = run(tmp_path)
    assert r["summary"]["cases"] == 32
    assert r["summary"]["changed_pairs"] == 32
    assert r["source"]["format"] == "md"
    assert r["summary"]["human_review"]["unreviewed_cases"] == 32


def test_json_top_level_cases(tmp_path: Path) -> None:
    p = setup_root(tmp_path)
    rows = [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines()]
    output = tmp_path / "reports/public/f4_07/human_review.json"
    output.parent.mkdir(parents=True)
    output.write_text(json.dumps({"cases": rows}, ensure_ascii=False), encoding="utf-8")
    report = run(tmp_path, input_path="reports/public/f4_07/human_review.json")
    assert report["summary"]["cases"] == 32


def test_multi_candidate_requires_explicit_input(tmp_path: Path) -> None:
    p = setup_root(tmp_path)
    p2 = tmp_path / "data/processed/f4_07/human_review.jsonl"
    p2.parent.mkdir(parents=True, exist_ok=True)
    p2.write_bytes(p.read_bytes())
    with pytest.raises(DiagnosticsError, match="несколько"):
        run(tmp_path)
    assert run(tmp_path, input_path="data/processed/f4_07/cpu/human_review.jsonl")["status"] == "PARTIAL"


def test_missing_input_no_synthetic_output(tmp_path: Path) -> None:
    c = tmp_path / "configs/f5_02_llm_diagnostics.json"
    c.parent.mkdir(parents=True)
    c.write_bytes(CONFIG_PATH.read_bytes())
    with pytest.raises(DiagnosticsError, match="Не найден"):
        run(tmp_path)
    assert not (tmp_path / "reports/public/f5_02").exists()


def test_check_detects_changes(tmp_path: Path) -> None:
    p = setup_root(tmp_path)
    run(tmp_path)
    p.write_text(p.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(DiagnosticsError, match="Устаревший"):
        run(tmp_path, check=True)
    run(tmp_path)
    run(tmp_path, check=True)


def test_require_expert_blocks_before_write(tmp_path: Path) -> None:
    setup_root(tmp_path)
    with pytest.raises(DiagnosticsError, match="PARTIAL"):
        run(tmp_path, require_expert=True)
    assert not (tmp_path / "reports/public/f5_02").exists()


def test_invalid_source_count(tmp_path: Path) -> None:
    p = setup_root(tmp_path)
    p.write_text("\n".join(p.read_text(encoding="utf-8").splitlines()[:31]), encoding="utf-8")
    with pytest.raises(DiagnosticsError, match="ожидается 32"):
        run(tmp_path)


def test_invalid_criteria_count(tmp_path: Path) -> None:
    p = setup_root(tmp_path)
    rows = p.read_text(encoding="utf-8").splitlines()
    rec = json.loads(rows[0]);rec["facts"] = ["value"]
    rows[0] = json.dumps(rec, ensure_ascii=False)
    p.write_text("\n".join(rows), encoding="utf-8")
    with pytest.raises(DiagnosticsError, match="по 2"):
        run(tmp_path)


def test_duplicate_case_id(tmp_path: Path) -> None:
    p = setup_root(tmp_path)
    lines = p.read_text(encoding="utf-8").splitlines()
    lines[1] = lines[0]
    p.write_text("\n".join(lines), encoding="utf-8")
    with pytest.raises(DiagnosticsError, match="Дублирован"):
        run(tmp_path)


@pytest.mark.parametrize("value", [3, "2", 1.5, -1, True, float("nan")])
def test_invalid_score_rejected(value: object) -> None:
    with pytest.raises(DiagnosticsError):
        _fact_scores([value, 2], ("a", "b"), "case_x")


def test_score_null_allowed() -> None:
    assert _fact_scores({"x": None, "y": 2}, ("y", "x"), "case_x") == [2, None]


def test_score_keys_must_match() -> None:
    with pytest.raises(DiagnosticsError):
        _fact_scores({"x": 1, "z": 2}, ("x", "y"), "case_x")


@pytest.mark.parametrize("bad_path", ["../outside", "/tmp/data", "C:/outside", "a/./b", "a//b", "a\\b", "./test"])
def test_unsafe_paths(tmp_path: Path, bad_path: str) -> None:
    with pytest.raises(DiagnosticsError):
        _path(tmp_path, bad_path, "test")


def test_symlink_escape(tmp_path: Path) -> None:
    other = tmp_path.parent / "outside_f5_02_tmp"
    other.mkdir(exist_ok=True)
    link = tmp_path / "escape"
    try:
        link.symlink_to(other, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("Символьные ссылки недоступны")
    with pytest.raises(DiagnosticsError):
        _path(tmp_path, "escape/secrets.json", "test")


def test_invalid_config_threshold(tmp_path: Path) -> None:
    setup_root(tmp_path)
    p = tmp_path / "configs/f5_02_llm_diagnostics.json"
    content = json.loads(p.read_text(encoding="utf-8"))
    content["low_overlap_f1_threshold"] = 1.5
    p.write_text(json.dumps(content), encoding="utf-8")
    with pytest.raises(DiagnosticsError):
        load_config(tmp_path)


def test_invalid_config_output_path(tmp_path: Path) -> None:
    setup_root(tmp_path)
    p = tmp_path / "configs/f5_02_llm_diagnostics.json"
    content = json.loads(p.read_text(encoding="utf-8"))
    content["outputs"]["json"] = "reports/private/abc.json"
    p.write_text(json.dumps(content), encoding="utf-8")
    with pytest.raises(DiagnosticsError):
        load_config(tmp_path)


def test_invalid_jsonl(tmp_path: Path) -> None:
    p = setup_root(tmp_path)
    p.write_text("{bad", encoding="utf-8")
    with pytest.raises(DiagnosticsError, match="JSONL"):
        run(tmp_path)


def test_bad_jsonl_record(tmp_path: Path) -> None:
    p = setup_root(tmp_path)
    rows = p.read_text(encoding="utf-8").splitlines()
    rows[0] = json.dumps({"bad": "schema"})
    p.write_text("\n".join(rows), encoding="utf-8")
    with pytest.raises(DiagnosticsError, match="case_id"):
        run(tmp_path)


def test_duplicate_section_md_rejected(tmp_path: Path) -> None:
    p = setup_root(tmp_path, use_markdown=True)
    s = p.read_text(encoding="utf-8").replace("**Base:**", "**Base:**\nhello\n**Base:**", 1)
    p.write_text(s, encoding="utf-8")
    with pytest.raises(DiagnosticsError, match="дублирован"):
        run(tmp_path)


def test_proxies_never_count_as_expert(tmp_path: Path) -> None:
    setup_root(tmp_path)
    report = run(tmp_path)
    assert report["summary"]["average_token_f1_base"] is not None
    assert report["summary"]["human_review"]["base_average_score"] is None
    assert report["status"] == "PARTIAL"


def test_no_language_model_import_needed() -> None:
    import sys
    assert "transformers" not in sys.modules
    assert "torch" not in sys.modules


def test_cli_build_check(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    setup_root(tmp_path)
    assert main(["--root", str(tmp_path)]) == 0
    assert main(["--root", str(tmp_path), "--check"]) == 0
    assert "F5.2 PARTIAL" in capsys.readouterr().out


def test_cli_require_expert_failure(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    setup_root(tmp_path)
    assert main(["--root", str(tmp_path), "--require-expert"]) == 2
    assert "PARTIAL" in capsys.readouterr().err


def test_cli_missing_source_failure(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    cfg = tmp_path / "configs/f5_02_llm_diagnostics.json"
    cfg.parent.mkdir(parents=True)
    cfg.write_bytes(CONFIG_PATH.read_bytes())
    assert main(["--root", str(tmp_path)]) == 2
    assert "Не найден" in capsys.readouterr().err


def test_no_text_in_report(tmp_path: Path) -> None:
    setup_root(tmp_path, reviewed=True)
    report = to_json(run(tmp_path))
    assert "Контекст значения" not in report
    assert "Как проверить значение" not in report
    assert "Значение 0" not in report


def test_markdown_parser_example() -> None:
    text = """# F4.7
## ev_py_01 — python
**Вопрос:**
Что такое x?
**Данные:**
(нет)
**Эталон:**
x есть число
**Факты для проверки:**
- `fact_one`: первый
- `fact_two`: второй
**Base:**

x
**LoRA:**

x число
---
"""
    parsed = _parse_md(text)
    assert parsed[0]["facts"][0]["id"] == "fact_one"
    assert parsed[0]["lora_response"] == "x число"


def test_nested_review_auto_discovery(tmp_path: Path) -> None:
    original = setup_root(tmp_path)
    nested = tmp_path / "outputs/f4_07_comparison/cpu-v2/human_review.jsonl"
    nested.parent.mkdir(parents=True, exist_ok=True)
    nested.write_bytes(original.read_bytes())
    original.unlink()
    report = run(tmp_path)
    assert report["source"]["path"] == "outputs/f4_07_comparison/cpu-v2/human_review.jsonl"
    assert report["summary"]["cases"] == 32
