# Назначение файла: F1.4: проверки нормализации, фильтрации и аудита очистки на выдуманных записях.
# Примечание: комментарии не влияют на работу программы.

"""F1.4: synthetic-only deterministic cleaning tests. No dataset download."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from scripts.clean_data import CleaningError, run
from src.data.cleaning import clean_records, clean_text


# Создаёт тестовую запись с вопросом, ответом и служебными полями.
def record(identifier="1", context="one two three four five six seven eight",
           answer="alpha beta gamma delta epsilon zeta eta theta", domain="linux", **extra):
    return {"id": identifier, "domain": domain, "context": context,
            "answer": answer, "source_row": 0, "rejected": "Weak **reply**", **extra}


# Проверяет соответствие нормализации правилам исходного конвейера.
def test_normalization_matches_original_pipeline():
    assert clean_text(" A&nbsp; <p>B</p> [guide](https://example.com) `code` #tag* ") == "A B guide code tag"


# Проверяет удаление URL и свёртывание повторяющихся пробелов.
def test_url_removed_and_whitespace_collapsed():
    assert clean_text("A   https://example.com/path\nB\t C") == "A B C"


# Проверяет порядок декодирования HTML-сущностей и удаления разметки.
def test_entities_converted_before_html_stripping():
    assert clean_text("&lt;b&gt; Hello &lt;/b&gt;") == "Hello"


# Проверяет сохранение полей происхождения и неизменность входных записей.
def test_preserves_provenance_and_does_not_modify_input():
    row = record(context="one two three four five six seven **eight**")
    previous = copy.deepcopy(row)
    cleaned, stats = clean_records([row], min_words=8)
    assert row == previous
    assert cleaned[0]["id"] == "1" and cleaned[0]["domain"] == "linux"
    assert cleaned[0]["source_row"] == 0
    assert cleaned[0]["context"].endswith("eight")
    assert stats["output_rows"] == 1


# Проверяет фильтрацию строк без обязательных полей и объектов неверного типа.
def test_discard_raw_missing_and_non_dict():
    rows, stats = clean_records([None, {"id": "x"}, record()], min_words=8)
    assert len(rows) == 1
    assert stats["invalid_or_missing"] == 2


# Проверяет удаление точных повторов вопросов без учёта регистра.
def test_context_dedup_casefold_after_cleaning():
    first = record("x", context="ONE two three four five six seven eight")
    second = record("y", context="one  two three four five six seven eight")
    rows, stats = clean_records([first, second], min_words=8)
    assert len(rows) == 1 and stats["duplicates_removed"] == 1


# Проверяет удаление повторных ID даже при разных текстах вопросов.
def test_duplicate_ids_removed_even_with_different_context():
    rows, stats = clean_records([
        record("same"), record("same", context="new many other words for a different long question")
    ], min_words=8)
    assert len(rows) == 1 and stats["duplicates_removed"] == 1


# Проверяет фильтр минимальной длины вопроса и ответа.
def test_short_context_and_short_answer_both_removed():
    rows, stats = clean_records([
        record("a", context="only two"),
        record("b", answer="also short"),
        record("c"),
    ], min_words=8)
    assert len(rows) == 1 and stats["short_removed"] == 2


# Проверяет, что отброшенный ID не блокирует следующую допустимую запись.
def test_short_unaccepted_id_does_not_block_later_valid_record():
    rows, stats = clean_records([
        record("shared", context="too short"), record("shared")
    ], min_words=8)
    assert len(rows) == 1 and stats["short_removed"] == 1
    assert stats["duplicates_removed"] == 0


# Проверяет обработку необязательного поля отклонённого ответа.
def test_rejected_optional_or_non_string():
    rows, _ = clean_records([record("1", rejected=None)], min_words=8)
    assert rows[0]["rejected"] == ""


# Проверяет сохранение баланса счётчиков и повторяемость результата.
def test_count_conservation_and_repeatability():
    raw = [record("1"), record("2", context="short"), record("1"), None]
    first, stats = clean_records(raw, min_words=8)
    second, stats2 = clean_records(raw, min_words=8)
    assert (first, stats) == (second, stats2)
    assert stats == {"input_rows": 4, "invalid_or_missing": 1,
                     "duplicates_removed": 1, "short_removed": 1, "output_rows": 1}


# Проверяет отказ при некорректном пороге длины текста.
@pytest.mark.parametrize("limit", [0, -2, 1.5, True])
def test_invalid_min_words(limit):
    with pytest.raises(ValueError, match="min_words"):
        clean_records([record()], min_words=limit)


# Создаёт временный проект с синтетическим набором для тестов CLI.
@pytest.fixture
def sample_project(tmp_path: Path):
    cfg = tmp_path / "configs/local_cpu/lab.yaml"
    cfg.parent.mkdir(parents=True)
    cfg.write_text("seed: 42\ndata:\n  input: data/raw/subset.jsonl\n  min_words: 8\n", encoding="utf-8")
    raw = tmp_path / "data/raw/subset.jsonl"
    raw.parent.mkdir(parents=True)
    examples = [record("1"), record("2", context="short"),
                record("3", context="new text that has at least eight separate token words")]
    raw.write_text("".join(json.dumps(r) + "\n" for r in examples), encoding="utf-8")
    return tmp_path, raw


# Проверяет полный offline-конвейер, аудит и неизменность выходных байтов.
def test_end_to_end_offline_audit_and_stable_bytes(sample_project):
    root, raw = sample_project
    report = run(root, expect_input=3, expect_output=2)
    output_path = root / "data/interim/cleaned.jsonl"
    audit_path = root / "reports/data_cleaning_audit.json"
    assert report["source_sha256"] == hashlib.sha256(raw.read_bytes()).hexdigest()
    assert report["cleaning"]["short_removed"] == 1
    assert len(output_path.read_text(encoding="utf-8").splitlines()) == 2
    assert "NOT RUN (F1.6)" == report["near_duplicate_grouping"]
    saved = output_path.read_bytes(), audit_path.read_bytes()
    assert run(root, expect_input=3, expect_output=2) == report
    assert (output_path.read_bytes(), audit_path.read_bytes()) == saved
    # audit contains no question/answer text
    assert "alpha beta" not in audit_path.read_text(encoding="utf-8")


# Проверяет отсутствие скрытого перехода к образцу при отсутствии входного файла.
def test_input_missing_no_silent_fallback(tmp_path):
    cfg = tmp_path / "configs/local_cpu/lab.yaml"
    cfg.parent.mkdir(parents=True)
    cfg.write_text("data:\n  input: data/raw/subset.jsonl\n  min_words: 8\n", encoding="utf-8")
    with pytest.raises(CleaningError, match="missing full F1.3 subset"):
        run(tmp_path)
    assert not (tmp_path / "data/interim/cleaned.jsonl").exists()


# Проверяет отказ без записи при неверном ожидаемом числе результатов.
def test_wrong_expected_count_writes_nothing(sample_project):
    root, _ = sample_project
    with pytest.raises(CleaningError, match="output row count mismatch"):
        run(root, expect_input=3, expect_output=3040)
    assert not (root / "data/interim/cleaned.jsonl").exists()
    assert not (root / "reports/data_cleaning_audit.json").exists()


# Проверяет контроль числа входных строк до создания артефактов.
def test_incorrect_input_count_fails_before_writing(sample_project):
    root, _ = sample_project
    with pytest.raises(CleaningError, match="input row count mismatch"):
        run(root, expect_input=3066)
    assert not (root / "data/interim/cleaned.jsonl").exists()


# Проверяет защиту от совпадения путей входа и выходных файлов.
def test_reject_overwriting_input_or_audit(sample_project):
    root, _ = sample_project
    with pytest.raises(CleaningError, match="different paths"):
        run(root, output_path=Path("data/raw/subset.jsonl"))
