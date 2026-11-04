"""F3.9: тесты аудита итогового отчёта, защиты от подмены и CLI smoke.

В тестовом payload сознательно имитируется схема pretrained F3.7,
но ранжирования созданы вручную. Они применяются ТОЛЬКО в tmp/test-памяти;
публичные метрики из этой фикстуры публиковать нельзя.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from transformers_f3.embeddings import ModelSpec  # noqa: E402
from transformers_f3.retrieval_benchmark import evaluate_rankings, load_frozen_benchmark  # noqa: E402
from transformers_f3.retrieval_comparison import ComparisonConfig, METHODS  # noqa: E402
from transformers_f3 import phase_report as f3  # noqa: E402


@pytest.fixture(scope="module")
def benchmark():
    """F3.6 позволяет пересчитать каждый тестовый rank, не полагаясь на JSON."""
    return load_frozen_benchmark()


@pytest.fixture(scope="module")
def config():
    """У тестового входа та же строго фиксированная конфигурация F3.7."""
    return ComparisonConfig.from_json()


@pytest.fixture(scope="module")
def evidence_payload(benchmark, config):
    """Создать полный синтетический JSON именно структуры F3.7 (80 queries).

    Ранги намеренно заданы вручную, поэтому это тест парсера, а не сравнение
    настоящих pretrained embeddings. Не сохранять в reports/public.
    """
    known_docs = [doc.doc_id for doc in benchmark.documents]
    relevant = benchmark.relevance_map()
    rankings = {method: {} for method in METHODS}
    for query in benchmark.queries:
        correct = sorted(relevant[query.query_id])
        distractor = next(doc for doc in known_docs if doc not in correct)
        rankings["bm25"][query.query_id] = [correct[0], distractor, correct[1]]
        rankings["vector"][query.query_id] = [distractor, correct[0], correct[1]]
        rankings["hybrid"][query.query_id] = [correct[0], correct[1], distractor]
    evaluated = {
        method: {k: evaluate_rankings(benchmark, rankings[method], k=k) for k in config.top_ks}
        for method in METHODS
    }
    spec = ModelSpec.from_json(ROOT / "configs" / "f3_02_model.json")
    rows = []
    for query in benchmark.queries:
        rows.append({
            "query_id": query.query_id,
            "domain": query.domain,
            "kind": query.kind,
            "rankings": {method: rankings[method][query.query_id] for method in METHODS},
            "per_query_metrics": {
                method: {
                    str(k): {
                        "recall_at_k": next(row.recall_at_k for row in evaluated[method][k].per_query if row.query_id == query.query_id),
                        "reciprocal_rank_at_k": next(row.reciprocal_rank_at_k for row in evaluated[method][k].per_query if row.query_id == query.query_id),
                    }
                    for k in config.top_ks
                }
                for method in METHODS
            },
        })
    return {
        "schema_version": 1,
        "benchmark_id": benchmark.benchmark_id,
        "benchmark_manifest_sha256": benchmark.manifest_sha256,
        "config": config.to_dict(),
        "config_sha256": config.sha256,
        "backend": "pretrained_transformer",  # Имитация контракта только в unit-тесте.
        "model_metadata": {
            "model_id": spec.model_id, "revision": spec.revision, "license": spec.license,
            "embedding_dim": spec.embedding_dim, "pooling": spec.pooling,
            "normalize": spec.normalize, "model_manifest_sha256": "c" * 64,
            "library_versions": {"torch": "2.10.0+cpu", "transformers": "4.50.0", "huggingface_hub": "0.35.0"},
            "numpy_version": "2.3.0",
        },
        "n_queries": len(benchmark.queries),
        "methods": list(METHODS),
        "metrics": {
            method: {
                str(k): {
                    "recall_at_k": evaluated[method][k].recall_at_k,
                    "mrr_at_k": evaluated[method][k].mrr_at_k,
                    "n_queries": evaluated[method][k].n_queries,
                }
                for k in config.top_ks
            }
            for method in METHODS
        },
        "per_query_rankings": rows,
        "doc_sources": {doc.doc_id: doc.source_id for doc in benchmark.documents},
    }


@pytest.fixture(scope="module")
def valid_audit(evidence_payload):
    """Один раз вычислить корректный аудиторский результат для форматирования."""
    return f3.validate_comparison(evidence_payload, "a" * 64)


def _clone(data):
    """Копия тестового JSON предотвращает взаимное влияние негативных кейсов."""
    return deepcopy(data)


def test_full_evidence_passes_all_queries(valid_audit):
    """Не допускаем публикацию результатов, рассчитанных на части запросов."""
    assert valid_audit.n_queries == 80
    assert valid_audit.n_documents == 88
    assert valid_audit.n_judgments == 160


def test_metrics_are_recomputed_not_hardcoded(valid_audit):
    """Ранги вручную дают BM25=0.5 и Hybrid=1.0 по Recall@1 при 2 qrels."""
    assert valid_audit.metrics["bm25"]["1"]["recall_at_k"] == 0.5
    assert valid_audit.metrics["vector"]["1"]["recall_at_k"] == 0.0
    assert valid_audit.metrics["hybrid"]["5"]["recall_at_k"] == 1.0


def test_counts_of_disagreements_are_exhaustive(valid_audit):
    """Никакие неудобные запросы не исключаются из сравнений с BM25."""
    for stats in valid_audit.disagreements.values():
        assert stats["better"] + stats["worse"] + stats["equal"] == 80


def test_each_group_has_three_methods(valid_audit):
    """Диагностика на lexical/paraphrase и доменах полная для всех методов."""
    assert set(valid_audit.groups) == {"kind", "domain"}
    assert set(valid_audit.groups["kind"]) == {"lexical", "paraphrase"}
    assert set(valid_audit.groups["domain"]) == {"databases", "linux", "systems", "programming"}
    for groups in valid_audit.groups.values():
        for stats in groups.values():
            assert set(stats) == set(METHODS)


def test_groups_reconstruct_macro_score(valid_audit):
    """Усреднённые по группам метрики согласованы с макро-метрикой по 80."""
    for method in METHODS:
        groups = valid_audit.groups["kind"]
        weighted = sum(stats[method]["recall_at_5"] * stats[method]["n_queries"] for stats in groups.values()) / 80
        assert weighted == pytest.approx(valid_audit.metrics[method]["5"]["recall_at_k"])


def test_rag_synthetic_contract(valid_audit):
    """Цитата и fallback проверяются, но не заявляются как answer quality."""
    assert all(valid_audit.rag_smoke.values())


def test_markdown_contains_genuine_source_fingerprint(valid_audit):
    """Отчёт должен позволять связать каждую цифру с конкретным JSON."""
    md = f3.build_phase_markdown(valid_audit)
    assert "`" + "a" * 64 + "`" in md
    assert "F3.6" in md and "F3.8" in md
    assert "Recall@5" in md and "MRR@5" in md
    assert "синтетич" in md


def test_markdown_includes_all_nine_stages(valid_audit):
    """Финальный отчёт не пропускает пройденные темы и F3.9."""
    md = f3.build_phase_markdown(valid_audit)
    for step in range(1, 10):
        assert f"F3.{step}" in md


def test_markdown_has_no_git_commands_or_install_guides(valid_audit):
    """MD-материал строго алгоритмический, без инструкций по публикации."""
    md = f3.build_phase_markdown(valid_audit)
    for banned in ("git add", "git commit", "git push", "pip install", "PowerShell", "собеседовани"):
        assert banned not in md


def test_markdown_has_no_trailing_whitespace(valid_audit):
    """Регрессия ошибки F3.7: git diff --check должен оставаться чистым."""
    lines = f3.build_phase_markdown(valid_audit).splitlines()
    assert all(line == line.rstrip(" \t") for line in lines)


def test_markdown_uses_actual_precision_in_audit(valid_audit):
    """Округление до четырёх знаков применяется только при отображении."""
    doc = f3.evidence_as_dict(valid_audit)
    assert doc["metrics_recomputed"]["bm25"]["5"]["recall_at_k"] == valid_audit.metrics["bm25"]["5"]["recall_at_k"]


def test_evidence_json_is_portable_without_local_paths(valid_audit):
    """В публикационный JSON не попадают абсолютные Windows/Linux пути."""
    dump = json.dumps(f3.evidence_as_dict(valid_audit), ensure_ascii=False)
    assert "F:\\" not in dump and "/mnt/data/" not in dump
    assert "ranking" not in dump


def test_report_does_not_claim_production_quality(valid_audit):
    """Высокая макро-метрика на синтетике не означает production readiness."""
    md = f3.build_phase_markdown(valid_audit)
    assert "не доказанное" in md or "не доказано" in md
    assert "не** измерение" in md or "**не** измерение" in md


@pytest.mark.parametrize("modified", [
    "schema_version", "backend", "benchmark_id", "benchmark_manifest_sha256", "config_sha256",
    "n_queries", "methods", "doc_sources", "metrics", "per_query_rankings", "model_metadata",
])
def test_missing_required_root_fields_rejected(evidence_payload, modified):
    """Отсутствие любого существенного раздела F3.7 блокирует отчёт."""
    data = _clone(evidence_payload)
    data.pop(modified)
    with pytest.raises((ValueError, TypeError)):
        f3.validate_comparison(data, "a" * 64)


@pytest.mark.parametrize("backend", ["synthetic_test_vectors", "other", "", None])
def test_fake_or_missing_pretrained_backend_rejected(evidence_payload, backend):
    """Числа офлайн-демонстрации не могут стать публичными ML-метриками."""
    data = _clone(evidence_payload)
    data["backend"] = backend
    with pytest.raises(ValueError, match="pretrained"):
        f3.validate_comparison(data, "a" * 64)


@pytest.mark.parametrize("k,value", [("1", 0.73), ("5", 0.123), ("10", 0.001)])
def test_modified_global_recall_rejected(evidence_payload, k, value):
    """Неправильная макро-Recall не пройдёт независимый пересчёт qrels."""
    data = _clone(evidence_payload)
    data["metrics"]["bm25"][k]["recall_at_k"] = value
    with pytest.raises(ValueError, match="расхождение"):
        f3.validate_comparison(data, "a" * 64)


@pytest.mark.parametrize("k,value", [("1", 0.73), ("5", 0.123), ("10", 0.001)])
def test_modified_global_mrr_rejected(evidence_payload, k, value):
    """MRR проверяется отдельно от Recall по рангу первого релевантного."""
    data = _clone(evidence_payload)
    data["metrics"]["hybrid"][k]["mrr_at_k"] = value
    with pytest.raises(ValueError, match="расхождение"):
        f3.validate_comparison(data, "a" * 64)


@pytest.mark.parametrize("field", ["recall_at_k", "reciprocal_rank_at_k"])
def test_tampered_individual_metric_rejected(evidence_payload, field):
    """Даже при неизменной общей метрике искажение одной строки блокируется."""
    data = _clone(evidence_payload)
    data["per_query_rankings"][0]["per_query_metrics"]["vector"]["5"][field] = 0.1234567
    with pytest.raises(ValueError, match="расхождение"):
        f3.validate_comparison(data, "a" * 64)


@pytest.mark.parametrize("field,value", [
    ("model_id", "untrusted/encoder"),
    ("revision", "0" * 40),
    ("license", "unknown"),
    ("embedding_dim", 128),
    ("pooling", "CLS"),
    ("normalize", False),
])
def test_model_spec_mismatch_rejected(evidence_payload, field, value):
    """Отчёт не смешивает модели и параметры различных экспериментов."""
    data = _clone(evidence_payload)
    data["model_metadata"][field] = value
    with pytest.raises(ValueError, match="модель"):
        f3.validate_comparison(data, "a" * 64)


@pytest.mark.parametrize("invalid", ["123", "x" * 64, "A" * 64, None, False])
def test_invalid_model_manifest_rejected(evidence_payload, invalid):
    """Отпечаток версии модели должен иметь строгий синтаксис SHA-256."""
    data = _clone(evidence_payload)
    data["model_metadata"]["model_manifest_sha256"] = invalid
    with pytest.raises(ValueError, match="SHA-256"):
        f3.validate_comparison(data, "a" * 64)


@pytest.mark.parametrize("name", ["torch", "transformers", "huggingface_hub"])
def test_missing_library_version_rejected(evidence_payload, name):
    """Версии инструментов инференса необходимы для воспроизводимости."""
    data = _clone(evidence_payload)
    data["model_metadata"]["library_versions"].pop(name)
    with pytest.raises(ValueError, match="версия"):
        f3.validate_comparison(data, "a" * 64)


@pytest.mark.parametrize("bad", ["1.2.3` INJECT", "", None, False])
def test_unsafe_numpy_version_rejected(evidence_payload, bad):
    """Небезопасные строки версий не попадают в Markdown/JSON."""
    data = _clone(evidence_payload)
    data["model_metadata"]["numpy_version"] = bad
    with pytest.raises(ValueError, match="NumPy"):
        f3.validate_comparison(data, "a" * 64)


@pytest.mark.parametrize("variant", ["missing", "duplicate", "unknown", "swapped", "too_many"])
def test_bad_ranking_lists_rejected(evidence_payload, variant):
    """Все выдачи должны принадлежать корпусу, без подставленных ID и дублей."""
    data = _clone(evidence_payload)
    rows = data["per_query_rankings"]
    key = "bm25"
    if variant == "missing":
        rows[0]["rankings"].pop(key)
    elif variant == "duplicate":
        rows[0]["rankings"][key].append(rows[0]["rankings"][key][0])
    elif variant == "unknown":
        rows[0]["rankings"][key][0] = "outside-corpus"
    elif variant == "swapped":
        rows[0]["rankings"][key][0], rows[0]["rankings"][key][1] = (
            rows[0]["rankings"][key][1], rows[0]["rankings"][key][0]
        )
    else:
        rows[0]["rankings"][key].extend(["missing"] * 15)
    with pytest.raises(ValueError):
        f3.validate_comparison(data, "a" * 64)


@pytest.mark.parametrize("variant", ["remove", "duplicate", "reorder", "wrong_domain", "wrong_kind"])
def test_bad_query_coverage_rejected(evidence_payload, variant):
    """Не допускаем сокращать знаменатель за счёт неудобных запросов."""
    data = _clone(evidence_payload)
    rows = data["per_query_rankings"]
    if variant == "remove":
        rows.pop(0)
    elif variant == "duplicate":
        rows[0] = deepcopy(rows[1])
    elif variant == "reorder":
        rows[0], rows[1] = rows[1], rows[0]
    elif variant == "wrong_domain":
        rows[0]["domain"] = "unknown"
    else:
        rows[0]["kind"] = "unknown"
    with pytest.raises(ValueError):
        f3.validate_comparison(data, "a" * 64)


@pytest.mark.parametrize("key", ["benchmark_id", "benchmark_manifest_sha256", "config_sha256"])
def test_frozen_fingerprint_mismatch_rejected(evidence_payload, key):
    """Разные версии benchmark/протокола нельзя смешивать в одной таблице."""
    data = _clone(evidence_payload)
    data[key] = "0" * 64
    with pytest.raises(ValueError):
        f3.validate_comparison(data, "a" * 64)


@pytest.mark.parametrize("corruption", ["missing", "different_source", "new_doc"])
def test_source_provenance_mismatch_rejected(evidence_payload, corruption):
    """Связь doc_id → source_id проверяется по фактическому frozen корпусу."""
    data = _clone(evidence_payload)
    sources = data["doc_sources"]
    first = next(iter(sources))
    if corruption == "missing":
        sources.pop(first)
    elif corruption == "different_source":
        sources[first] = "forged"
    else:
        sources["forged"] = "source"
    with pytest.raises(ValueError, match="source_id"):
        f3.validate_comparison(data, "a" * 64)


@pytest.mark.parametrize("invalid", ["bad", "X" * 64, None])
def test_source_sha_must_be_canonical(evidence_payload, invalid):
    """Отпечаток входного JSON не может быть фиктивным форматом."""
    with pytest.raises(ValueError, match="SHA-256"):
        f3.validate_comparison(evidence_payload, invalid)


def test_reject_duplicate_json_key(tmp_path):
    """В JSON с повторными ключами нельзя доверять заявленным метрикам."""
    path = tmp_path / "duplicate.json"
    path.write_text('{"n_queries":80,"n_queries":2}', encoding="utf-8")
    with pytest.raises(ValueError, match="Повтор"):
        f3._read_evidence(path)


@pytest.mark.parametrize("bad", ['{"v":NaN}', '{"v":Infinity}', '[]', '{"broken":'])
def test_reject_nonstandard_or_invalid_json(tmp_path, bad):
    """Неполные и нестандартные JSON не должны создавать отчёт."""
    path = tmp_path / "invalid.json"
    path.write_text(bad, encoding="utf-8")
    with pytest.raises((ValueError, json.JSONDecodeError)):
        f3._read_evidence(path)


@pytest.mark.parametrize("bad", [b'\xef\xbb\xbf{}', b'{\r"v":1}'])
def test_reject_bom_and_bare_cr(tmp_path, bad):
    """UTF-8 без BOM защищает от неочевидных расхождений на Windows."""
    path = tmp_path / "invalid.json"
    path.write_bytes(bad)
    with pytest.raises(ValueError, match="UTF-8"):
        f3._read_evidence(path)


def test_normalize_crlf_sha_on_windows(tmp_path):
    """Checkout CRLF не меняет каноническую контрольную сумму evidence JSON."""
    first = tmp_path / "lf.json"
    second = tmp_path / "crlf.json"
    first.write_bytes(b'{\n"v":1\n}\n')
    second.write_bytes(b'{\r\n"v":1\r\n}\r\n')
    a, ha = f3._read_evidence(first)
    b, hb = f3._read_evidence(second)
    assert a == b and ha == hb


def test_missing_phase_files_fail_closed(evidence_payload, tmp_path):
    """Отчёт нельзя получить из отдельного JSON без реальных модулей F3."""
    with pytest.raises(FileNotFoundError, match="Не найдены"):
        f3.validate_comparison(evidence_payload, "a" * 64, root=tmp_path)


def test_report_requires_distinct_input_and_output_paths(tmp_path):
    """Не допускаем случайно перезаписать исходный JSON F3.7."""
    path = tmp_path / "any.json"
    with pytest.raises(ValueError, match="различаться"):
        f3.write_phase_report(path, audit_json=path, root=ROOT)


def test_invalid_input_never_creates_public_artifacts(tmp_path, evidence_payload):
    """Отсутствие pretrained-результата блокирует запись двух выходных файлов."""
    bad = _clone(evidence_payload)
    bad["backend"] = "synthetic_test_vectors"
    source = tmp_path / "incoming.json"
    md = tmp_path / "out" / "F3_PHASE_REPORT.md"
    audit = tmp_path / "out" / "F3_PHASE_EVIDENCE.json"
    source.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError, match="pretrained"):
        f3.write_phase_report(source, report_md=md, audit_json=audit, root=ROOT)
    assert not md.exists() and not audit.exists()


def test_full_write_is_idempotent_and_utf8(tmp_path, evidence_payload):
    """Одинаковый вход даёт точь-в-точь тот же Markdown и JSON с LF."""
    source = tmp_path / "from_f3_07.json"
    source.write_text(json.dumps(evidence_payload, ensure_ascii=False), encoding="utf-8", newline="\n")
    md = tmp_path / "report.md"
    data = tmp_path / "evidence.json"
    got_md, got_data, audit = f3.write_phase_report(source, report_md=md, audit_json=data, root=ROOT)
    initial = (md.read_bytes(), data.read_bytes())
    f3.write_phase_report(source, report_md=md, audit_json=data, root=ROOT)
    assert initial == (md.read_bytes(), data.read_bytes())
    assert got_md == md and got_data == data
    assert audit.source_sha256 == hashlib.sha256(source.read_bytes()).hexdigest()
    assert b"\r\n" not in initial[0] and b"\r\n" not in initial[1]
    assert json.loads(data.read_text(encoding="utf-8"))["audit_status"] == "PASS"


def test_export_does_not_include_source_rankings(tmp_path, evidence_payload):
    """Компактный итоговый аудит не дублирует 240 поисковых списков F3.7."""
    source = tmp_path / "result.json"
    source.write_text(json.dumps(evidence_payload), encoding="utf-8")
    md, data, _ = f3.write_phase_report(source, report_md=tmp_path / "md.md", audit_json=tmp_path / "out.json", root=ROOT)
    assert "per_query_rankings" not in data.read_text(encoding="utf-8")
    assert "query_id" not in md.read_text(encoding="utf-8")


def test_cli_smoke_with_cp1251_parent_console():
    """Регрессия Windows UnicodeDecodeError: дочерний Python всегда UTF-8."""
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "f3_phase_report.py"), "--smoke"],
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8", env=env,
        timeout=25, check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "smoke: PASS" in proc.stdout
    assert "реальные retrieval-метрики не рассчитывались" in proc.stdout


def test_cli_missing_real_artifact_exits_nonzero(tmp_path):
    """Отсутствие F3.7 real JSON запрещает формировать фиктивный итоговый отчёт."""
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "f3_phase_report.py"),
         "--comparison-json", str(tmp_path / "missing.json"),
         "--report-md", str(tmp_path / "fake.md")],
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8", env=env,
        timeout=25, check=False,
    )
    assert proc.returncode != 0
    assert "F3.9 FAIL" in proc.stderr
    assert not (tmp_path / "fake.md").exists()
