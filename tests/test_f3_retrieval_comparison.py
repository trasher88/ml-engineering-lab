"""F3.7: проверяем формулы, честность выборки и отсутствие искусственных метрик.

Все тестовые embeddings созданы вручную или вычислены как хеши слов:
предобученную модель, Интернет и конфигурации обученных моделей не загружаем.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from transformers_f3.retrieval_benchmark import (  # noqa: E402
    BenchmarkDocument, BenchmarkQuery, FrozenBenchmark, RelevanceJudgment,
    load_frozen_benchmark, evaluate_rankings,
)
from transformers_f3 import retrieval_comparison as comp  # noqa: E402


@pytest.fixture(scope="module")
def config() -> comp.ComparisonConfig:
    """Фиксированный протокол; повторные чтения не зависят от working directory."""
    return comp.ComparisonConfig.from_json()


@pytest.fixture(scope="module")
def frozen() -> FrozenBenchmark:
    """Проверенный F3.6 benchmark: ни один запрос не должен исчезать из оценки."""
    return load_frozen_benchmark()


@pytest.fixture
def toy() -> FrozenBenchmark:
    """Независимая разметка из четырёх документов и трёх вопросов."""
    return FrozenBenchmark(
        "toy", (
            BenchmarkDocument("a", "s1", "databases", "postgres index"),
            BenchmarkDocument("b", "s2", "databases", "linux shell"),
            BenchmarkDocument("c", "s1", "databases", "sql query"),
            BenchmarkDocument("d", "s3", "databases", "advanced network protocol"),
        ), (
            BenchmarkQuery("q1", "databases", "lexical", "postgres index"),
            BenchmarkQuery("q2", "databases", "paraphrase", "query performance"),
            BenchmarkQuery("q3", "databases", "lexical", "unknown nonexistent"),
        ), (
            RelevanceJudgment("q1", "a", 1),
            RelevanceJudgment("q1", "c", 1),
            RelevanceJudgment("q2", "c", 1),
            RelevanceJudgment("q3", "d", 1),
        ), "toy-fingerprint",
    )


@pytest.fixture
def vector_data() -> tuple[np.ndarray, np.ndarray]:
    """Три ручных Query вектора, заведомо не являющихся trained embeddings."""
    return (
        np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0.5, 0.5]], dtype=np.float32),
        np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=np.float32),
    )


@pytest.fixture
def toy_result(toy: FrozenBenchmark, vector_data: tuple[np.ndarray, np.ndarray], config: comp.ComparisonConfig) -> comp.ComparisonResult:
    """Объединённые ранги и метрики, которые нельзя публиковать как модельные."""
    return comp.compare_benchmark(
        toy, *vector_data, config, backend="synthetic_test_vectors",
        model_metadata={"origin": "hand-crafted"},
    )


def test_protocol_sha_is_frozen(config: comp.ComparisonConfig) -> None:
    """Изменение конфигурации до сравнения должно дать новый протокол."""
    assert config.sha256 == comp.FROZEN_CONFIG_SHA256
    assert hashlib.sha256(comp.default_comparison_config().read_bytes()).hexdigest() == config.sha256


def test_config_expected_values(config: comp.ComparisonConfig) -> None:
    """Сопоставимость требует заранее известных k, RRF и BM25 параметров."""
    assert config.top_ks == (1, 5, 10)
    assert config.candidate_pool_size == 20
    assert (config.bm25_k1, config.bm25_b) == (1.5, 0.75)
    assert (config.rrf_k, config.lexical_weight, config.vector_weight) == (60, 1.0, 1.0)
    assert config.batch_size == 16


def test_config_roundtrip_values(config: comp.ComparisonConfig) -> None:
    """Опубликованный JSON результата содержит все параметры без потери."""
    assert list(config.to_dict()) == ["top_ks", "candidate_pool_size", "bm25_k1", "bm25_b", "rrf_k", "lexical_weight", "vector_weight", "batch_size"]
    assert config.to_dict()["top_ks"] == [1, 5, 10]


def test_config_read_outside_cwd(config: comp.ComparisonConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Запуск из любого каталога обращается к конфигу внутри проекта."""
    monkeypatch.chdir(tmp_path)
    assert comp.ComparisonConfig.from_json() == config


@pytest.mark.parametrize("field,value", [
    ("top_ks", []), ("top_ks", [5, 1]), ("top_ks", [1, 1]), ("top_ks", [1, True]),
    ("candidate_pool_size", 2), ("candidate_pool_size", 0),
    ("batch_size", -1), ("batch_size", True), ("rrf_k", 0),
    ("bm25_k1", -2.0), ("bm25_b", -0.1), ("bm25_b", 1.1),
    ("vector_weight", 0.0), ("lexical_weight", float("inf")),
])
def test_invalid_config_rejected(tmp_path: Path, config: comp.ComparisonConfig, field: str, value: object) -> None:
    """Непроверенные параметры нельзя ввести через пользовательский JSON."""
    data = {"schema_version": 1, **config.to_dict()}
    data[field] = value
    path = tmp_path / "broken.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises((ValueError, TypeError)):
        comp.ComparisonConfig.from_json(path)


@pytest.mark.parametrize("corruption", [
    '{"schema_version":1,"schema_version":1}',
    '{"schema_version":2}',
    '{"schema_version":1,"unexpected":5}',
    'garbage',
])
def test_corrupted_config_rejected(tmp_path: Path, corruption: str) -> None:
    """Повтор JSON-ключей, неизвестная схема и неполные конфиги — ошибка."""
    path = tmp_path / "config.json"
    path.write_text(corruption, encoding="utf-8")
    with pytest.raises((ValueError, json.JSONDecodeError)):
        comp.ComparisonConfig.from_json(path)


def test_config_crlf_is_stable(tmp_path: Path) -> None:
    """Git checkout с CRLF не меняет SHA заранее утверждённого протокола."""
    payload = comp.default_comparison_config().read_bytes().replace(b"\n", b"\r\n")
    path = tmp_path / "config.json"
    path.write_bytes(payload)
    assert comp.ComparisonConfig.from_json(path).sha256 == comp.FROZEN_CONFIG_SHA256


@pytest.mark.parametrize("bad", [b'\xef\xbb\xbf{}', b'{\r"x":1}'])
def test_rejects_bom_and_bare_cr(tmp_path: Path, bad: bytes) -> None:
    """Межплатформенный SHA требует однозначного UTF-8."""
    path = tmp_path / "config.json"
    path.write_bytes(bad)
    with pytest.raises(ValueError):
        comp.ComparisonConfig.from_json(path)


def test_config_immutable(config: comp.ComparisonConfig) -> None:
    """Параметры не должны изменяться во время обработки разных вопросов."""
    with pytest.raises(FrozenInstanceError):
        config.rrf_k = 2  # type: ignore[misc]


def test_generate_rankings_no_qrels_argument(toy: FrozenBenchmark, vector_data: tuple[np.ndarray, np.ndarray], config: comp.ComparisonConfig) -> None:
    """Структура интерфейса не допускает передачи релевантности в ранжировщик."""
    rankings = comp.generate_rankings(toy.documents, toy.queries, *vector_data, config)
    assert set(rankings.as_dict()) == set(comp.METHODS)
    assert all(set(method) == {"q1", "q2", "q3"} for method in rankings.as_dict().values())


def test_vector_first_is_known_unit_direction(toy_result: comp.ComparisonResult) -> None:
    """Для единичного ручного вектора первый dense-документ предсказуем."""
    assert toy_result.rankings.vector["q1"][0] == "a"
    assert toy_result.rankings.vector["q2"][0] == "c"
    assert toy_result.rankings.vector["q3"][0] == "b"


def test_bm25_uses_words_not_qrels(toy_result: comp.ComparisonResult) -> None:
    """Неизвестный запрос без терминов возвращает пустой список, а не qrels."""
    assert toy_result.rankings.bm25["q3"] == ()
    assert toy_result.rankings.bm25["q1"][0] == "a"


def test_rankings_have_no_duplicate_ids(toy_result: comp.ComparisonResult) -> None:
    """Несколько фрагментов одного source_id допустимы, дубли doc_id — нет."""
    for method in toy_result.rankings.as_dict().values():
        for ranks in method.values():
            assert len(ranks) == len(set(ranks))
            assert set(ranks) <= {"a", "b", "c", "d"}


def test_relevance_does_not_change_search(toy: FrozenBenchmark, vector_data: tuple[np.ndarray, np.ndarray], config: comp.ComparisonConfig) -> None:
    """Перестановка ground truth влияет на метрики, но НЕ на rankings."""
    original = comp.compare_benchmark(toy, *vector_data, config, backend="synthetic_test_vectors", model_metadata={"origin": "synthetic"})
    altered = replace(toy, judgments=(RelevanceJudgment("q1", "d", 1), RelevanceJudgment("q2", "a", 1), RelevanceJudgment("q3", "c", 1)))
    changed = comp.compare_benchmark(altered, *vector_data, config, backend="synthetic_test_vectors", model_metadata={"origin": "synthetic"})
    assert original.rankings == changed.rankings
    assert original.evaluations["vector"][1].recall_at_k != changed.evaluations["vector"][1].recall_at_k


def test_results_use_f3_06_exact_evaluator(toy: FrozenBenchmark, toy_result: comp.ComparisonResult) -> None:
    """Нет новой версии формул Recall/MRR, используются функции F3.6."""
    for method, rankings in toy_result.rankings.as_dict().items():
        for k in (1, 5, 10):
            assert toy_result.evaluations[method][k] == evaluate_rankings(toy, rankings, k=k)


def test_manual_recall_with_two_relevant(toy_result: comp.ComparisonResult) -> None:
    """У q1 два релевантных документа, один в top-1 даёт Recall@1=0.5."""
    q1 = next(row for row in toy_result.evaluations["vector"][1].per_query if row.query_id == "q1")
    assert q1.relevant_count == 2
    assert q1.hits_at_k == 1
    assert q1.recall_at_k == 0.5
    assert q1.reciprocal_rank_at_k == 1.0


def test_empty_bm25_ranking_contributes_zero(toy_result: comp.ComparisonResult) -> None:
    """Даже неудачный запрос входит в знаменатель макро-метрик."""
    row = next(x for x in toy_result.evaluations["bm25"][5].per_query if x.query_id == "q3")
    assert (row.hits_at_k, row.recall_at_k, row.reciprocal_rank_at_k) == (0, 0.0, 0.0)


@pytest.mark.parametrize("method", ["bm25", "vector", "hybrid"])
@pytest.mark.parametrize("k", [1, 5, 10])
def test_evaluates_all_queries(toy_result: comp.ComparisonResult, method: str, k: int) -> None:
    """Один и тот же набор запросов оценивается для всех методов и top-k."""
    evaluation = toy_result.evaluations[method][k]
    assert evaluation.n_queries == len(evaluation.per_query) == 3
    assert {x.query_id for x in evaluation.per_query} == {"q1", "q2", "q3"}
    assert 0 <= evaluation.recall_at_k <= 1 and 0 <= evaluation.mrr_at_k <= 1


def test_metric_monotonicity(toy_result: comp.ComparisonResult) -> None:
    """При большем K Recall и MRR не должны уменьшаться."""
    for method in comp.METHODS:
        evaluations = toy_result.evaluations[method]
        assert evaluations[1].recall_at_k <= evaluations[5].recall_at_k <= evaluations[10].recall_at_k
        assert evaluations[1].mrr_at_k <= evaluations[5].mrr_at_k <= evaluations[10].mrr_at_k


def test_backend_requires_explicit_label(toy: FrozenBenchmark, vector_data: tuple[np.ndarray, np.ndarray], config: comp.ComparisonConfig) -> None:
    """Тестовые оценки нельзя назвать измерением pretrained без смены маркировки."""
    with pytest.raises(ValueError, match="backend"):
        comp.compare_benchmark(toy, *vector_data, config, backend="unspecified", model_metadata={"origin": "toy"})


def test_missing_model_metadata_forbidden(toy: FrozenBenchmark, vector_data: tuple[np.ndarray, np.ndarray], config: comp.ComparisonConfig) -> None:
    """След model/version обязателен для связки score с моделью."""
    with pytest.raises(ValueError, match="model_metadata"):
        comp.compare_benchmark(toy, *vector_data, config, backend="synthetic_test_vectors", model_metadata={})


@pytest.mark.parametrize("corruption", [
    "not_numpy", "bad_docs", "bad_queries", "wrong_dim", "integer", "nan", "inf", "zero", "wrong_rank",
])
def test_bad_vectors_refused(toy: FrozenBenchmark, vector_data: tuple[np.ndarray, np.ndarray], config: comp.ComparisonConfig, corruption: str) -> None:
    """Ошибочная матрица или несовпадение строк и ID не искажают метрики."""
    docs, queries = (v.copy() for v in vector_data)
    if corruption == "not_numpy":
        docs = docs.tolist()  # type: ignore[assignment]
    elif corruption == "bad_docs":
        docs = docs[:-1]
    elif corruption == "bad_queries":
        queries = queries[:-1]
    elif corruption == "wrong_dim":
        queries = queries[:, :-1]
    elif corruption == "integer":
        docs = docs.astype(np.int64)
    elif corruption == "nan":
        docs[0, 0] = np.nan
    elif corruption == "inf":
        docs[0, 0] = np.inf
    elif corruption == "zero":
        queries[0] = 0
    elif corruption == "wrong_rank":
        queries = queries[0]
    with pytest.raises((TypeError, ValueError)):
        comp.generate_rankings(toy.documents, toy.queries, docs, queries, config)


def test_doc_order_determines_vector_row_binding(toy: FrozenBenchmark, vector_data: tuple[np.ndarray, np.ndarray], config: comp.ComparisonConfig) -> None:
    """Если меняем порядок документов и строк embeddings согласованно, выдача не меняется."""
    documents = toy.documents[::-1]
    embeddings = vector_data[0][::-1].copy()
    before = comp.generate_rankings(toy.documents, toy.queries, *vector_data, config)
    after = comp.generate_rankings(documents, toy.queries, embeddings, vector_data[1], config)
    assert before == after


def test_query_order_independent(toy: FrozenBenchmark, vector_data: tuple[np.ndarray, np.ndarray], config: comp.ComparisonConfig) -> None:
    """Результат не зависит от порядка обхода запросов, если вектора привязаны верно."""
    before = comp.generate_rankings(toy.documents, toy.queries, *vector_data, config)
    after = comp.generate_rankings(toy.documents, toy.queries[::-1], vector_data[0], vector_data[1][::-1].copy(), config)
    assert before == after


def test_errors_on_empty_docs_or_queries(toy: FrozenBenchmark, vector_data: tuple[np.ndarray, np.ndarray], config: comp.ComparisonConfig) -> None:
    """Без корпуса/вопросов эксперимент не имеет осмысленного знаменателя."""
    with pytest.raises(ValueError):
        comp.generate_rankings((), toy.queries, *vector_data, config)
    with pytest.raises(ValueError):
        comp.generate_rankings(toy.documents, (), *vector_data, config)


def test_dedup_docs_required(toy: FrozenBenchmark, vector_data: tuple[np.ndarray, np.ndarray], config: comp.ComparisonConfig) -> None:
    """Повторные doc_id нарушают сопоставление индекса и векторов."""
    documents = (toy.documents[0],) * 4
    with pytest.raises(ValueError):
        comp.generate_rankings(documents, toy.queries, *vector_data, config)


def test_report_refuses_synthetic_result(toy_result: comp.ComparisonResult) -> None:
    """Ни Markdown, ни автоматический экспорт toy metrics не являются публичными."""
    with pytest.raises(ValueError, match="pretrained"):
        comp.build_markdown_report(toy_result)


def test_export_refuses_synthetic_result(toy_result: comp.ComparisonResult, tmp_path: Path) -> None:
    """Офлайн-проверка не создаёт подложные evidence-файлы."""
    with pytest.raises(ValueError, match="синтетическим"):
        comp.write_comparison_artifacts(toy_result, tmp_path / "results")
    assert not (tmp_path / "results").exists()


def test_json_audit_includes_all_rankings(toy_result: comp.ComparisonResult) -> None:
    """Аудит должен хранить каждый ranking, а не только сводную таблицу."""
    result = comp.result_as_dict(toy_result)
    assert result["schema_version"] == 1
    assert result["n_queries"] == 3
    assert result["methods"] == ["bm25", "vector", "hybrid"]
    assert len(result["per_query_rankings"]) == 3
    assert set(result["metrics"]) == set(comp.METHODS)
    assert result["backend"] == "synthetic_test_vectors"
    assert result["benchmark_manifest_sha256"] == "toy-fingerprint"


def test_published_report_generated_from_real_labeled_data(toy_result: comp.ComparisonResult, tmp_path: Path) -> None:
    """Генератор строит Markdown из вычисленных оценок, не из шаблонных чисел."""
    labeled = replace(toy_result, backend="pretrained_transformer", model_metadata={
        "model_id": "fake/manual-for-report-tests", "revision": "0" * 40, "license": "test-only",
        "model_manifest_sha256": "synthetic-test", "library_versions": {"torch": "test"},
    })
    report, evidence = comp.write_comparison_artifacts(labeled, tmp_path / "public")
    rendered = report.read_text(encoding="utf-8")
    audit = json.loads(evidence.read_text(encoding="utf-8"))
    assert "BM25 vs Vector vs Hybrid" in rendered
    assert "toy" in rendered
    assert "fake/manual-for-report-tests" in rendered
    assert "## Диагностика разногласий" in rendered
    assert "## Ограничения" in rendered
    assert "Это **авторский синтетический**" in rendered
    assert audit["metrics"]["vector"]["5"]["recall_at_k"] == toy_result.evaluations["vector"][5].recall_at_k


def test_report_lists_improvements_and_degradations(toy_result: comp.ComparisonResult) -> None:
    """Негативные результаты не скрываются из итогового сравнения."""
    labeled = replace(toy_result, backend="pretrained_transformer")
    text = comp.build_markdown_report(labeled)
    assert "Примеры улучшений" in text
    assert "Примеры ухудшений" in text
    assert "без изменений" in text


def test_report_deterministic(toy_result: comp.ComparisonResult) -> None:
    """Одни и те же ранги должны давать одинаковые bytes отчёта."""
    labeled = replace(toy_result, backend="pretrained_transformer")
    assert comp.build_markdown_report(labeled) == comp.build_markdown_report(labeled)


def test_frozen_benchmark_integration_counts(frozen: FrozenBenchmark) -> None:
    """F3.7 использует все реальные записи frozen F3.6 без новых qrels."""
    assert len(frozen.documents) == 88
    assert len(frozen.queries) == 80
    assert len(frozen.judgments) == 160


def _stable_bow_embedding(text: str) -> np.ndarray:
    """Непредобученный и несемантический хеш токенов, только для интеграции."""
    from transformers_f3.bm25 import tokenize
    vec = np.zeros(32, dtype=np.float32)
    for token in tokenize(text):
        position = int.from_bytes(hashlib.sha256(token.encode()).digest()[:2], "big") % 32
        vec[position] += 1
    return vec if np.any(vec) else np.ones(32, dtype=np.float32)


@pytest.fixture(scope="module")
def frozen_synthetic_result(frozen: FrozenBenchmark, config: comp.ComparisonConfig) -> comp.ComparisonResult:
    """Прогнать все 80 запросов без моделей, сети и использования qrels в индексе."""
    docs = np.stack([_stable_bow_embedding(d.text) for d in frozen.documents])
    qs = np.stack([_stable_bow_embedding(q.text) for q in frozen.queries])
    return comp.compare_benchmark(
        frozen, docs, qs, config, backend="synthetic_test_vectors",
        model_metadata={"origin": "stable-hashed-bow; not pretrained, not semantic"},
    )


@pytest.mark.parametrize("method", comp.METHODS)
@pytest.mark.parametrize("k", [1, 5, 10])
def test_frozen_all_queries_and_scores(frozen_synthetic_result: comp.ComparisonResult, method: str, k: int) -> None:
    """Покрытие фиксированного benchmark: каждую метрику считаем по 80 запросам."""
    row = frozen_synthetic_result.evaluations[method][k]
    assert row.n_queries == len(row.per_query) == 80
    assert 0.0 <= row.recall_at_k <= 1.0
    assert 0.0 <= row.mrr_at_k <= 1.0


def test_frozen_rankings_keep_identical_query_ids(frozen_synthetic_result: comp.ComparisonResult, frozen: FrozenBenchmark) -> None:
    """Не теряем ни одного неудобного запроса в сравнении трёх систем."""
    expected = {q.query_id for q in frozen.queries}
    assert all(set(method) == expected for method in frozen_synthetic_result.rankings.as_dict().values())


def test_frozen_rankings_only_existing_docs(frozen_synthetic_result: comp.ComparisonResult, frozen: FrozenBenchmark) -> None:
    """Никакие результаты не выходят за пределы одного corpus freeze."""
    valid = {doc.doc_id for doc in frozen.documents}
    assert all(set(ranking) <= valid for method in frozen_synthetic_result.rankings.as_dict().values() for ranking in method.values())


def test_frozen_no_public_report(frozen_synthetic_result: comp.ComparisonResult) -> None:
    """Числа от hashing-BOW невозможно случайно сохранить как pretrained отчет."""
    with pytest.raises(ValueError):
        comp.build_markdown_report(frozen_synthetic_result)


def test_frozen_no_hidden_label_in_metadata(frozen_synthetic_result: comp.ComparisonResult) -> None:
    """Списки поиска не содержат релевантности, только позиции документов."""
    evidence = comp.result_as_dict(frozen_synthetic_result)
    assert "qrels" not in evidence
    assert all(set(row["rankings"]) == set(comp.METHODS) for row in evidence["per_query_rankings"])


def test_cli_smoke_without_pretrained_or_model_download() -> None:
    """CLI тестируем в независимом процессе с принудительным UTF-8 на Windows."""
    environment = os.environ.copy()
    environment["PYTHONIOENCODING"] = "utf-8"
    environment["PYTHONUTF8"] = "1"
    command = [sys.executable, str(ROOT / "scripts" / "retrieval_comparison.py")]
    result = subprocess.run(command, cwd=ROOT, env=environment, capture_output=True,
                            text=True, encoding="utf-8", errors="strict", check=False, timeout=30)
    assert result.returncode == 0, result.stderr
    assert "smoke: PASS" in result.stdout
    assert "НЕ оценка качества" in result.stdout


def test_cli_rejects_network_without_real() -> None:
    """Флаг скачивания никогда не должен быть допустим по умолчанию."""
    environment = os.environ.copy()
    environment["PYTHONIOENCODING"] = "utf-8"
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "retrieval_comparison.py"), "--download"],
        cwd=ROOT, env=environment, capture_output=True, text=True, encoding="utf-8",
        errors="strict", check=False, timeout=20,
    )
    assert result.returncode != 0
    assert "--real" in result.stderr


def test_cli_smoke_writes_no_report(tmp_path: Path) -> None:
    """Простая проверка алгоритма не создаёт публикационные JSON/MD."""
    environment = os.environ.copy()
    environment["PYTHONIOENCODING"] = "utf-8"
    command = [sys.executable, str(ROOT / "scripts" / "retrieval_comparison.py"),
               "--out-dir", str(tmp_path / "results")]
    result = subprocess.run(command, cwd=ROOT, env=environment, capture_output=True,
                            text=True, encoding="utf-8", check=False, timeout=30)
    assert result.returncode == 0, result.stderr
    assert not (tmp_path / "results").exists()


def test_report_full_80_query_format_is_complete(
    frozen_synthetic_result: comp.ComparisonResult, tmp_path: Path
) -> None:
    """Проверить формат реального отчёта на всех доменах без заявления pretrained-метрик.

    Двойник с поддельной меткой backend создаётся исключительно в tmp_path:
    публичная команда CLI не умеет публиковать тестовые векторы.
    """
    test_only = replace(frozen_synthetic_result, backend="pretrained_transformer", model_metadata={
        "model_id": "tests/fake-only", "revision": "0" * 40,
        "license": "test-only", "model_manifest_sha256": "test-sha",
        "library_versions": {"torch": "test", "transformers": "test", "huggingface_hub": "test"},
    })
    md, evidence = comp.write_comparison_artifacts(test_only, tmp_path / "generated")
    content = md.read_text(encoding="utf-8")
    data = json.loads(evidence.read_text(encoding="utf-8"))
    assert len(data["per_query_rankings"]) == 80
    assert len(data["doc_sources"]) == 88
    assert all(len(row["per_query_metrics"]) == 3 for row in data["per_query_rankings"])
    assert "| paraphrase | vector | 40 |" in content
    assert "| databases | hybrid | 20 |" in content
    assert "| programming | bm25 | 20 |" in content
    assert "## Диагностика разногласий" in content
    assert all(k in data["metrics"]["vector"] for k in ("1", "5", "10"))


def test_config_cannot_silently_change_on_default_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Изменение даже корректного параметра требует нового pinned protocol."""
    content = comp.default_comparison_config().read_text(encoding="utf-8")
    path = tmp_path / "changed.json"
    path.write_text(content.replace('"rrf_k": 60', '"rrf_k": 61'), encoding="utf-8")
    monkeypatch.setattr(comp, "default_comparison_config", lambda: path)
    with pytest.raises(ValueError, match="SHA-256 pinned"):
        comp.ComparisonConfig.from_json()


def test_source_id_mapping_is_preserved(toy_result: comp.ComparisonResult) -> None:
    """Одинаковые source_id двух фрагментов сохраняются, doc_id не сливаются."""
    result = comp.result_as_dict(toy_result)
    assert result["doc_sources"]["a"] == result["doc_sources"]["c"] == "s1"
    assert len(result["doc_sources"]) == 4
