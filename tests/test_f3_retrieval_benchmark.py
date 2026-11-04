"""Тесты F3.6: целостность замороженного корпуса и математика retrieval-метрик.

Тесты автономны и не используют pretrained weights, сеть или рабочий датасет.
Примеры оценок в тестах ручные; их нельзя выдавать за качество модели.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import FrozenInstanceError, replace
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from transformers_f3 import retrieval_benchmark as rb  # noqa: E402


@pytest.fixture(scope="module")
def frozen() -> rb.FrozenBenchmark:
    """Загружать публичный тестовый корпус один раз; проверка SHA обязательна."""
    return rb.load_frozen_benchmark()


@pytest.fixture
def copied(tmp_path: Path) -> Path:
    """Изолированная копия для проверок порчи без модификации эталона."""
    root = tmp_path / "frozen"
    shutil.copytree(rb.default_benchmark_dir(), root)
    return root


@pytest.fixture
def toy() -> rb.FrozenBenchmark:
    """Независимый игрушечный пример 2 запроса / 4 документа / 3 qrels."""
    documents = tuple(
        rb.BenchmarkDocument(did, "source-" + did, "databases", "text " + did)
        for did in ("a", "b", "c", "d")
    )
    queries = (
        rb.BenchmarkQuery("q1", "databases", "lexical", "query one"),
        rb.BenchmarkQuery("q2", "databases", "paraphrase", "query two"),
    )
    qrels = (
        rb.RelevanceJudgment("q1", "a", 1),
        rb.RelevanceJudgment("q1", "c", 1),
        rb.RelevanceJudgment("q2", "d", 1),
    )
    return rb.FrozenBenchmark("toy", documents, queries, qrels, "toy")


def test_default_directory_is_under_project() -> None:
    """Путь относителен к модулю, а не к текущей рабочей папке."""
    assert rb.default_benchmark_dir() == ROOT / "tests" / "fixtures" / "f3_06_benchmark_v1"


def test_public_benchmark_is_frozen_version(frozen: rb.FrozenBenchmark) -> None:
    """Случайная замена идентификатора набора делает результаты несравнимыми."""
    assert frozen.benchmark_id == rb.BENCHMARK_ID
    assert frozen.manifest_sha256 == rb.FROZEN_MANIFEST_SHA256
    assert len(frozen.manifest_sha256) == 64


def test_benchmark_counts_are_exact(frozen: rb.FrozenBenchmark) -> None:
    """Знаменатель будущей оценки всегда включает все 80 вопросов."""
    assert (len(frozen.documents), len(frozen.queries), len(frozen.judgments)) == (88, 80, 160)


@pytest.mark.parametrize("domain", ["databases", "linux", "systems", "programming"])
def test_domain_balance(frozen: rb.FrozenBenchmark, domain: str) -> None:
    """Каждый домен F1 получает одинаковую долю синтетических примеров."""
    assert sum(doc.domain == domain for doc in frozen.documents) == 22
    assert sum(q.domain == domain for q in frozen.queries) == 20


@pytest.mark.parametrize("kind", ["lexical", "paraphrase"])
def test_query_style_balance(frozen: rb.FrozenBenchmark, kind: str) -> None:
    """Прямые и перефразированные вопросы представлены равномерно."""
    assert sum(q.kind == kind for q in frozen.queries) == 40
    for domain in ("databases", "linux", "systems", "programming"):
        assert sum(q.kind == kind and q.domain == domain for q in frozen.queries) == 10


def test_ids_are_unique_without_embedded_domain_hint(frozen: rb.FrozenBenchmark) -> None:
    """ID не должны напрямую выдавать тематику или ответ на вопрос."""
    assert len({d.doc_id for d in frozen.documents}) == 88
    assert len({q.query_id for q in frozen.queries}) == 80
    assert all(d.doc_id.startswith("doc-") and len(d.doc_id) == 20 for d in frozen.documents)
    assert all(q.query_id.startswith("qry-") and len(q.query_id) == 20 for q in frozen.queries)
    assert all(d.domain not in d.doc_id for d in frozen.documents)
    assert all(q.domain not in q.query_id for q in frozen.queries)


def test_query_texts_are_not_copies_of_document_texts(frozen: rb.FrozenBenchmark) -> None:
    """Нет дословной утечки ответа копированием целого документа в запрос."""
    texts = {doc.text.casefold() for doc in frozen.documents}
    assert not texts.intersection(q.text.casefold() for q in frozen.queries)


def test_binary_qrels_have_two_relevant_chunks_each(frozen: rb.FrozenBenchmark) -> None:
    """По каждому вопросу точно два релевантных фрагмента одной статьи."""
    grouped = frozen.relevance_map()
    sources = {doc.doc_id: doc.source_id for doc in frozen.documents}
    assert len(grouped) == 80
    assert all(len(relevant) == 2 for relevant in grouped.values())
    assert all(len({sources[doc_id] for doc_id in relevant}) == 1 for relevant in grouped.values())
    assert all(type(qrel.relevance) is int and qrel.relevance == 1 for qrel in frozen.judgments)


def test_eight_distractors_have_no_positive_judgments(frozen: rb.FrozenBenchmark) -> None:
    """Лексически похожие нерелевантные документы не получают qrels."""
    positive = {j.doc_id for j in frozen.judgments}
    assert len({d.doc_id for d in frozen.documents} - positive) == 8


def test_query_relevance_map_is_a_copy(frozen: rb.FrozenBenchmark) -> None:
    """Клиент не может изменить внутреннюю разметку через полученную карту."""
    mapping = frozen.relevance_map()
    first = frozen.queries[0].query_id
    mapping[first] = frozenset()
    assert len(frozen.relevance_map()[first]) == 2


def test_frozen_dataclasses_reject_attribute_assignment(frozen: rb.FrozenBenchmark) -> None:
    """Контракт результата исключает случайное изменение ID и метрик."""
    with pytest.raises(FrozenInstanceError):
        frozen.queries[0].query_id = "changed"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        frozen.benchmark_id = "changed"  # type: ignore[misc]


def test_repeated_load_is_deterministic(frozen: rb.FrozenBenchmark) -> None:
    """Изменение случайного seed не должно влиять на набор или порядок строк."""
    assert rb.load_frozen_benchmark() == frozen


def test_manifest_lists_all_content_hashes() -> None:
    """Все три содержательных JSONL контролируются manifest, включая qrels."""
    root = rb.default_benchmark_dir()
    man = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    assert set(man["files_sha256"]) == {"documents.jsonl", "queries.jsonl", "qrels.jsonl"}
    for filename, sha in man["files_sha256"].items():
        assert hashlib.sha256((root / filename).read_bytes()).hexdigest() == sha


def test_manifest_sha_is_pinned_not_only_self_reported() -> None:
    """Корпус нельзя заменить, просто пересчитав manifest и его вложенные SHA."""
    raw = rb._canonical_bytes(rb.default_benchmark_dir() / "manifest.json")
    assert hashlib.sha256(raw).hexdigest() == rb.FROZEN_MANIFEST_SHA256


@pytest.mark.parametrize("filename", ["manifest.json", "documents.jsonl", "queries.jsonl", "qrels.jsonl"])
def test_loader_accepts_crlf_windows_checkout(copied: Path, filename: str) -> None:
    """Git core.autocrlf не должен ломать равенство содержимого SHA-256."""
    path = copied / filename
    path.write_bytes(path.read_bytes().replace(b"\n", b"\r\n"))
    assert len(rb.load_frozen_benchmark(copied).queries) == 80


@pytest.mark.parametrize("filename", ["manifest.json", "documents.jsonl", "queries.jsonl", "qrels.jsonl"])
def test_loader_rejects_bom(copied: Path, filename: str) -> None:
    """UTF-8 BOM может незаметно изменить hash и JSON-парсер."""
    path = copied / filename
    path.write_bytes(b"\xef\xbb\xbf" + path.read_bytes())
    with pytest.raises(ValueError, match="BOM"):
        rb.load_frozen_benchmark(copied)


@pytest.mark.parametrize("filename", ["manifest.json", "documents.jsonl", "queries.jsonl", "qrels.jsonl"])
def test_loader_rejects_bare_carriage_return(copied: Path, filename: str) -> None:
    """Разрешено только LF или CRLF, но не произвольный перевод строки."""
    path = copied / filename
    path.write_bytes(path.read_bytes() + b"\r")
    with pytest.raises(ValueError, match="переносы строк"):
        rb.load_frozen_benchmark(copied)


@pytest.mark.parametrize("filename", ["manifest.json", "documents.jsonl", "queries.jsonl", "qrels.jsonl"])
def test_loader_rejects_invalid_utf8(copied: Path, filename: str) -> None:
    """Скрытая замена сломанного текста символом replacement недопустима."""
    path = copied / filename
    path.write_bytes(path.read_bytes() + b"\xff")
    with pytest.raises(UnicodeDecodeError):
        rb.load_frozen_benchmark(copied)


@pytest.mark.parametrize("filename", ["documents.jsonl", "queries.jsonl", "qrels.jsonl"])
def test_loader_rejects_mutated_data(copied: Path, filename: str) -> None:
    """Даже небольшая правка текста без смены версии нарушает freeze."""
    path = copied / filename
    path.write_bytes(path.read_bytes().replace(b'"', b'" ', 1))
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        rb.load_frozen_benchmark(copied)


def test_loader_rejects_changed_manifest(copied: Path) -> None:
    """Пересчёт вложенных SHA не должен позволить тихо заменить benchmark."""
    path = copied / "manifest.json"
    path.write_bytes(path.read_bytes().replace(b"2026-11-04", b"2026-11-05"))
    with pytest.raises(ValueError, match="manifest F3.6 v1"):
        rb.load_frozen_benchmark(copied)


@pytest.mark.parametrize("filename", ["manifest.json", "documents.jsonl", "queries.jsonl", "qrels.jsonl"])
def test_missing_file_fails_closed(copied: Path, filename: str) -> None:
    """При отсутствии разметки не допускаем fallback к пустому benchmark."""
    (copied / filename).unlink()
    with pytest.raises(OSError):
        rb.load_frozen_benchmark(copied)


@pytest.mark.parametrize("source", [b'{"a":1,"a":2}', b'{"outer":{"x":1,"x":2}}'])
def test_duplicate_json_keys_are_rejected(source: bytes) -> None:
    """Дубликаты ключей нельзя трактовать по правилу «последний победил»."""
    with pytest.raises(ValueError, match="Дублированный JSON-ключ"):
        rb._decode_json(source)


@pytest.mark.parametrize("data", [b"", b'{}', b'{}\n\n', b'\n{}\n'])
def test_jsonl_invalid_termination_or_empty_lines(data: bytes) -> None:
    """Каждая строка является самостоятельной записью, завершённой LF."""
    with pytest.raises(ValueError):
        rb._read_jsonl(data, filename="test.jsonl", fields={"text"})


@pytest.mark.parametrize("data", [b'[]\n', b'{"a":1}\n', b'{"text":1,"extra":1}\n'])
def test_jsonl_schema_must_match_exactly(data: bytes) -> None:
    """Новые поля требуют нового протокола, а не молчаливой обработки."""
    with pytest.raises(ValueError, match="схема"):
        rb._read_jsonl(data, filename="test.jsonl", fields={"text"})


def test_jsonl_accepts_correct_record() -> None:
    """UTF-8 JSON Lines читаются без зависимостей от pandas и numpy."""
    result = rb._read_jsonl(b'{"text":"hello"}\n', filename="test.jsonl", fields={"text"})
    assert result == [{"text": "hello"}]


@pytest.mark.parametrize("bad_value", [None, 2, 0.0, "", " \t ", [], {}])
def test_required_text_rejects_invalid_values(bad_value: object) -> None:
    """Непустые ID необходимы для точной адресации qrels."""
    with pytest.raises(ValueError, match="непустую строку"):
        rb._text(bad_value, "id")


def test_bad_document_count_is_rejected(frozen: rb.FrozenBenchmark) -> None:
    """Удаление фрагмента меняет базу оценки, даже если запросы остались."""
    with pytest.raises(ValueError, match="88 документов"):
        rb._validate_benchmark(frozen.documents[:-1], frozen.queries, frozen.judgments)


def test_unknown_qrel_doc_is_rejected(frozen: rb.FrozenBenchmark) -> None:
    """Нельзя ссылаться на несуществующий документ при вычислении Recall."""
    modified = (replace(frozen.judgments[0], doc_id="missing"),) + frozen.judgments[1:]
    with pytest.raises(ValueError, match="несуществующий"):
        rb._validate_benchmark(frozen.documents, frozen.queries, modified)


def test_unknown_qrel_query_is_rejected(frozen: rb.FrozenBenchmark) -> None:
    """Неизвестный вопрос мог бы изменить знаменатель незаметно."""
    modified = (replace(frozen.judgments[0], query_id="missing"),) + frozen.judgments[1:]
    with pytest.raises(ValueError, match="несуществующий"):
        rb._validate_benchmark(frozen.documents, frozen.queries, modified)


def test_duplicate_qrels_are_rejected(frozen: rb.FrozenBenchmark) -> None:
    """Одна и та же пара query/doc не может считаться дважды."""
    modified = (frozen.judgments[1],) + frozen.judgments[1:]
    with pytest.raises(ValueError, match="Повторяющаяся пара"):
        rb._validate_benchmark(frozen.documents, frozen.queries, modified)


@pytest.mark.parametrize("invalid_relevance", [0, 2, True, -1, "1"])
def test_qrels_reject_nonbinary_label(frozen: rb.FrozenBenchmark, invalid_relevance: object) -> None:
    """В v1 qrels строго бинарны; никакого скрытого graded relevance."""
    modified = (replace(frozen.judgments[0], relevance=invalid_relevance),) + frozen.judgments[1:]
    with pytest.raises(ValueError, match="бинарные"):
        rb._validate_benchmark(frozen.documents, frozen.queries, modified)


def test_qrel_from_another_domain_is_rejected(frozen: rb.FrozenBenchmark) -> None:
    """Перекрёстная ошибочная ссылка между доменами обнаруживается."""
    query_domain = next(q.domain for q in frozen.queries if q.query_id == frozen.judgments[0].query_id)
    foreign = next(doc for doc in frozen.documents if doc.domain != query_domain)
    modified = (replace(frozen.judgments[0], doc_id=foreign.doc_id),) + frozen.judgments[1:]
    with pytest.raises(ValueError, match="другой домен"):
        rb._validate_benchmark(frozen.documents, frozen.queries, modified)


def test_query_with_one_relevant_doc_rejected(frozen: rb.FrozenBenchmark) -> None:
    """Объём разметки не должен «потерять» один из двух фрагментов."""
    a = frozen.judgments[0]
    b = next(j for j in frozen.judgments if j.query_id != a.query_id)
    modified = (replace(a, query_id=b.query_id),) + frozen.judgments[1:]
    # Новый qrel уже будет дубликатом или нарушит число релевантных.
    with pytest.raises(ValueError):
        rb._validate_benchmark(frozen.documents, frozen.queries, modified)


def test_toy_math_for_recall_and_mrr(toy: rb.FrozenBenchmark) -> None:
    """q1: hit 1/2 на ранге 2; q2: 1/1 на ранге 1; оба macro = 0.75."""
    result = rb.evaluate_rankings(toy, {"q1": ["b", "a", "c"], "q2": ["d", "b"]}, k=2)
    assert result.n_queries == 2
    assert result.k == 2
    assert result.recall_at_k == pytest.approx(0.75)
    assert result.mrr_at_k == pytest.approx(0.75)
    assert result.per_query[0] == rb.QueryMetrics("q1", 2, 2, 1, 0.5, 0.5)
    assert result.per_query[1] == rb.QueryMetrics("q2", 1, 2, 1, 1.0, 1.0)


def test_metric_changes_with_k(toy: rb.FrozenBenchmark) -> None:
    """Увеличение top-k может добавить релевантный документ и не ухудшает recall."""
    ranks = {"q1": ["b", "a", "c"], "q2": ["d", "b"]}
    values = [rb.evaluate_rankings(toy, ranks, k=k) for k in (1, 2, 3)]
    assert [v.recall_at_k for v in values] == pytest.approx([0.5, 0.75, 1.0])
    assert [v.mrr_at_k for v in values] == pytest.approx([0.5, 0.75, 0.75])


def test_empty_rankings_do_not_change_denominator(toy: rb.FrozenBenchmark) -> None:
    """Отсутствие выдачи означает нулевые метрики, но НЕ удаление вопроса."""
    result = rb.evaluate_rankings(toy, {"q1": [], "q2": []}, k=5)
    assert (result.n_queries, result.recall_at_k, result.mrr_at_k) == (2, 0.0, 0.0)
    assert all(v.retrieved_at_k == 0 for v in result.per_query)


def test_one_missing_relevant_doc_is_not_full_recall(toy: rb.FrozenBenchmark) -> None:
    """Hit@K=1 не эквивалентен Recall@K=1 при нескольких qrels."""
    result = rb.evaluate_rankings(toy, {"q1": ["a"], "q2": ["b"]}, k=3)
    assert result.per_query[0].recall_at_k == 0.5
    assert result.per_query[0].reciprocal_rank_at_k == 1.0
    assert result.per_query[1].reciprocal_rank_at_k == 0.0


def test_exact_k_position_is_included(toy: rb.FrozenBenchmark) -> None:
    """Релевантный документ на месте k учитывается в Recall и MRR."""
    result = rb.evaluate_rankings(toy, {"q1": ["b", "a"], "q2": ["b", "c", "d"]}, k=3)
    assert result.per_query[1].reciprocal_rank_at_k == pytest.approx(1 / 3)


def test_beyond_cutoff_does_not_count(toy: rb.FrozenBenchmark) -> None:
    """Документы ниже k не должны влиять на оценку top-k."""
    result = rb.evaluate_rankings(toy, {"q1": ["b", "a"], "q2": ["a", "d"]}, k=1)
    assert result.per_query[0].reciprocal_rank_at_k == 0.0
    assert result.per_query[1].reciprocal_rank_at_k == 0.0


def test_all_relevant_at_top_yield_one(toy: rb.FrozenBenchmark) -> None:
    """Верхняя граница обеих метрик при идеальной выдаче — 1."""
    scores = rb.evaluate_rankings(toy, {"q1": ["a", "c"], "q2": ["d"]}, k=2)
    assert (scores.recall_at_k, scores.mrr_at_k) == (1.0, 1.0)


def test_rankings_are_not_sorted_by_doc_id(toy: rb.FrozenBenchmark) -> None:
    """Порядок поисковой выдачи важен; функция его не пересортировывает."""
    a = rb.evaluate_rankings(toy, {"q1": ["a", "b"], "q2": ["d", "c"]}, k=2)
    b = rb.evaluate_rankings(toy, {"q1": ["b", "a"], "q2": ["c", "d"]}, k=2)
    assert a.mrr_at_k > b.mrr_at_k
    assert a.recall_at_k == b.recall_at_k


def test_metrics_are_immutable(toy: rb.FrozenBenchmark) -> None:
    """После расчёта отчёт не допускает подмену значения отдельного вопроса."""
    score = rb.evaluate_rankings(toy, {"q1": [], "q2": []})
    with pytest.raises(FrozenInstanceError):
        score.mrr_at_k = 1.0  # type: ignore[misc]


@pytest.mark.parametrize("bad_k", [0, -1, 2.0, True, None, "5"])
def test_invalid_cutoff_rejected(toy: rb.FrozenBenchmark, bad_k: object) -> None:
    """Оценка имеет смысл только для целочисленной положительной глубины."""
    with pytest.raises(ValueError, match="положительным Python int"):
        rb.evaluate_rankings(toy, {"q1": [], "q2": []}, k=bad_k)


@pytest.mark.parametrize("bad_ranks", [None, [], ["a"], "ranks"])
def test_rankings_must_be_mapping(toy: rb.FrozenBenchmark, bad_ranks: object) -> None:
    """Нельзя перепутать список документов с картой запросов."""
    with pytest.raises(TypeError, match="отображением"):
        rb.evaluate_rankings(toy, bad_ranks)


@pytest.mark.parametrize("rankings", [
    {},
    {"q1": []},
    {"q2": []},
    {"q1": [], "q2": [], "q3": []},
    {"q1": [], "q3": []},
])
def test_all_queries_must_be_evaluated(toy: rb.FrozenBenchmark, rankings: dict) -> None:
    """Запрещено исключать трудные вопросы из знаменателя или добавлять новые."""
    with pytest.raises(ValueError, match="все вопросы"):
        rb.evaluate_rankings(toy, rankings)


@pytest.mark.parametrize("bad_rank", ["a", None, set(), {"a"}, 1])
def test_each_ranking_must_be_sequence_of_doc_ids(toy: rb.FrozenBenchmark, bad_rank: object) -> None:
    """Ranked list — позиционный list/tuple, а не строка или множество."""
    with pytest.raises(TypeError, match="list/tuple"):
        rb.evaluate_rankings(toy, {"q1": bad_rank, "q2": []})


@pytest.mark.parametrize("rank", [["a", "a"], ["a", "b", "a"]])
def test_duplicate_doc_id_in_ranking_is_error(toy: rb.FrozenBenchmark, rank: list[str]) -> None:
    """Повтор одного документа не должен считаться несколькими попаданиями."""
    with pytest.raises(ValueError, match="Дубли"):
        rb.evaluate_rankings(toy, {"q1": rank, "q2": []})


@pytest.mark.parametrize("bad_id", ["unknown", "", 42, None])
def test_unknown_doc_id_is_error(toy: rb.FrozenBenchmark, bad_id: object) -> None:
    """Оценка выдачи другого корпуса не должна выглядеть корректной."""
    with pytest.raises(ValueError, match="Неизвестный doc_id"):
        rb.evaluate_rankings(toy, {"q1": [bad_id], "q2": []})


def test_full_benchmark_empty_search_yields_zero_with_80_queries(frozen: rb.FrozenBenchmark) -> None:
    """Все 80 query_id оцениваются даже при полностью пустой выдаче."""
    output = rb.evaluate_rankings(frozen, {q.query_id: [] for q in frozen.queries}, k=5)
    assert (output.n_queries, output.recall_at_k, output.mrr_at_k) == (80, 0.0, 0.0)
    assert len(output.per_query) == 80


def test_bm25_index_can_consume_frozen_documents(frozen: rb.FrozenBenchmark) -> None:
    """F3.3 индексирует документы benchmark без преобразования идентификаторов."""
    from transformers_f3.bm25 import BM25Index, Document
    docs = [Document(row.doc_id, row.text) for row in frozen.documents]
    index = BM25Index(docs)
    hits = index.search(frozen.queries[0].text, top_k=5)
    assert index.stats.n_documents == 88
    assert len(hits) <= 5
    assert all(hit.doc_id in {doc.doc_id for doc in frozen.documents} for hit in hits)
    # Это исключительно контракт совместимости: quality-метрики здесь НЕ вычисляются.


def test_cli_smoke_in_fresh_process_cp1251() -> None:
    """Явный UTF-8 на дочернем процессе защищает от Windows cp1251/cp866."""
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "retrieval_benchmark_demo.py")],
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8", env=env,
        check=False, timeout=20,
    )
    assert result.returncode == 0, result.stderr
    assert "queries: 80" in result.stdout
    assert "smoke: PASS" in result.stdout
    assert "TOY ONLY Recall@2: 0.7500" in result.stdout
    assert "real-world evaluation" in result.stdout
