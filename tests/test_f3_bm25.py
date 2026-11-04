"""F3.3: офлайн unit- и smoke-тесты индексации, BM25 и ранжирования.

Никаких реальных train/validation/test данных, сетевых запросов или весов
моделей: проверки строятся на вручную заданном синтетическом корпусе.
"""

from __future__ import annotations

from collections import Counter
import math
import os
from pathlib import Path
import subprocess
import sys

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from transformers_f3.bm25 import (  # noqa: E402
    BM25Index, Document, IndexStats, SearchHit, tokenize,
)


@pytest.fixture
def index() -> BM25Index:
    """Корпус с контролируемыми tf, df, длинами и пересечениями термов."""
    return BM25Index([
        Document("a", "postgres postgres index"),
        Document("b", "postgres backup"),
        Document("c", "linux kernel"),
    ])


@pytest.mark.parametrize("raw,expected", [
    ("PostgreSQL INDEX", ("postgresql", "index")),
    ("  linux   kernel  ", ("linux", "kernel")),
    ("SQL-server/v1.0", ("sql", "server", "v1", "0")),
    ("ДАННЫЕ и Сервер", ("данные", "и", "сервер")),
    ("Ｐｙｔｈｏｎ", ("python",)),
    ("Straße", ("strasse",)),
    ("nvidia_gpu CUDA", ("nvidia_gpu", "cuda")),
    ("one one ONE", ("one", "one", "one")),
    ("", ()),
    ("!? ...", ()),
    ("123 456", ("123", "456")),
])
def test_tokenize(raw: str, expected: tuple[str, ...]) -> None:
    """Одинаковая Unicode-нормализация текста и запроса сохраняет повторы."""
    assert tokenize(raw) == expected


@pytest.mark.parametrize("value", [None, [], 0, 7.2, {}, b"linux"])
def test_tokenize_rejects_nonstrings(value: object) -> None:
    """BM25 не должен молча превращать произвольные объекты в слова."""
    with pytest.raises(TypeError):
        tokenize(value)  # type: ignore[arg-type]


@pytest.mark.parametrize("doc_id", ["", "   ", "\t", None, 4, []])
def test_document_requires_stable_id(doc_id: object) -> None:
    """Без устойчивого ID нельзя корректно связать найденный текст с источником."""
    with pytest.raises(ValueError):
        Document(doc_id, "linux")  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [None, 10, [], object()])
def test_document_requires_string_text(value: object) -> None:
    """Данные документа не должны неявно сериализоваться через str()."""
    with pytest.raises(TypeError):
        Document("id", value)  # type: ignore[arg-type]


def test_empty_document_allowed() -> None:
    """Документ без лексических токенов безопасно остаётся в корпусе."""
    assert Document("empty", "").text == ""


@pytest.mark.parametrize("docs,expected_error", [
    ([Document("a", "one"), Document("a", "two")], ValueError),
    ([Document("a", "one"), "invalid"], TypeError),
    ([123], TypeError),
])
def test_rejects_invalid_corpus(docs: list[object], expected_error: type[Exception]) -> None:
    """Нельзя испортить postings неоднозначными ID и неизвестными типами."""
    with pytest.raises(expected_error):
        BM25Index(docs)  # type: ignore[arg-type]


@pytest.mark.parametrize("argument,value,error", [
    ("k1", -0.1, ValueError),
    ("k1", 0.0, ValueError),
    ("k1", float("inf"), ValueError),
    ("k1", float("nan"), ValueError),
    ("k1", True, TypeError),
    ("k1", "1.5", TypeError),
    ("b", -0.01, ValueError),
    ("b", 1.01, ValueError),
    ("b", float("-inf"), ValueError),
    ("b", float("nan"), ValueError),
    ("b", False, TypeError),
    ("b", "0.75", TypeError),
])
def test_invalid_parameters(argument: str, value: object, error: type[Exception]) -> None:
    """k1 и b обязаны иметь физически интерпретируемые значения."""
    with pytest.raises(error):
        BM25Index([Document("a", "word")], **{argument: value})


def test_boundary_parameters_are_valid() -> None:
    """b=0 отключает length normalization, b=1 включает её полностью."""
    for b in (0.0, 1.0):
        for k1 in (0.1, 1.5, 10):
            assert BM25Index([Document("a", "hi")], k1=k1, b=b).b == b


def test_corpus_statistics(index: BM25Index) -> None:
    """Длина включает повторы; avgdl вычисляется на всех документах."""
    assert index.stats == IndexStats(3, 7, 5, 7 / 3)
    assert index.document_length("a") == 3
    assert index.document_length("b") == 2
    assert index.document_length("c") == 2


def test_inverted_postings_and_document_frequency(index: BM25Index) -> None:
    """tf — частота в конкретном документе, df — число документов."""
    assert index.postings("POSTGRES") == (("a", 2), ("b", 1))
    assert index.document_frequency("postgres") == 2
    assert index.postings("index") == (("a", 1),)
    assert index.postings("nonexistent") == ()
    assert index.document_frequency("nonexistent") == 0


@pytest.mark.parametrize("term", ["", "one two", "???", "postgres index"])
def test_single_term_api_rejects_phrases(index: BM25Index, term: str) -> None:
    """Нельзя ошибочно трактовать целый запрос как один IDF-терм."""
    with pytest.raises(ValueError):
        index.document_frequency(term)
    with pytest.raises(ValueError):
        index.idf(term)
    with pytest.raises(ValueError):
        index.postings(term)


def test_postings_return_immutable_snapshot(index: BM25Index) -> None:
    """Чтение списка кандидатов не должно изменять внутренний индекс."""
    before = index.postings("postgres")
    mutable = list(before)
    mutable.append(("nonexistent", 100))
    assert index.postings("postgres") == before


def test_idf_hand_computed(index: BM25Index) -> None:
    """IDF сверяется с независимой формулой при N=3, df=2 и df=1."""
    assert index.idf("postgres") == pytest.approx(math.log(1.6), abs=1e-12)
    assert index.idf("index") == pytest.approx(math.log(1 + 2.5 / 1.5), abs=1e-12)
    assert index.idf("unknown") == 0


def test_hand_computed_tf_and_length_normalization(index: BM25Index) -> None:
    """Сверяем коэффициент BM25 для tf=2 и tf=1 с независимым эталоном."""
    avgdl = 7 / 3
    idf = math.log(1.6)
    norm_a = 1.5 * (0.25 + 0.75 * (3 / avgdl))
    norm_b = 1.5 * (0.25 + 0.75 * (2 / avgdl))
    expected_a = idf * 2 * 2.5 / (2 + norm_a)
    expected_b = idf * 1 * 2.5 / (1 + norm_b)
    assert index.score("postgres", "a") == pytest.approx(expected_a, abs=1e-12)
    assert index.score("postgres", "b") == pytest.approx(expected_b, abs=1e-12)
    assert index.score("postgres", "c") == 0


def test_multiple_query_terms_add_score(index: BM25Index) -> None:
    """BM25 аддитивен по различным query terms: не Softmax и не cosine."""
    combined = index.score("postgres index", "a")
    separate = index.score("postgres", "a") + index.score("index", "a")
    assert combined == pytest.approx(separate, abs=1e-12)
    assert combined > index.score("postgres", "a")


def test_duplicate_query_terms_do_not_double_count(index: BM25Index) -> None:
    """В выбранном baseline повтор токена запроса не удваивает его вклад."""
    assert index.score("postgres postgres", "a") == index.score("postgres", "a")
    assert index.search("postgres postgres") == index.search("postgres")


def test_score_rejects_missing_document(index: BM25Index) -> None:
    """Ошибку source-id нельзя выдавать за нулевую релевантность."""
    with pytest.raises(KeyError):
        index.score("postgres", "unknown")


def test_get_document_returns_original_source(index: BM25Index) -> None:
    """Поиск сохраняет source-id, а текст не подменяется нормализованной версией."""
    assert index.get_document("a") == Document("a", "postgres postgres index")
    with pytest.raises(KeyError):
        index.get_document("bad")


@pytest.mark.parametrize("query", ["", " ", "?!", "no_matching_words"])
def test_empty_or_oov_query_returns_no_candidates(index: BM25Index, query: str) -> None:
    """Если нет общих термов, выдача пустая, а не произвольный top-k."""
    assert index.search(query) == []


def test_all_empty_corpus_and_empty_index() -> None:
    """Не делим на нулевой avgdl ни при пустом корпусе, ни при пустых текстах."""
    for docs, expected_n in [([], 0), ([Document("e1", ""), Document("e2", "??")], 2)]:
        ranker = BM25Index(docs)
        assert ranker.stats.n_documents == expected_n
        assert ranker.stats.avgdl == 0
        assert ranker.stats.vocabulary_size == 0
        assert ranker.search("postgres") == []
        assert ranker.idf("postgres") == 0


def test_mixed_empty_document_affects_avgdl() -> None:
    """Пустые документы участвуют в N и avgdl, как определено контрактом."""
    ranker = BM25Index([Document("a", "linux"), Document("e", "")])
    assert ranker.stats.n_documents == 2
    assert ranker.stats.avgdl == 0.5
    assert ranker.search("linux")[0].doc_id == "a"
    assert ranker.score("linux", "e") == 0


@pytest.mark.parametrize("top_k", [0, -1, 2.5, True, False, "3", None])
def test_invalid_top_k(index: BM25Index, top_k: object) -> None:
    """Размер выдачи должен быть положительным целым числом, не bool/float."""
    with pytest.raises(ValueError):
        index.search("postgres", top_k=top_k)  # type: ignore[arg-type]


def test_top_k_cutoff_and_positive_scores(index: BM25Index) -> None:
    """Возвращаем не больше top-k совпадений, каждый имеет положительный score."""
    assert len(index.search("postgres", top_k=1)) == 1
    assert len(index.search("postgres", top_k=20)) == 2
    assert all(h.score > 0 for h in index.search("postgres"))


def test_search_hit_rank_is_one_based(index: BM25Index) -> None:
    """Ранги начинаются с 1, а source-id не теряется при сортировке."""
    hits = index.search("postgres", top_k=4)
    assert [h.rank for h in hits] == [1, 2]
    assert all(isinstance(h, SearchHit) for h in hits)
    assert all(h.doc_id in ("a", "b", "c") for h in hits)


def test_search_scores_match_direct_score(index: BM25Index) -> None:
    """Inverted index и прямой скоринг реализуют одну и ту же формулу."""
    for hit in index.search("postgres index", top_k=10):
        assert hit.score == pytest.approx(index.score("postgres index", hit.doc_id), abs=1e-12)


def test_ties_are_broken_by_document_id() -> None:
    """Одинаковые score не дают случайный порядок из-за порядка ввода корпуса."""
    ranker = BM25Index([Document("z", "linux"), Document("b", "linux"), Document("a", "linux")])
    assert [h.doc_id for h in ranker.search("linux")] == ["a", "b", "z"]


def test_input_corpus_order_does_not_affect_ranking() -> None:
    """Сравнение экспериментов воспроизводимо после перестановки документов."""
    docs = [Document("c", "linux kernel"), Document("a", "linux linux"), Document("b", "linux memory")]
    left = BM25Index(docs).search("linux memory", top_k=10)
    right = BM25Index(reversed(docs)).search("linux memory", top_k=10)
    assert left == right


def test_generator_is_consumed_once() -> None:
    """Входным корпусом может быть одноразовый генератор Document."""
    ranker = BM25Index(Document(str(i), f"term {i}") for i in range(5))
    assert ranker.stats.n_documents == 5
    assert len(ranker.search("term", top_k=10)) == 5


def test_length_normalization_can_be_disabled() -> None:
    """При b=0 частота tf=1 даёт равный вклад независимо от длины текста."""
    docs = [Document("short", "postgres"), Document("long", "postgres many other words again")]
    ranker = BM25Index(docs, b=0)
    assert ranker.score("postgres", "short") == ranker.score("postgres", "long")


def test_length_normalization_changes_ranking() -> None:
    """При b>0 короткий документ с тем же tf получает больший балл."""
    docs = [Document("short", "postgres"), Document("long", "postgres many other words again")]
    ranker = BM25Index(docs, b=0.75)
    assert ranker.score("postgres", "short") > ranker.score("postgres", "long")


def test_term_frequency_saturation() -> None:
    """Дополнительные повторы увеличивают вес с уменьшающейся отдачей."""
    ranker = BM25Index([
        Document("one", "postgres"),
        Document("two", "postgres postgres"),
        Document("three", "postgres postgres postgres"),
    ], b=0)
    s1 = ranker.score("postgres", "one")
    s2 = ranker.score("postgres", "two")
    s3 = ranker.score("postgres", "three")
    assert 0 < s1 < s2 < s3
    assert s2 - s1 > s3 - s2


def test_no_mutation_between_queries(index: BM25Index) -> None:
    """Поиск другой фразы не дообучает и не перестраивает индекс."""
    stats_before = index.stats
    postings_before = index.postings("postgres")
    first = index.search("postgres")
    index.search("linux kernel")
    assert index.stats == stats_before
    assert index.postings("postgres") == postings_before
    assert index.search("postgres") == first


def test_large_top_k_returns_only_matching_documents() -> None:
    """Не дополняем результаты нулевыми совпадениями ради достижения K."""
    ranker = BM25Index([Document(str(i), "linux" if i == 2 else "other") for i in range(8)])
    assert [h.doc_id for h in ranker.search("linux", top_k=100)] == ["2"]


def test_unicode_casefold_retrieval() -> None:
    """Кириллица и Unicode-эквиваленты находят тот же текст после NFKC."""
    ranker = BM25Index([Document("ru", "ПРИВЕТ Python"), Document("other", "linux")])
    assert ranker.search("привет")[0].doc_id == "ru"
    assert ranker.search("Ｐｙｔｈｏｎ")[0].doc_id == "ru"


def test_independent_reference_brute_force() -> None:
    """Sparse postings сопоставлены с независимым полным перебором документов.

    Этот тест не использует внутренние posting lists для расчёта эталона:
    tf и df вычисляются заново из текстов синтетического корпуса.
    """
    docs = [
        Document("d3", "postgres query query index"),
        Document("d1", "linux postgres server"),
        Document("d2", "query query tuning"),
        Document("d4", ""),
        Document("d5", "docker linux"),
    ]
    query = "Postgres query index index"
    k1, b = 1.7, 0.6
    ranker = BM25Index(docs, k1=k1, b=b)
    counters = {d.doc_id: Counter(tokenize(d.text)) for d in docs}
    lengths = {key: sum(counts.values()) for key, counts in counters.items()}
    avgdl = sum(lengths.values()) / len(docs)
    expected = []
    for document in docs:
        value = 0.0
        for term in set(tokenize(query)):
            tf = counters[document.doc_id][term]
            if not tf:
                continue
            df = sum(int(term in counter) for counter in counters.values())
            idf = math.log(1 + (len(docs) - df + .5) / (df + .5))
            value += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * lengths[document.doc_id] / avgdl))
        if value > 0:
            expected.append((document.doc_id, value))
    expected.sort(key=lambda pair: (-pair[1], pair[0]))
    hits = ranker.search(query, top_k=20)
    assert [h.doc_id for h in hits] == [item[0] for item in expected]
    for hit, (_, expected_score) in zip(hits, expected):
        assert hit.score == pytest.approx(expected_score, abs=1e-12)


def test_demo_cli_smoke_without_model_download() -> None:
    """Демонстрация запускается как CLI в чистом процессе и не требует данных."""
    project = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, str(project / "scripts" / "bm25_demo.py")],
        cwd=project,
        capture_output=True,
        text=True,
        encoding="utf-8",
        # Дочерний Python обязан писать UTF-8 даже при Windows cp1251/cp866.
        # encoding= выше определяет только декодирование в родительском процессе.
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        check=False,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
    assert "smoke: PASS" in result.stdout
    assert "doc-db-01" in result.stdout
    assert "df(postgresql)" in result.stdout
