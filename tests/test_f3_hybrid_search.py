"""F3.5: независимые unit- и smoke-проверки Hybrid Search и RRF.

Корпус и векторы полностью синтетические. Тесты не загружают модели,
не обращаются к реальным данным и не оценивают Recall@k или MRR.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
import math
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

from transformers_f3.bm25 import BM25Index, Document  # noqa: E402
from transformers_f3.hybrid_search import HybridHit, HybridSearcher  # noqa: E402
from transformers_f3.vector_search import VectorIndex  # noqa: E402


@pytest.fixture
def documents() -> list[Document]:
    """Малый корпус, в котором есть чисто lexical и чисто dense кандидаты."""
    return [
        Document("lexical", "postgresql медленные запросы"),
        Document("semantic", "оптимизация производительности sql"),
        Document("shared", "postgresql оптимизация запросов"),
        Document("unrelated", "linux системные журналы"),
    ]


@pytest.fixture
def embeddings() -> np.ndarray:
    """Синтетические направления, не созданные предобученной моделью."""
    return np.array([[0.7, 0.7], [1, 0], [0.9, 0.1], [0, 1]], dtype=np.float64)


@pytest.fixture
def indices(documents: list[Document], embeddings: np.ndarray) -> tuple[BM25Index, VectorIndex]:
    """Оба независимых индекса с одним множеством doc_id и source_id."""
    return (
        BM25Index(documents),
        VectorIndex(
            [d.doc_id for d in documents], embeddings,
            source_ids=["src-a", "src-b", "src-a", "src-c"],
        ),
    )


@pytest.fixture
def hybrid(indices: tuple[BM25Index, VectorIndex]) -> HybridSearcher:
    """RRF без весовых предпочтений и с пулом на весь корпус."""
    return HybridSearcher(*indices, candidate_pool_size=4)


def q(x: float = 1.0, y: float = 0.0) -> np.ndarray:
    """Построить 1D numpy embedding, валидный для индекса F3.4."""
    return np.array([x, y], dtype=np.float64)


def test_constructor_retains_exact_indices(indices: tuple[BM25Index, VectorIndex]) -> None:
    """Оркестратор использует уже проверенные BM25 и VectorIndex без дубликатов."""
    obj = HybridSearcher(*indices)
    assert obj.lexical_index is indices[0]
    assert obj.vector_index is indices[1]
    assert (obj.rrf_k, obj.candidate_pool_size) == (60, 20)
    assert (obj.lexical_weight, obj.vector_weight) == (1.0, 1.0)


def test_includes_lexical_and_dense_only_candidates(hybrid: HybridSearcher) -> None:
    """Объединение двух пулов — union, а не intersection кандидатов."""
    hits = hybrid.search("postgresql медленные запросы", q(), top_k=4)
    assert {h.doc_id for h in hits} == {"lexical", "semantic", "shared", "unrelated"}
    by_id = {h.doc_id: h for h in hits}
    assert by_id["semantic"].bm25_rank is None
    assert by_id["semantic"].vector_rank == 1
    assert by_id["lexical"].bm25_rank == 1
    assert by_id["lexical"].vector_rank == 3
    assert by_id["unrelated"].bm25_rank is None


def test_dedupe_by_doc_id_but_not_source_id(hybrid: HybridSearcher) -> None:
    """Ранг объединяем по doc_id, а два фрагмента src-a сохраняем отдельно."""
    hits = hybrid.search("postgresql медленные запросы", q(), top_k=4)
    assert len({h.doc_id for h in hits}) == len(hits) == 4
    assert sum(h.source_id == "src-a" for h in hits) == 2
    assert {h.doc_id for h in hits if h.source_id == "src-a"} == {"lexical", "shared"}


def test_rrf_manual_numeric_reference(hybrid: HybridSearcher) -> None:
    """Эталон: 1/(60+1)+1/(60+3) для lexical, 2/(60+2) для shared."""
    by_id = {hit.doc_id: hit for hit in hybrid.search("postgresql медленные запросы", q(), top_k=4)}
    assert by_id["lexical"].rrf_score == pytest.approx(1 / 61 + 1 / 63, abs=1e-15)
    assert by_id["shared"].rrf_score == pytest.approx(1 / 62 + 1 / 62, abs=1e-15)
    assert by_id["semantic"].rrf_score == pytest.approx(1 / 61, abs=1e-15)
    assert by_id["unrelated"].rrf_score == pytest.approx(1 / 64, abs=1e-15)


def test_ordering_of_hand_computed_rrf(hybrid: HybridSearcher) -> None:
    """RRF даёт lexical чуть выше shared, затем semantic и unrelated."""
    hits = hybrid.search("postgresql медленные запросы", q(), top_k=4)
    assert [h.doc_id for h in hits] == ["lexical", "shared", "semantic", "unrelated"]
    assert [h.rank for h in hits] == [1, 2, 3, 4]
    assert all(isinstance(h, HybridHit) for h in hits)
    assert [h.rrf_score for h in hits] == sorted([h.rrf_score for h in hits], reverse=True)


def test_original_scores_are_kept_but_not_combined(hybrid: HybridSearcher) -> None:
    """Исходные метрики доступны для диагностики, RRF использует лишь rank."""
    hits = {h.doc_id: h for h in hybrid.search("postgresql медленные запросы", q(), top_k=4)}
    lexical = hybrid.lexical_index.search("postgresql медленные запросы", top_k=4)
    vector = hybrid.vector_index.search(q(), top_k=4)
    assert hits["lexical"].bm25_score == pytest.approx(lexical[0].score)
    assert hits["semantic"].cosine_score == pytest.approx(vector[0].score)
    assert hits["semantic"].bm25_score is None
    assert hits["unrelated"].bm25_score is None
    assert hits["unrelated"].cosine_score == pytest.approx(0.0)


def test_weighted_rrf_uses_weight_in_numerator(indices: tuple[BM25Index, VectorIndex]) -> None:
    """При удвоенном dense-весе shared получает 1/62 + 2/62."""
    hyb = HybridSearcher(*indices, candidate_pool_size=4, vector_weight=2.0)
    hits = {h.doc_id: h for h in hyb.search("postgresql медленные запросы", q(), top_k=4)}
    assert hits["shared"].rrf_score == pytest.approx(3 / 62)
    assert hits["lexical"].rrf_score == pytest.approx(1 / 61 + 2 / 63)
    assert hits["shared"].rank == 1


def test_lexical_weight_can_change_order(indices: tuple[BM25Index, VectorIndex]) -> None:
    """Увеличение lexical_weight усиливает BM25 ранг без сложения BM25 score."""
    obj = HybridSearcher(*indices, candidate_pool_size=4, lexical_weight=3)
    hits = obj.search("postgresql медленные запросы", q(), top_k=4)
    assert hits[0].doc_id == "lexical"
    assert hits[0].rrf_score == pytest.approx(3 / 61 + 1 / 63)


@pytest.mark.parametrize("rrf_k", [1, 2, 10, 60, 120, 1000])
def test_rrf_k_manual(indices: tuple[BM25Index, VectorIndex], rrf_k: int) -> None:
    """Константа RRF сдвигает знаменатель, но не меняет исходные ранги."""
    hyb = HybridSearcher(*indices, rrf_k=rrf_k, candidate_pool_size=4)
    hits = {h.doc_id: h for h in hyb.search("postgresql медленные запросы", q(), top_k=4)}
    assert hits["shared"].rrf_score == pytest.approx(2 / (rrf_k + 2))
    assert hits["semantic"].rrf_score == pytest.approx(1 / (rrf_k + 1))


@pytest.mark.parametrize("top_k", [1, 2, 3, 4])
def test_top_k_is_prefix(hybrid: HybridSearcher, top_k: int) -> None:
    """Фиксированный candidate pool обеспечивает стабильный префикс выдачи."""
    full = hybrid.search("postgresql медленные запросы", q(), top_k=4)
    assert hybrid.search("postgresql медленные запросы", q(), top_k=top_k) == full[:top_k]


def test_candidate_pool_changes_available_union(indices: tuple[BM25Index, VectorIndex]) -> None:
    """При pool=1 видим двух лидеров; pool=2 позволяет общему shared победить."""
    small = HybridSearcher(*indices, candidate_pool_size=1)
    medium = HybridSearcher(*indices, candidate_pool_size=2)
    hits_small = small.search("postgresql медленные запросы", q(), top_k=1)
    hits_medium = medium.search("postgresql медленные запросы", q(), top_k=2)
    assert hits_small[0].doc_id == "lexical"  # ничья, doc_id ASC
    assert hits_medium[0].doc_id == "shared"
    assert hits_medium[0].bm25_rank == hits_medium[0].vector_rank == 2


def test_pool_larger_than_corpus(hybrid: HybridSearcher, indices: tuple[BM25Index, VectorIndex]) -> None:
    """Глубина 100 не создаёт несуществующие документы."""
    big = HybridSearcher(*indices, candidate_pool_size=100)
    hits = big.search("postgresql медленные запросы", q(), top_k=100)
    assert len(hits) == 4
    assert [h.doc_id for h in hits] == [h.doc_id for h in hybrid.search("postgresql медленные запросы", q(), top_k=4)]


def test_stable_source_id_and_rank(hybrid: HybridSearcher) -> None:
    """Сохраняются provenance и согласованная нумерация итоговых позиций."""
    for hit in hybrid.search("postgresql медленные запросы", q(), top_k=4):
        assert hit.source_id == hybrid.vector_index.get_source_id(hit.doc_id)
        assert hit.rank >= 1


def test_empty_text_still_allows_dense_candidates(hybrid: HybridSearcher) -> None:
    """Пустой BM25 запрос не препятствует полностью векторной выдаче."""
    hits = hybrid.search("", q(), top_k=4)
    assert [h.doc_id for h in hits] == ["semantic", "shared", "lexical", "unrelated"]
    assert all(h.bm25_rank is None for h in hits)
    assert hits[0].rrf_score == pytest.approx(1 / 61)


def test_unknown_terms_dense_fallback(hybrid: HybridSearcher) -> None:
    """Отсутствие лексического overlap не означает пустой hybrid-результат."""
    hits = hybrid.search("термин_которого_нет", q(), top_k=3)
    assert len(hits) == 3
    assert [h.doc_id for h in hits] == ["semantic", "shared", "lexical"]


def test_vector_min_score_keeps_only_selected_dense_hits(hybrid: HybridSearcher) -> None:
    """Порог dense применяет прежний VectorIndex, не удаляя BM25 кандидатов."""
    hits = hybrid.search("postgresql медленные запросы", q(), top_k=4, vector_min_score=1.0)
    ranks = {hit.doc_id: hit for hit in hits}
    assert ranks["semantic"].vector_rank == 1
    assert ranks["lexical"].vector_rank is None
    assert ranks["shared"].vector_rank is None
    assert ranks["lexical"].bm25_rank == 1
    assert "unrelated" not in ranks


def test_vector_without_candidates_preserves_bm25(indices: tuple[BM25Index, VectorIndex]) -> None:
    """Когда cosine threshold исключил dense-пул, лексический список остаётся."""
    hybrid = HybridSearcher(*indices, candidate_pool_size=4)
    hits = hybrid.search("postgresql", q(1, .01), top_k=3, vector_min_score=1.0)
    assert {h.doc_id for h in hits} == {"lexical", "shared"}
    assert all(h.vector_rank is None for h in hits)


def test_empty_both_indices() -> None:
    """Пустой корпус возвращает [], не NaN и не ошибку деления на ноль."""
    lexical = BM25Index([])
    vector = VectorIndex([], np.empty((0, 2), dtype=np.float64), source_ids=[])
    hyb = HybridSearcher(lexical, vector)
    assert hyb.search("anything", q()) == []


def test_sparse_empty_document_remains_dense() -> None:
    """Документ без слов индексируется как вектор и не теряется при fusion."""
    lexical = BM25Index([Document("empty", "!!!"), Document("text", "database")])
    vector = VectorIndex(["empty", "text"], np.eye(2), source_ids=["a", "b"])
    hyb = HybridSearcher(lexical, vector)
    ids = [h.doc_id for h in hyb.search("database", q(), top_k=2)]
    assert set(ids) == {"empty", "text"}


def test_identical_scores_tie_break_by_id() -> None:
    """При одинаковых рангах RRF сравниваем doc_id, а не порядок вставки."""
    docs = [Document("z", "cat"), Document("a", "dog")]
    vector = VectorIndex(["z", "a"], np.array([[1., 0.], [0., 1.]]), source_ids=["s", "s"])
    obj = HybridSearcher(BM25Index(docs), vector, candidate_pool_size=1)
    # BM25 отдаёт z (cat); dense запрос [0,1] отдаёт a (dog).
    hits = obj.search("cat", q(0, 1), top_k=1)
    assert hits[0].doc_id == "a"  # RRF равен, a < z


def test_index_input_order_permutation_no_effect(documents: list[Document], embeddings: np.ndarray) -> None:
    """Перестановка документов вместе с embeddings/source_ids не меняет выдачу."""
    sources = ["src-a", "src-b", "src-a", "src-c"]
    a = HybridSearcher(
        BM25Index(documents), VectorIndex([d.doc_id for d in documents], embeddings, source_ids=sources),
        candidate_pool_size=4,
    )
    perm = [2, 0, 3, 1]
    docs = [documents[i] for i in perm]
    b = HybridSearcher(
        BM25Index(docs), VectorIndex([d.doc_id for d in docs], embeddings[perm], source_ids=[sources[i] for i in perm]),
        candidate_pool_size=4,
    )
    assert a.search("postgresql медленные запросы", q(), top_k=4) == b.search("postgresql медленные запросы", q(), top_k=4)


def test_repeat_calls_deterministic(hybrid: HybridSearcher) -> None:
    """Повторные обращения не изменяют внутренние индексы или tie-break."""
    before = hybrid.search("postgresql", q(), top_k=4)
    for _ in range(10):
        assert hybrid.search("postgresql", q(), top_k=4) == before


def test_vector_query_scale_invariant(hybrid: HybridSearcher) -> None:
    """L2-нормировка в F3.4 защищает ranking при масштабировании query."""
    assert hybrid.search("postgresql", q(1, 2), top_k=4) == hybrid.search("postgresql", q(10, 20), top_k=4)


def test_bm25_query_casefold_inherited(hybrid: HybridSearcher) -> None:
    """Гибридный поиск не заменяет NFKC/casefold токенизатор F3.3."""
    assert hybrid.search("PostgreSQL", q(), top_k=4) == hybrid.search("postgresql", q(), top_k=4)


def test_scores_do_not_need_common_scale(hybrid: HybridSearcher) -> None:
    """RRF зависит только от рангов, а BM25 и cosine имеют разные шкалы."""
    hits = hybrid.search("postgresql", q(), top_k=4)
    assert all(0 < h.rrf_score < 1 for h in hits)
    assert all(h.rrf_score < (2 / (60 + 1)) + 1e-14 for h in hits)


def test_hybrid_hits_immutable(hybrid: HybridSearcher) -> None:
    """Выданные результаты нельзя изменить после ранжирования."""
    h = hybrid.search("postgresql", q(), top_k=1)[0]
    with pytest.raises(FrozenInstanceError):
        h.rrf_score = 99  # type: ignore[misc]


@pytest.mark.parametrize("invalid", [0, -1, True, False, 1.5, "60", None, np.int64(60), [], {}])
def test_invalid_rrf_k(indices: tuple[BM25Index, VectorIndex], invalid: object) -> None:
    """Параметр сглаживания должен быть положительным Python int, не bool."""
    with pytest.raises(ValueError):
        HybridSearcher(*indices, rrf_k=invalid)  # type: ignore[arg-type]


@pytest.mark.parametrize("invalid", [0, -1, True, False, 2.1, "5", None, np.int64(2)])
def test_invalid_candidate_pool(indices: tuple[BM25Index, VectorIndex], invalid: object) -> None:
    """Пул задаёт число кандидатов из каждого поиска, только Python int > 0."""
    with pytest.raises(ValueError):
        HybridSearcher(*indices, candidate_pool_size=invalid)  # type: ignore[arg-type]


@pytest.mark.parametrize("invalid,expected", [
    (0, ValueError), (-1, ValueError), (-0.1, ValueError),
    (float("inf"), ValueError), (float("-inf"), ValueError), (float("nan"), ValueError),
    (True, TypeError), (False, TypeError), ("1", TypeError), (None, TypeError),
    ([], TypeError), (object(), TypeError),
])
@pytest.mark.parametrize("name", ["lexical_weight", "vector_weight"])
def test_invalid_weights(indices: tuple[BM25Index, VectorIndex], invalid: object,
                         expected: type[Exception], name: str) -> None:
    """Веса должны быть строго положительными конечными вещественными числами."""
    with pytest.raises(expected):
        HybridSearcher(*indices, **{name: invalid})


@pytest.mark.parametrize("invalid", [0, -1, True, False, 1.5, "2", None, np.int64(2), 5])
def test_invalid_result_top_k(hybrid: HybridSearcher, invalid: object) -> None:
    """Нельзя запрашивать top_k вне интервала от 1 до глубины pool=4."""
    with pytest.raises(ValueError):
        hybrid.search("postgresql", q(), top_k=invalid)  # type: ignore[arg-type]


@pytest.mark.parametrize("invalid", [None, 7, [], np.array([1]), {}])
def test_invalid_query_text(hybrid: HybridSearcher, invalid: object) -> None:
    """Лексический запрос обязан оставаться строкой, не преобразуем объект в str."""
    with pytest.raises(TypeError):
        hybrid.search(invalid, q())  # type: ignore[arg-type]


@pytest.mark.parametrize("invalid", [
    np.array([0., 0.]),
    np.array([1., 2., 3.]),
    np.array([[1., 0.]]),
    np.array([1., float("nan")]),
    np.array([1., float("inf")]),
    np.array([1, 0], dtype=np.int64),
])
def test_bad_vector_query_is_rejected_by_f34(hybrid: HybridSearcher, invalid: np.ndarray) -> None:
    """Нельзя обходить проверки размерности/finite/dtype из F3.4."""
    with pytest.raises((ValueError, TypeError)):
        hybrid.search("postgresql", invalid)


@pytest.mark.parametrize("value", [-1.1, 1.01, float("nan"), "0.5", True])
def test_vector_threshold_validation_delegated(hybrid: HybridSearcher, value: object) -> None:
    """Недопустимый min_score должен быть отвергнут прежним VectorIndex."""
    with pytest.raises((ValueError, TypeError)):
        hybrid.search("postgresql", q(), vector_min_score=value)  # type: ignore[arg-type]


def test_index_set_mismatch_same_length_rejected(documents: list[Document], embeddings: np.ndarray) -> None:
    """Одинаковое N не доказывает равенство doc_id, подмена запрещена."""
    lexical = BM25Index(documents)
    vector = VectorIndex(["lexical", "semantic", "shared", "another"], embeddings, source_ids=["x"] * 4)
    with pytest.raises(ValueError, match="doc_id"):
        HybridSearcher(lexical, vector)


def test_index_set_mismatch_different_length_rejected(documents: list[Document]) -> None:
    """Утерянный документ векторного индекса нельзя молча игнорировать."""
    lexical = BM25Index(documents)
    vector = VectorIndex(["lexical"], np.array([[1., 0.]]), source_ids=["s"])
    with pytest.raises(ValueError, match="doc_id"):
        HybridSearcher(lexical, vector)


def test_incorrect_index_types_rejected(indices: tuple[BM25Index, VectorIndex]) -> None:
    """Гибридный компонент принимает только интерфейсы предыдущих этапов."""
    with pytest.raises(TypeError):
        HybridSearcher(None, indices[1])  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        HybridSearcher(indices[0], None)  # type: ignore[arg-type]


def test_no_query_mutation(hybrid: HybridSearcher) -> None:
    """Гибридный поиск не редактирует числовой вектор запроса по ссылке."""
    query = q(2, 1)
    saved = query.copy()
    hybrid.search("postgresql", query, top_k=4)
    assert np.array_equal(query, saved)


def test_no_corpus_mutation(documents: list[Document], embeddings: np.ndarray) -> None:
    """Создание и использование гибридного объекта не изменяет входные данные."""
    before_docs = list(documents)
    before_vecs = embeddings.copy()
    lex = BM25Index(documents)
    dense = VectorIndex([d.doc_id for d in documents], embeddings, source_ids=["a", "b", "a", "c"])
    obj = HybridSearcher(lex, dense, candidate_pool_size=4)
    obj.search("postgresql", q(), top_k=4)
    assert documents == before_docs
    assert np.array_equal(embeddings, before_vecs)


def test_no_hidden_training_required(hybrid: HybridSearcher) -> None:
    """Retrieval возвращает документы без параметров fit/train и разметки."""
    assert not hasattr(hybrid, "fit")
    assert not hasattr(hybrid, "train")
    assert hybrid.search("postgresql", q(), top_k=1)[0].doc_id in hybrid.vector_index.doc_ids


def test_cli_smoke_windows_safe_utf8() -> None:
    """CLI принудительно выводит UTF-8, обходя cp1251 и cp866 Windows.

    Без PYTHONIOENCODING дочерний Python выбирает системную консольную
    кодировку, а тест ошибочно пытается декодировать её как UTF-8 (F3.3).
    """
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "hybrid_search_demo.py")],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="strict",
        env=env,
        timeout=20,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "smoke: PASS" in result.stdout
    assert "source_id=" in result.stdout
    assert "bm25_rank=" in result.stdout
    assert "vector_rank=" in result.stdout
    assert "model=" not in result.stdout


def test_one_query_uses_fixed_ranks_not_relevance_label(hybrid: HybridSearcher) -> None:
    """RRF не интерпретирует свои оценки как P(relevant | query, doc)."""
    hits = hybrid.search("postgresql медленные запросы", q(), top_k=4)
    assert all(math.isfinite(h.rrf_score) for h in hits)
    assert all(h.rrf_score > 0 for h in hits)
    assert all(h.bm25_rank is not None or h.vector_rank is not None for h in hits)
