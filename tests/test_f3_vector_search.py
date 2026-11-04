"""F3.4: проверка exact cosine retrieval, provenance и граничных случаев.

Только синтетические числовые векторы и локальный CLI без pretrained-весов.
Математические эталоны задаются независимо от самой реализации индекса.
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from transformers_f3.vector_search import (  # noqa: E402
    VectorHit, VectorIndex, VectorIndexStats,
)


@pytest.fixture
def index() -> VectorIndex:
    """Матрица: совпадение, частичное совпадение, ортогональность, антипод."""
    return VectorIndex(
        ["same", "near", "orthogonal", "opposite"],
        np.array([[1, 0], [3, 4], [0, 1], [-1, 0]], dtype=np.float32),
        source_ids=["src-a", "src-a", "src-b", "src-c"],
    )


def q(*coords: float, dtype: np.dtype | type = np.float64) -> np.ndarray:
    """Сформировать заведомо новый одномерный числовой embedding."""
    return np.array(coords, dtype=dtype)


def test_stats_exact_size(index: VectorIndex) -> None:
    """Форма корпуса [4,2] и float64-хранилище определяют объём памяти."""
    assert index.stats == VectorIndexStats(4, 2, 4 * 2 * 8)
    assert index.doc_ids == ("same", "near", "orthogonal", "opposite")


def test_cosine_hand_calculation(index: VectorIndex) -> None:
    """cos([1,0],[3,4]) = 3/5, cos антипода = -1, ортогонального = 0."""
    assert index.score(q(1, 0), "same") == pytest.approx(1.0)
    assert index.score(q(1, 0), "near") == pytest.approx(3.0 / 5.0)
    assert index.score(q(1, 0), "orthogonal") == pytest.approx(0.0)
    assert index.score(q(1, 0), "opposite") == pytest.approx(-1.0)


def test_cosine_manual_nonaxis() -> None:
    """Независимый эталон: [1,2]·[2,1]/(√5√5) = 4/5."""
    ind = VectorIndex(["d"], np.array([[1., 2.]]), source_ids=["s"])
    assert ind.score(q(2, 1), "d") == pytest.approx(0.8)


def test_rank_and_source_id(index: VectorIndex) -> None:
    """Выдача строго ранжирована и сохраняет provenance каждого фрагмента."""
    hits = index.search(q(1, 0), top_k=4)
    assert [(hit.doc_id, hit.source_id, hit.rank) for hit in hits] == [
        ("same", "src-a", 1), ("near", "src-a", 2),
        ("orthogonal", "src-b", 3), ("opposite", "src-c", 4),
    ]
    assert [hit.score for hit in hits] == pytest.approx([1, 0.6, 0, -1])
    assert all(isinstance(hit, VectorHit) for hit in hits)


def test_docs_with_common_source_not_collapsed(index: VectorIndex) -> None:
    """Фрагменты одного источника различаются по doc_id, но не теряют source_id."""
    assert index.get_source_id("same") == index.get_source_id("near") == "src-a"
    assert index.get_source_id("opposite") == "src-c"


def test_unknown_doc_id(index: VectorIndex) -> None:
    """Неизвестный ID не должен молча соответствовать другой строке матрицы."""
    with pytest.raises(KeyError):
        index.get_source_id("missing")
    with pytest.raises(KeyError):
        index.score(q(1, 0), "missing")


@pytest.mark.parametrize("top_k,expected", [(1, ["same"]), (2, ["same", "near"]),
                                            (3, ["same", "near", "orthogonal"]),
                                            (4, ["same", "near", "orthogonal", "opposite"]),
                                            (20, ["same", "near", "orthogonal", "opposite"])])
def test_top_k(index: VectorIndex, top_k: int, expected: list[str]) -> None:
    """top_k ограничивает результат, но не меняет формулу cosine score."""
    assert [hit.doc_id for hit in index.search(q(1, 0), top_k=top_k)] == expected


@pytest.mark.parametrize("threshold,expected", [
    (1.0, ["same"]),
    (0.7, ["same"]),
    (0.59, ["same", "near"]),
    (0.0, ["same", "near", "orthogonal"]),
    (-0.9, ["same", "near", "orthogonal"]),
    (-1.0, ["same", "near", "orthogonal", "opposite"]),
])
def test_min_score(index: VectorIndex, threshold: float, expected: list[str]) -> None:
    """Фильтр min_score применяется после вычисления cosine, порог включительный."""
    assert [hit.doc_id for hit in index.search(q(1, 0), top_k=10, min_score=threshold)] == expected


@pytest.mark.parametrize("invalid", [0, -1, 1.0, "3", None, True, False, [], np.int64(2)])
def test_invalid_top_k(index: VectorIndex, invalid: object) -> None:
    """top_k обязан быть положительным Python int; bool не считается количеством."""
    with pytest.raises(ValueError):
        index.search(q(1, 0), top_k=invalid)  # type: ignore[arg-type]


@pytest.mark.parametrize("invalid,error", [
    ("0.5", TypeError), (True, TypeError), (False, TypeError),
    ([], TypeError), (object(), TypeError),
    (float("nan"), ValueError), (float("inf"), ValueError),
    (float("-inf"), ValueError), (1.1, ValueError), (-1.1, ValueError),
])
def test_invalid_min_score(index: VectorIndex, invalid: object, error: type[Exception]) -> None:
    """Некорректные пороги нельзя неявно интерпретировать как float."""
    with pytest.raises(error):
        index.search(q(1, 0), min_score=invalid)  # type: ignore[arg-type]


def test_ties_ordered_by_doc_id() -> None:
    """При одинаковом score сравниваем doc_id, а не порядок вставки."""
    ind = VectorIndex(
        ["z", "a", "m"],
        np.array([[1., 0.], [2., 0.], [3., 0.]]),
        source_ids=["s1", "s2", "s3"],
    )
    assert [hit.doc_id for hit in ind.search(q(1, 0))] == ["a", "m", "z"]


def test_permutation_invariance() -> None:
    """Перестановка строк вместе с ID не меняет ранжирование top-k."""
    arr = np.array([[1., 0.], [0., 1.], [0.5, 0.5]])
    ids = ["a", "b", "c"]
    sources = ["s-a", "s-b", "s-c"]
    a = VectorIndex(ids, arr, source_ids=sources)
    p = [2, 0, 1]
    b = VectorIndex([ids[i] for i in p], arr[p], source_ids=[sources[i] for i in p])
    assert a.search(q(1, 0), top_k=3) == b.search(q(1, 0), top_k=3)


def test_input_mutation_cannot_change_index() -> None:
    """Оригинальный массив копируется: поздняя запись не меняет embedding."""
    arr = np.array([[1., 0.], [0., 1.]], dtype=np.float64)
    ind = VectorIndex(["a", "b"], arr, source_ids=["src", "src"])
    before = ind.search(q(1, 0))
    arr[:] = [[0., 1.], [1., 0.]]
    assert ind.search(q(1, 0)) == before


def test_float32_float64_agree() -> None:
    """Выбор исходной точности сохраняет математическую структуру рангов."""
    data = [[2., 1.], [-2., 1.], [1., 2.]]
    ids = ["a", "b", "c"]
    kwargs = {"source_ids": ids}
    a = VectorIndex(ids, np.array(data, dtype=np.float32), **kwargs)
    b = VectorIndex(ids, np.array(data, dtype=np.float64), **kwargs)
    assert [h.doc_id for h in a.search(q(1, 0))] == [h.doc_id for h in b.search(q(1, 0))]
    assert [h.score for h in a.search(q(1, 0))] == pytest.approx([h.score for h in b.search(q(1, 0))], abs=1e-7)


def test_query_scaling_invariance(index: VectorIndex) -> None:
    """Умножение на положительную константу не меняет cosine similarity."""
    assert index.search(q(1, 2)) == index.search(q(50, 100))


def test_document_scaling_invariance() -> None:
    """Cosine сравнивает направления, а не длину embeddings."""
    ind = VectorIndex(["a", "b"], np.array([[1., 2.], [4., 8.]]), source_ids=["s", "s"])
    assert ind.score(q(1, 2), "a") == pytest.approx(1)
    assert ind.score(q(1, 2), "b") == pytest.approx(1)


def test_negative_query_reverses_alignment(index: VectorIndex) -> None:
    """Противоположный запрос должен переставить местами антиподы."""
    assert index.search(q(-1, 0), top_k=1)[0].doc_id == "opposite"


def test_large_magnitudes_no_overflow() -> None:
    """Стабильная нормировка исключает переполнение норм конечных float64."""
    huge = np.array([[1e300, 0.], [0., 1e300]])
    ind = VectorIndex(["a", "b"], huge, source_ids=["s", "t"])
    assert ind.score(q(1e300, 0), "a") == pytest.approx(1)


def test_tiny_magnitudes_no_underflow() -> None:
    """Векторы ниже обычного диапазона квадрата нормы сохраняют направление."""
    tiny = np.array([[1e-300, 0.], [0., 1e-300]])
    ind = VectorIndex(["a", "b"], tiny, source_ids=["s", "t"])
    assert ind.score(q(1e-300, 0), "a") == pytest.approx(1)


@pytest.mark.parametrize("array,error", [
    ([[1., 2.]], TypeError),
    (np.array([1., 2.]), ValueError),
    (np.ones((1, 2, 3)), ValueError),
    (np.ones((2, 0)), ValueError),
    (np.array([[1, 2]], dtype=np.int64), TypeError),
    (np.array([[True, False]]), TypeError),
    (np.array([[1., 2.]], dtype=np.float16), TypeError),
    (np.array([[1., np.nan]]), ValueError),
    (np.array([[1., np.inf]]), ValueError),
    (np.array([[0., 0.]]), ValueError),
])
def test_invalid_corpus_vectors(array: object, error: type[Exception]) -> None:
    """Форма, dtype, конечность и ненулевость — обязательные инварианты."""
    with pytest.raises(error):
        VectorIndex(["a"], array, source_ids=["s"])  # type: ignore[arg-type]


@pytest.mark.parametrize("invalid,error", [
    (["", "b"], ValueError), ([" ", "b"], ValueError),
    (["a", "a"], ValueError), (["a"], ValueError),
    (["a", 2], ValueError), (None, TypeError),
    (("a", "b", "c"), ValueError), ("ab", TypeError),
])
def test_invalid_doc_ids(invalid: object, error: type[Exception]) -> None:
    """У каждого embedding должен быть один уникальный устойчивый ID."""
    with pytest.raises(error):
        VectorIndex(invalid, np.eye(2), source_ids=["x", "y"])  # type: ignore[arg-type]


@pytest.mark.parametrize("invalid,error", [
    ([""], ValueError), ([" ", "src"], ValueError),
    (["src"], ValueError), (None, TypeError),
    ("xy", TypeError), (["src", 2], ValueError),
    (("one", "two", "three"), ValueError),
])
def test_invalid_source_ids(invalid: object, error: type[Exception]) -> None:
    """Каждая запись обязана иметь непустой ID происхождения."""
    with pytest.raises(error):
        VectorIndex(["a", "b"], np.eye(2), source_ids=invalid)  # type: ignore[arg-type]


@pytest.mark.parametrize("invalid,error", [
    (None, TypeError), ([1., 0.], TypeError),
    (np.array([[1., 0.]]), ValueError),
    (np.array([1., 0., 0.]), ValueError),
    (np.array([1, 0], dtype=np.int64), TypeError),
    (np.array([1., 0.], dtype=np.float16), TypeError),
    (np.array([np.nan, 1.]), ValueError),
    (np.array([1., np.inf]), ValueError),
    (np.array([0., 0.]), ValueError),
])
def test_invalid_queries(index: VectorIndex, invalid: object, error: type[Exception]) -> None:
    """Запрос должен быть одним валидным embedding, а не batch или null."""
    with pytest.raises(error):
        index.search(invalid)  # type: ignore[arg-type]


def test_empty_index_is_valid() -> None:
    """Нулевой корпус при известном D не даёт выдачу и сохраняет shape."""
    empty = VectorIndex([], np.empty((0, 3), dtype=np.float32), source_ids=[])
    assert empty.stats == VectorIndexStats(0, 3, 0)
    assert empty.search(q(1, 0, 0)) == []
    assert empty.doc_ids == ()


def test_empty_index_rejects_bad_query_shape() -> None:
    """Даже при пустом корпусе неверный D не должен скрываться."""
    empty = VectorIndex([], np.empty((0, 3), dtype=np.float32), source_ids=[])
    with pytest.raises(ValueError):
        empty.search(q(1, 0))


def test_no_threshold_keeps_zero_and_negative(index: VectorIndex) -> None:
    """Dense retrieval отличается от BM25: отрицательные cosine не отбрасываются."""
    assert len(index.search(q(1, 0), top_k=10)) == 4
    assert index.search(q(1, 0), top_k=10)[-1].score == pytest.approx(-1.0)


def test_search_matches_individual_score(index: VectorIndex) -> None:
    """Векторизированное ранжирование совпадает с индивидуальным cosine."""
    hits = index.search(q(1., 1.), top_k=4)
    for hit in hits:
        assert hit.score == pytest.approx(index.score(q(1., 1.), hit.doc_id), abs=1e-14)


def test_determinism_without_training(index: VectorIndex) -> None:
    """Повтор одинакового поискового запроса возвращает те же числа и порядок."""
    baseline = index.search(q(1.5, 2.5), top_k=3)
    for _ in range(5):
        assert index.search(q(1.5, 2.5), top_k=3) == baseline


def test_numpy_encoder_output_contract() -> None:
    """F3.2-контракт [N,D] float32 совместим с индексом без дообучения."""
    # Такие же форма и тип, какие возвращает encoder.encode(...).numpy().
    documents = np.array([[1., 0., 2.], [0., 1., 1.]], dtype=np.float32)
    ind = VectorIndex(["d1", "d2"], documents, source_ids=["s1", "s2"])
    query = np.array([1., 0., 1.], dtype=np.float32)
    assert ind.search(query, top_k=1)[0].doc_id == "d1"


def test_search_does_not_write_any_artifacts(index: VectorIndex, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Проверка не затрагивает data, кеши моделей и файловые источники."""
    monkeypatch.chdir(tmp_path)
    assert index.search(q(1, 0), top_k=2)
    assert list(tmp_path.iterdir()) == []


def test_cli_smoke_windows_utf8() -> None:
    """CLI должен работать на Windows даже с cp1251/cp866 по умолчанию."""
    project = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    result = subprocess.run(
        [sys.executable, str(project / "scripts" / "vector_search_demo.py")],
        cwd=project, capture_output=True, text=True,
        encoding="utf-8", env=env, check=False, timeout=20,
    )
    assert result.returncode == 0, result.stderr
    assert "smoke: PASS" in result.stdout
    assert "source_id=" in result.stdout
    assert "cosine=" in result.stdout


def test_immutable_result_data(index: VectorIndex) -> None:
    """Dataclass результата не может менять rank или ID после ранжирования."""
    from dataclasses import FrozenInstanceError
    result = index.search(q(1, 0), top_k=1)[0]
    with pytest.raises(FrozenInstanceError):
        result.rank = 9  # type: ignore[misc]
