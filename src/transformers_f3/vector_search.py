"""F3.4: точный поиск ближайших документов по dense embeddings через NumPy.

Индекс принимает готовые векторы: их происхождение (синтетика или F3.2)
не меняет формулу cosine similarity. Документы связываются с устойчивыми
``doc_id`` и ``source_id`` без хранения исходных текстов и без обращения к сети.
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
from numbers import Real
from typing import Sequence

import numpy as np


@dataclass(frozen=True, slots=True)
class VectorHit:
    """Результат поиска с идентификаторами документа и источника.

    Атрибуты:
        doc_id: Уникальный ID индексированной записи или фрагмента.
        source_id: ID исходного источника; может повторяться у его фрагментов.
        score: Cosine similarity от -1 до 1, НЕ вероятность релевантности.
        rank: Номер в ранжированной выдаче, начиная с единицы.
    """

    doc_id: str
    source_id: str
    score: float
    rank: int


@dataclass(frozen=True, slots=True)
class VectorIndexStats:
    """Размеры корпуса и матрицы для оценки ресурсов поиска.

    Атрибуты:
        n_documents: Число индексированных записей.
        embedding_dim: Число координат у каждого embedding.
        storage_bytes: Размер внутренней нормализованной матрицы float64.
    """

    n_documents: int
    embedding_dim: int
    storage_bytes: int


def _unit_rows(vectors: np.ndarray, *, name: str, ndim: int) -> np.ndarray:
    """Проверить массив и устойчиво L2-нормировать строки без переполнения.

    Аргументы:
        vectors: Вещественный ndarray формы [N, D] или [D] для запроса.
        name: Название аргумента для сообщения об ошибке.
        ndim: Ожидаемое число измерений (1 или 2).
    Результат:
        Новая матрица float64 с ненулевыми векторами единичной длины.

    Масштабирование по максимальному модулю сначала защищает вычисление
    нормы для очень больших и очень малых конечных значений.
    """
    if not isinstance(vectors, np.ndarray):
        raise TypeError(f"{name} должен быть numpy.ndarray")
    if vectors.ndim != ndim:
        raise ValueError(f"{name} должен иметь {ndim} измерения")
    if vectors.dtype not in (np.dtype("float32"), np.dtype("float64")):
        raise TypeError(f"{name} должен иметь dtype float32 или float64")
    if vectors.shape[-1] == 0:
        raise ValueError(f"{name} не может иметь embedding_dim=0")
    if not np.isfinite(vectors).all():
        raise ValueError(f"{name} содержит NaN или Inf")

    matrix = np.array(vectors, dtype=np.float64, copy=True).reshape(-1, vectors.shape[-1])
    if matrix.shape[0] == 0:
        return matrix
    maxima = np.max(np.abs(matrix), axis=1, keepdims=True)
    if np.any(maxima == 0):
        raise ValueError(f"{name} содержит нулевой embedding")
    scaled = matrix / maxima
    norms = np.linalg.norm(scaled, axis=1, keepdims=True)
    return scaled / norms


def _validated_ids(ids: Sequence[str], *, name: str, count: int, unique: bool) -> tuple[str, ...]:
    """Проверить, что список IDs строго соответствует строкам матрицы."""
    if not isinstance(ids, (list, tuple)):
        raise TypeError(f"{name} должен быть list или tuple")
    if len(ids) != count:
        raise ValueError(f"Число {name} не совпадает с числом embeddings")
    if any(not isinstance(value, str) or not value.strip() for value in ids):
        raise ValueError(f"{name} должен содержать непустые строковые идентификаторы")
    if unique and len(set(ids)) != count:
        raise ValueError(f"{name} содержит повторяющийся ID")
    return tuple(ids)


class VectorIndex:
    """Точный индекс cosine similarity без approximate nearest neighbors.

    Аргументы:
        doc_ids: Уникальные устойчивые ID документов или фрагментов.
        embeddings: Матрица float32/float64 [N, D]; например результат
            ``PretrainedTextEncoder.encode(texts).numpy()`` из F3.2.
        source_ids: ID источников в том же порядке; повторения допустимы.

    Порядок ``doc_ids``, ``source_ids`` и строк ``embeddings`` должен совпадать.
    Индекс копирует и нормирует векторы один раз; изменение внешнего массива
    не влияет на поиск. Поиск не изменяет индекс и не читает файлов.
    """

    def __init__(
        self,
        doc_ids: Sequence[str],
        embeddings: np.ndarray,
        *,
        source_ids: Sequence[str],
    ) -> None:
        unit = _unit_rows(embeddings, name="embeddings", ndim=2)
        count, dimension = unit.shape
        self._doc_ids = _validated_ids(doc_ids, name="doc_ids", count=count, unique=True)
        self._source_ids = _validated_ids(source_ids, name="source_ids", count=count, unique=False)
        self._id_to_row = {doc_id: row for row, doc_id in enumerate(self._doc_ids)}
        # Помечаем массив read-only, чтобы ссылка внутри индекса не менялась.
        unit.flags.writeable = False
        self._unit_embeddings = unit
        self._stats = VectorIndexStats(count, dimension, int(unit.nbytes))

    @property
    def stats(self) -> VectorIndexStats:
        """Вернуть неизменяемую статистику размеров индекса."""
        return self._stats

    @property
    def doc_ids(self) -> tuple[str, ...]:
        """Вернуть ID в исходном порядке строк матрицы."""
        return self._doc_ids

    def get_source_id(self, doc_id: str) -> str:
        """Получить источник по doc_id; для отсутствующего ID — KeyError."""
        return self._source_ids[self._id_to_row[doc_id]]

    def _query_unit(self, query_embedding: np.ndarray) -> np.ndarray:
        """Нормировать один запрос и проверить совпадение размерностей."""
        query = _unit_rows(query_embedding, name="query_embedding", ndim=1)
        if query.shape[1] != self._stats.embedding_dim:
            raise ValueError("Размерность query_embedding не совпадает с embedding_dim индекса")
        return query[0]

    def score(self, query_embedding: np.ndarray, doc_id: str) -> float:
        """Вычислить точную cosine similarity запроса с одним документом.

        Аргументы:
            query_embedding: Float32/float64 numpy-вектор [D].
            doc_id: ID уже индексированного документа.
        Результат:
            Число из [-1, 1] с учётом погрешности округления.
        """
        if doc_id not in self._id_to_row:
            raise KeyError(doc_id)
        query = self._query_unit(query_embedding)
        value = float(self._unit_embeddings[self._id_to_row[doc_id]] @ query)
        return float(np.clip(value, -1.0, 1.0))

    def search(
        self,
        query_embedding: np.ndarray,
        *,
        top_k: int = 5,
        min_score: float | None = None,
    ) -> list[VectorHit]:
        """Вычислить точные cosine scores всех документов и вернуть top-k.

        Аргументы:
            query_embedding: Ненулевой вектор запроса формы [D].
            top_k: Положительный предел числа результатов.
            min_score: Необязательный порог cosine в диапазоне [-1, 1].
        Результат:
            Список VectorHit, score DESC, при равенстве doc_id ASC.

        В отличие от BM25, даже отрицательные scores могут участвовать в
        выдаче: это не лексические совпадения. Порог — явная фильтрация,
        не решение о ground-truth релевантности.
        """
        if type(top_k) is not int or top_k < 1:
            raise ValueError("top_k должен быть положительным целым числом")
        if min_score is not None:
            if isinstance(min_score, (bool, np.bool_)) or not isinstance(min_score, Real):
                raise TypeError("min_score должен быть вещественным числом")
            if not math.isfinite(float(min_score)) or not -1.0 <= min_score <= 1.0:
                raise ValueError("min_score должен быть конечным числом в [-1, 1]")

        query = self._query_unit(query_embedding)
        if self._stats.n_documents == 0:
            return []
        # Один NumPy matvec вместо Python-цикла по документам: [N,D]@[D]→[N].
        scores = np.clip(self._unit_embeddings @ query, -1.0, 1.0)
        # heapq.nsmallest хранит O(k) кандидатов; ID разрешает точные ничьи.
        candidates = (
            (index, float(value))
            for index, value in enumerate(scores)
            if min_score is None or value >= min_score
        )
        winners = heapq.nsmallest(
            top_k, candidates, key=lambda item: (-item[1], self._doc_ids[item[0]])
        )
        return [
            VectorHit(
                doc_id=self._doc_ids[row],
                source_id=self._source_ids[row],
                score=score,
                rank=rank,
            )
            for rank, (row, score) in enumerate(winners, start=1)
        ]
