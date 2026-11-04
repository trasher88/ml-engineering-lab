"""F3.5: гибридный поиск BM25 + NumPy Vector Search с Reciprocal Rank Fusion.

Компонент объединяет два уже построенных индекса F3.3 и F3.4: не меняет
их скоринг, не запрашивает сеть и не требует обучения модели. Ранги
сливаются по устойчивому doc_id; исходный source_id берётся из VectorIndex.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real

import numpy as np

from .bm25 import BM25Index
from .vector_search import VectorIndex


@dataclass(frozen=True, slots=True)
class HybridHit:
    """Один документ после объединения двух независимых ранжированных списков.

    Атрибуты:
        doc_id: Устойчивый ID записи; используется для объединения выдач.
        source_id: ID исходного источника; у разных фрагментов может совпадать.
        rrf_score: Взвешенная сумма обратных рангов; НЕ вероятность.
        rank: Итоговая позиция, начиная с единицы.
        bm25_rank: Ранг в лексическом пуле или None, если совпадения нет.
        vector_rank: Ранг в векторном пуле или None, если кандидат отсутствует.
        bm25_score: Исходная оценка BM25 либо None; не участвует в RRF.
        cosine_score: Исходная косинусная близость либо None; не участвует в RRF.
    """

    doc_id: str
    source_id: str
    rrf_score: float
    rank: int
    bm25_rank: int | None
    vector_rank: int | None
    bm25_score: float | None
    cosine_score: float | None


def _positive_weight(name: str, value: float) -> float:
    """Проверить положительность и конечность веса отдельного ранжировщика.

    Аргументы:
        name: Имя параметра для диагностического сообщения.
        value: Вес вклада одного из двух списков в формулу RRF.
    Результат:
        Приведённое к float положительное конечное число.
    """
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} должен быть вещественным числом")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} должен быть конечным положительным числом")
    return result


class HybridSearcher:
    """Оркестратор гибридного поиска с проверяемым контрактом двух индексов.

    Аргументы:
        lexical_index: Готовый BM25Index из F3.3.
        vector_index: Готовый VectorIndex из F3.4.
        rrf_k: Положительная константа сглаживания RRF (обычно 60).
        candidate_pool_size: Глубина выборки *из каждого* индекса.
        lexical_weight: Положительный вес вклада BM25 в RRF.
        vector_weight: Положительный вес вклада Vector Search в RRF.

    Оба индекса обязаны содержать строго одинаковый набор doc_id.
    Соответствие текстов и строк матрицы embeddings остаётся обязанностью
    вызывающего кода при построении индексов. source_id не дедуплицируем:
    два фрагмента одного источника являются разными кандидатами.
    """

    def __init__(
        self,
        lexical_index: BM25Index,
        vector_index: VectorIndex,
        *,
        rrf_k: int = 60,
        candidate_pool_size: int = 20,
        lexical_weight: float = 1.0,
        vector_weight: float = 1.0,
    ) -> None:
        if not isinstance(lexical_index, BM25Index):
            raise TypeError("lexical_index должен иметь тип BM25Index")
        if not isinstance(vector_index, VectorIndex):
            raise TypeError("vector_index должен иметь тип VectorIndex")
        if type(rrf_k) is not int or rrf_k <= 0:
            raise ValueError("rrf_k должен быть положительным Python int")
        if type(candidate_pool_size) is not int or candidate_pool_size <= 0:
            raise ValueError("candidate_pool_size должен быть положительным Python int")
        lex_weight = _positive_weight("lexical_weight", lexical_weight)
        vec_weight = _positive_weight("vector_weight", vector_weight)

        # Без проверки состава корпусов одинаковый doc_id мог бы ошибочно
        # обозначать разные записи, а отсутствующий документ — теряться.
        if lexical_index.stats.n_documents != vector_index.stats.n_documents:
            raise ValueError("Индексы должны содержать один набор doc_id")
        for doc_id in vector_index.doc_ids:
            try:
                lexical_index.get_document(doc_id)
            except KeyError as exc:
                raise ValueError("Индексы должны содержать один набор doc_id") from exc

        self.lexical_index = lexical_index
        self.vector_index = vector_index
        self.rrf_k = rrf_k
        self.candidate_pool_size = candidate_pool_size
        self.lexical_weight = lex_weight
        self.vector_weight = vec_weight

    def search(
        self,
        query_text: str,
        query_embedding: np.ndarray,
        *,
        top_k: int = 5,
        vector_min_score: float | None = None,
    ) -> list[HybridHit]:
        """Получить top-k документов после объединения рангов BM25 и Vector.

        Аргументы:
            query_text: Исходный текст запроса для BM25.
            query_embedding: Вектор *того же* запроса формы [D], что и F3.4.
            top_k: Количество результатов, от 1 до candidate_pool_size.
            vector_min_score: Необязательный порог F3.4 для dense-пула.
        Результат:
            Детерминированные HybridHit по rrf_score DESC, затем doc_id ASC.

        Значения BM25 score и cosine не складываем: они имеют разный масштаб.
        RRF использует только позиции в списках: вес/(rrf_k + rank).
        Нулевые результаты BM25 и отсутствующие dense-кандидаты вносят 0.
        """
        if not isinstance(query_text, str):
            raise TypeError("query_text должен быть строкой")
        if type(top_k) is not int or not 1 <= top_k <= self.candidate_pool_size:
            raise ValueError("top_k должен быть в диапазоне [1, candidate_pool_size]")

        lexical_hits = self.lexical_index.search(query_text, top_k=self.candidate_pool_size)
        vector_hits = self.vector_index.search(
            query_embedding, top_k=self.candidate_pool_size, min_score=vector_min_score
        )
        lexical_by_id = {hit.doc_id: hit for hit in lexical_hits}
        vector_by_id = {hit.doc_id: hit for hit in vector_hits}

        # Внешний union по doc_id: один кандидат может присутствовать в двух
        # списках, но обязан появиться в итоговой выдаче не более одного раза.
        merged: list[tuple[str, float]] = []
        for doc_id in sorted(lexical_by_id.keys() | vector_by_id.keys()):
            lexical = lexical_by_id.get(doc_id)
            vector = vector_by_id.get(doc_id)
            parts = (
                self.lexical_weight / (self.rrf_k + lexical.rank) if lexical else 0.0,
                self.vector_weight / (self.rrf_k + vector.rank) if vector else 0.0,
            )
            merged.append((doc_id, math.fsum(parts)))

        # При одинаковой сумме рангов doc_id задаёт независимый от вставки
        # порядок: важно для регрессии и повторяемости экспериментов.
        ranked = sorted(merged, key=lambda item: (-item[1], item[0]))[:top_k]
        result: list[HybridHit] = []
        for rank, (doc_id, fused_score) in enumerate(ranked, start=1):
            lexical = lexical_by_id.get(doc_id)
            vector = vector_by_id.get(doc_id)
            result.append(
                HybridHit(
                    doc_id=doc_id,
                    source_id=self.vector_index.get_source_id(doc_id),
                    rrf_score=fused_score,
                    rank=rank,
                    bm25_rank=lexical.rank if lexical else None,
                    vector_rank=vector.rank if vector else None,
                    bm25_score=lexical.score if lexical else None,
                    cosine_score=vector.score if vector else None,
                )
            )
        return result
