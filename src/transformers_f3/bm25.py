"""F3.3: детерминированный лексический поиск BM25 по инвертированному индексу.

Реализован вариант Okapi BM25 с положительным сглаженным IDF Робертсона.
Индекс строится исключительно из переданных документов и хранится в памяти.
Модели, интернет, размеченная релевантность и сторонние библиотеки не нужны.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import heapq
import math
import re
from typing import Iterable
import unicodedata


# Одна и та же нормализация используется при индексации и обработке запроса.
# \w охватывает латиницу, кириллицу, цифры и подчёркивания; знаки пунктуации
# отделяют токены. Это намеренно простой baseline без stemming/lemmatization.
_TOKEN_PATTERN = re.compile(r"\w+", flags=re.UNICODE)


def tokenize(text: str) -> tuple[str, ...]:
    """Получить лексические токены после NFKC-нормализации и casefold.

    Аргументы:
        text: Исходный документ либо поисковый запрос.
    Результат:
        Кортеж токенов в порядке появления, включая повторения слов.

    Пустая или состоящая из знаков пунктуации строка даёт пустой кортеж.
    """
    if not isinstance(text, str):
        raise TypeError("text должен быть строкой")
    return tuple(_TOKEN_PATTERN.findall(unicodedata.normalize("NFKC", text).casefold()))


@dataclass(frozen=True, slots=True)
class Document:
    """Единица корпуса с устойчивым идентификатором и исходным текстом.

    Аргументы:
        doc_id: Уникальный непустой идентификатор; регистр имеет значение.
        text: Текст документа, который требуется индексировать.
    """

    doc_id: str
    text: str

    def __post_init__(self) -> None:
        """Отсеять неверные документы до построения индекса."""
        if not isinstance(self.doc_id, str) or not self.doc_id.strip():
            raise ValueError("doc_id должен быть непустой строкой")
        if not isinstance(self.text, str):
            raise TypeError("text документа должен быть строкой")


@dataclass(frozen=True, slots=True)
class SearchHit:
    """Результат поиска без вероятностной интерпретации BM25 score.

    Поля:
        doc_id: Идентификатор документа в исходном корпусе.
        score: Ненормированная положительная оценка BM25 (не вероятность).
        rank: Позиция в выдаче, начиная с 1.
    """

    doc_id: str
    score: float
    rank: int


@dataclass(frozen=True, slots=True)
class IndexStats:
    """Статистика корпуса, необходимая для проверки формулы BM25.

    Поля:
        n_documents: Количество индексированных документов, включая пустые.
        total_tokens: Число всех токенов корпуса с повторениями.
        vocabulary_size: Число уникальных токенов.
        avgdl: Средняя длина документа в токенах.
    """

    n_documents: int
    total_tokens: int
    vocabulary_size: int
    avgdl: float


class BM25Index:
    """Неизменяемый после построения inverted index и Okapi BM25 ranker.

    Аргументы:
        documents: Итерация Document; идентификаторы должны быть уникальны.
        k1: Положительный параметр насыщения вклада term frequency.
        b: Степень нормализации по длине документа, от 0 до 1.

    Для неизвестных термов IDF равен 0. Для пустого запроса, корпуса без
    совпадений или top-k без подходящих документов возвращается пустая выдача.
    Query term frequency не учитывается: повторные слова запроса схлопываются.
    """

    def __init__(
        self,
        documents: Iterable[Document],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        for name, value in (("k1", k1), ("b", b)):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} должен быть вещественным числом")
            if not math.isfinite(value):
                raise ValueError(f"{name} должен быть конечным")
        if k1 <= 0:
            raise ValueError("k1 должен быть больше нуля")
        if not 0 <= b <= 1:
            raise ValueError("b должен находиться в диапазоне [0, 1]")

        self.k1 = float(k1)
        self.b = float(b)
        self._documents: dict[str, Document] = {}
        self._lengths: dict[str, int] = {}
        # term -> {doc_id: tf}. В поиске просматриваем только postings
        # совпавших термов вместо полного корпуса документов.
        self._postings: dict[str, dict[str, int]] = {}
        total_tokens = 0
        for document in documents:
            if not isinstance(document, Document):
                raise TypeError("documents должен содержать только Document")
            if document.doc_id in self._documents:
                raise ValueError(f"Повторяющийся doc_id: {document.doc_id!r}")
            tokens = tokenize(document.text)
            frequencies = Counter(tokens)
            self._documents[document.doc_id] = document
            self._lengths[document.doc_id] = len(tokens)
            total_tokens += len(tokens)
            for term, tf in frequencies.items():
                self._postings.setdefault(term, {})[document.doc_id] = tf

        n_documents = len(self._documents)
        self._stats = IndexStats(
            n_documents=n_documents,
            total_tokens=total_tokens,
            vocabulary_size=len(self._postings),
            avgdl=total_tokens / n_documents if n_documents else 0.0,
        )
        # IDF зависит только от построенного корпуса: вычисляем один раз.
        # Благодаря кешу скоринг не сортирует postings заново для каждого tf.
        self._idf = {
            term: math.log1p((n_documents - len(posting) + 0.5) / (len(posting) + 0.5))
            for term, posting in self._postings.items()
        }

    @property
    def stats(self) -> IndexStats:
        """Вернуть снимок статистики корпуса для интерпретации BM25."""
        return self._stats

    def get_document(self, doc_id: str) -> Document:
        """Найти исходный документ по идентификатору; иначе KeyError."""
        return self._documents[doc_id]

    def document_length(self, doc_id: str) -> int:
        """Получить длину документа в токенах (включая повторы)."""
        return self._lengths[doc_id]

    @staticmethod
    def _single_term(term: str) -> str:
        """Нормализовать единственный токен для диагностических методов."""
        terms = tokenize(term)
        if len(terms) != 1:
            raise ValueError("Ожидается ровно один лексический токен")
        return terms[0]

    def postings(self, term: str) -> tuple[tuple[str, int], ...]:
        """Вернуть отсортированные пары (doc_id, tf) для конкретного терма.

        Возвращаем копию в кортеже, не позволяя изменить индекс снаружи.
        """
        normalized = self._single_term(term)
        return tuple(sorted(self._postings.get(normalized, {}).items()))

    def document_frequency(self, term: str) -> int:
        """Сколько разных документов содержит данный терм (не сумма tf)."""
        normalized = self._single_term(term)
        return len(self._postings.get(normalized, {}))

    def idf(self, term: str) -> float:
        """Рассчитать сглаженный положительный Robertson IDF.

        IDF(t) = log(1 + (N - df(t) + 0.5) / (df(t) + 0.5)).
        Для отсутствующего в индексе терма договорённо возвращаем 0.
        """
        normalized = self._single_term(term)
        return self._idf.get(normalized, 0.0)

    def _term_contribution(self, term: str, doc_id: str, tf: int) -> float:
        """Вклад одного терма с насыщением частоты и поправкой на длину."""
        if tf <= 0:
            return 0.0
        # Если tf>0, в индексе есть хотя бы один токен, значит avgdl>0.
        length = self._lengths[doc_id]
        norm = self.k1 * (1.0 - self.b + self.b * length / self._stats.avgdl)
        return self._idf[term] * (tf * (self.k1 + 1.0)) / (tf + norm)

    def score(self, query: str, doc_id: str) -> float:
        """Вычислить BM25(query, document) без ранжирования других документов.

        Аргументы:
            query: Лексический поисковый запрос.
            doc_id: Идентификатор документа в текущем индексе.
        Результат:
            Конечная неотрицательная сумма вкладов уникальных термов.
        """
        if doc_id not in self._documents:
            raise KeyError(doc_id)
        terms = sorted(set(tokenize(query)))
        return math.fsum(
            self._term_contribution(term, doc_id, self._postings.get(term, {}).get(doc_id, 0))
            for term in terms
        )

    def search(self, query: str, *, top_k: int = 5) -> list[SearchHit]:
        """Найти top-k совпавших документов по BM25 через inverted index.

        Аргументы:
            query: Текст запроса; токенизируется так же, как документы.
            top_k: Максимальное количество документов (положительное целое).
        Результат:
            SearchHit в порядке score DESC, при равенстве — doc_id ASC.

        Нулевые оценки и документы без пересекающихся термов в выдачу не
        включаем. Параметр top_k не влияет на рассчитанные score.
        """
        if type(top_k) is not int or top_k < 1:
            raise ValueError("top_k должен быть положительным целым числом")
        terms = sorted(set(tokenize(query)))
        if not terms:
            return []
        # Обходим лишь posting lists совпавших термов; IDF уже закеширован
        # при построении индекса и не требует вычисления на каждом документе.
        scores: dict[str, list[float]] = {}
        for term in terms:
            posting = self._postings.get(term)
            if not posting:
                continue
            for doc_id, tf in sorted(posting.items()):
                scores.setdefault(doc_id, []).append(self._term_contribution(term, doc_id, tf))
        # math.fsum даёт воспроизводимую и более устойчивую сумму вкладов.
        ranked = heapq.nsmallest(
            top_k,
            ((doc_id, math.fsum(parts)) for doc_id, parts in scores.items()),
            key=lambda entry: (-entry[1], entry[0]),
        )
        return [SearchHit(doc_id=doc_id, score=value, rank=rank)
                for rank, (doc_id, value) in enumerate(ranked, start=1)]
