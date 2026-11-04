"""F3.6: замороженная синтетическая выборка и строгие метрики retrieval.

Проверяет SHA-256 содержимого, ID документов и запросов, полноту qrels,
а также определяет macro Recall@K и MRR@K. Модели и интернет не нужны.
Ни поиск, ни подгонка параметров по frozen-разметке не выполняются.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping, Sequence


BENCHMARK_ID = "f3.6-synthetic-retrieval-v1"
# Печатный SHA-256 фиксирует всю спецификацию набора: версии, объёмы,
# контрольные суммы трёх файлов. Изменение корпуса требует новой версии.
FROZEN_MANIFEST_SHA256 = "a584868b8818552815151a85dfde714a762f14d959813e607bb4319cec70756c"
_DATA_FILES = ("documents.jsonl", "queries.jsonl", "qrels.jsonl")
_DOMAINS = frozenset({"databases", "linux", "systems", "programming"})
_QUERY_KINDS = frozenset({"lexical", "paraphrase"})


@dataclass(frozen=True, slots=True)
class BenchmarkDocument:
    """Документ корпуса; source_id группирует два фрагмента одной темы.

    Поля:
        doc_id: Непустой идентификатор *фрагмента* для ранжирования.
        source_id: Идентификатор источника, допускающий повторения.
        domain: Один из четырёх технических доменов F1.
        text: Синтетический технический текст на английском языке.
    """

    doc_id: str
    source_id: str
    domain: str
    text: str


@dataclass(frozen=True, slots=True)
class BenchmarkQuery:
    """Вопрос benchmark: lexical и paraphrase — тип формулировки, не ответ.

    Поля:
        query_id: Устойчивый идентификатор запроса.
        domain: Тематический домен, только для диагностической разбивки.
        kind: lexical (прямой) или paraphrase (перефразированный).
        text: Сам вопрос, передаваемый поисковой системе.
    """

    query_id: str
    domain: str
    kind: str
    text: str


@dataclass(frozen=True, slots=True)
class RelevanceJudgment:
    """Независимая от результатов поиска бинарная оценка query → doc_id.

    relevance равна 1. Отсутствие пары означает нерелевантность в рамках
    конкретно этой синтетической разметки, а не общую оценку полезности.
    """

    query_id: str
    doc_id: str
    relevance: int


@dataclass(frozen=True, slots=True)
class FrozenBenchmark:
    """Неизменяемое представление файлов версии v1 после полного аудита.

    Поля:
        benchmark_id: Зафиксированный идентификатор версии.
        documents: Корпус в фиксированном порядке файловых записей.
        queries: Все 80 вопросов, которые обязаны участвовать в оценке.
        judgments: Бинарная разметка, не полученная из BM25/embeddings.
        manifest_sha256: Контрольная сумма подтверждённого manifest.
    """

    benchmark_id: str
    documents: tuple[BenchmarkDocument, ...]
    queries: tuple[BenchmarkQuery, ...]
    judgments: tuple[RelevanceJudgment, ...]
    manifest_sha256: str

    def relevance_map(self) -> dict[str, frozenset[str]]:
        """Вернуть новую таблицу релевантных doc_id для каждого запроса.

        Результат: mapping query_id → множество ID. Копия не изменяет
        внутреннюю разметку benchmark при обращении извне.
        """
        result: dict[str, set[str]] = {item.query_id: set() for item in self.queries}
        for item in self.judgments:
            result[item.query_id].add(item.doc_id)
        return {key: frozenset(ids) for key, ids in result.items()}


@dataclass(frozen=True, slots=True)
class QueryMetrics:
    """Проверяемые метрики одного вопроса на первых k результатах.

    hits_at_k: Число найденных релевантных документов в top-k.
    recall_at_k: hits_at_k / число всех релевантных документов.
    reciprocal_rank_at_k: 1 / ранг первого релевантного документа или 0.
    """

    query_id: str
    relevant_count: int
    retrieved_at_k: int
    hits_at_k: int
    recall_at_k: float
    reciprocal_rank_at_k: float


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """Макро-усреднение по ВСЕМ запросам benchmark, а не по документам.

    Поля:
        k: Глубина каждого поискового списка.
        n_queries: Количество оценённых запросов.
        recall_at_k: Среднее индивидуальных Recall@K.
        mrr_at_k: Среднее Reciprocal Rank@K.
        per_query: Детальные результаты для последующего error analysis.
    """

    k: int
    n_queries: int
    recall_at_k: float
    mrr_at_k: float
    per_query: tuple[QueryMetrics, ...]


def default_benchmark_dir() -> Path:
    """Получить путь к публичному синтетическому набору в дереве проекта.

    Результат: Путь ``tests/fixtures/f3_06_benchmark_v1``; чтение выполняется
    отдельно в load_frozen_benchmark, а директория может отсутствовать.
    """
    return Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "f3_06_benchmark_v1"


def _canonical_bytes(path: Path) -> bytes:
    """Нормализовать только CRLF → LF для устойчивости Git checkout на Windows.

    Содержимое UTF-8 остаётся неизменным; bare CR и BOM запрещаются. При этом
    одинаковые данные под Windows и Linux имеют одинаковый SHA-256.
    """
    raw = path.read_bytes().replace(b"\r\n", b"\n")
    if b"\r" in raw or raw.startswith(b"\xef\xbb\xbf"):
        raise ValueError(f"Неподдерживаемые переносы строк или BOM: {path.name}")
    raw.decode("utf-8", errors="strict")
    return raw


def _hash(data: bytes) -> str:
    """Вычислить канонический шестнадцатеричный SHA-256."""
    return hashlib.sha256(data).hexdigest()


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """Запретить повторные JSON-ключи, которые обычный json.loads перезаписывает."""
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Дублированный JSON-ключ: {key}")
        result[key] = value
    return result


def _decode_json(data: bytes) -> object:
    """Декодировать UTF-8 JSON без молчаливой потери повторных ключей."""
    return json.loads(data.decode("utf-8"), object_pairs_hook=_unique_object)


def _read_jsonl(data: bytes, *, filename: str, fields: set[str]) -> list[dict[str, object]]:
    """Прочитать строгие JSON Lines без пустых строк и лишних полей.

    Аргументы:
        data: Уже проверенное содержимое файла.
        filename: Имя, используемое в диагностике ошибок.
        fields: Точное множество допустимых JSON-ключей.
    Результат:
        Список записей в порядке их сохранения на диске.
    """
    if not data or not data.endswith(b"\n"):
        raise ValueError(f"{filename}: ожидался непустой JSONL, заканчивающийся LF")
    result: list[dict[str, object]] = []
    for line_number, line in enumerate(data.split(b"\n")[:-1], start=1):
        if not line:
            raise ValueError(f"{filename}:{line_number}: пустая строка")
        record = _decode_json(line)
        if not isinstance(record, dict) or set(record) != fields:
            raise ValueError(f"{filename}:{line_number}: неверная схема записи")
        result.append(record)
    return result


def _text(value: object, name: str) -> str:
    """Обязательная непустая строка без пробельных псевдо-ID."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} должен содержать непустую строку")
    return value


def _validate_benchmark(
    docs: tuple[BenchmarkDocument, ...],
    queries: tuple[BenchmarkQuery, ...],
    judgments: tuple[RelevanceJudgment, ...],
) -> None:
    """Проверить целостность ссылок, разметки и баланса frozen-корпуса.

    Все проверки используют только опубликованные данные: никакой ответ
    модели не может изменить релевантность или исключить неудобный вопрос.
    """
    if (len(docs), len(queries), len(judgments)) != (88, 80, 160):
        raise ValueError("F3.6 v1 требует 88 документов, 80 запросов и 160 qrels")
    doc_map = {doc.doc_id: doc for doc in docs}
    query_map = {query.query_id: query for query in queries}
    if len(doc_map) != len(docs) or len(query_map) != len(queries):
        raise ValueError("Найдены дубли doc_id или query_id")
    if any(doc.domain not in _DOMAINS for doc in docs):
        raise ValueError("Неизвестный домен документа")
    if any(query.domain not in _DOMAINS or query.kind not in _QUERY_KINDS for query in queries):
        raise ValueError("Неизвестный домен или стиль запроса")
    if any(sum(q.domain == domain for q in queries) != 20 for domain in _DOMAINS):
        raise ValueError("Каждый домен должен содержать ровно 20 запросов")
    if any(sum(d.domain == domain for d in docs) != 22 for domain in _DOMAINS):
        raise ValueError("Каждый домен должен содержать ровно 22 документа")
    if (sum(q.kind == "lexical" for q in queries), sum(q.kind == "paraphrase" for q in queries)) != (40, 40):
        raise ValueError("Должно быть по 40 запросов каждого стиля")
    if any(sum(q.domain == domain and q.kind == kind for q in queries) != 10
           for domain in _DOMAINS for kind in _QUERY_KINDS):
        raise ValueError("Для каждого домена нужны 10 запросов каждого стиля")
    if len({doc.text.casefold().strip() for doc in docs}) != len(docs):
        raise ValueError("Дубли текстов документов")
    if len({query.text.casefold().strip() for query in queries}) != len(queries):
        raise ValueError("Дубли текстов запросов")
    if {doc.text.casefold().strip() for doc in docs} & {q.text.casefold().strip() for q in queries}:
        raise ValueError("Текст запроса не должен дословно совпадать с документом")

    qrel_ids: set[tuple[str, str]] = set()
    grouped: dict[str, set[str]] = {qid: set() for qid in query_map}
    for judgment in judgments:
        if judgment.query_id not in query_map or judgment.doc_id not in doc_map:
            raise ValueError("Qrel ссылается на несуществующий запрос или документ")
        if type(judgment.relevance) is not int or judgment.relevance != 1:
            raise ValueError("Frozen v1 содержит только бинарные положительные qrels")
        pair = (judgment.query_id, judgment.doc_id)
        if pair in qrel_ids:
            raise ValueError("Повторяющаяся пара query_id/doc_id")
        qrel_ids.add(pair)
        if query_map[judgment.query_id].domain != doc_map[judgment.doc_id].domain:
            raise ValueError("Релевантный документ имеет другой домен")
        grouped[judgment.query_id].add(judgment.doc_id)
    if any(len(ids) != 2 for ids in grouped.values()):
        raise ValueError("У каждого из 80 запросов должны быть ровно два релевантных doc_id")
    if any(len({doc_map[did].source_id for did in ids}) != 1 for ids in grouped.values()):
        raise ValueError("Два релевантных фрагмента должны принадлежать одному source_id")
    if len({did for ids in grouped.values() for did in ids}) != 80:
        raise ValueError("Ожидается 80 размеченных фрагментов и 8 distractors")


def load_frozen_benchmark(root: str | Path | None = None) -> FrozenBenchmark:
    """Проверить и загрузить именно frozen synthetic benchmark F3.6 v1.

    Аргументы:
        root: Каталог с manifest.json и тремя JSONL. Если None, используется
            стандартный путь внутри tests/fixtures репозитория.
    Результат:
        FrozenBenchmark с 88 документами, 80 запросами и 160 qrels.
    Исключения:
        OSError при недоступных файлах; ValueError при нарушении версии,
        SHA-256, структуры или ссылочной целостности.
    """
    directory = Path(root) if root is not None else default_benchmark_dir()
    manifest_data = _canonical_bytes(directory / "manifest.json")
    actual_manifest_sha = _hash(manifest_data)
    if actual_manifest_sha != FROZEN_MANIFEST_SHA256:
        raise ValueError("manifest F3.6 v1 изменён: требуется новая версия benchmark")
    manifest = _decode_json(manifest_data)
    if not isinstance(manifest, dict) or set(manifest) != {
        "benchmark_id", "schema_version", "freeze_date", "provenance",
        "relevance", "counts", "files_sha256",
    }:
        raise ValueError("Неверная структура manifest.json")
    if (manifest["benchmark_id"] != BENCHMARK_ID or
        type(manifest["schema_version"]) is not int or manifest["schema_version"] != 1 or
        manifest["counts"] != {"documents": 88, "queries": 80, "qrels": 160} or
        manifest["relevance"] != "binary-document-level"):
        raise ValueError("Неверная версия или объём frozen benchmark")
    hashes = manifest["files_sha256"]
    if not isinstance(hashes, dict) or set(hashes) != set(_DATA_FILES):
        raise ValueError("Отсутствуют контрольные суммы frozen-файлов")
    raw: dict[str, bytes] = {}
    for filename in _DATA_FILES:
        data = _canonical_bytes(directory / filename)
        if _hash(data) != hashes[filename]:
            raise ValueError(f"SHA-256 mismatch: {filename}")
        raw[filename] = data

    document_records = _read_jsonl(
        raw["documents.jsonl"], filename="documents.jsonl",
        fields={"doc_id", "source_id", "domain", "text"},
    )
    query_records = _read_jsonl(
        raw["queries.jsonl"], filename="queries.jsonl",
        fields={"query_id", "domain", "kind", "text"},
    )
    qrel_records = _read_jsonl(
        raw["qrels.jsonl"], filename="qrels.jsonl",
        fields={"query_id", "doc_id", "relevance"},
    )
    docs = tuple(BenchmarkDocument(
        _text(rec["doc_id"], "doc_id"), _text(rec["source_id"], "source_id"),
        _text(rec["domain"], "domain"), _text(rec["text"], "document.text"),
    ) for rec in document_records)
    queries = tuple(BenchmarkQuery(
        _text(rec["query_id"], "query_id"), _text(rec["domain"], "domain"),
        _text(rec["kind"], "kind"), _text(rec["text"], "query.text"),
    ) for rec in query_records)
    judgments = tuple(RelevanceJudgment(
        _text(rec["query_id"], "qrels.query_id"), _text(rec["doc_id"], "qrels.doc_id"),
        rec["relevance"],
    ) for rec in qrel_records)
    _validate_benchmark(docs, queries, judgments)
    return FrozenBenchmark(BENCHMARK_ID, docs, queries, judgments, actual_manifest_sha)


def evaluate_rankings(
    benchmark: FrozenBenchmark,
    rankings: Mapping[str, Sequence[str]],
    *,
    k: int = 5,
) -> EvaluationResult:
    """Оценить переданные списки по frozen ground truth без запуска retriever.

    Аргументы:
        benchmark: Загруженный и проверенный замороженный корпус.
        rankings: Для КАЖДОГО query_id — ordered list/tuple найденных doc_id.
            Позиции начинаются с 1. Пустой список допускается, но запрос
            всё равно входит в знаменатель общих метрик.
        k: Положительное число первых результатов, участвующих в расчёте.
    Результат:
        Macro Recall@K и MRR@K и покомпонентный отчёт по всем 80 запросам.

    Неизвестные, повторяющиеся doc_id и отсутствующие вопросы вызывают
    ошибку, чтобы нельзя было завысить метрики за счёт пропуска сложных задач.
    Отрицательных qrels в файле нет: все неуказанные пары трактуются как 0.
    """
    if not isinstance(benchmark, FrozenBenchmark):
        raise TypeError("benchmark должен иметь тип FrozenBenchmark")
    if type(k) is not int or k <= 0:
        raise ValueError("k должен быть положительным Python int")
    if not isinstance(rankings, Mapping):
        raise TypeError("rankings должен быть отображением query_id → doc_ids")
    known_queries = {query.query_id for query in benchmark.queries}
    if set(rankings) != known_queries:
        missing = len(known_queries - set(rankings))
        extra = len(set(rankings) - known_queries)
        raise ValueError(f"Необходимо оценить все вопросы: missing={missing}, extra={extra}")
    valid_docs = {doc.doc_id for doc in benchmark.documents}
    qrels = benchmark.relevance_map()
    rows: list[QueryMetrics] = []
    for query in benchmark.queries:
        ranked = rankings[query.query_id]
        if not isinstance(ranked, (tuple, list)):
            raise TypeError("Каждая выдача должна быть list/tuple doc_id")
        if len(set(ranked)) != len(ranked):
            raise ValueError(f"Дубли doc_id в выдаче {query.query_id}")
        if any(not isinstance(did, str) or did not in valid_docs for did in ranked):
            raise ValueError(f"Неизвестный doc_id в выдаче {query.query_id}")
        relevant = qrels[query.query_id]
        if not relevant:
            raise ValueError(f"У запроса {query.query_id} нет размеченной релевантности")
        hits = [index for index, doc_id in enumerate(ranked[:k], start=1) if doc_id in relevant]
        rows.append(QueryMetrics(
            query_id=query.query_id,
            relevant_count=len(relevant),
            retrieved_at_k=min(len(ranked), k),
            hits_at_k=len(hits),
            recall_at_k=len(hits) / len(relevant),
            reciprocal_rank_at_k=(1.0 / hits[0]) if hits else 0.0,
        ))
    count = len(rows)
    if not count:
        raise ValueError("Benchmark не содержит запросов")
    return EvaluationResult(
        k=k,
        n_queries=count,
        recall_at_k=math.fsum(row.recall_at_k for row in rows) / count,
        mrr_at_k=math.fsum(row.reciprocal_rank_at_k for row in rows) / count,
        per_query=tuple(rows),
    )
