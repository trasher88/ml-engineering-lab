"""F3.7: честное сравнение BM25, точного Vector Search и Hybrid RRF.

Использует неизменяемый benchmark F3.6 и уже реализованные индексы F3.3–F3.5.
Разметка qrels применяется только ПОСЛЕ построения всех выдач. Синтетические
векторы тестируют интеграцию, но не считаются результатом pretrained-модели.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .bm25 import BM25Index, Document
from .hybrid_search import HybridSearcher
from .retrieval_benchmark import (
    BenchmarkDocument, BenchmarkQuery, EvaluationResult, FrozenBenchmark,
    evaluate_rankings,
)
from .vector_search import VectorIndex


METHODS = ("bm25", "vector", "hybrid")
# Фиксируем протокол до оценки frozen benchmark; смена JSON требует новой версии.
FROZEN_CONFIG_SHA256 = "6682562919c46b608823689a34dfb5983ed671ad21e3ec844fbcacdb4e315fb7"
_EXPECTED_CONFIG_KEYS = frozenset({
    "schema_version", "top_ks", "candidate_pool_size", "bm25_k1", "bm25_b",
    "rrf_k", "lexical_weight", "vector_weight", "batch_size",
})


def default_comparison_config() -> Path:
    """Вернуть путь к зафиксированной конфигурации F3.7 в проекте."""
    return Path(__file__).resolve().parents[2] / "configs" / "f3_07_comparison.json"


def _strict_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Отклонить повторяющиеся ключи JSON вместо молчаливой перезаписи."""
    result: dict[str, Any] = {}
    for name, value in pairs:
        if name in result:
            raise ValueError(f"Повтор JSON-ключа: {name}")
        result[name] = value
    return result


def _finite_number(value: object, *, name: str, minimum: float, maximum: float | None = None) -> float:
    """Валидировать параметр до построения индексов, исключая bool и NaN."""
    if type(value) not in (float, int) or not math.isfinite(float(value)):
        raise ValueError(f"{name}: ожидалось конечное число")
    result = float(value)
    if result < minimum or (maximum is not None and result > maximum):
        raise ValueError(f"{name}: значение вне допустимого диапазона")
    return result


@dataclass(frozen=True, slots=True)
class ComparisonConfig:
    """Заранее фиксируемые гиперпараметры трёх поисковых методов.

    Поля:
        top_ks: Глубины метрик; список строго возрастает.
        candidate_pool_size: Число кандидатов каждого ранжировщика перед RRF.
        bm25_k1, bm25_b: Параметры BM25 F3.3.
        rrf_k, lexical_weight, vector_weight: Параметры Hybrid F3.5.
        batch_size: Пакет pretrained encoder для CPU inference.
        sha256: SHA-256 исходного канонического JSON конфигурации.
    """

    top_ks: tuple[int, ...]
    candidate_pool_size: int
    bm25_k1: float
    bm25_b: float
    rrf_k: int
    lexical_weight: float
    vector_weight: float
    batch_size: int
    sha256: str

    def __post_init__(self) -> None:
        """Отсечь заведомо некорректную или неоднозначную конфигурацию."""
        if (not isinstance(self.top_ks, tuple) or not self.top_ks
                or any(type(k) is not int or k < 1 for k in self.top_ks)
                or tuple(sorted(set(self.top_ks))) != self.top_ks):
            raise ValueError("top_ks должны быть уникальными возрастающими положительными int")
        for field in ("candidate_pool_size", "rrf_k", "batch_size"):
            value = getattr(self, field)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{field} должен быть положительным int")
        if self.candidate_pool_size < max(self.top_ks):
            raise ValueError("candidate_pool_size должен быть >= max(top_ks)")
        _finite_number(self.bm25_k1, name="bm25_k1", minimum=1e-12)
        _finite_number(self.bm25_b, name="bm25_b", minimum=0.0, maximum=1.0)
        _finite_number(self.lexical_weight, name="lexical_weight", minimum=1e-12)
        _finite_number(self.vector_weight, name="vector_weight", minimum=1e-12)
        if not isinstance(self.sha256, str) or len(self.sha256) != 64 or any(c not in "0123456789abcdef" for c in self.sha256):
            raise ValueError("sha256 должен быть каноническим отпечатком конфигурации")

    @classmethod
    def from_json(cls, path: str | Path | None = None) -> "ComparisonConfig":
        """Прочитать конфигурацию UTF-8, проверить схему и вычислить SHA-256.

        Аргументы:
            path: JSON с заранее установленными значениями; None — проектный.
        Результат:
            Валидированная неизменяемая конфигурация.
        """
        location = Path(path) if path is not None else default_comparison_config()
        raw = location.read_bytes().replace(b"\r\n", b"\n")
        if b"\r" in raw or raw.startswith(b"\xef\xbb\xbf"):
            raise ValueError("Конфигурация должна быть UTF-8 с LF/CRLF и без BOM")
        fingerprint = hashlib.sha256(raw).hexdigest()
        if path is None and fingerprint != FROZEN_CONFIG_SHA256:
            raise ValueError("SHA-256 pinned конфигурации F3.7 изменён: нужен новый протокол")
        data = json.loads(raw.decode("utf-8"), object_pairs_hook=_strict_json_keys)
        if not isinstance(data, dict) or set(data) != _EXPECTED_CONFIG_KEYS:
            raise ValueError("Неизвестные или отсутствующие поля конфигурации")
        if type(data["schema_version"]) is not int or data["schema_version"] != 1:
            raise ValueError("Неподдерживаемая версия конфигурации")
        if not isinstance(data["top_ks"], list):
            raise ValueError("top_ks должен быть JSON-массивом")
        return cls(
            top_ks=tuple(data["top_ks"]),
            candidate_pool_size=data["candidate_pool_size"],
            bm25_k1=data["bm25_k1"], bm25_b=data["bm25_b"],
            rrf_k=data["rrf_k"], lexical_weight=data["lexical_weight"],
            vector_weight=data["vector_weight"], batch_size=data["batch_size"],
            sha256=fingerprint,
        )

    def to_dict(self) -> dict[str, Any]:
        """Вернуть параметры эксперимента для проверки независимо от SHA."""
        return {
            "top_ks": list(self.top_ks),
            "candidate_pool_size": self.candidate_pool_size,
            "bm25_k1": self.bm25_k1,
            "bm25_b": self.bm25_b,
            "rrf_k": self.rrf_k,
            "lexical_weight": self.lexical_weight,
            "vector_weight": self.vector_weight,
            "batch_size": self.batch_size,
        }


@dataclass(frozen=True, slots=True)
class Rankings:
    """Три полные выдачи, рассчитанные ДО чтения qrels.

    Порядок ключей: bm25, vector, hybrid; у каждого ровно один список doc_id
    для каждого запроса. Вектора привязаны к документам и запросам по порядку.
    """

    bm25: dict[str, tuple[str, ...]]
    vector: dict[str, tuple[str, ...]]
    hybrid: dict[str, tuple[str, ...]]

    def as_dict(self) -> dict[str, dict[str, tuple[str, ...]]]:
        """Представить списки для последующего независимого скоринга."""
        return {name: getattr(self, name) for name in METHODS}


def _validate_embeddings(embeddings: np.ndarray, n: int, name: str) -> np.ndarray:
    """Проверить форму/тип вектора и отсутствие нечисловых значений."""
    if not isinstance(embeddings, np.ndarray):
        raise TypeError(f"{name} должен быть numpy.ndarray")
    if embeddings.ndim != 2 or embeddings.shape[0] != n or embeddings.shape[1] < 1:
        raise ValueError(f"{name} должен иметь форму [{n}, D], D > 0")
    if embeddings.dtype not in (np.dtype("float32"), np.dtype("float64")):
        raise TypeError(f"{name} должен иметь dtype float32/float64")
    if not np.isfinite(embeddings).all() or (np.linalg.norm(embeddings.astype("float64"), axis=1) == 0).any():
        raise ValueError(f"{name} содержит NaN/Inf или нулевой вектор")
    return embeddings


def generate_rankings(
    documents: Sequence[BenchmarkDocument],
    queries: Sequence[BenchmarkQuery],
    document_embeddings: np.ndarray,
    query_embeddings: np.ndarray,
    config: ComparisonConfig,
) -> Rankings:
    """Сформировать три выдачи БЕЗ ДОСТУПА к меткам релевантности.

    Аргументы:
        documents, queries: Тексты и ID в фиксированном порядке benchmark F3.6.
        document_embeddings: Вектора только документов [N_docs, D].
        query_embeddings: Вектора только запросов [N_queries, D].
        config: Зафиксированные параметры без выбора по frozen qrels.
    Результат:
        Для каждого метода все query_id → top-max(top_ks) doc_id.
    """
    if not isinstance(config, ComparisonConfig):
        raise TypeError("config должен быть ComparisonConfig")
    if not documents or not queries:
        raise ValueError("Необходимо передать непустые документы и запросы")
    docs = tuple(documents)
    qs = tuple(queries)
    if any(not isinstance(doc, BenchmarkDocument) for doc in docs):
        raise TypeError("documents содержит не BenchmarkDocument")
    if any(not isinstance(query, BenchmarkQuery) for query in qs):
        raise TypeError("queries содержит не BenchmarkQuery")
    doc_ids = [doc.doc_id for doc in docs]
    query_ids = [query.query_id for query in qs]
    if len(set(doc_ids)) != len(docs) or len(set(query_ids)) != len(qs):
        raise ValueError("Повторяющиеся document или query ID")
    doc_matrix = _validate_embeddings(document_embeddings, len(docs), "document_embeddings")
    query_matrix = _validate_embeddings(query_embeddings, len(qs), "query_embeddings")
    if doc_matrix.shape[1] != query_matrix.shape[1]:
        raise ValueError("Размерности document и query embeddings не совпадают")

    lex = BM25Index((Document(item.doc_id, item.text) for item in docs), k1=config.bm25_k1, b=config.bm25_b)
    vec = VectorIndex(doc_ids, doc_matrix, source_ids=[item.source_id for item in docs])
    hybrid = HybridSearcher(
        lex, vec, rrf_k=config.rrf_k,
        candidate_pool_size=config.candidate_pool_size,
        lexical_weight=config.lexical_weight, vector_weight=config.vector_weight,
    )
    depth = max(config.top_ks)
    method_rankings: dict[str, dict[str, tuple[str, ...]]] = {name: {} for name in METHODS}
    for row, query in enumerate(qs):
        embedding = query_matrix[row]
        # Индексы одни на все запросы. Измеряем *тот же* корпус и не меняем
        # параметры в зависимости от текста или результата qrels.
        method_rankings["bm25"][query.query_id] = tuple(hit.doc_id for hit in lex.search(query.text, top_k=depth))
        method_rankings["vector"][query.query_id] = tuple(hit.doc_id for hit in vec.search(embedding, top_k=depth))
        method_rankings["hybrid"][query.query_id] = tuple(
            hit.doc_id for hit in hybrid.search(query.text, embedding, top_k=depth)
        )
    return Rankings(**method_rankings)


@dataclass(frozen=True, slots=True)
class ComparisonResult:
    """Полный результат трёх методов для одной версии данных и параметров.

    Синтетические embeddings допустимы только как тесты; backend фиксируется
    отдельно, чтобы нельзя было перепутать результат с pretrained encoder.
    """

    benchmark_id: str
    benchmark_sha256: str
    config: ComparisonConfig
    backend: str
    model_metadata: Mapping[str, Any]
    evaluations: dict[str, dict[int, EvaluationResult]]
    rankings: Rankings
    queries: tuple[BenchmarkQuery, ...]
    doc_sources: Mapping[str, str]


def compare_benchmark(
    benchmark: FrozenBenchmark,
    document_embeddings: np.ndarray,
    query_embeddings: np.ndarray,
    config: ComparisonConfig,
    *,
    backend: str,
    model_metadata: Mapping[str, Any],
) -> ComparisonResult:
    """Сравнить методы на всех запросах одного frozen benchmark.

    Сначала вызывается generate_rankings, которому не передаются qrels;
    только затем F3.6 оценивает результаты для заданных top-k. Никаких
    подборов гиперпараметров на разметке здесь нет.
    """
    if not isinstance(benchmark, FrozenBenchmark):
        raise TypeError("benchmark должен быть FrozenBenchmark")
    if backend not in ("pretrained_transformer", "synthetic_test_vectors"):
        raise ValueError("backend должен однозначно указывать происхождение embeddings")
    if not isinstance(model_metadata, Mapping) or not model_metadata:
        raise ValueError("Нужны непустые model_metadata")
    rankings = generate_rankings(
        benchmark.documents, benchmark.queries, document_embeddings, query_embeddings, config
    )
    evaluations = {
        method: {
            k: evaluate_rankings(benchmark, ranks, k=k)
            for k in config.top_ks
        }
        for method, ranks in rankings.as_dict().items()
    }
    return ComparisonResult(
        benchmark_id=benchmark.benchmark_id,
        benchmark_sha256=benchmark.manifest_sha256,
        config=config,
        backend=backend,
        model_metadata=dict(model_metadata),
        evaluations=evaluations,
        rankings=rankings,
        queries=benchmark.queries,
        doc_sources={doc.doc_id: doc.source_id for doc in benchmark.documents},
    )


def _group_metrics(result: ComparisonResult, method: str, k: int, *, key: str, value: str) -> tuple[float, float, int]:
    """Макро-усреднить индивидуальные метрики для домена или стиля запроса."""
    by_id = {q.query_id: q for q in result.queries}
    selected = [row for row in result.evaluations[method][k].per_query
                if getattr(by_id[row.query_id], key) == value]
    if not selected:
        raise ValueError("Нет запросов для диагностической группы")
    return (
        math.fsum(row.recall_at_k for row in selected) / len(selected),
        math.fsum(row.reciprocal_rank_at_k for row in selected) / len(selected),
        len(selected),
    )


def result_as_dict(result: ComparisonResult) -> dict[str, Any]:
    """Создать JSON-аудит всех 80 списков с метриками и версиями.

    Результат содержит только синтетические тексты/ID benchmark, ранги и
    метаданные модели. Токены, локальные пути, веса и приватные данные не входят.
    """
    all_rankings = result.rankings.as_dict()
    metrics: dict[str, Any] = {}
    for method in METHODS:
        metrics[method] = {}
        for k in result.config.top_ks:
            evaluation = result.evaluations[method][k]
            metrics[method][str(k)] = {
                "recall_at_k": evaluation.recall_at_k,
                "mrr_at_k": evaluation.mrr_at_k,
                "n_queries": evaluation.n_queries,
            }
    per_query = []
    for q in result.queries:
        per_query.append({
            "query_id": q.query_id,
            "domain": q.domain,
            "kind": q.kind,
            "rankings": {method: list(all_rankings[method][q.query_id]) for method in METHODS},
            "per_query_metrics": {
                method: {
                    str(k): {
                        "recall_at_k": next(row.recall_at_k for row in result.evaluations[method][k].per_query if row.query_id == q.query_id),
                        "reciprocal_rank_at_k": next(row.reciprocal_rank_at_k for row in result.evaluations[method][k].per_query if row.query_id == q.query_id),
                    }
                    for k in result.config.top_ks
                }
                for method in METHODS
            },
        })
    return {
        "schema_version": 1,
        "benchmark_id": result.benchmark_id,
        "benchmark_manifest_sha256": result.benchmark_sha256,
        "config": result.config.to_dict(),
        "config_sha256": result.config.sha256,
        "backend": result.backend,
        "model_metadata": dict(result.model_metadata),
        "n_queries": len(result.queries),
        "methods": list(METHODS),
        "metrics": metrics,
        "per_query_rankings": per_query,
        "doc_sources": dict(result.doc_sources),
    }


def _format_metric(value: float) -> str:
    """Округлить только отображение, сохранив полную точность в JSON."""
    return f"{value:.4f}"


def build_markdown_report(result: ComparisonResult) -> str:
    """Построить сравнительный отчёт с диагностикой без подгонки метрик.

    В Markdown выводится только фактически посчитанное; синтетический
    backend маркируется отдельно и не может masquerade как pretrained.
    """
    if result.backend != "pretrained_transformer":
        raise ValueError("Публичный сравнительный отчёт требует реальный pretrained encoder")
    lines = [
        "# F3.7 — BM25 vs Vector vs Hybrid: отчёт эксперимента",
        "",
        "## Условия сравнения",
        "",
        f"- Benchmark: `{result.benchmark_id}` (SHA-256 manifest: `{result.benchmark_sha256}`).",
        f"- Конфигурация F3.7 (SHA-256): `{result.config.sha256}`.",
        f"- Encoder: `{result.model_metadata.get('model_id', '<unknown>')}` @ `{result.model_metadata.get('revision', '<unknown>')}`.",
        f"- Лицензия, заявленная в model card: `{result.model_metadata.get('license', '<unknown>')}`.",
        f"- Manifest с SHA файлов модели: `{result.model_metadata.get('model_manifest_sha256', '<unknown>')}`.",
        f"- Библиотеки: {', '.join(f'`{name}={version}`' for name, version in sorted(result.model_metadata.get('library_versions', {}).items()))}.",
        f"- Всего запросов: **{len(result.queries)}**; используется весь frozen набор, не выборка удачных случаев.",
        "- Режим: **pretrained Transformer, CPU inference, без обучения и подбора параметров по qrels**.",
        f"- BM25: `k1={result.config.bm25_k1}`, `b={result.config.bm25_b}`; exact cosine; ",
        f"  RRF: `k={result.config.rrf_k}`, веса `{result.config.lexical_weight}`/`{result.config.vector_weight}`, ",
        f"  candidate pool `{result.config.candidate_pool_size}` на поисковик.",
        "",
        "## Общие результаты",
        "",
        "| Метод | K | Recall@K | MRR@K |",
        "|---|---:|---:|---:|",
    ]
    for method in METHODS:
        for k in result.config.top_ks:
            row = result.evaluations[method][k]
            lines.append(f"| {method.upper()} | {k} | {_format_metric(row.recall_at_k)} | {_format_metric(row.mrr_at_k)} |")
    for name, field, values in (
        ("По стилю запроса", "kind", ("lexical", "paraphrase")),
        ("По техническому домену", "domain", ("databases", "linux", "programming", "systems")),
    ):
        lines.extend(("", f"## {name} (K=5)", "", "| Группа | Метод | N | Recall@5 | MRR@5 |", "|---|---|---:|---:|---:|"))
        if 5 not in result.config.top_ks:
            raise ValueError("Для диагностического отчёта требуется K=5")
        # Для frozen F3.6 присутствуют все группы; маленькие unit-примеры
        # не обязаны их покрывать, поэтому пустые группы не подменяем нулями.
        available = {getattr(q, field) for q in result.queries}
        for value in values:
            if value not in available:
                continue
            for method in METHODS:
                recall, mrr, n = _group_metrics(result, method, 5, key=field, value=value)
                lines.append(f"| {value} | {method} | {n} | {_format_metric(recall)} | {_format_metric(mrr)} |")

    # Диагностические примеры не влияют на выбор параметров; сортировка
    # по величине разницы показывает и улучшения, и ухудшения без замалчивания.
    rows = {method: {row.query_id: row for row in result.evaluations[method][5].per_query} for method in METHODS}
    query_map = {q.query_id: q for q in result.queries}
    lines.extend(("", "## Диагностика разногласий (K=5)", "",
                  "Дельта Recall@5 считается относительно BM25; примеры иллюстрируют различия, а не являются отдельной оценкой качества.", ""))
    for method in ("vector", "hybrid"):
        deltas = [(rows[method][qid].recall_at_k - rows["bm25"][qid].recall_at_k, qid)
                  for qid in query_map]
        wins = sorted((item for item in deltas if item[0] > 0), key=lambda t: (-t[0], t[1]))
        losses = sorted((item for item in deltas if item[0] < 0), key=lambda t: (t[0], t[1]))
        ties = len(deltas) - len(wins) - len(losses)
        lines.append(f"### {method.upper()} относительно BM25: улучшений {len(wins)}, ухудшений {len(losses)}, без изменений {ties}")
        lines.append("")
        for caption, cases in (("Примеры улучшений", wins[:3]), ("Примеры ухудшений", losses[:3])):
            lines.append(f"**{caption}:**")
            if not cases:
                lines.append("Нет случаев на фиксированном наборе.")
            else:
                for delta, qid in cases:
                    q = query_map[qid]
                    lines.append(f"- `{qid}` ({q.kind}, {q.domain}): ΔRecall@5={delta:+.2f}; BM25 top-5={list(result.rankings.bm25[qid][:5])}; {method} top-5={list(getattr(result.rankings, method)[qid][:5])}.")
            lines.append("")

    lines.extend((
        "## Ограничения",
        "",
        "- Это **авторский синтетический** benchmark из F3.6, а не независимое измерение производственного поиска.",
        "- Qrels двоичные и не охватывают все потенциально полезные документы; отсутствие qrel означает 0 только по контракту набора.",
        "- Модель F3.2 не обучалась на этом benchmark; конфигурация F3.7 была зафиксирована до оценки.",
        "- Vector использует exact cosine, BM25 использует простую токенизацию без морфологии, Hybrid объединяет РАНГИ, а не вероятности.",
        "- Изменение frozen-разметки, версии encoder либо гиперпараметров требует отдельного эксперимента, а не перезаписи данного результата.",
        "- Если hybrid оказался слабее одного из методов, результат остаётся фактическим: сочетание рангов не гарантирует прироста.",
        "- Полные 80 ранжированных списков каждого метода и метаданные доступны в сопутствующем JSON-артефакте.",
        "",
    ))
    return "\n".join(lines)


def write_comparison_artifacts(result: ComparisonResult, out_dir: Path) -> tuple[Path, Path]:
    """Сохранить только реальный отчёт/JSON вне каталога моделей/кеша.

    Аргументы:
        result: Результат с явно подтверждённым pretrained backend.
        out_dir: Место публикационных артефактов, выбранное пользователем.
    Результат:
        Пути двух файлов UTF-8. Автоматические Git-операции не выполняются.
    """
    if result.backend != "pretrained_transformer":
        raise ValueError("Запрещён публичный отчёт по синтетическим test vectors")
    destination = Path(out_dir)
    destination.mkdir(parents=True, exist_ok=True)
    report = destination / "F3_07_comparison.md"
    evidence = destination / "F3_07_comparison.json"
    text = build_markdown_report(result)
    json_content = json.dumps(result_as_dict(result), ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    report.write_text(text, encoding="utf-8", newline="\n")
    evidence.write_text(json_content, encoding="utf-8", newline="\n")
    return report, evidence
