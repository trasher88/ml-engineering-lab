"""F3.9: аудит доказательств Фазы 3 и итоговый воспроизводимый отчёт.

Не запускает модели, не выбирает гиперпараметры и не создаёт метрики из
синтетического smoke. Повторно вычисляет Recall@K/MRR@K из полных выдач F3.7
на замороженной разметке F3.6 и проверяет конфигурации F3.2/F3.7.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping

from .embeddings import ModelSpec
from .rag_provenance import GenerationDraft, RAGConfig, RAGPipeline, SourceDocument
from .retrieval_benchmark import FrozenBenchmark, evaluate_rankings, load_frozen_benchmark
from .retrieval_comparison import ComparisonConfig, METHODS, FROZEN_CONFIG_SHA256


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_METRIC_TOLERANCE = 1e-12
# Отпечатки исходных опубликованных конфигов; для обновления нужен новый протокол.
_F3_02_MODEL_CONFIG_SHA256 = "807530eb91e92eaf56a8517012adec35e8b4dd184d5dca8640d7f20a29661610"
_F3_08_RAG_CONFIG_SHA256 = "03f14a21229ef485dd4cd3ecbbb4bff0f23e6d6f7f5f920959954f4ae579d786"
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_VERSION_PATTERN = re.compile(r"^[\w.+-]{1,100}$", re.ASCII)
_REQUIRED_PREVIOUS_FILES = (
    "src/transformers_f3/attention.py",
    "src/transformers_f3/embeddings.py",
    "src/transformers_f3/model_cache.py",
    "src/transformers_f3/bm25.py",
    "src/transformers_f3/vector_search.py",
    "src/transformers_f3/hybrid_search.py",
    "src/transformers_f3/retrieval_benchmark.py",
    "src/transformers_f3/retrieval_comparison.py",
    "src/transformers_f3/rag_provenance.py",
    "configs/f3_02_model.json",
    "configs/f3_07_comparison.json",
    "configs/f3_08_rag.json",
    "docs/steps/F3_01_transformer_attention.md",
    "docs/steps/F3_02_pretrained_embeddings.md",
    "docs/steps/F3_03_bm25_indexing.md",
    "docs/steps/F3_04_vector_search_numpy.md",
    "docs/steps/F3_05_hybrid_rrf.md",
    "docs/steps/F3_06_frozen_retrieval_benchmark.md",
    "docs/steps/F3_07_bm25_vector_hybrid_comparison.md",
    "docs/steps/F3_08_rag_provenance.md",
)
_STAGES = (
    ("F3.1", "Self-Attention и masks", "Проверяемый механизм внимания на PyTorch"),
    ("F3.2", "Pretrained embeddings и кеш", "Pinned encoder с manifest и локальным CPU inference"),
    ("F3.3", "BM25", "Лексический индекс и top-k выдача"),
    ("F3.4", "Exact vector search", "Dense embeddings, cosine и source_id"),
    ("F3.5", "Hybrid RRF", "Объединение рангов двух независимых методов"),
    ("F3.6", "Frozen benchmark", "80 запросов, 88 документов и 160 qrels"),
    ("F3.7", "Retrieval comparison", "Аудируемое сопоставление Recall@K/MRR@K"),
    ("F3.8", "RAG provenance", "Дословные цитаты и extractive fallback"),
)


@dataclass(frozen=True, slots=True)
class PhaseAudit:
    """Проверенные агрегаты одного F3.7-прогона, без приватных текстов.

    Поля:
        metrics: Пересчитанные макро Recall@K и MRR@K для трёх методов.
        groups: Метрики по типам запросов и техническим доменам при K=5.
        disagreements: Улучшения/ухудшения относительно BM25 по запросам.
        model: Публичные сведения о pinned encoder из F3.2.
        rag_smoke: Результаты автономных проверок цитат и fallback F3.8.
        source_sha256: Отпечаток входного JSON F3.7 (с нормализацией CRLF).
    """

    source_sha256: str
    benchmark_id: str
    benchmark_sha256: str
    config_sha256: str
    n_queries: int
    n_documents: int
    n_judgments: int
    model: Mapping[str, Any]
    metrics: Mapping[str, Mapping[str, Mapping[str, float | int]]]
    groups: Mapping[str, Mapping[str, Mapping[str, Mapping[str, float | int]]]]
    disagreements: Mapping[str, Mapping[str, int]]
    rag_smoke: Mapping[str, bool]


def _canonical_project_sha(path: Path) -> str:
    """Зафиксировать хеш конфигурации независимо от CRLF при checkout Windows.

    Запрещён BOM и одиночный CR; такая проверка не изменяет исходный файл.
    """
    raw = path.read_bytes().replace(b"\r\n", b"\n")
    if raw.startswith(b"\xef\xbb\xbf") or b"\r" in raw:
        raise ValueError(f"Неканонический UTF-8 конфигурации: {path.name}")
    raw.decode("utf-8", errors="strict")
    return sha256(raw).hexdigest()


def _strict_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """Запретить дубли JSON-ключей, скрывающие подмену параметров или метрик."""
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Повторяющийся JSON-ключ: {key}")
        result[key] = value
    return result


def _no_constants(value: str) -> None:
    """Не считать NaN/Infinity допустимыми числами для отчётных метрик."""
    raise ValueError(f"Недопустимое JSON-значение: {value}")


def _read_evidence(path: str | Path) -> tuple[dict[str, Any], str]:
    """Прочитать UTF-8 JSON F3.7 и вернуть проверяемый SHA-256 входа.

    Аргументы:
        path: Путь к реальному ``F3_07_comparison.json`` пользователя.
    Результат:
        Словарь с сохранённой схемой и канонический SHA-256 исходного JSON.
    """
    raw = Path(path).read_bytes().replace(b"\r\n", b"\n")
    if raw.startswith(b"\xef\xbb\xbf") or b"\r" in raw:
        raise ValueError("F3.7 JSON: ожидался UTF-8 без BOM и одиночных CR")
    payload = json.loads(raw.decode("utf-8"), object_pairs_hook=_strict_pairs, parse_constant=_no_constants)
    if not isinstance(payload, dict):
        raise ValueError("F3.7 JSON: корень должен быть объектом")
    return payload, sha256(raw).hexdigest()


def _same_real_number(actual: object, expected: float, *, context: str) -> None:
    """Сверить точные вычисления, не позволяя bool и нечисловые значения."""
    if type(actual) not in (int, float) or not math.isfinite(actual):
        raise ValueError(f"{context}: ожидалось конечное число")
    if not math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=_METRIC_TOLERANCE):
        raise ValueError(f"{context}: расхождение с повторным вычислением")


def _exact_sha(value: object, *, field: str) -> str:
    """Проверить, что поле содержит канонический 64-символьный SHA-256."""
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{field}: неверный SHA-256")
    return value


def _check_model_metadata(payload: object, spec: ModelSpec) -> dict[str, Any]:
    """Проверить декларацию pretrained-encoder относительно pinned F3.2.

    SHA модели из F3.7 проверяется только по формату: для криптографической
    сверки файлов модели необходим исходный кеш F3.2, который сюда не входит.
    """
    if not isinstance(payload, dict):
        raise ValueError("F3.7: отсутствуют model_metadata")
    expected = {
        "model_id": spec.model_id,
        "revision": spec.revision,
        "license": spec.license,
        "embedding_dim": spec.embedding_dim,
        "pooling": spec.pooling,
        "normalize": spec.normalize,
    }
    for key, value in expected.items():
        if type(payload.get(key)) is not type(value) or payload[key] != value:
            raise ValueError(f"F3.7: модель не соответствует закреплённой F3.2: {key}")
    fingerprint = _exact_sha(payload.get("model_manifest_sha256"), field="model_manifest_sha256")
    libraries = payload.get("library_versions")
    if not isinstance(libraries, dict) or not libraries:
        raise ValueError("F3.7: отсутствуют версии библиотек encoder")
    for name in ("torch", "transformers", "huggingface_hub"):
        version = libraries.get(name)
        if not isinstance(version, str) or _VERSION_PATTERN.fullmatch(version) is None:
            raise ValueError(f"F3.7: отсутствует корректная версия {name}")
    numpy_version = payload.get("numpy_version")
    if not isinstance(numpy_version, str) or _VERSION_PATTERN.fullmatch(numpy_version) is None:
        raise ValueError("F3.7: неизвестная версия NumPy")
    return {
        **expected,
        "model_manifest_sha256": fingerprint,
        "library_versions": {name: libraries[name] for name in ("torch", "transformers", "huggingface_hub")},
        "numpy_version": numpy_version,
    }


def _assert_phase_files(root: Path) -> None:
    """Проверить, что генератор отчёта не запущен вместо собранной F3."""
    missing = [name for name in _REQUIRED_PREVIOUS_FILES if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError("Не найдены исходники/документация F3.1–F3.8: " + ", ".join(missing))


def check_rag_smoke() -> dict[str, bool]:
    """Проверить цитаты и fallback F3.8 на независимом игрушечном корпусе.

    Результат не является оценкой answer quality или генеративной LLM.
    """
    cfg = RAGConfig.from_json()
    documents = (
        SourceDocument("toy-db", "source:toy-db", "Check PostgreSQL index usage for slow SQL queries."),
        SourceDocument("toy-linux", "source:toy-linux", "Linux journalctl displays service logs."),
    )
    basic = RAGPipeline(documents, config=cfg)
    extracted = basic.answer("PostgreSQL index queries")
    fallback = (
        extracted.mode == "extractive_fallback" and len(extracted.evidence) > 0
        and all(basic.verify_evidence(e) for e in extracted.evidence)
        and extracted.answer == f"{extracted.evidence[0].quote} [1]"
    )
    untrusted = RAGPipeline(
        documents, config=cfg,
        generator=lambda req: GenerationDraft("Performance improved 1000 times.", "[1]"),
    )
    unsupported = untrusted.answer("PostgreSQL index queries")
    rejection = unsupported.mode == "extractive_fallback" and unsupported.reason == "unsupported_generation"
    no_evidence = basic.answer("unseenuniquetoken987654")
    abstention = no_evidence.mode == "no_evidence" and not no_evidence.evidence
    result = {"verifiable_quote": fallback, "unsupported_generation_rejected": rejection, "no_evidence_abstention": abstention}
    if not all(result.values()):
        raise ValueError("F3.8: не пройден независимый smoke для provenance/fallback")
    return result


def validate_comparison(payload: Mapping[str, Any], source_sha256: str, *,
                        root: str | Path | None = None) -> PhaseAudit:
    """Аудировать F3.7 по неизменённым qrels и заново рассчитать все метрики.

    Аргументы:
        payload: Полный JSON, выданный ``result_as_dict`` этапа F3.7.
        source_sha256: SHA-256 прочитанного JSON; вычисляет вызывающий код.
        root: Корень проекта; по умолчанию расположение этого модуля.
    Результат:
        PhaseAudit с повторно измеренными величинами и проверками F3.8.
    Исключения:
        ValueError при любом несоответствии frozen-набора, модели или оценок.
    """
    project = Path(root) if root is not None else _PROJECT_ROOT
    _assert_phase_files(project)
    _exact_sha(source_sha256, field="source_sha256")
    if not isinstance(payload, Mapping):
        raise TypeError("F3.7: ожидался JSON-объект")
    keys = {
        "schema_version", "benchmark_id", "benchmark_manifest_sha256", "config", "config_sha256",
        "backend", "model_metadata", "n_queries", "methods", "metrics", "per_query_rankings", "doc_sources",
    }
    if set(payload) != keys or type(payload["schema_version"]) is not int or payload["schema_version"] != 1:
        raise ValueError("F3.7: неподдерживаемая структура артефакта")
    if payload["backend"] != "pretrained_transformer":
        raise ValueError("F3.9: нет подтверждённого pretrained-прогона F3.7; synthetic smoke не публикуется")

    benchmark: FrozenBenchmark = load_frozen_benchmark(project / "tests" / "fixtures" / "f3_06_benchmark_v1")
    config = ComparisonConfig.from_json(project / "configs" / "f3_07_comparison.json")
    if config.sha256 != FROZEN_CONFIG_SHA256:
        raise ValueError("F3.7: изменена утверждённая конфигурация retrieval")
    if _canonical_project_sha(project / "configs" / "f3_02_model.json") != _F3_02_MODEL_CONFIG_SHA256:
        raise ValueError("F3.2: изменена утверждённая спецификация encoder")
    if _canonical_project_sha(project / "configs" / "f3_08_rag.json") != _F3_08_RAG_CONFIG_SHA256:
        raise ValueError("F3.8: изменена утверждённая конфигурация RAG")
    spec = ModelSpec.from_json(project / "configs" / "f3_02_model.json")
    if payload["benchmark_id"] != benchmark.benchmark_id or payload["benchmark_manifest_sha256"] != benchmark.manifest_sha256:
        raise ValueError("F3.7: benchmark_id/SHA-256 не совпадают с frozen F3.6")
    if payload["config_sha256"] != config.sha256 or payload["config"] != config.to_dict():
        raise ValueError("F3.7: конфигурация/контрольная сумма не совпадают с F3.7")
    if type(payload["n_queries"]) is not int or payload["n_queries"] != len(benchmark.queries):
        raise ValueError("F3.7: количество запросов не соответствует benchmark")
    if payload["methods"] != list(METHODS):
        raise ValueError("F3.7: неподдерживаемый набор поисковых методов")
    expected_sources = {doc.doc_id: doc.source_id for doc in benchmark.documents}
    if payload["doc_sources"] != expected_sources:
        raise ValueError("F3.7: нарушено соответствие doc_id → source_id")
    model = _check_model_metadata(payload["model_metadata"], spec)
    per_query = payload["per_query_rankings"]
    if not isinstance(per_query, list) or len(per_query) != len(benchmark.queries):
        raise ValueError("F3.7: неполная покомпонентная выдача")

    lists: dict[str, dict[str, list[str]]] = {method: {} for method in METHODS}
    by_id: dict[str, dict[str, Any]] = {}
    docs_ids = set(expected_sources)
    for q, row in zip(benchmark.queries, per_query, strict=True):
        if not isinstance(row, dict) or set(row) != {"query_id", "domain", "kind", "rankings", "per_query_metrics"}:
            raise ValueError("F3.7: неверная структура записи запроса")
        if (row["query_id"], row["domain"], row["kind"]) != (q.query_id, q.domain, q.kind):
            raise ValueError("F3.7: query_id/domain/kind изменены или пропущены")
        if not isinstance(row["rankings"], dict) or set(row["rankings"]) != set(METHODS):
            raise ValueError("F3.7: в запросе отсутствуют методы ранжирования")
        if not isinstance(row["per_query_metrics"], dict) or set(row["per_query_metrics"]) != set(METHODS):
            raise ValueError("F3.7: в запросе отсутствуют покомпонентные метрики")
        for method in METHODS:
            ranked = row["rankings"][method]
            if not isinstance(ranked, list) or len(ranked) > max(config.top_ks):
                raise ValueError("F3.7: неверный формат/размер top-k выдачи")
            if any(type(item) is not str or item not in docs_ids for item in ranked) or len(ranked) != len(set(ranked)):
                raise ValueError("F3.7: неизвестный или повторный doc_id в выдаче")
            lists[method][q.query_id] = ranked
        by_id[q.query_id] = row

    metrics: dict[str, dict[str, dict[str, float | int]]] = {}
    groups: dict[str, dict[str, dict[str, dict[str, float | int]]]] = {"kind": {}, "domain": {}}
    per_method_k5: dict[str, dict[str, float]] = {}
    for method in METHODS:
        metric_by_k = payload["metrics"]
        if not isinstance(metric_by_k, dict) or set(metric_by_k) != set(METHODS):
            raise ValueError("F3.7: агрегаты по методам отсутствуют")
        if not isinstance(metric_by_k[method], dict) or set(metric_by_k[method]) != {str(k) for k in config.top_ks}:
            raise ValueError("F3.7: агрегаты по K неполны")
        metrics[method] = {}
        for k in config.top_ks:
            measured = evaluate_rankings(benchmark, lists[method], k=k)
            expected = metric_by_k[method][str(k)]
            if not isinstance(expected, dict) or set(expected) != {"recall_at_k", "mrr_at_k", "n_queries"}:
                raise ValueError("F3.7: неверная схема агрегатных метрик")
            if type(expected["n_queries"]) is not int or expected["n_queries"] != measured.n_queries:
                raise ValueError("F3.7: неправильный знаменатель макро-метрик")
            _same_real_number(expected["recall_at_k"], measured.recall_at_k, context=f"{method} Recall@{k}")
            _same_real_number(expected["mrr_at_k"], measured.mrr_at_k, context=f"{method} MRR@{k}")
            metrics[method][str(k)] = {
                "recall_at_k": measured.recall_at_k,
                "mrr_at_k": measured.mrr_at_k,
                "n_queries": measured.n_queries,
            }
            for record in measured.per_query:
                stored = by_id[record.query_id]["per_query_metrics"][method]
                if not isinstance(stored, dict) or set(stored) != {str(n) for n in config.top_ks}:
                    raise ValueError("F3.7: покомпонентные метрики неполны")
                given = stored[str(k)]
                if not isinstance(given, dict) or set(given) != {"recall_at_k", "reciprocal_rank_at_k"}:
                    raise ValueError("F3.7: неизвестная схема индивидуальных метрик")
                _same_real_number(given["recall_at_k"], record.recall_at_k, context=f"{method}/{record.query_id} Recall@{k}")
                _same_real_number(given["reciprocal_rank_at_k"], record.reciprocal_rank_at_k,
                                  context=f"{method}/{record.query_id} RR@{k}")
            if k == 5:
                per_method_k5[method] = {item.query_id: item.recall_at_k for item in measured.per_query}
                for field in ("kind", "domain"):
                    for name in sorted({getattr(query, field) for query in benchmark.queries}):
                        ids = [query.query_id for query in benchmark.queries if getattr(query, field) == name]
                        by_records = [row for row in measured.per_query if row.query_id in ids]
                        groups[field].setdefault(name, {})[method] = {
                            "n_queries": len(by_records),
                            "recall_at_5": math.fsum(item.recall_at_k for item in by_records) / len(by_records),
                            "mrr_at_5": math.fsum(item.reciprocal_rank_at_k for item in by_records) / len(by_records),
                        }
    if 5 not in config.top_ks:
        raise ValueError("F3.9: не найден K=5 для диагностики")
    disagreement: dict[str, dict[str, int]] = {}
    for method in ("vector", "hybrid"):
        differences = [per_method_k5[method][q.query_id] - per_method_k5["bm25"][q.query_id]
                       for q in benchmark.queries]
        disagreement[method] = {
            "better": sum(value > _METRIC_TOLERANCE for value in differences),
            "worse": sum(value < -_METRIC_TOLERANCE for value in differences),
            "equal": sum(abs(value) <= _METRIC_TOLERANCE for value in differences),
        }
    return PhaseAudit(
        source_sha256=source_sha256,
        benchmark_id=benchmark.benchmark_id,
        benchmark_sha256=benchmark.manifest_sha256,
        config_sha256=config.sha256,
        n_queries=len(benchmark.queries),
        n_documents=len(benchmark.documents),
        n_judgments=len(benchmark.judgments),
        model=model,
        metrics=metrics,
        groups=groups,
        disagreements=disagreement,
        rag_smoke=check_rag_smoke(),
    )


def audit_file(comparison_json: str | Path, *, root: str | Path | None = None) -> PhaseAudit:
    """Прочитать конкретный результат реального F3.7 и проверить его целиком."""
    payload, fingerprint = _read_evidence(comparison_json)
    return validate_comparison(payload, fingerprint, root=root)


def _format_fraction(value: float) -> str:
    """Отобразить долю с четырьмя знаками, сохраняя исходную точность в JSON."""
    return f"{value:.4f}"


def build_phase_markdown(audit: PhaseAudit) -> str:
    """Составить технический отчёт только из проверенных числовых результатов.

    Отчёт включает связи между алгоритмами, интерпретацию метрик и
    ограничения. В нём нет установки пакетов, Git-команд или вопросов.
    """
    methods = ("bm25", "vector", "hybrid")
    lines = [
        "# ML Engineering Lab — F3.9: итоговый отчёт Фазы 3",
        "",
        "## 1. Назначение и инженерный результат",
        "",
        "Фаза 3 построила цепочку от Self-Attention и предобученных dense embeddings",
        "до лексического, векторного и гибридного поиска, оценки на фиксированных",
        "релевантностях и ограниченного RAG-конвейера с подтверждением источников.",
        "Основное достижение — воспроизводимый эксперимент, а не доказанное",
        "превосходство Transformer или готовая production RAG-система.",
        "",
        "## 2. Связь этапов и архитектура",
        "",
        "```mermaid",
        "flowchart LR",
        "    A[F3.1 Attention] --> B[F3.2 Pretrained embeddings]",
        "    B --> D[F3.4 Vector search]",
        "    C[F3.3 BM25] --> E[F3.5 Hybrid RRF]",
        "    D --> E",
        "    C --> G[F3.7 Evaluation]",
        "    D --> G",
        "    E --> G",
        "    F[F3.6 Frozen qrels] --> G",
        "    C --> H[F3.8 RAG provenance]",
        "    E --> H",
        "    G --> I[F3.9 Audit & report]",
        "    H --> I",
        "```",
        "",
        "| Этап | Алгоритм/компонент | Проверяемый выход |",
        "|---|---|---|",
    ]
    for name, subject, output in _STAGES:
        lines.append(f"| {name} | {subject} | {output} |")
    lines.extend([
        "",
        "## 3. Источники измерений и неизменность данных",
        "",
        "- Источник измерений: `reports/public/f3_07/F3_07_comparison.json`,",
        f"  канонический SHA-256: `{audit.source_sha256}`.",
        f"- F3.6: `{audit.benchmark_id}`, manifest SHA-256 `{audit.benchmark_sha256}`.",
        f"- Объём: **{audit.n_documents}** документов, **{audit.n_queries}** запросов, **{audit.n_judgments}** бинарных qrels.",
        f"- Замороженный протокол F3.7: SHA-256 `{audit.config_sha256}`.",
        f"- Encoder: `{audit.model['model_id']}` @ `{audit.model['revision']}`.",
        f"- Лицензия из pinned спецификации: `{audit.model['license']}`; размерность `{audit.model['embedding_dim']}`.",
        f"- Признаки encoder: `{audit.model['pooling']}` pooling, L2 normalize `{str(audit.model['normalize']).lower()}`.",
        f"- Отпечаток manifest модели, сохранённый F3.7: `{audit.model['model_manifest_sha256']}`.",
        "- Сверены все запросы, `doc_id`/`source_id`, точные выдачи и метрики каждого запроса.",
        "  Никакой запрос не исключён из макро-усреднения.",
        "",
        "## 4. Определения и математика retrieval-оценки",
        "",
        "Для запроса q и множества размеченных релевантных документов R(q):",
        "",
        r"\[\operatorname{Recall@K}(q)=\frac{|\operatorname{topK}(q)\cap R(q)|}{|R(q)|}\]",
        "",
        r"\[\operatorname{RR@K}(q)=\begin{cases}1/r_1,& r_1\leq K,\\0,&\text{иначе}\end{cases}\]",
        "",
        "Здесь `r₁` — ранг первого релевантного документа. Macro Recall@K и",
        "MRR@K — средние этих величин по всем 80 запросам. Например, если",
        "есть 2 релевантных документа и в top-5 найден только один на позиции 3,",
        "то Recall@5 = 1/2 = 0.5, а RR@5 = 1/3 ≈ 0.3333.",
        "",
        "## 5. Фактические результаты F3.7",
        "",
        "Результаты ниже заново рассчитаны по ранжированным спискам F3.7 на qrels F3.6.",
        "Округление применяется только для отображения, не для аудита.",
        "",
        "| Метод | K | Recall@K | MRR@K |",
        "|---|---:|---:|---:|",
    ])
    for method in methods:
        for k, row in sorted(audit.metrics[method].items(), key=lambda item: int(item[0])):
            lines.append(f"| {method.upper()} | {k} | {_format_fraction(float(row['recall_at_k']))} | {_format_fraction(float(row['mrr_at_k']))} |")
    difference = (audit.metrics["hybrid"]["5"]["recall_at_k"] - audit.metrics["bm25"]["5"]["recall_at_k"]) * 100
    lines.extend([
        "",
        f"Hybrid минус BM25 по Recall@5: **{difference:+.2f} процентного пункта**.",
        "Отрицательная дельта допустима: RRF объединяет порядки документов",
        "и не гарантирует прирост по каждому методу или набору данных.",
        "",
        "### 5.1. Различия между группами (K=5)",
        "",
        "| Разрез | Группа | Метод | N | Recall@5 | MRR@5 |",
        "|---|---|---|---:|---:|---:|",
    ])
    for field, display in (("kind", "Формулировка"), ("domain", "Домен")):
        for group, values in sorted(audit.groups[field].items()):
            for method in methods:
                rec = values[method]
                lines.append(f"| {display} | {group} | {method} | {rec['n_queries']} | {_format_fraction(float(rec['recall_at_5']))} | {_format_fraction(float(rec['mrr_at_5']))} |")
    lines.extend([
        "",
        "### 5.2. Разногласия по запросам при K=5",
        "",
        "Считается изменение Recall@5 одного и того же запроса относительно BM25.",
        "Равенство означает одинаковую долю обнаруженных релевантных документов,",
        "а не одинаковые списки или оценки сходства.",
        "",
        "| Метод vs BM25 | Лучше | Хуже | Без изменений |",
        "|---|---:|---:|---:|",
    ])
    for method in ("vector", "hybrid"):
        counts = audit.disagreements[method]
        lines.append(f"| {method} | {counts['better']} | {counts['worse']} | {counts['equal']} |")
    lines.extend([
        "",
        "## 6. Проверка RAG/provenance F3.8",
        "",
        "На отдельных синтетических текстах проверены: точная цитата с",
        "`source_id` и SHA-256, отклонение неподтверждённой генерации и отказ",
        "при отсутствии найденного свидетельства. Результат smoke:",
        "",
    ])
    for key, text in (
        ("verifiable_quote", "Дословная цитата и целостность источника"),
        ("unsupported_generation_rejected", "Неподтверждённая генерация → extractive fallback"),
        ("no_evidence_abstention", "Нет документов → no_evidence"),
    ):
        lines.append(f"- {text}: **{'PASS' if audit.rag_smoke[key] else 'FAIL'}**.")
    lines.extend([
        "",
        "Это функциональная проверка на игрушечных данных, **не** измерение",
        "точности, faithfulness или answer quality генеративной модели.",
        "",
        "## 7. Ограничения и корректная интерпретация",
        "",
        "1. Frozen benchmark F3.6 создан вручную и синтетичен: 80 запросов не",
        "   представляют статистическую выборку всех реальных обращений пользователей.",
        "2. Бинарные qrels содержат только отмеченные автором релевантные пары;",
        "   потенциально полезные, но неразмеченные документы считаются нерелевантными",
        "   в пределах этого протокола.",
        "3. Изменение qrels, параметров, версии encoder либо корпуса требует нового",
        "   зафиксированного эксперимента. Подбор параметров по test-qrels недопустим.",
        "4. BM25 реализует лексическое совпадение, exact vector — косинусную",
        "   близость dense embeddings, RRF — слияние рангов; ни одно значение",
        "   score не является вероятностью истинности или релевантности.",
        "5. SHA-256 фиксирует неизменность сравниваемого содержимого относительно",
        "   сохранённых отпечатков; он не подтверждает происхождение первоисточника.",
        "6. Версия encoder и manifest берутся из F3.7 и сверяются с pinned",
        "   спецификацией F3.2; физические веса и локальный кеш на этом шаге",
        "   независимо не перечитываются. Из аудита нельзя заключить, что",
        "   encoder в самом деле исполнялся при создании JSON.",
        "7. F3.8 пока не содержит обязательной LLM и измерения корректности",
        "   полных генеративных ответов. Дословная цитата не обязательно релевантна",
        "   вопросу, а простой контроль подстроки не заменяет factuality checks.",
        "8. Нет доказательства производственной производительности, устойчивости",
        "   к prompt injection, приватности и безопасности реального RAG-сервиса.",
        "9. Исторические значения F1/F2 не приравниваются к retrieval-метрикам:",
        "   классификация и поиск имеют разные цели, единицы оценки и разметку.",
        "10. Статус Windows-регрессии и GitHub Actions нельзя получить из JSON F3.7;",
        "    он подтверждается отдельно после выполнения соответствующих проверок.",
        "",
        "## 8. Выводы",
        "",
        "Фаза 3 даёт проверяемую реализацию Self-Attention, embeddings и трёх",
        "способов retrieval, а также механизм для безопасного представления",
        "дословных источников. Приведённые метрики относятся только к frozen",
        "синтетической коллекции. Итоговый выбор поискового метода для реальных",
        "технических запросов требует независимого тематического benchmark.",
        "",
        "## 9. Файлы, с которыми стоит познакомиться",
        "",
        "| Файл | Что изучить |",
        "|---|---|",
        "| `src/transformers_f3/attention.py` | Q/K/V, масштабирование, Softmax, padding masks |",
        "| `src/transformers_f3/embeddings.py` | Encoder, masked mean pooling, нормализация выходных векторов |",
        "| `src/transformers_f3/model_cache.py` | Pinned revision, локальная проверка весов и manifest |",
        "| `src/transformers_f3/bm25.py` | TF/DF/IDF, BM25 и инвертированный индекс |",
        "| `src/transformers_f3/vector_search.py` | Exact cosine search, top-k, doc_id/source_id |",
        "| `src/transformers_f3/hybrid_search.py` | RRF, candidate pool, дедупликация |",
        "| `src/transformers_f3/retrieval_benchmark.py` | Frozen qrels, Recall@K, MRR@K, SHA-256 |",
        "| `src/transformers_f3/retrieval_comparison.py` | Формирование выдач отдельно от вычисления метрик |",
        "| `src/transformers_f3/rag_provenance.py` | Evidence, validate quote, source provenance и fallback |",
        "| `src/transformers_f3/phase_report.py` | Независимый повторный расчёт метрик и fail-closed аудит |",
        "| `scripts/f3_phase_report.py` | Формирование отчёта из подтверждённого JSON F3.7 |",
        "| `tests/test_f3_phase_report.py` | Согласованность метрик, защита от подмены, офлайн smoke |",
        "",
    ])
    text = "\n".join(lines)
    if any(line.rstrip(" \t") != line for line in text.splitlines()):
        raise AssertionError("Генератор отчёта создал trailing whitespace")
    return text.rstrip() + "\n"


def evidence_as_dict(audit: PhaseAudit) -> dict[str, Any]:
    """Создать небольшой машинно-проверяемый аудит без исходных текстов/весов."""
    return {
        "schema_version": 1,
        "phase": "F3",
        "audit_status": "PASS",
        "comparison_source": "reports/public/f3_07/F3_07_comparison.json",
        "comparison_sha256": audit.source_sha256,
        "benchmark_id": audit.benchmark_id,
        "benchmark_manifest_sha256": audit.benchmark_sha256,
        "comparison_config_sha256": audit.config_sha256,
        "counts": {
            "documents": audit.n_documents,
            "queries": audit.n_queries,
            "qrels": audit.n_judgments,
        },
        "model_metadata": dict(audit.model),
        "metrics_recomputed": dict(audit.metrics),
        "group_metrics": dict(audit.groups),
        "per_query_disagreements": dict(audit.disagreements),
        "rag_synthetic_contract_smoke": dict(audit.rag_smoke),
        "audit_checks": [
            "frozen_benchmark_and_qrels_sha256",
            "pinned_f3_07_config",
            "pinned_f3_02_model_metadata",
            "all_80_queries_recomputed",
            "all_retrieved_doc_ids_and_source_ids",
            "aggregates_and_per_query_metric_consistency",
            "rag_quote_fallback_and_abstention_synthetic_smoke",
        ],
        "scope": "Synthetic F3.6 benchmark; no production or LLM answer-quality claim",
    }


def write_phase_report(comparison_json: str | Path, *,
                       report_md: str | Path | None = None,
                       audit_json: str | Path | None = None,
                       root: str | Path | None = None) -> tuple[Path, Path, PhaseAudit]:
    """Сохранить итоговый отчёт и аудит только после успешной полной проверки.

    Аргументы:
        comparison_json: Исходный F3.7 JSON, рассчитанный с pretrained encoder.
        report_md: Выходной Markdown; по умолчанию ``docs/reports/F3_PHASE_REPORT.md``.
        audit_json: Проверка F3.9 в JSON без исходных текстов и приватных данных.
        root: Корень проекта, необходим для тестов в изолированной копии.
    Результат:
        Пути двух файлов и PhaseAudit. При невалидном входе ничего не создаёт.
    """
    project = Path(root) if root is not None else _PROJECT_ROOT
    md = Path(report_md) if report_md is not None else project / "docs" / "reports" / "F3_PHASE_REPORT.md"
    data = Path(audit_json) if audit_json is not None else project / "reports" / "public" / "f3_09" / "F3_PHASE_EVIDENCE.json"
    source = Path(comparison_json)
    if len({p.resolve() for p in (md, data, source)}) != 3:
        raise ValueError("Пути исходного JSON, Markdown и выходного аудита должны различаться")
    audit = audit_file(source, root=project)
    markdown = build_phase_markdown(audit)
    evidence = json.dumps(evidence_as_dict(audit), ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    md.parent.mkdir(parents=True, exist_ok=True)
    data.parent.mkdir(parents=True, exist_ok=True)
    md.write_text(markdown, encoding="utf-8", newline="\n")
    data.write_text(evidence, encoding="utf-8", newline="\n")
    return md, data, audit
