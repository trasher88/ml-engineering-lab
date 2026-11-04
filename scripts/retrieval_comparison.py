"""F3.7: офлайн smoke или реальное сравнение retrieval на frozen F3.6.

Режим по умолчанию использует только маленькие тестовые векторы без оценок
качества Transformer. Флаг --real берёт pinned pretrained-модель F3.2 из
пользовательского кеша и создаёт проверяемые файлы отчёта и метрик.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from transformers_f3.embeddings import ModelSpec  # noqa: E402
from transformers_f3.model_cache import default_cache_dir, load_pretrained_encoder  # noqa: E402
from transformers_f3.retrieval_benchmark import (  # noqa: E402
    BenchmarkDocument, BenchmarkQuery, FrozenBenchmark, RelevanceJudgment,
    load_frozen_benchmark,
)
from transformers_f3.retrieval_comparison import (  # noqa: E402
    ComparisonConfig, compare_benchmark, write_comparison_artifacts,
)


def smoke() -> None:
    """Проверить сквозной контракт на игрушечных embeddings без pretrained."""
    benchmark = FrozenBenchmark(
        "toy-not-a-public-measurement",
        (
            BenchmarkDocument("d1", "s1", "databases", "postgres index"),
            BenchmarkDocument("d2", "s2", "databases", "linux kernel"),
            BenchmarkDocument("d3", "s3", "databases", "sql slow query"),
        ),
        (
            BenchmarkQuery("q1", "databases", "lexical", "postgres index"),
            BenchmarkQuery("q2", "databases", "paraphrase", "database query performance"),
        ),
        (
            RelevanceJudgment("q1", "d1", 1),
            RelevanceJudgment("q2", "d3", 1),
        ),
        "toy-only",
    )
    config = ComparisonConfig.from_json()
    docs = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]], dtype=np.float32)
    queries = np.array([[1., 0., 0.], [0., 0., 1.]], dtype=np.float32)
    result = compare_benchmark(
        benchmark, docs, queries, config, backend="synthetic_test_vectors",
        model_metadata={"origin": "hand-written deterministic vectors; not pretrained"},
    )
    assert len(result.rankings.bm25) == len(result.rankings.vector) == len(result.rankings.hybrid) == 2
    assert result.rankings.vector["q1"][0] == "d1"
    assert result.rankings.vector["q2"][0] == "d3"
    print("F3.7 — synthetic smoke: BM25 + exact cosine + RRF")
    print("Вектора заданы вручную; это НЕ оценка качества pretrained-модели.")
    print("smoke: PASS")


def real_run(*, download: bool, out_dir: Path, cache_dir: Path | None) -> None:
    """Вычислить сравнение на всех 80 запросах с pinned encoder из F3.2.

    Аргументы:
        download: Только явное True разрешает сеть для скачивания модели.
        out_dir: Выходные UTF-8 отчёт/JSON с реально рассчитанными метриками.
        cache_dir: Необязательный существующий пользовательский кеш.
    """
    spec = ModelSpec.from_json(PROJECT_ROOT / "configs" / "f3_02_model.json")
    config = ComparisonConfig.from_json()
    benchmark = load_frozen_benchmark()
    root = cache_dir if cache_dir is not None else default_cache_dir()
    manifest_path = root.parent / "manifest.json"
    # Реальная модель из F3.2 проверяется по pinned revision и содержимому
    # snapshot. Без --download любые HTTP-запросы запрещены контрактом F3.2.
    encoder, manifest = load_pretrained_encoder(
        spec, cache_dir=root, allow_download=download, manifest_path=manifest_path
    )
    docs = encoder.encode([doc.text for doc in benchmark.documents], batch_size=config.batch_size)
    queries = encoder.encode([query.text for query in benchmark.queries], batch_size=config.batch_size)
    if tuple(docs.shape) != (88, spec.embedding_dim) or tuple(queries.shape) != (80, spec.embedding_dim):
        raise ValueError("Преобразователь вернул embeddings неожиданной размерности")
    # Оригинальные F3.2 тензоры CPU передаются в F3.4 без смены порядка ID.
    doc_matrix = docs.detach().cpu().numpy()
    query_matrix = queries.detach().cpu().numpy()
    metadata = {
        "model_id": spec.model_id,
        "revision": spec.revision,
        "license": spec.license,
        "embedding_dim": spec.embedding_dim,
        "pooling": spec.pooling,
        "normalize": spec.normalize,
        "model_manifest_sha256": hashlib.sha256(
            json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "library_versions": manifest["library_versions"],
        "numpy_version": np.__version__,
    }
    result = compare_benchmark(
        benchmark, doc_matrix, query_matrix, config,
        backend="pretrained_transformer", model_metadata=metadata,
    )
    # Сначала выполнены все запросы, затем рассчитана каждая метрика.
    report, evidence = write_comparison_artifacts(result, out_dir)
    print(f"Benchmark: {benchmark.benchmark_id} ({len(benchmark.queries)} запросов)")
    print(f"Model: {spec.model_id}@{spec.revision}")
    print(f"Config SHA-256: {config.sha256}")
    for method in ("bm25", "vector", "hybrid"):
        row = result.evaluations[method][5]
        print(f"{method:>6}: Recall@5={row.recall_at_k:.4f}, MRR@5={row.mrr_at_k:.4f}")
    print(f"Report: {report}")
    print(f"Evidence: {evidence}")
    print("real benchmark: PASS; synthetic dataset, not production quality")


def main() -> None:
    """Разделить сетевой pretrained-прогон и полностью автономную проверку."""
    parser = argparse.ArgumentParser(description="F3.7 — BM25 / Vector / Hybrid comparison")
    parser.add_argument("--real", action="store_true", help="Сравнить на реальном pinned pretrained encoder F3.2")
    parser.add_argument("--download", action="store_true", help="Разрешить первую загрузку pinned encoder (только с --real)")
    parser.add_argument("--cache-dir", type=Path, help="Пользовательский кеш pretrained weights")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "reports" / "public" / "f3_07",
                        help="Куда сохранить рассчитанный отчёт и полный JSON-аудит")
    args = parser.parse_args()
    if not args.real and (args.download or args.cache_dir is not None):
        parser.error("--download и --cache-dir допустимы только вместе с --real")
    if args.real:
        real_run(download=args.download, out_dir=args.out_dir, cache_dir=args.cache_dir)
    else:
        smoke()


if __name__ == "__main__":
    main()
