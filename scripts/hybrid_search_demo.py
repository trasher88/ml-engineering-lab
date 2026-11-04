"""F3.5: демонстрация гибридного BM25 + cosine поиска и RRF.

По умолчанию используются синтетические документы и вручную заданные
векторы — это проверка формул, а не измерение семантической релевантности.
Режим --real использует уже закешированную модель F3.2 без скачивания.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from transformers_f3.bm25 import BM25Index, Document  # noqa: E402
from transformers_f3.hybrid_search import HybridSearcher  # noqa: E402
from transformers_f3.vector_search import VectorIndex  # noqa: E402


def run_synthetic() -> None:
    """Показать объединение списка совпадений и списка близких векторов."""
    documents = [
        Document("lexical", "postgresql медленные запросы"),
        Document("semantic", "оптимизация производительности sql"),
        Document("shared", "postgresql оптимизация запросов"),
        Document("unrelated", "linux системные журналы"),
    ]
    # Векторы подобраны вручную; смысл английских/русских слов эти оси
    # не описывают и никакого качества pretrained-модели не измеряют.
    vectors = np.array([
        [0.7, 0.7],
        [1.0, 0.0],
        [0.9, 0.1],
        [0.0, 1.0],
    ], dtype=np.float64)
    ids = [doc.doc_id for doc in documents]
    sources = ["synthetic:db-guide", "synthetic:sql-guide", "synthetic:db-guide", "synthetic:linux-guide"]
    lexical = BM25Index(documents)
    dense = VectorIndex(ids, vectors, source_ids=sources)
    hybrid = HybridSearcher(lexical, dense, candidate_pool_size=4)
    query = "postgresql медленные запросы"
    embedding = np.array([1.0, 0.0], dtype=np.float64)

    print("F3.5 — BM25 + Vector Search + RRF (synthetic)")
    print(f"docs={len(ids)}; query={query!r}; rrf_k={hybrid.rrf_k}")
    for hit in hybrid.search(query, embedding, top_k=4):
        print(
            f"rank={hit.rank} doc_id={hit.doc_id} source_id={hit.source_id} "
            f"rrf={hit.rrf_score:.8f} bm25_rank={hit.bm25_rank} vector_rank={hit.vector_rank}"
        )
    assert hybrid.search(query, embedding, top_k=4) == hybrid.search(query, embedding, top_k=4)
    print("smoke: PASS; synthetic ranks are not evaluation metrics")


def run_pretrained() -> None:
    """Объединить реальные embeddings из локального кеша F3.2 без сети."""
    from transformers_f3.embeddings import ModelSpec
    from transformers_f3.model_cache import load_pretrained_encoder

    spec = ModelSpec.from_json(PROJECT_ROOT / "configs" / "f3_02_model.json")
    encoder, _manifest = load_pretrained_encoder(spec, allow_download=False)
    texts = [
        "PostgreSQL index and query planning",
        "How to back up a PostgreSQL database",
        "Linux kernel logs and process monitoring",
        "Python generators and list comprehensions",
    ]
    ids = ["db-index", "db-backup", "linux-logs", "python-generators"]
    sources = ["synthetic:database", "synthetic:database", "synthetic:linux", "synthetic:python"]
    question = "How to optimize a PostgreSQL SQL query?"
    documents = [Document(doc_id, text) for doc_id, text in zip(ids, texts)]
    embeddings = encoder.encode(texts, batch_size=2).numpy()
    query_embedding = encoder.encode([question], batch_size=1).numpy()[0]
    hybrid = HybridSearcher(
        BM25Index(documents), VectorIndex(ids, embeddings, source_ids=sources),
        candidate_pool_size=4,
    )
    print("F3.5 — cached pretrained embeddings + BM25 + RRF")
    print(f"model={spec.model_id}@{spec.revision}; query={question}")
    for hit in hybrid.search(question, query_embedding, top_k=4):
        print(
            f"rank={hit.rank} doc_id={hit.doc_id} source_id={hit.source_id} "
            f"rrf={hit.rrf_score:.8f} bm25_rank={hit.bm25_rank} vector_rank={hit.vector_rank}"
        )
    print("smoke: PASS; no independent relevance labels")


def main() -> None:
    """Запустить синтетический пример либо inference уже загруженной модели."""
    parser = argparse.ArgumentParser(description="F3.5 — Hybrid Search + RRF")
    parser.add_argument("--real", action="store_true", help="Использовать локальный кеш F3.2 без скачивания")
    args = parser.parse_args()
    if args.real:
        run_pretrained()
    else:
        run_synthetic()


if __name__ == "__main__":
    main()
