"""F3.3: воспроизводимый офлайн-smoke инвертированного индекса и BM25.

Только синтетические технические документы: нет файлового ввода,
необученных или pretrained моделей, сетевого доступа и оценки Recall@K.
"""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from transformers_f3.bm25 import BM25Index, Document  # noqa: E402


def main() -> None:
    """Индексировать учебные тексты и показать top-k и статистику корпуса."""
    documents = [
        Document("doc-db-01", "PostgreSQL slow query explain analyze and index"),
        Document("doc-db-02", "PostgreSQL backup restore database"),
        Document("doc-linux-01", "Linux kernel logs and memory allocation"),
        Document("doc-python-01", "Python list comprehension and iterators"),
        Document("doc-db-03", "SQL query plan and PostgreSQL index tuning"),
    ]
    query = "postgresql slow query index"
    index = BM25Index(documents, k1=1.5, b=0.75)
    stats = index.stats
    print("F3.3 — BM25 lexical retrieval (синтетический корпус)")
    print(f"documents: {stats.n_documents}; tokens: {stats.total_tokens}; "
          f"vocabulary: {stats.vocabulary_size}; avgdl: {stats.avgdl:.3f}")
    print(f"query: {query!r}")
    print(f"df(postgresql): {index.document_frequency('postgresql')}")
    print(f"idf(postgresql): {index.idf('postgresql'):.6f}")
    print(f"postings(index): {index.postings('index')}")
    hits = index.search(query, top_k=3)
    for hit in hits:
        print(f"rank={hit.rank} doc_id={hit.doc_id} bm25={hit.score:.6f}")
    assert hits and hits[0].doc_id == "doc-db-01", "Синтетический эталон поиска изменился"
    assert hits == index.search(query, top_k=3), "Ранжирование должно быть повторяемым"
    assert index.search("unknown_nonexistent_term") == [], "OOV-запрос не должен давать выдачу"
    print("smoke: PASS; BM25 score — не вероятность релевантности")


if __name__ == "__main__":
    main()
