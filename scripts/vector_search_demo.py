"""F3.4: офлайн-smoke NumPy vector search и optional cached F3.2 encoder.

Синтетический режим использует вручную заданные векторы, а не реальные
семантические признаки. В режиме --real можно использовать только локально
закешированную и закреплённую по SHA модель F3.2; сеть не вызывается.
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

from transformers_f3.vector_search import VectorIndex  # noqa: E402


def run_synthetic() -> None:
    """Показать точный ранжированный результат на контролируемых векторах."""
    # Вручную созданные координаты: оси являются математическим примером,
    # а не измерением качества языковой модели или настоящей релевантности.
    embeddings = np.array([
        [1.0, 0.0, 0.0],
        [0.6, 0.8, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)
    index = VectorIndex(
        ["db-index-01", "db-backup-02", "linux-01", "python-01"],
        embeddings,
        source_ids=["synthetic:db-guide", "synthetic:db-guide",
                    "synthetic:linux-guide", "synthetic:python-guide"],
    )
    query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    print("F3.4 — NumPy exact vector search (synthetic)")
    print(f"vectors: {index.stats.n_documents}; dimension: {index.stats.embedding_dim}")
    for hit in index.search(query, top_k=3):
        print(f"rank={hit.rank} doc_id={hit.doc_id} source_id={hit.source_id} cosine={hit.score:.6f}")
    assert index.search(query, top_k=1)[0].doc_id == "db-index-01"
    assert index.search(query, top_k=3) == index.search(query, top_k=3)
    print("smoke: PASS; synthetic scores are not relevance probabilities")


def run_pretrained() -> None:
    """Выполнить retrieval с pretrained encoder F3.2, не обращаясь к сети.

    Требуется ранее загруженная фиксированная ревизия модели; предобученный
    inference не создаёт разметку релевантности и не вычисляет Recall@k.
    """
    from transformers_f3.embeddings import ModelSpec
    from transformers_f3.model_cache import load_pretrained_encoder

    spec = ModelSpec.from_json(PROJECT_ROOT / "configs" / "f3_02_model.json")
    encoder, _manifest = load_pretrained_encoder(spec, allow_download=False)
    texts = [
        "PostgreSQL indexing and slow SQL query plans",
        "How to back up PostgreSQL databases",
        "Linux kernel logs and processes",
        "Python comprehensions and iterators",
    ]
    doc_ids = ["db-index-01", "db-backup-02", "linux-01", "python-01"]
    sources = ["synthetic:db-notes", "synthetic:db-notes", "synthetic:linux-notes", "synthetic:python-notes"]
    question = "How can I speed up a PostgreSQL SQL query?"
    # Encoder F3.2 возвращает CPU torch.Tensor: копируем числа в NumPy,
    # не обучая модель и не изменяя frozen tokenizer / weights.
    vectors = encoder.encode(texts, batch_size=2).numpy()
    query = encoder.encode([question], batch_size=1).numpy()[0]
    index = VectorIndex(doc_ids, vectors, source_ids=sources)
    print("F3.4 — cached pretrained encoder + NumPy exact search")
    print(f"model: {spec.model_id}@{spec.revision}")
    print(f"embedding dim: {index.stats.embedding_dim}; query: {question}")
    for hit in index.search(query, top_k=3):
        print(f"rank={hit.rank} doc_id={hit.doc_id} source_id={hit.source_id} cosine={hit.score:.6f}")
    print("smoke: PASS; ranking has no independent relevance labels")


def main() -> None:
    """Выбрать синтетическую демонстрацию или offline pretrained inference."""
    parser = argparse.ArgumentParser(description="F3.4 — NumPy Vector Search")
    parser.add_argument("--real", action="store_true", help="Использовать модель F3.2 из локального кеша без скачивания")
    args = parser.parse_args()
    if args.real:
        run_pretrained()
    else:
        run_synthetic()


if __name__ == "__main__":
    main()
