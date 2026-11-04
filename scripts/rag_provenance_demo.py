"""F3.8: офлайн-демонстрация RAG, ссылок и резервного извлечения цитат.

Используются только синтетические технические заметки. Для hybrid
заданы ручные векторы; ни LLM, ни pretrained encoder не скачиваются.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from transformers_f3.rag_provenance import (  # noqa: E402
    GenerationDraft, GenerationRequest, RAGConfig, RAGPipeline, SourceDocument,
    build_generation_prompt,
)
from transformers_f3.vector_search import VectorIndex  # noqa: E402


def main() -> None:
    """Показать извлечение цитаты, provenance и отказ от галлюцинации."""
    parser = argparse.ArgumentParser(description="F3.8: RAG & provenance (offline)")
    parser.add_argument("--hybrid", action="store_true", help="Использовать ручные векторы и RRF")
    parser.add_argument("--real", action="store_true", help="Использовать закешированный encoder F3.2 без скачивания")
    parser.add_argument("--generator-demo", action="store_true", help="Подать намеренно неподтверждённый draft")
    args = parser.parse_args()

    documents = [
        SourceDocument("db-plan", "synthetic:postgres", "PostgreSQL EXPLAIN ANALYZE shows the execution plan. Check index usage for slow queries."),
        SourceDocument("db-index", "synthetic:postgres", "A B-tree index can speed up selective equality filters in PostgreSQL."),
        SourceDocument("linux-log", "synthetic:linux", "journalctl shows logs for systemd services."),
    ]
    config = RAGConfig.from_json()
    question = "PostgreSQL slow queries index"
    if args.real:
        from transformers_f3.embeddings import ModelSpec
        from transformers_f3.model_cache import load_pretrained_encoder

        # Только явно подготовленный локальный кеш: сеть запрещена контрактом F3.2.
        spec = ModelSpec.from_json(PROJECT_ROOT / "configs" / "f3_02_model.json")
        encoder, _manifest = load_pretrained_encoder(spec, allow_download=False)
        vectors = encoder.encode([item.text for item in documents], batch_size=2).detach().cpu().numpy()
        embedding = encoder.encode([question], batch_size=1).detach().cpu().numpy()[0]
        config = replace(config, retrieval_mode="hybrid")
    elif args.hybrid:
        config = replace(config, retrieval_mode="hybrid")
        # Векторы вручную: исключительно интерфейсный тест, не семантический поиск.
        vectors = np.array([[1., 0.], [0.9, 0.1], [0., 1.]], dtype=np.float64)
        embedding = np.array([1., 0.], dtype=np.float64)
    else:
        vectors = None
        embedding = None

    if vectors is not None:
        index = VectorIndex(
            [item.doc_id for item in documents], vectors,
            source_ids=[item.source_id for item in documents],
        )
    else:
        index = None

    def fictional_generator(_request):
        """Вернуть непроверенное заявление, которое обязано быть отклонено."""
        return GenerationDraft("PostgreSQL всегда ускоряется в 100 раз.", "[1]")

    pipeline = RAGPipeline(
        documents, config=config, vector_index=index,
        generator=fictional_generator if args.generator_demo else None,
    )
    result = pipeline.answer(question, query_embedding=embedding)
    assert result.mode == "extractive_fallback"
    assert result.evidence and all(pipeline.verify_evidence(item) for item in result.evidence)
    assert result.answer.endswith("[1]")
    assert "100 раз" not in result.answer

    print("F3.8 — RAG & source provenance (synthetic corpus, CPU-only)")
    print("mode:", result.mode, "reason:", result.reason)
    print("retrieval:", result.retrieval_mode, "doc_ids:", result.retrieved_doc_ids)
    print("answer:", result.answer)
    for item in result.evidence:
        print(f"{item.citation_id} doc_id={item.doc_id} source_id={item.source_id} span=[{item.start}:{item.end}] sha256={item.document_sha256[:12]}...")
    prompt = build_generation_prompt(GenerationRequest(question, result.evidence))
    assert "untrusted_evidence" in prompt
    print("smoke: PASS")


if __name__ == "__main__":
    main()
