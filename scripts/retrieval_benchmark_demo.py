"""F3.6: аудит замороженной синтетической разметки и числовой пример метрик.

Демонстрация не использует BM25, Transformer, Vector Search и сеть.
Реальные показатели качества будут измеряться только на F3.7.
"""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from transformers_f3.retrieval_benchmark import (  # noqa: E402
    BenchmarkDocument,
    BenchmarkQuery,
    FrozenBenchmark,
    RelevanceJudgment,
    evaluate_rankings,
    load_frozen_benchmark,
)


def main() -> None:
    """Проверить 80 вопросов и отдельно показать арифметику на игрушечных ID."""
    benchmark = load_frozen_benchmark()
    print("F3.6 — frozen SYNTHETIC benchmark, not real-world evaluation")
    print(f"benchmark_id: {benchmark.benchmark_id}")
    print(f"manifest_sha256: {benchmark.manifest_sha256}")
    print(f"documents: {len(benchmark.documents)}")
    print(f"queries: {len(benchmark.queries)}")
    print(f"positive_qrels: {len(benchmark.judgments)}")

    # Независимая от frozen-корпуса игрушечная разметка показывает знаменатель
    # Recall@K и позицию первого попадания для MRR@K. Никаких BM25 scores.
    toy = FrozenBenchmark(
        "toy-only", (
            BenchmarkDocument("a", "s1", "databases", "alpha"),
            BenchmarkDocument("b", "s2", "databases", "beta"),
            BenchmarkDocument("c", "s3", "databases", "gamma"),
            BenchmarkDocument("d", "s4", "databases", "delta"),
        ), (
            BenchmarkQuery("q1", "databases", "lexical", "toy query one"),
            BenchmarkQuery("q2", "databases", "paraphrase", "toy query two"),
        ), (
            RelevanceJudgment("q1", "a", 1),
            RelevanceJudgment("q1", "c", 1),
            RelevanceJudgment("q2", "d", 1),
        ), "toy-not-a-frozen-manifest",
    )
    scores = evaluate_rankings(toy, {"q1": ["b", "a", "c"], "q2": ["d", "b"]}, k=2)
    print(f"TOY ONLY Recall@2: {scores.recall_at_k:.4f}")
    print(f"TOY ONLY MRR@2: {scores.mrr_at_k:.4f}")
    assert scores.recall_at_k == 0.75 and scores.mrr_at_k == 0.75
    print("smoke: PASS; retrieval model quality has not been measured")


if __name__ == "__main__":
    main()
