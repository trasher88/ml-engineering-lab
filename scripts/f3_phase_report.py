"""F3.9: аудит сохранённого сравнения F3.7 и отчёт Фазы 3.

По умолчанию читает реальный JSON F3.7 и создаёт два воспроизводимых
артефакта. Режим --smoke проверяет только синтетический контракт F3.8:
он не подменяет и не записывает реальные retrieval-метрики.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from transformers_f3.phase_report import check_rag_smoke, write_phase_report  # noqa: E402


def main() -> None:
    """Разграничить автономную проверку и реальный финальный отчёт."""
    parser = argparse.ArgumentParser(description="F3.9 — verified phase report from pretrained F3.7 evidence")
    parser.add_argument("--smoke", action="store_true", help="Только синтетический RAG/provenance smoke без отчёта")
    parser.add_argument("--comparison-json", type=Path,
                        default=ROOT / "reports" / "public" / "f3_07" / "F3_07_comparison.json",
                        help="JSON, полученный в F3.7 с --real")
    parser.add_argument("--report-md", type=Path, default=ROOT / "docs" / "reports" / "F3_PHASE_REPORT.md")
    parser.add_argument("--audit-json", type=Path, default=ROOT / "reports" / "public" / "f3_09" / "F3_PHASE_EVIDENCE.json")
    args = parser.parse_args()
    if args.smoke:
        checks = check_rag_smoke()
        print("F3.9 — синтетический smoke F3.8:")
        for name in checks:
            print(f"  {name}: PASS")
        print("smoke: PASS; реальные retrieval-метрики не рассчитывались")
        return
    try:
        md, data, audit = write_phase_report(
            args.comparison_json, report_md=args.report_md, audit_json=args.audit_json, root=ROOT
        )
    except (OSError, ValueError, TypeError) as exc:
        parser.exit(1, f"F3.9 FAIL — отчёт не опубликован: {exc}\n")
    print(f"F3.9 — frozen benchmark: {audit.n_queries} queries, {audit.n_documents} documents")
    for method in ("bm25", "vector", "hybrid"):
        m = audit.metrics[method]["5"]
        print(f"{method:>6}: Recall@5={m['recall_at_k']:.4f}, MRR@5={m['mrr_at_k']:.4f}")
    print(f"Отчёт: {md}")
    print(f"Аудит: {data}")
    print("phase report: PASS; benchmark synthetic, production quality not established")


if __name__ == "__main__":
    main()
