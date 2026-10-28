"""F2.8: собрать публичный итог Фазы 2 из локальных агрегатов F2.6/F2.7.

Команда не обучает модели, не открывает исходные Q&A и не копирует ID.
Изменение README выполняется точечно, без перезаписи остального файла.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.pytorch_f2.final_report import (
    PhaseReportError, build_from_artifacts, sha256_report,
    update_readme, validate_comparison, validate_diagnostics,
    verify_source_fingerprints,
)


def _self_check() -> None:
    """Проверить механизмы F2.8 на маленьких безопасных числовых фикстурах."""
    metrics = {"accuracy": 1.0, "macro_f1": 0.5, "n": 1,
               "labels": ["linux", "systems"], "confusion_matrix": [[0, 0], [0, 1]]}
    comparison = {
        "source": "synthetic", "split_counts": {"train": 4, "val": 1, "test": 1},
        "split_sha256": {"train": "synthetic-train", "val": "synthetic-val", "test": "synthetic-test"},
        "comparison_invariants_pass": True,
        "protocol": {"test_used_for_selection": False, "classical_fit_split": "train",
                     "neural_selection_split": "val", "neural_selection_metric": "macro_f1",
                     "neural_best_epoch": 1, "neural_validation_macro_f1": 0.5},
        "comparison": {"samples": 1, "labels": ["linux", "systems"],
                       "classical": metrics, "neural": metrics,
                       "delta_neural_minus_classical": {"accuracy": 0.0, "macro_f1": 0.0},
                       "winning_model_by_macro_f1": "tie",
                       "paired_errors": {"both_correct": 1, "both_wrong": 0,
                                         "only_classical_wrong": 0,
                                         "only_neural_wrong": 0,
                                         "prediction_disagreements": 0}},
        "complexity": {"neural_parameter_count": 2978},
    }
    diagnostic = {
        "format_version": 1, "source": "synthetic", "trained_split": "train",
        "selection_split": "val", "test_evaluated": False,
        "selection_metric": "val_macro_f1",
        "split_counts": {"train": 4, "val": 1, "test": 1},
        "epochs": 1, "best_epoch": 1, "best_val_macro_f1": 0.5,
        "history": [{"epoch": 1, "train_loss": 0.75, "val_loss": 0.72,
                     "val_accuracy": 1.0, "val_macro_f1": 0.5}],
        "interpretation": {"signal": "insufficient_epochs"},
    }
    validate_comparison(comparison, require_real=False)
    validate_diagnostics(diagnostic, require_real=False)
    current = "# ML Engineering Lab\n\nПлан последовательного раскрытия модулей: пример\n"
    updated = update_readme(current)
    if update_readme(updated) != updated or "F2_PHASE_REPORT.md" not in updated:
        raise PhaseReportError("README patch is not idempotent")


def run(*, root: Path, update_navigation: bool = False) -> dict:
    """Проверить два локальных результата и записать итоговую публичную версию.

    Проверка завершится ошибкой, если результаты не из реального split F1,
    отсутствуют метрики F2.6/7 или нарушены их инварианты. При ошибке
    нечего публиковать, поскольку синтетические метрики не подменяют реальные.
    """
    root = Path(root)
    markdown = build_from_artifacts(
        root / "outputs" / "f2_6_diagnostics",
        root / "outputs" / "f2_7_comparison" / "comparison.json",
        require_real=True,
    )
    artifact = json.loads((root / "outputs" / "f2_7_comparison" / "comparison.json").read_text(encoding="utf-8"))
    verify_source_fingerprints(root, artifact)
    readme = root / "README.md"
    changed_readme = None
    original_readme_bytes = None
    if update_navigation:
        original_readme_bytes = readme.read_bytes()
        has_bom = original_readme_bytes.startswith(b"\xef\xbb\xbf")
        original = original_readme_bytes.decode("utf-8-sig" if has_bom else "utf-8")
        line_ending = "\r\n" if "\r\n" in original else "\n"
        patched = update_readme(original.replace("\r\n", "\n"))
        changed_readme = ((b"\xef\xbb\xbf" if has_bom else b"") +
                          patched.replace("\n", line_ending).encode("utf-8"))
    report = root / "docs" / "reports" / "F2_PHASE_REPORT.md"
    if not report.parent.is_dir():
        raise PhaseReportError("missing docs/reports directory")
    report.write_text(markdown, encoding="utf-8", newline="\n")
    if changed_readme is not None:
        # Не меняем кодировку/BOM и окончания строк в остальной части README.
        if original_readme_bytes != changed_readme:
            readme.write_bytes(changed_readme)
    if report.read_text(encoding="utf-8") != markdown:
        raise PhaseReportError("F2 report round-trip failed")
    return {"report": str(report), "report_sha256": sha256_report(markdown),
            "readme_updated": update_navigation,
            "data_source": "local_F2_6_and_F2_7_aggregates", "no_training": True}


def main() -> int:
    """Выполнить audit/generation или быстрый синтетический smoke test."""
    parser = argparse.ArgumentParser(description="F2.8: проверяемый итог фазы PyTorch")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--update-readme", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    try:
        if args.self_check:
            _self_check()
            result = {"source": "synthetic", "self_check": True, "no_training": True}
        else:
            result = run(root=args.root, update_navigation=args.update_readme)
    except (PhaseReportError, OSError, ValueError, UnicodeError) as exc:
        print(f"F2.8 Final Phase Report CHECK: FAIL ({exc})")
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.check:
        print("F2.8 Final Phase Report CHECK: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
