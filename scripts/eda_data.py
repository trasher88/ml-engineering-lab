# Назначение файла: F1.5: CLI для статистики и графиков по очищенным вопросам.
# Сохраняет агрегаты, а не строки датасета.

"""F1.5 EDA on locally cleaned data; emits aggregate-only JSON and charts.

Run: python -m scripts.eda_data --expect-rows 3040
The original texts are never logged, serialized to reports or checked in.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

from src.common.io import ROOT, read_jsonl
from src.data.eda import EDAError, length_arrays, summarize_cleaned


# Ошибка чтения данных, расчёта статистики или сохранения EDA.
class EDARunError(EDAError):
    """Bad input, invalid output path or mismatch against expected data."""


# Считает SHA-256 очищенного корпуса для аудита воспроизводимости.
def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


# Записывает JSON-аудит атомарно.
def _write_json_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", newline="\n", dir=path.parent,
            prefix=".eda-", suffix=".tmp", delete=False
        ) as output:
            temp = Path(output.name)
            output.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        os.replace(temp, path)
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)


# Строит диаграмму распределения классов без исходных текстов.
def _plot_bars(data: dict[str, int], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    try:
        x = list(data)
        y = [data[k] for k in x]
        bars = ax.bar(x, y)
        ax.bar_label(bars, padding=3)
        ax.set(title="F1.5 — Class distribution (cleaned corpus)", ylabel="Rows")
        ax.set_ylim(0, max(y) * 1.15)
        fig.tight_layout()
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=160)
    finally:
        plt.close(fig)


# Строит гистограмму длин вопросов и ответов.
def _plot_lengths(questions: list[int], answers: list[int], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    try:
        # Log scale accommodates very long technical questions without hiding
        # the body of the distribution. This is not a per-split plot (F1.6).
        import numpy as np
        all_lengths = questions + answers
        bins = np.geomspace(1, max(all_lengths) + 1, 28)
        ax.hist(questions, bins=bins, alpha=0.55, label="Question")
        ax.hist(answers, bins=bins, alpha=0.55, label="Answer")
        ax.set(xscale="log", xlabel="Whitespace-separated words (log scale)",
               ylabel="Rows", title="F1.5 — Text lengths before split")
        ax.legend()
        fig.tight_layout()
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=160)
    finally:
        plt.close(fig)


# Запускает полный EDA по очищенному корпусу и сохраняет агрегаты.
def run(root: Path = ROOT, *,
        input_path: Path = Path("data/interim/cleaned.jsonl"),
        audit_path: Path = Path("reports/data_eda_audit.json"),
        figures_dir: Path = Path("docs/visuals/f1"),
        expect_rows: int | None = None) -> dict:
    """Read F1.4 results and create reproducible aggregate-only artifacts."""
    root = Path(root)
    source = root / input_path
    report = root / audit_path
    figs = root / figures_dir
    if not source.is_file():
        raise EDARunError(f"missing full F1.4 cleaned corpus: {source}")
    if source.resolve() == report.resolve() or source.resolve().is_relative_to(figs.resolve()):
        raise EDARunError("input and output paths must be different")
    if expect_rows is not None and (type(expect_rows) is not int or expect_rows < 1):
        raise EDARunError("expect_rows must be a positive integer")

    try:
        rows = read_jsonl(source)
    except json.JSONDecodeError as exc:
        raise EDARunError("invalid JSONL in F1.4 output") from exc
    if expect_rows is not None and len(rows) != expect_rows:
        raise EDARunError(f"row count mismatch: expected {expect_rows}, found {len(rows)}")
    summary = summarize_cleaned(rows)
    summary["source_file"] = str(input_path).replace("\\", "/")
    summary["source_sha256"] = _sha256(source)
    summary["plot_files"] = [
        str(figures_dir / "class_distribution.png").replace("\\", "/"),
        str(figures_dir / "text_length.png").replace("\\", "/"),
    ]
    questions, answers = length_arrays(rows)
    # All validation is complete before writing. Generated figures and audit
    # remain local: .gitignore excludes them from the public repository.
    _plot_bars(summary["class_distribution"], figs / "class_distribution.png")
    _plot_lengths(questions, answers, figs / "text_length.png")
    _write_json_atomic(report, summary)
    return summary


# Обрабатывает параметры CLI и печатает результаты EDA.
def main() -> int:
    parser = argparse.ArgumentParser(description="F1.5 — EDA on cleaned technical Q&A")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--input", type=Path, default=Path("data/interim/cleaned.jsonl"))
    parser.add_argument("--audit", type=Path, default=Path("reports/data_eda_audit.json"))
    parser.add_argument("--figures", type=Path, default=Path("docs/visuals/f1"))
    parser.add_argument("--expect-rows", type=int)
    args = parser.parse_args()
    try:
        summary = run(args.root, input_path=args.input, audit_path=args.audit,
                      figures_dir=args.figures, expect_rows=args.expect_rows)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"EDA: FAIL: {exc}")
        return 1
    print(f"EDA: PASS; rows={summary['total_rows']}; "
          f"classes={len(summary['class_distribution'])}; "
          f"source_sha256={summary['source_sha256'][:12]}...")
    print("Class distribution: " + ", ".join(
        f"{name}={count}" for name, count in summary["class_distribution"].items()
    ))
    print("Plots: class_distribution.png, text_length.png (local only)")
    print("Near-duplicate grouping / train-val-test split: NOT RUN (F1.6)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
