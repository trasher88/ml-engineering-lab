"""Fix three stale/order-dependent tests in ml-engineering-lab.

Run from the repository root: .\\.venv\\Scripts\\python.exe fix_ml_lab_3_tests.py
Changes only the three tests explicitly named below. No README or docs changes.
"""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path.cwd()

CHANGES: dict[str, list[tuple[str, str]]] = {
    "tests/test_f4_model_selection.py": [
        ('"## 2. Что реализовано"', '"## 2. Реализация: файлы этапа"'),
        ('"## 3. Последовательность реализации"', '"## 3. Последовательность действий"'),
        ('"## 8. Основная мысль этапа"', '"## 8. Основная мысль раздела"'),
        ('"## 9. Файлы, с которыми стоит познакомиться"', '"## 9. Файлы этапа"'),
    ],
    "tests/test_f4_base_model_evaluation.py": [
        ('"## 2. Что реализовано"', '"## 2. Реализация: файлы этапа"'),
    ],
}

OLD_IMPORT_TEST = '''def test_no_language_model_import_needed() -> None:
    import sys
    assert "transformers" not in sys.modules
    assert "torch" not in sys.modules
'''

NEW_IMPORT_TEST = '''def test_no_language_model_import_needed() -> None:
    """Check F5.2 imports in a fresh interpreter, independently of test order."""
    import subprocess
    import sys

    probe = (
        "import sys\\n"
        "import src.evaluation_f5.llm_diagnostics\\n"
        "import scripts.f5_llm_diagnostics\\n"
        "assert 'transformers' not in sys.modules\\n"
        "assert 'torch' not in sys.modules\\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert result.returncode == 0, result.stderr
'''


def replace_once_or_already_done(data: str, old: str, new: str, filename: str) -> str:
    count = data.count(old)
    if count == 1:
        return data.replace(old, new, 1)
    if count == 0 and new in data:
        return data
    raise RuntimeError(
        f"{filename}: expected exactly one old snippet or an already-patched snippet; found {count}. "
        "No files were changed."
    )


def main() -> int:
    updates: dict[Path, str] = {}
    for rel, replacements in CHANGES.items():
        path = ROOT / rel
        original = path.read_text(encoding="utf-8")
        revised = original
        for old, new in replacements:
            revised = replace_once_or_already_done(revised, old, new, rel)
        updates[path] = revised

    rel = "tests/test_f5_llm_diagnostics.py"
    path = ROOT / rel
    data = path.read_text(encoding="utf-8")
    updates[path] = replace_once_or_already_done(data, OLD_IMPORT_TEST, NEW_IMPORT_TEST, rel)

    # Validate the actual documentation before updating the tests.
    doc_requirements = {
        "docs/steps/F4_02_model_selection.md": (
            "## 1. Цель этапа", "## 2. Реализация: файлы этапа",
            "## 3. Последовательность действий", "## 4. Алгоритм и математический пример",
            "## 5. Схема работы", "## 6. Локальная проверка PowerShell",
            "## 7. Что доказано тестами", "## 8. Основная мысль раздела",
            "## 9. Файлы этапа",
        ),
        "docs/steps/F4_04_base_model_evaluation.md": (
            "## 1. Цель этапа", "## 2. Реализация: файлы этапа",
            "## 3. Последовательность реализации", "## 4. Алгоритм и математический пример",
            "## 5. Схема работы", "## 6. Локальная проверка PowerShell",
            "## 7. Что доказано тестами и какие ограничения остаются",
            "## 8. Основная мысль этапа", "## 9. Файлы этапа",
        ),
    }
    for rel, headings in doc_requirements.items():
        content = (ROOT / rel).read_text(encoding="utf-8")
        missing = [heading for heading in headings if heading not in content]
        if missing:
            raise RuntimeError(f"{rel}: current documentation has different headings: {missing!r}. No files changed.")

    for path, revised in updates.items():
        old = path.read_text(encoding="utf-8")
        if revised != old:
            path.write_text(revised, encoding="utf-8")
            print("UPDATED", path.relative_to(ROOT))
        else:
            print("ALREADY OK", path.relative_to(ROOT))
    print("Done. Only three tests were touched; docs and README.md were not modified.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
