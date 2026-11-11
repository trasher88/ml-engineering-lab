"""F5.6 — типизированная схема проверок и provenance набора тестов.

Не запускает обучение, Docker или сетевые запросы. Исполняет только
список pytest-файлов, который явно сохранён в JSON-конфигурации.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
from typing import Any

CONFIG_PATH = "configs/f5_06_test_matrix.json"
PROFILES = ("unit", "integration", "regression")
# Явный контроль покрытия: новый regression-профиль не может забыть фазу.
REQUIRED_INPUTS = (
    "src/api_f5/app.py",
    "src/api_f5/demo_service.py",
    "src/evaluation_f5/unified_report.py",
    "src/evaluation_f5/llm_diagnostics.py",
    "src/evaluation_f5/safety_benchmark.py",
    "configs/f5_04_api.json",
    "tests/fixtures/f5_04_api_demo_v1.json",
    "tests/fixtures/f5_03_safety_cases_v1.json",
    "requirements-f5_04.txt",
    "requirements-f5_05.txt",
    "Dockerfile",
    "reports/public/f5_03/F5_03_safety_evidence.json",
    "docs/steps/F5_01_unified_evaluation.md",
    "docs/steps/F5_02_llm_diagnostics.md",
    "docs/steps/F5_03_safety_benchmark.md",
    "docs/steps/F5_04_fastapi.md",
    "docs/steps/F5_05_docker_inference.md",
)

REQUIRED_TESTS = (
    "tests/test_f5_unified_report.py",
    "tests/test_f5_llm_diagnostics.py",
    "tests/test_f5_safety_benchmark.py",
    "tests/test_f5_api.py",
    "tests/test_f5_docker.py",
    "tests/test_f5_ci_contract.py",
    "tests/test_f5_ci_integration.py",
    "tests/test_f5_ci_regression.py",
)


class MatrixError(ValueError):
    """Ошибка схемы, нарушенная граница репозитория или пропавший тест."""


@dataclass(frozen=True)
class Matrix:
    """Проверенный манифест и перечень исходников для воспроизведения."""
    root: Path
    profiles: dict[str, tuple[str, ...]]
    input_paths: tuple[str, ...]
    manifest_sha256: str
    file_hashes: dict[str, str]


def _relpath(root: Path, raw: Any, prefix: str | None = None) -> str:
    if not isinstance(raw, str) or not raw or "\\" in raw or "//" in raw or raw.strip() != raw:
        raise MatrixError(f"Некорректный относительный путь: {raw!r}")
    path = PurePosixPath(raw)
    if path.is_absolute() or any(part in (".", "..") for part in raw.split("/")) or ":" in raw:
        raise MatrixError(f"Путь за пределами проекта: {raw!r}")
    if prefix is not None and (not raw.startswith(prefix) or not raw.endswith(".py")):
        raise MatrixError(f"Путь теста вне разрешённого каталога: {raw!r}")
    target = root / raw
    if not target.is_file() or not target.resolve().is_relative_to(root.resolve()):
        raise MatrixError(f"Нет требуемого файла внутри репозитория: {raw}")
    return raw


def _strings(value: Any, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value or any(not isinstance(x, str) for x in value):
        raise MatrixError(f"{name}: ожидается непустой список строк")
    if len(value) != len(set(value)):
        raise MatrixError(f"{name}: повторяющиеся пути")
    return tuple(value)


def load_matrix(root: Path) -> Matrix:
    """Проверить содержимое конфигурации и SHA-256 всех зависимостей."""
    root = root.resolve()
    config = root / CONFIG_PATH
    try:
        data = config.read_bytes()
        payload = json.loads(data)
    except (OSError, ValueError) as exc:
        raise MatrixError(f"Не удалось прочитать {CONFIG_PATH}: {exc}") from exc
    if not isinstance(payload, dict) or set(payload) != {"schema_version", "project", "scope", "python_minor", "profiles", "required_inputs"}:
        raise MatrixError("Неизвестная схема конфигурации F5.6")
    if (type(payload['schema_version']) is not int or payload['schema_version'] != 1
            or payload['project'] != 'ml-engineering-lab' or payload['python_minor'] != '3.13'
            or not isinstance(payload['scope'], str) or not payload['scope']):
        raise MatrixError("Неверная версия схемы / Python / проекта")
    profiles = payload['profiles']
    if not isinstance(profiles, dict) or set(profiles) != set(PROFILES):
        raise MatrixError("Набор профилей должен содержать unit/integration/regression")
    checked = {
        name: tuple(_relpath(root, value, 'tests/test_f5_') for value in _strings(profiles[name], name))
        for name in PROFILES
    }
    if set(checked['regression']) != set(REQUIRED_TESTS):
        raise MatrixError("regression обязан покрывать все восемь файлов F5.1–F5.6")
    if not (set(checked['unit']) | set(checked['integration'])) <= set(checked['regression']):
        raise MatrixError("unit/integration не входят в regression")
    if set(checked['unit']) & set(checked['integration']):
        raise MatrixError("Один тестовый файл указан сразу в двух узких профилях")
    if set(checked['unit']) | set(checked['integration']) != set(REQUIRED_TESTS) - {'tests/test_f5_ci_regression.py'}:
        raise MatrixError("Профили unit/integration не охватывают заданные тесты")
    inputs = tuple(_relpath(root, p) for p in _strings(payload['required_inputs'], 'required_inputs'))
    if set(inputs) != set(REQUIRED_INPUTS):
        raise MatrixError("Неполный или неверный перечень runtime-входов")
    relative = sorted(set(inputs) | set(REQUIRED_TESTS))
    file_hashes = {path: sha256((root/path).read_bytes()).hexdigest() for path in relative}
    return Matrix(root, checked, inputs, sha256(data).hexdigest(), file_hashes)


def profile_provenance(matrix: Matrix, profile: str) -> dict[str, Any]:
    if profile not in PROFILES:
        raise MatrixError(f"Неизвестный профиль: {profile}")
    tests = matrix.profiles[profile]
    return {
        'schema_version':1,
        'profile':profile,
        'python_minor':'3.13',
        'manifest_sha256':matrix.manifest_sha256,
        'files_sha256':{path:matrix.file_hashes[path] for path in sorted(set(tests) | set(matrix.input_paths))},
        'test_files':list(tests),
    }


def validate_evidence(matrix: Matrix, report: Any) -> None:
    """Проверить атрибуцию набора тестов и отсутствие устаревших артефактов."""
    if not isinstance(report, dict) or report.get('schema_version') != 1:
        raise MatrixError('Неизвестный формат evidence')
    source = profile_provenance(matrix, report.get('profile', ''))
    for field in source:
        if report.get(field) != source[field]:
            raise MatrixError(f"Evidence устарел или изменён: {field}")
    counts = report.get('pytest', {})
    if (not isinstance(counts, dict) or
            any(type(counts.get(k)) is not int or counts[k] < 0
                for k in ('tests','failures','errors','skipped'))):
        raise MatrixError('Некорректные результаты pytest')
    if (counts['tests'] < 1 or counts['failures'] or counts['errors'] or
            counts['skipped'] >= counts['tests'] or
            type(counts.get('exit_code')) is not int or counts['exit_code'] != 0):
        raise MatrixError('Есть ошибки или нет выполненных тестов')
    if report.get('status') != 'PASS' or report.get('backend') != 'synthetic_contract':
        raise MatrixError('Нет подтверждённого PASS контракта')
