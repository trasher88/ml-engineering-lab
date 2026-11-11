"""F5.6 — план, выполнение и проверка тестовых профилей F5.

Команды (из корня):
    python -m scripts.f5_ci plan
    python -m scripts.f5_ci run --profile regression
    python -m scripts.f5_ci http-smoke

По умолчанию никакие результаты не создаются в репозитории.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
from urllib.error import URLError
from urllib.request import urlopen
import xml.etree.ElementTree as ET

from src.testing_f5.ci_matrix import MatrixError, load_matrix, profile_provenance, validate_evidence

ROOT = Path(__file__).resolve().parents[1]


def _counts(junit: Path) -> dict[str, int]:
    try:
        xml = ET.parse(junit).getroot()
        suites = [xml] if xml.tag == 'testsuite' else list(xml.findall('testsuite'))
        if not suites:
            raise ValueError('Отсутствуют testsuite')
        result = {name:0 for name in ('tests','failures','errors','skipped')}
        for suite in suites:
            for field in result:
                result[field] += int(suite.attrib[field])
    except (ET.ParseError, OSError, ValueError, KeyError, TypeError) as exc:
        raise MatrixError(f"Некорректный pytest JUnit XML: {exc}") from exc
    if result['tests'] <= 0 or sum(result[k] for k in ('failures','errors','skipped')) > result['tests']:
        raise MatrixError('Несогласованные счётчики pytest JUnit XML')
    return result


def run_profile(root: Path, profile: str, output: Path | None = None) -> int:
    """Запустить выбранные тесты; сохранить SHA-привязанный результат по запросу."""
    matrix = load_matrix(root)
    source = profile_provenance(matrix, profile)
    if sys.version_info[:2] != (3, 13):
        raise MatrixError('Для воспроизведения F5.6 требуется Python 3.13')
    with tempfile.TemporaryDirectory(prefix='f5_06_pytest_') as temp:
        xml = Path(temp)/'results.xml'
        args = [sys.executable, '-m', 'pytest', *matrix.profiles[profile], '-q',
                '--disable-warnings', '--junitxml', str(xml)]
        try:
            completed = subprocess.run(args, cwd=matrix.root, capture_output=True,
                                       text=True, encoding='utf-8', errors='replace',
                                       timeout=360, check=False)
        except subprocess.TimeoutExpired as exc:
            raise MatrixError('Тестовый профиль превысил 360 секунд') from exc
        if completed.stdout:
            print(completed.stdout.rstrip())
        if completed.stderr:
            print(completed.stderr.rstrip(), file=sys.stderr)
        counts = _counts(xml) if xml.exists() else dict(tests=0, failures=1, errors=0, skipped=0)
        passed = (completed.returncode == 0 and counts['tests'] > 0 and
                  counts['failures'] == 0 and counts['errors'] == 0 and
                  counts['skipped'] < counts['tests'])
        report = {**source, 'backend':'synthetic_contract', 'pytest':{**counts, 'exit_code':completed.returncode},
                  'status':'PASS' if passed else 'FAIL'}
        if output is not None:
            dest = output.resolve()
            dest.parent.mkdir(parents=True,exist_ok=True)
            dest.write_text(json.dumps(report,ensure_ascii=False,sort_keys=True,indent=2)+'\n', encoding='utf-8',newline='\n')
        print(f"F5.6 {report['status']}: profile={profile}; tests={counts['tests']}; "
              f"failures={counts['failures']}; errors={counts['errors']}; "
              f"skipped={counts['skipped']}; model=NOT EXECUTED; Docker=NOT EXECUTED")
        return 0 if passed else 1


def http_smoke(root: Path, timeout: float = 30.0) -> int:
    """Настоящий локальный TCP-smoke: отдельный Uvicorn-процесс + 8 HTTP-запросов."""
    if not 5 <= timeout <= 120:
        raise MatrixError('timeout должен находиться в диапазоне 5..120 секунд')
    load_matrix(root)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(('127.0.0.1', 0))
        port = listener.getsockname()[1]
    base_url = f'http://127.0.0.1:{port}'
    cmd = [sys.executable, '-m', 'scripts.f5_api', 'serve', '--host', '127.0.0.1', '--port', str(port)]
    with tempfile.TemporaryFile(mode='w+t', encoding='utf-8') as log:
        process = subprocess.Popen(cmd, cwd=root, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + timeout
            ready = False
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    break
                try:
                    with urlopen(base_url + '/health', timeout=0.6) as response:
                        if response.status == 200:
                            ready = True
                            break
                except (OSError, URLError, TimeoutError):
                    time.sleep(0.15)
            if not ready:
                log.seek(0)
                raise MatrixError('Uvicorn не запустился: ' + log.read(1200))
            from scripts.f5_api import run_http_smoke
            n = run_http_smoke(base_url)
            if n != 8:
                raise MatrixError(f'Неполный HTTP smoke: {n}/8')
            print('F5.6 HTTP PASS: TCP=8/8; backend=synthetic_demo; LLM=NOT EXECUTED')
            return n
        finally:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='F5.6: CPU-first тесты и CI evidence')
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('plan', help='Проверить список и SHA исходников без выполнения тестов')
    run = sub.add_parser('run', help='Запустить выбранный набор pytest')
    run.add_argument('--profile', choices=('unit','integration','regression'), default='regression')
    run.add_argument('--output', type=Path, help='Необязательный путь для JSON evidence, лучше вне Git')
    check = sub.add_parser('check', help='Сверить provenance и статусы сохранённого evidence')
    check.add_argument('--evidence', required=True, type=Path)
    smoke = sub.add_parser('http-smoke', help='Запустить HTTP сервер и 8 реальных TCP-проверок')
    smoke.add_argument('--timeout', type=float, default=30.0)
    options = parser.parse_args(argv)
    try:
        if options.command == 'http-smoke':
            http_smoke(ROOT, options.timeout)
            return 0
        matrix = load_matrix(ROOT)
        if options.command == 'plan':
            for name, paths in matrix.profiles.items():
                print(f"{name}: {len(paths)} test files; {', '.join(paths)}")
            print(f"F5.6 PLAN PASS: manifest_sha256={matrix.manifest_sha256}; "
                  f"tracked_source_files={len(matrix.file_hashes)}")
            return 0
        if options.command == 'check':
            report = json.loads(options.evidence.read_text(encoding='utf-8'))
            validate_evidence(matrix, report)
            print(f"F5.6 EVIDENCE PASS: profile={report['profile']}; tests={report['pytest']['tests']}")
            return 0
        return run_profile(ROOT, options.profile, options.output)
    except (MatrixError, OSError, ValueError, RuntimeError) as exc:
        print(f"F5.6 FAIL: {exc}", file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
