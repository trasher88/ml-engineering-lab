"""F5.6: unit-тесты манифеста, provenance, ошибок CLI и JUnit XML."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest

from src.testing_f5.ci_matrix import (
    CONFIG_PATH, REQUIRED_TESTS, MatrixError, load_matrix, profile_provenance, validate_evidence,
)
from scripts import f5_ci

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def matrix_root(tmp_path: Path) -> Path:
    source = load_matrix(ROOT)
    for rel in (CONFIG_PATH, *source.file_hashes):
        destination = tmp_path / rel
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT / rel, destination)
    return tmp_path


def edit_config(root: Path, update) -> None:
    path = root / CONFIG_PATH
    obj = json.loads(path.read_text(encoding='utf-8'))
    update(obj)
    path.write_text(json.dumps(obj, ensure_ascii=False), encoding='utf-8')


def test_real_matrix_is_complete_and_cpu_only() -> None:
    matrix = load_matrix(ROOT)
    assert set(matrix.profiles['regression']) == set(REQUIRED_TESTS)
    assert len(matrix.profiles['unit']) == 4
    assert len(matrix.profiles['integration']) == 3
    assert len(matrix.file_hashes) >= 19
    assert len(matrix.manifest_sha256) == 64
    assert not any('qwen' in name.lower() or 'models/' in name.lower() for name in matrix.input_paths)


def test_sha256_is_exact_and_provenance_is_stable() -> None:
    m = load_matrix(ROOT)
    p = profile_provenance(m,'unit')
    assert p == profile_provenance(load_matrix(ROOT),'unit')
    path = m.profiles['unit'][0]
    assert p['files_sha256'][path] == sha256((ROOT/path).read_bytes()).hexdigest()
    assert sorted(p['files_sha256']) == list(p['files_sha256'])


@pytest.mark.parametrize('profile', ['Unit', '', 'all', '../regression'])
def test_invalid_profile_rejected(profile: str) -> None:
    with pytest.raises(MatrixError):
        profile_provenance(load_matrix(ROOT), profile)


@pytest.mark.parametrize('name', ['../../escape.py', '/tmp/foo.py', 'tests\\test_f5_bad.py',
                                  'tests/test_f5_missing.py', 'tests//test_f5_bad.py',
                                  'tests/test_f5_api.py/../test_f5_api.py', 'C:/oops.py'])
def test_disallowed_test_path_rejected(matrix_root: Path, name: str) -> None:
    edit_config(matrix_root, lambda p: p['profiles']['unit'].__setitem__(0,name))
    with pytest.raises(MatrixError):
        load_matrix(matrix_root)


def test_duplicate_test_rejected(matrix_root: Path) -> None:
    edit_config(matrix_root, lambda p: p['profiles']['unit'].append(p['profiles']['unit'][0]))
    with pytest.raises(MatrixError,match='повторяющиеся'):
        load_matrix(matrix_root)


def test_uncovered_phase_rejected(matrix_root: Path) -> None:
    edit_config(matrix_root, lambda p: p['profiles']['regression'].remove('tests/test_f5_docker.py'))
    with pytest.raises(MatrixError,match='regression'):
        load_matrix(matrix_root)


def test_runtime_source_omission_rejected(matrix_root: Path) -> None:
    edit_config(matrix_root, lambda p: p['required_inputs'].remove('Dockerfile'))
    with pytest.raises(MatrixError, match='Неполный'):
        load_matrix(matrix_root)


def test_runtime_source_missing_rejected(matrix_root: Path) -> None:
    (matrix_root / 'src/api_f5/app.py').unlink()
    with pytest.raises(MatrixError,match='Нет требуемого'):
        load_matrix(matrix_root)


def test_wrong_schema_rejected(matrix_root: Path) -> None:
    edit_config(matrix_root, lambda p: p.__setitem__('schema_version',True))
    with pytest.raises(MatrixError,match='версия'):
        load_matrix(matrix_root)


def good_evidence(root: Path) -> dict:
    return {**profile_provenance(load_matrix(root),'unit'), 'backend':'synthetic_contract',
            'status':'PASS', 'pytest':{'tests':45,'failures':0,'errors':0,'skipped':1,'exit_code':0}}


def test_verified_evidence_accepts_skip_without_fabrication(matrix_root: Path) -> None:
    validate_evidence(load_matrix(matrix_root), good_evidence(matrix_root))


@pytest.mark.parametrize('mutation', [
    lambda e: e['pytest'].__setitem__('tests',0),
    lambda e: e['pytest'].__setitem__('failures',1),
    lambda e: e['pytest'].__setitem__('errors',1),
    lambda e: e['pytest'].__setitem__('skipped',100),
    lambda e: e['pytest'].__setitem__('skipped',45),
    lambda e: e['pytest'].__setitem__('exit_code',1),
    lambda e: e['pytest'].__setitem__('tests',True),
    lambda e: e.__setitem__('status','PARTIAL'),
    lambda e: e.__setitem__('backend','real_qwen'),
    lambda e: e.__setitem__('manifest_sha256','f'*64),
    lambda e: e.__setitem__('test_files',[]),
])
def test_invalid_evidence_rejected(matrix_root: Path, mutation) -> None:
    report = good_evidence(matrix_root)
    mutation(report)
    with pytest.raises(MatrixError):
        validate_evidence(load_matrix(matrix_root), report)


def test_stale_evidence_rejected_after_test_change(matrix_root: Path) -> None:
    evidence = good_evidence(matrix_root)
    with (matrix_root / 'tests/test_f5_api.py').open('a',encoding='utf-8') as file:
        file.write('\n# revised\n')
    # unit profile не включает API; поменять включённый тест
    with (matrix_root / 'tests/test_f5_unified_report.py').open('a',encoding='utf-8') as file:
        file.write('\n# revised\n')
    with pytest.raises(MatrixError,match='files_sha256'):
        validate_evidence(load_matrix(matrix_root), evidence)


def test_junit_counts_valid_and_corrupted_rejected(tmp_path: Path) -> None:
    junit = tmp_path / 'results.xml'
    junit.write_text('<testsuites><testsuite tests="7" failures="1" errors="0" skipped="1"/></testsuites>',encoding='utf-8')
    assert f5_ci._counts(junit) == {'tests':7,'failures':1,'errors':0,'skipped':1}
    junit.write_text('<testsuite tests="1"/>',encoding='utf-8')
    with pytest.raises(MatrixError,match='JUnit'):
        f5_ci._counts(junit)


def test_run_profile_mocked_success_with_saved_evidence(monkeypatch, matrix_root: Path, tmp_path: Path, capsys) -> None:
    def fake_run(cmd, **kwargs):
        xml = Path(cmd[-1])
        xml.write_text('<testsuites><testsuite tests="9" failures="0" errors="0" skipped="1"/></testsuites>',encoding='utf-8')
        assert cmd[:3] == [f5_ci.sys.executable, '-m', 'pytest']
        assert kwargs['cwd'] == matrix_root.resolve()
        assert kwargs['timeout'] == 360
        return SimpleNamespace(returncode=0, stdout='9 passed', stderr='')
    monkeypatch.setattr(f5_ci.subprocess,'run',fake_run)
    evidence_path = tmp_path / 'saved.json'
    assert f5_ci.run_profile(matrix_root,'unit',evidence_path) == 0
    report = json.loads(evidence_path.read_text(encoding='utf-8'))
    validate_evidence(load_matrix(matrix_root),report)
    assert 'F5.6 PASS:' in capsys.readouterr().out


def test_run_profile_mocked_failure_not_pass(monkeypatch, matrix_root: Path, tmp_path: Path, capsys) -> None:
    def fake_run(cmd, **kwargs):
        Path(cmd[-1]).write_text('<testsuite tests="2" failures="1" errors="0" skipped="0"/>',encoding='utf-8')
        return SimpleNamespace(returncode=1,stdout='FAILED one',stderr='')
    monkeypatch.setattr(f5_ci.subprocess,'run',fake_run)
    output=tmp_path/'failure.json'
    assert f5_ci.run_profile(matrix_root,'unit',output) == 1
    assert json.loads(output.read_text(encoding='utf-8'))['status'] == 'FAIL'
    assert 'F5.6 FAIL:' in capsys.readouterr().out


def test_plan_cli_and_missing_evidence(tmp_path: Path,capsys) -> None:
    assert f5_ci.main(['plan']) == 0
    assert 'F5.6 PLAN PASS' in capsys.readouterr().out
    assert f5_ci.main(['check','--evidence',str(tmp_path/'absent.json')]) == 2
    assert 'F5.6 FAIL:' in capsys.readouterr().err


def test_evidence_file_is_optional_by_default(monkeypatch,matrix_root: Path,tmp_path: Path) -> None:
    def fake_run(cmd, **kwargs):
        Path(cmd[-1]).write_text('<testsuite tests="1" failures="0" errors="0" skipped="0"/>',encoding='utf-8')
        return SimpleNamespace(returncode=0, stdout='ok',stderr='')
    monkeypatch.setattr(f5_ci.subprocess,'run',fake_run)
    assert f5_ci.run_profile(matrix_root,'unit') == 0
    assert not (matrix_root/'reports/public/f5_06').exists()
