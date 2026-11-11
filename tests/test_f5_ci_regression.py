"""F5.6: сквозные инварианты публикационных контрактов F5.1–F5.5."""

from __future__ import annotations

import json
from pathlib import Path

from src.testing_f5.ci_matrix import load_matrix
from src.api_f5.app import create_app
from fastapi.testclient import TestClient

ROOT=Path(__file__).resolve().parents[1]


def test_f5_phase_files_are_retained() -> None:
    needed=(
       'docs/steps/F5_01_unified_evaluation.md',
       'docs/steps/F5_02_llm_diagnostics.md',
       'docs/steps/F5_03_safety_benchmark.md',
       'docs/steps/F5_04_fastapi.md',
       'docs/steps/F5_05_docker_inference.md',
    )
    assert all((ROOT/p).is_file() for p in needed)


def test_safety_contract_remains_not_real_model() -> None:
    # Результат reference-rule-gate не следует интерпретировать как real LLM.
    data=json.loads((ROOT/'reports/public/f5_03/F5_03_safety_evidence.json').read_text(encoding='utf-8'))
    assert data['status']=='PARTIAL'
    assert data['contract_status']=='PASS'
    assert data['metrics']['overall']['cases']==28
    assert data['run']['external_origin_verified'] is False


def test_llm_sources_not_embedded_in_api_demo() -> None:
    config=json.loads((ROOT/'configs/f5_04_api.json').read_text(encoding='utf-8'))
    assert config['mode']=='synthetic_demo'
    assert config['expected_training_examples']==24
    assert config['expected_documents']==12
    with TestClient(create_app(ROOT)) as client:
        health=client.get('/health').json()
        assert health['mode']=='synthetic_demo'


def test_ci_matrix_uses_only_f5_test_sources() -> None:
    matrix=load_matrix(ROOT)
    assert sorted(matrix.file_hashes)==sorted(set(matrix.file_hashes))
    assert all('/test_f5_' in file for file in matrix.profiles['regression'])
    assert all((ROOT/p).is_file() for p in matrix.file_hashes)


def test_docker_smoke_is_explicitly_opt_in() -> None:
    txt=(ROOT/'scripts/f5_docker_smoke.py').read_text(encoding='utf-8')
    assert 'run_container_smoke' in txt
    workflow=(ROOT/'.github/workflows/f5_06_ci.yml').read_text(encoding='utf-8')
    assert 'docker build' not in workflow
    assert 'docker run' not in workflow
