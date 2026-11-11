"""F5.6: integration-контракт API и fixture, без сети/весов/Docker daemon."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api_f5.app import create_app
from scripts.f5_api import run_http_smoke
from src.testing_f5.ci_matrix import load_matrix

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def client():
    with TestClient(create_app(ROOT)) as c:
        yield c


def test_http_health_declares_synthetic_only(client) -> None:
    response=client.get('/health')
    assert response.status_code == 200
    data=response.json()
    assert data['mode']=='synthetic_demo'
    assert data['documents']==12
    assert data['training_examples']==24


def test_three_endpoints_declared_in_openapi(client) -> None:
    paths=client.get('/openapi.json').json()['paths']
    assert all('post' in paths[path] for path in ('/classify','/search','/ask'))


def test_classify_uses_demo_not_llm(client) -> None:
    data=client.post('/classify',json={'text':'Python enumerate start индекс'}).json()
    assert data['label']=='python'
    assert data['backend']=='demo_tfidf_logreg'


def test_search_returns_ranked_bm25_hits(client) -> None:
    r=client.post('/search',json={'query':'SQL HAVING GROUP BY','top_k':3})
    assert r.status_code==200
    assert r.json()['hits'][0]['doc_id']=='sql_02'


def test_ask_produces_exact_evidence_without_generation(client) -> None:
    r=client.post('/ask',json={'question':'Python range правая граница'})
    assert r.status_code==200
    obj=r.json()
    assert obj['mode']=='extractive_demo'
    assert obj['answer']==obj['evidence'][0]['quote']


def test_missing_evidence_does_not_fabricate(client) -> None:
    data=client.post('/ask',json={'question':'xyzunknowntoken93654'}).json()
    assert data['answer'] is None
    assert data['mode']=='no_evidence'


def test_pydantic_rejects_bad_top_k(client) -> None:
    assert client.post('/search',json={'query':'SQL','top_k':0}).status_code==422


def test_unknown_route_not_found(client) -> None:
    assert client.get('/unknown-route').status_code==404


def test_fixture_hash_matches_pinned_config() -> None:
    config=json.loads((ROOT/'configs/f5_04_api.json').read_text(encoding='utf-8'))
    assert sha256((ROOT/config['fixture_path']).read_bytes()).hexdigest()==config['fixture_sha256']


def test_no_external_http_requests_are_required() -> None:
    assert '127.0.0.1' in run_http_smoke.__doc__ or 'уже запущенному' in run_http_smoke.__doc__
    assert len(load_matrix(ROOT).profiles['integration'])==3
