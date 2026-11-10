"""F5.4: контракт FastAPI, реальные demo-вычисления и HTTP smoke через TCP."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time

from fastapi.testclient import TestClient
import httpx
import pytest

from scripts.f5_api import SmokeError, main, run_http_smoke
from src.api_f5.app import create_app
from src.api_f5.demo_service import APIFixtureError, DemoEngine, load_demo, tokens


ROOT = Path(__file__).resolve().parents[1]
CFG = "configs/f5_04_api.json"
DATA = "tests/fixtures/f5_04_api_demo_v1.json"


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(create_app(ROOT))


def local_root(tmp_path: Path) -> Path:
    for rel in (CFG, DATA):
        target = tmp_path / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, target)
    return tmp_path


def mutate_fixture(tmp_path: Path, mutation) -> Path:
    root = local_root(tmp_path)
    fpath = root / DATA
    data = json.loads(fpath.read_text(encoding="utf-8"))
    mutation(data)
    fpath.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    cfg_file = root / CFG
    cfg = json.loads(cfg_file.read_text(encoding="utf-8"))
    cfg["fixture_sha256"] = sha256(fpath.read_bytes()).hexdigest()
    cfg_file.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return root


def test_health(client: TestClient) -> None:
    result = client.get("/health")
    assert result.status_code == 200
    assert result.json() == {
        "status": "ready", "mode": "synthetic_demo", "fixture_id": "f5_04_api_synthetic_v1",
        "fixture_sha256": sha256((ROOT / DATA).read_bytes()).hexdigest(),
        "documents": 12, "training_examples": 24,
    }


def test_openapi_contract(client: TestClient) -> None:
    resp = client.get("/openapi.json")
    assert resp.status_code == 200
    schema = resp.json()
    assert schema["info"]["title"] == "ML Engineering Lab — F5.4 Demo API"
    assert "синтетических" in schema["info"]["description"].lower()
    assert all("post" in schema["paths"][path] for path in ("/classify", "/search", "/ask"))
    for req in ("ClassifyRequest", "SearchRequest", "AskRequest"):
        assert req in schema["components"]["schemas"]
    assert client.get("/docs").status_code == 200
    assert client.get("/redoc").status_code == 404


@pytest.mark.parametrize("body,expected", [
    ({"text": "Python enumerate start индекс"}, "python"),
    ({"text": "Linux chmod права доступа"}, "linux"),
    ({"text": "SQL SELECT WHERE таблица"}, "sql"),
    ({"text": "ML embeddings векторное представление"}, "ml"),
])
def test_classification_real_tfidf_logreg(client: TestClient, body: dict, expected: str) -> None:
    res = client.post("/classify", json=body)
    assert res.status_code == 200
    info = res.json()
    assert info["status"] == "predicted"
    assert info["label"] == expected
    assert info["backend"] == "demo_tfidf_logreg"
    assert 0.0 <= info["score"] <= 1.0


def test_classification_no_demo_features(client: TestClient) -> None:
    res = client.post("/classify", json={"text": "xyzunknowntoken93654"})
    assert res.status_code == 200
    assert res.json()["status"] == "no_features"
    assert res.json()["label"] is None and res.json()["score"] is None


def test_classification_deterministic(client: TestClient) -> None:
    body = {"text": "Python range список"}
    assert client.post("/classify", json=body).json() == client.post("/classify", json=body).json()


@pytest.mark.parametrize("url,body", [
    ("/classify", {"text": ""}),
    ("/classify", {"text": "    "}),
    ("/classify", {"text": "x" * 501}),
    ("/classify", {"text": 123}),
    ("/classify", {"text": "Python", "unexpected": 1}),
    ("/classify", {}),
    ("/search", {"query": ""}),
    ("/search", {"query": "pwd", "top_k": 0}),
    ("/search", {"query": "pwd", "top_k": 6}),
    ("/search", {"query": "pwd", "top_k": True}),
    ("/search", {"query": "pwd", "top_k": 1.5}),
    ("/search", {"query": "pwd", "top_k": "2"}),
    ("/search", {"query": "pwd", "x": 1}),
    ("/ask", {"question": "  "}),
    ("/ask", {"question": ["what"]}),
    ("/ask", {"question": "x" * 501}),
    ("/ask", {"question": "python", "unknown": "bad"}),
])
def test_bad_inputs_return_422(client: TestClient, url: str, body: dict) -> None:
    assert client.post(url, json=body).status_code == 422


def test_invalid_json_and_content_type(client: TestClient) -> None:
    assert client.post("/classify", content=b'{"text":', headers={"Content-Type": "application/json"}).status_code == 422
    assert client.post("/classify", content=b"text=Python", headers={"Content-Type": "text/plain"}).status_code == 422


def test_bad_method_and_unknown_route(client: TestClient) -> None:
    assert client.get("/classify").status_code == 405
    assert client.post("/missing", json={}).status_code == 404


def test_bm25_search_ranking_and_metadata(client: TestClient) -> None:
    response = client.post("/search", json={"query": "SQL HAVING GROUP BY", "top_k": 2})
    assert response.status_code == 200
    result = response.json()
    assert result["backend"] == "demo_bm25_exact"
    hits = result["hits"]
    assert hits[0]["doc_id"] == "sql_02"
    assert hits[0]["source_id"] == "synthetic://sql/groupby"
    assert all(hit["score"] > 0 for hit in hits)
    assert hits[0]["score"] > hits[1]["score"]
    assert result == client.post("/search", json={"query": "SQL HAVING GROUP BY", "top_k": 2}).json()


def test_bm25_topk_and_no_hits(client: TestClient) -> None:
    result = client.post("/search", json={"query": "python", "top_k": 1}).json()
    assert len(result["hits"]) == 1
    assert client.post("/search", json={"query": "xyzunknowntoken93654"}).json()["hits"] == []


def test_bm25_algorithm_idf_tf_and_normalization() -> None:
    cfg, data = load_demo(ROOT)
    engine = DemoEngine(cfg, data)
    doc = engine.docs[0]
    term = "pwd"
    freq = engine.doc_tokens[0][term]
    df = engine.df[term]
    n = len(engine.docs)
    import math
    expected = (math.log1p((n - df + 0.5) / (df + 0.5)) *
                freq * (1.5 + 1) /
                (freq + 1.5 * (1 - 0.75 + 0.75 * engine.lengths[0] / engine.average_length)))
    hits = engine.search(term, 5)
    assert hits[0].doc == doc
    assert hits[0].score == pytest.approx(expected, abs=1e-12)
    assert tokens("SQL SQL Python") == ["sql", "sql", "python"]


def test_ask_quote_matches_actual_source_bytes(client: TestClient) -> None:
    resp = client.post("/ask", json={"question": "Python range правая граница"})
    assert resp.status_code == 200
    result = resp.json()
    assert result["mode"] == "extractive_demo"
    assert result["backend"] == "demo_extractive_bm25"
    assert result["answer"] == result["evidence"][0]["quote"]
    evidence = result["evidence"][0]
    _, fixture = load_demo(ROOT)
    original = next(d["text"] for d in fixture["documents"] if d["doc_id"] == evidence["doc_id"])
    assert original[evidence["start"]:evidence["end"]] == evidence["quote"]
    assert evidence["text_sha256"] == sha256(original.encode("utf-8")).hexdigest()
    assert evidence["source_id"].startswith("synthetic://")


def test_ask_no_evidence(client: TestClient) -> None:
    res = client.post("/ask", json={"question": "xyzunknowntoken93654"})
    assert res.status_code == 200
    assert res.json()["mode"] == "no_evidence"
    assert res.json()["answer"] is None and res.json()["evidence"] == []


def test_ask_does_not_echo_injected_instruction(client: TestClient) -> None:
    test_text = "FAKE_INJECT_IGNORE_ALL. Python range правая граница"
    res = client.post("/ask", json={"question": test_text})
    assert res.status_code == 200
    assert "FAKE_INJECT" not in json.dumps(res.json())
    assert res.json()["answer"] == res.json()["evidence"][0]["quote"]


def test_immutable_fixture_sha256(tmp_path: Path) -> None:
    local_root(tmp_path)
    f = tmp_path / DATA
    f.write_bytes(f.read_bytes() + b" ")
    with pytest.raises(APIFixtureError, match="SHA-256"):
        create_app(tmp_path)


def test_config_wrong_schema(tmp_path: Path) -> None:
    local_root(tmp_path)
    p = tmp_path / CFG
    cfg = json.loads(p.read_text(encoding="utf-8"))
    cfg["evil_key"] = 123
    p.write_text(json.dumps(cfg), encoding="utf-8")
    with pytest.raises(APIFixtureError, match="схема"):
        load_demo(tmp_path)


def test_reject_config_path_change(tmp_path: Path) -> None:
    local_root(tmp_path)
    p = tmp_path / CFG
    cfg = json.loads(p.read_text(encoding="utf-8"))
    cfg["fixture_path"] = "../../outside.json"
    p.write_text(json.dumps(cfg), encoding="utf-8")
    with pytest.raises(APIFixtureError, match="протокол"):
        create_app(tmp_path)


def test_reject_duplicate_json_key(tmp_path: Path) -> None:
    local_root(tmp_path)
    config = tmp_path / CFG
    raw = config.read_text(encoding="utf-8")
    config.write_text(raw.replace('"schema_version": 1,', '"schema_version": 1, "schema_version": 1,'), encoding="utf-8")
    with pytest.raises(APIFixtureError, match="Повторён JSON-ключ"):
        create_app(tmp_path)


def test_reject_nonfinite_json_constant(tmp_path: Path) -> None:
    local_root(tmp_path)
    config = tmp_path / CFG
    raw = config.read_text(encoding="utf-8")
    config.write_text(raw.replace('"schema_version": 1,', '"schema_version": NaN,'), encoding="utf-8")
    with pytest.raises(APIFixtureError, match="JSON-константа"):
        create_app(tmp_path)


def test_reject_duplicate_documents(tmp_path: Path) -> None:
    def mutate(data: dict) -> None:
        data["documents"][1]["doc_id"] = data["documents"][0]["doc_id"]
    root = mutate_fixture(tmp_path, mutate)
    with pytest.raises(APIFixtureError, match="документ"):
        load_demo(root)


def test_reject_training_balance_change(tmp_path: Path) -> None:
    def mutate(data: dict) -> None:
        data["training_examples"][0]["label"] = "sql"
    root = mutate_fixture(tmp_path, mutate)
    with pytest.raises(APIFixtureError, match="несбалансирована"):
        load_demo(root)


def test_fixture_reproducible_between_app_instances() -> None:
    with TestClient(create_app(ROOT)) as a, TestClient(create_app(ROOT)) as b:
        for url, payload in (("/classify", {"text": "Linux chmod файл"}),
                             ("/search", {"query": "SQL WHERE"}),
                             ("/ask", {"question": "Python zip длина"})):
            assert a.post(url, json=payload).json() == b.post(url, json=payload).json()


def test_concurrent_repeated_queries(client: TestClient) -> None:
    def call(_: int) -> dict:
        return client.post("/search", json={"query": "Linux pwd"}).json()
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(call, range(12)))
    assert results and all(row == results[0] for row in results)


@pytest.mark.parametrize("url", [
    "https://127.0.0.1:8000", "http://example.com:8000", "http://127.0.0.1:8000/other",
    "http://127.0.0.1:8000?key=abc", "http://user:pass@127.0.0.1:8000",
])
def test_smoke_rejects_non_loopback_or_noncanonical_url(url: str) -> None:
    with pytest.raises(SmokeError, match="loopback"):
        run_http_smoke(url)


def test_cli_server_arg_validation() -> None:
    with pytest.raises(SystemExit) as exc:
        main(["serve", "--host", "0.0.0.0"])
    assert exc.value.code == 2
    with pytest.raises(SystemExit) as exc:
        main(["serve", "--port", "0"])
    assert exc.value.code == 2


def test_cli_smoke_without_server_returns_nonzero() -> None:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    assert main(["smoke", "--url", f"http://127.0.0.1:{port}"]) == 2


def test_real_tcp_http_smoke() -> None:
    """Отдельный процесс Uvicorn + реальный socket + проверка 8 HTTP-контрактов."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    proc = subprocess.Popen(
        [sys.executable, "-m", "scripts.f5_api", "serve", "--port", str(port)],
        cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
    )
    try:
        url = f"http://127.0.0.1:{port}"
        for _ in range(100):
            if proc.poll() is not None:
                pytest.fail(f"Сервер завершился: {proc.stderr.read()[:1000]}")
            try:
                with httpx.Client(timeout=0.25, trust_env=False) as client:
                    response = client.get(url + "/health")
                if response.status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            time.sleep(0.05)
        else:
            pytest.fail("Сервер не открыл TCP 127.0.0.1 до истечения окна проверки")
        assert run_http_smoke(url) == 8
    finally:
        proc.terminate()
        try:
            proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate(timeout=5)
