"""F5.5: контракты образа и smoke runner без ложного запуска Docker в CI.

Обычные тесты не обращаются к Docker CLI и не скачивают базовый образ.
Настоящая интеграция проверяется отдельно: scripts.f5_docker_smoke.
"""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import json
import shutil
import subprocess
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from scripts import f5_docker_smoke as f5
from src.api_f5.app import create_app


ROOT = Path(__file__).resolve().parents[1]
RUNTIME = (
    "src/api_f5/__init__.py",
    "src/api_f5/app.py",
    "src/api_f5/demo_service.py",
    "configs/f5_04_api.json",
    "tests/fixtures/f5_04_api_demo_v1.json",
)


def test_dockerfile_has_narrow_runtime_and_no_broad_copy() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "FROM python:3.13-slim-bookworm" in dockerfile
    assert "COPY . ." not in dockerfile
    assert "COPY src/ " not in dockerfile
    assert "USER app" in dockerfile
    assert "EXPOSE 8000" in dockerfile
    assert '"--factory"' in dockerfile
    assert '"src.api_f5.app:create_app"' in dockerfile
    assert '"0.0.0.0"' in dockerfile
    assert "HEALTHCHECK " in dockerfile
    assert "127.0.0.1:8000/health" in dockerfile
    assert "PYTHONDONTWRITEBYTECODE=1" in dockerfile
    assert "OPENBLAS_NUM_THREADS=1" in dockerfile
    assert "requirements-f5_05.txt" in dockerfile
    for item in RUNTIME:
        assert (ROOT / item).is_file(), item
    assert "COPY configs/f5_04_api.json" in dockerfile
    assert "COPY tests/fixtures/f5_04_api_demo_v1.json" in dockerfile


def test_context_deny_by_default_and_required_allowlist() -> None:
    lines = [x.strip() for x in (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
             if x.strip() and not x.lstrip().startswith("#")]
    assert lines[0] == "**"
    required = {"!Dockerfile", "!.dockerignore", "!requirements-f5_05.txt",
                "!src/", "!src/api_f5/", "!configs/", "!tests/", "!tests/fixtures/"}
    required |= {"!" + path for path in RUNTIME}
    assert required.issubset(set(lines))
    assert all(line == "**" or line.startswith("!") for line in lines)
    assert all("README" not in x and "mlruns" not in x for x in lines)
    assert not any(x.startswith("!.") and x not in {"!.dockerignore"} for x in lines)


def test_runtime_dependencies_are_only_pinned_cpu_packages() -> None:
    lines = [x.strip() for x in (ROOT / "requirements-f5_05.txt").read_text(encoding="utf-8").splitlines()
             if x.strip() and not x.startswith("#")]
    assert lines == ["fastapi==0.128.2", "uvicorn==0.48.0", "scikit-learn==1.8.0"]
    assert not any("torch" in x or "transformers" in x or "pytest" in x for x in lines)


def test_container_fixture_sha_is_exact_f5_04_source() -> None:
    config = json.loads((ROOT / "configs/f5_04_api.json").read_text(encoding="utf-8"))
    assert config["mode"] == "synthetic_demo"
    assert config["expected_training_examples"] == 24
    assert config["expected_documents"] == 12
    payload = (ROOT / config["fixture_path"]).read_bytes()
    assert sha256(payload).hexdigest() == config["fixture_sha256"]


def test_minimal_runtime_context_supports_real_f5_04_cpu_inference(tmp_path: Path) -> None:
    """Проверить только файлы, попадающие в образ, без полного репозитория."""
    for rel in RUNTIME:
        dst = tmp_path / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / rel, dst)
    with TestClient(create_app(tmp_path)) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["mode"] == "synthetic_demo"
        assert health.json()["training_examples"] == 24
        assert health.json()["documents"] == 12
        assert client.get("/openapi.json").status_code == 200
        classification = client.post("/classify", json={"text": "Python enumerate start индекс"})
        assert classification.status_code == 200
        assert classification.json()["label"] == "python"
        search = client.post("/search", json={"query": "SQL HAVING GROUP BY", "top_k": 2})
        assert search.status_code == 200
        assert search.json()["hits"][0]["doc_id"] == "sql_02"
        ask = client.post("/ask", json={"question": "Python range правая граница"})
        assert ask.status_code == 200
        assert ask.json()["mode"] == "extractive_demo"
        assert ask.json()["answer"] == ask.json()["evidence"][0]["quote"]
        assert client.post("/search", json={"query": "SQL", "top_k": 0}).status_code == 422
        assert client.get("/missing").status_code == 404


@pytest.mark.parametrize("image", ["", "image", "../x:tag", "x:tag;rm", "UPPER:tag", "x:tag with spaces", "a" * 130 + ":tag"])
def test_reject_bad_image_names_before_invoking_docker(image: str) -> None:
    with pytest.raises(f5.DockerSmokeError, match="image"):
        f5.run_container_smoke(image=image)


@pytest.mark.parametrize("port", [0, 1023, 65536, True])
def test_reject_bad_ports_before_invoking_docker(port) -> None:
    with pytest.raises(f5.DockerSmokeError, match="Порт"):
        f5.run_container_smoke(image="ml-engineering-lab:f5.5", port=port)


@pytest.mark.parametrize("timeout", [0, 9, 601, -1])
def test_reject_bad_timeout_before_invoking_docker(timeout: int) -> None:
    with pytest.raises(f5.DockerSmokeError, match="Timeout"):
        f5.run_container_smoke(image="ml-engineering-lab:f5.5", timeout=timeout)


def test_docker_command_is_argv_only(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = Mock(return_value=SimpleNamespace(returncode=0, stdout="28.0.0\n", stderr=""))
    monkeypatch.setattr(f5.subprocess, "run", fake)
    assert f5._docker("version", "--format", "{{.Server.Version}}") == "28.0.0"
    args, kwargs = fake.call_args
    assert args[0] == ["docker", "version", "--format", "{{.Server.Version}}"]
    assert "shell" not in kwargs
    assert kwargs["timeout"] == 30


def test_docker_command_reports_process_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(f5.subprocess, "run", Mock(return_value=SimpleNamespace(returncode=125, stdout="", stderr="bad daemon")))
    with pytest.raises(f5.DockerSmokeError, match="exit 125: bad daemon"):
        f5._docker("version")


def test_docker_command_reports_missing_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(f5.subprocess, "run", Mock(side_effect=FileNotFoundError("docker")))
    with pytest.raises(f5.DockerSmokeError, match="Docker CLI недоступен"):
        f5._docker("version")


def test_docker_command_reports_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(f5.subprocess, "run", Mock(side_effect=subprocess.TimeoutExpired("docker", 1)))
    with pytest.raises(f5.DockerSmokeError, match="timeout"):
        f5._docker("image", "inspect", "ml-engineering-lab:f5.5")


def test_real_docker_sequence_mock_requires_health_and_cleans_up(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    called: list[tuple[str, ...]] = []
    health = iter(("starting", "healthy"))

    def fake_docker(*args: str, **kwargs) -> str:
        called.append(args)
        if args[:2] == ("inspect", "--format") and args[2] == "{{.State.Health.Status}}":
            return next(health)
        if args[:2] == ("inspect", "--format") and args[2] == "{{.State.Running}}":
            return "true"
        if args[0] == "exec":
            return "999"
        return "sha256:fake"

    monkeypatch.setattr(f5, "_docker", fake_docker)
    monkeypatch.setattr(f5, "run_http_smoke", lambda url: 8)
    monkeypatch.setattr(f5.time, "sleep", lambda delay: None)
    assert f5.run_container_smoke(image="ml-engineering-lab:f5.5", port=18055, timeout=10) == 8
    assert "F5.5 DOCKER PASS:" in capsys.readouterr().out
    run = next(x for x in called if x[0] == "run")
    assert "127.0.0.1:18055:8000" in run
    assert "--read-only" in run and "--cap-drop" in run
    assert called[-1][0] == "stop"


@pytest.mark.parametrize("failure", ["unhealthy", "stopped", "uid0", "http"])
def test_failed_runtime_never_claims_pass_and_always_stops(
    monkeypatch: pytest.MonkeyPatch, capsys, failure: str
) -> None:
    calls: list[str] = []

    def fake_docker(*args: str, **kwargs) -> str:
        calls.append(args[0])
        if args[:3] == ("inspect", "--format", "{{.State.Health.Status}}"):
            return "unhealthy" if failure == "unhealthy" else ("starting" if failure == "stopped" else "healthy")
        if args[:3] == ("inspect", "--format", "{{.State.Running}}"):
            return "false" if failure == "stopped" else "true"
        if args[0] == "exec":
            return "0" if failure == "uid0" else "999"
        return "sha256:fake"

    def fake_http(url: str) -> int:
        if failure == "http":
            raise f5.SmokeError("contract mismatch")
        return 8

    monkeypatch.setattr(f5, "_docker", fake_docker)
    monkeypatch.setattr(f5, "run_http_smoke", fake_http)
    with pytest.raises(f5.DockerSmokeError):
        f5.run_container_smoke(image="ml-engineering-lab:f5.5", timeout=10)
    assert "stop" in calls
    assert "DOCKER PASS" not in capsys.readouterr().out


def test_cli_reports_missing_docker_as_not_verified(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    monkeypatch.setattr(f5, "_docker", lambda *args, **kwargs: (_ for _ in ()).throw(f5.DockerSmokeError("no docker")))
    assert f5.main(["--image", "ml-engineering-lab:f5.5"]) == 2
    assert "NOT VERIFIED / FAIL" in capsys.readouterr().err


def test_cli_reports_success_only_when_run_verified(monkeypatch: pytest.MonkeyPatch) -> None:
    values: list[dict] = []
    def fake_run(**kwargs):
        values.append(kwargs)
        return 8
    monkeypatch.setattr(f5, "run_container_smoke", fake_run)
    assert f5.main(["--image", "ml-engineering-lab:f5.5", "--port", "18080", "--timeout", "50"]) == 0
    assert values == [{"image": "ml-engineering-lab:f5.5", "port": 18080, "timeout": 50}]


def test_stop_failure_attempts_force_removal(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    calls: list[str] = []
    def fake_docker(*args: str, **kwargs) -> str:
        calls.append(args[0])
        if args[0] == "stop":
            raise f5.DockerSmokeError("cannot stop")
        if args[:3] == ("inspect", "--format", "{{.State.Health.Status}}"):
            return "healthy"
        if args[0] == "exec":
            return "10001"
        return "ok"
    monkeypatch.setattr(f5, "_docker", fake_docker)
    monkeypatch.setattr(f5, "run_http_smoke", lambda url: 8)
    assert f5.run_container_smoke(image="ml-engineering-lab:f5.5") == 8
    assert calls[-2:] == ["stop", "rm"]
    assert "DOCKER PASS" in capsys.readouterr().out


def test_both_cleanup_paths_failing_never_prints_pass(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    def fake_docker(*args: str, **kwargs) -> str:
        if args[0] in {"stop", "rm"}:
            raise f5.DockerSmokeError("cleanup unavailable")
        if args[:3] == ("inspect", "--format", "{{.State.Health.Status}}"):
            return "healthy"
        if args[0] == "exec":
            return "10001"
        return "ok"
    monkeypatch.setattr(f5, "_docker", fake_docker)
    monkeypatch.setattr(f5, "run_http_smoke", lambda url: 8)
    with pytest.raises(f5.DockerSmokeError, match="Cleanup контейнера не выполнен"):
        f5.run_container_smoke(image="ml-engineering-lab:f5.5")
    assert "DOCKER PASS" not in capsys.readouterr().out
