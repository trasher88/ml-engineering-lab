"""F5.5: реальный Docker run + проверка HTTP F5.4 и cleanup контейнера.

Требует установленный Docker CLI/daemon, собранный image и локальный Python
с зависимостями F5.4. В обычном CI без Docker не выполняется автоматически.

Пример из корня проекта:
    python -m scripts.f5_docker_smoke --image ml-engineering-lab:f5.5
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from uuid import uuid4

from scripts.f5_api import SmokeError, run_http_smoke


class DockerSmokeError(RuntimeError):
    """Не удалось подтвердить исполняемый контракт контейнера."""


_IMAGE_PATTERN = re.compile(r"[a-z0-9][a-z0-9./_-]*:[a-z0-9][a-z0-9._-]*\Z")


def _docker(*args: str, timeout: int = 30) -> str:
    """Выполнить команду Docker через аргументы без shell/interpolation."""
    try:
        result = subprocess.run(
            ["docker", *args], text=True, capture_output=True,
            encoding="utf-8", errors="replace", timeout=timeout, check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise DockerSmokeError(f"Docker CLI недоступен или timeout: {exc}") from exc
    if result.returncode != 0:
        details = (result.stderr or result.stdout).strip()[-900:]
        raise DockerSmokeError(f"docker {args[0]}: exit {result.returncode}: {details}")
    return result.stdout.strip()


def run_container_smoke(*, image: str, port: int = 18055, timeout: int = 90) -> int:
    """Подтвердить запуск из готового image и 8 HTTP-сценариев, затем удалить контейнер.

    При ошибке run/smoke/health выполняется принудительный cleanup. Не оставляет
    долгоживущего контейнера и не публикует порт на внешних интерфейсах.
    """
    if not _IMAGE_PATTERN.fullmatch(image) or len(image) > 128:
        raise DockerSmokeError("Нужен image в формате repository:tag, например ml-engineering-lab:f5.5")
    if not (isinstance(port, int) and not isinstance(port, bool) and 1024 <= port <= 65535):
        raise DockerSmokeError("Порт хоста должен быть в диапазоне 1024..65535")
    if not isinstance(timeout, int) or not 10 <= timeout <= 600:
        raise DockerSmokeError("Timeout должен быть в диапазоне 10..600 секунд")

    _docker("version", "--format", "{{.Server.Version}}")
    _docker("image", "inspect", image, "--format", "{{.Id}}")
    container = f"ml-lab-f5-05-{os.getpid()}-{uuid4().hex[:8]}"
    started = False
    error: Exception | None = None
    try:
        _docker(
            "run", "--detach", "--rm", "--name", container,
            "--publish", f"127.0.0.1:{port}:8000",
            "--read-only", "--tmpfs", "/tmp:rw,nosuid,size=64m",
            "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
            "--memory", "1g", "--pids-limit", "256",
            image, timeout=60,
        )
        started = True
        deadline = time.monotonic() + timeout
        health = "starting"
        while time.monotonic() < deadline:
            health = _docker("inspect", "--format", "{{.State.Health.Status}}", container)
            if health == "healthy":
                break
            if health == "unhealthy":
                raise DockerSmokeError("Docker healthcheck=unhealthy")
            state = _docker("inspect", "--format", "{{.State.Running}}", container)
            if state != "true":
                raise DockerSmokeError("Контейнер завершился до успешного healthcheck")
            time.sleep(1)
        if health != "healthy":
            raise DockerSmokeError(f"Ожидание HEALTHCHECK истекло: {health}")

        user_id = _docker("exec", container, "id", "-u")
        if not user_id.isdecimal() or int(user_id) == 0:
            raise DockerSmokeError(f"Контейнер работает не под non-root: uid={user_id!r}")
        try:
            checks = run_http_smoke(f"http://127.0.0.1:{port}")
        except SmokeError as exc:
            raise DockerSmokeError(f"HTTP smoke F5.4: {exc}") from exc
        if checks != 8:
            raise DockerSmokeError(f"Неверное количество HTTP-проверок: {checks}")
    except Exception as exc:
        error = exc
        raise
    finally:
        if started:
            try:
                _docker("stop", "--time", "3", container, timeout=20)
            except DockerSmokeError as cleanup_exc:
                # Если обычная остановка невозможна, попытаться принудительно удалить.
                try:
                    _docker("rm", "--force", container, timeout=20)
                except DockerSmokeError as force_exc:
                    problem = DockerSmokeError(
                        f"Cleanup контейнера не выполнен: stop={cleanup_exc}; rm={force_exc}"
                    )
                    if error is None:
                        raise problem from force_exc
                    print(f"F5.5 CLEANUP WARNING: {problem}", file=sys.stderr)
    print(f"F5.5 DOCKER PASS: image={image}; health=healthy; HTTP=8/8; "
          f"uid={user_id}; backend=synthetic_demo; LLM=NOT EXECUTED")
    return checks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="F5.5 Docker inference: реальный запуск и HTTP smoke")
    parser.add_argument("--image", default="ml-engineering-lab:f5.5")
    parser.add_argument("--port", type=int, default=18055)
    parser.add_argument("--timeout", type=int, default=90)
    args = parser.parse_args(argv)
    try:
        run_container_smoke(image=args.image, port=args.port, timeout=args.timeout)
    except DockerSmokeError as exc:
        print(f"F5.5 DOCKER NOT VERIFIED / FAIL: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
