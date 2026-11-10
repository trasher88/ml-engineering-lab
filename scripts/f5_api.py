"""F5.4: локальный FastAPI сервер и настоящий HTTP smoke через TCP.

Запуск из корня: python -m scripts.f5_api serve
Проверка во втором окне: python -m scripts.f5_api smoke
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from urllib.parse import urlsplit

from src.api_f5.app import create_app
from src.api_f5.demo_service import APIFixtureError


class SmokeError(RuntimeError):
    """HTTP-сервис не выполнил подтверждённый контракт F5.4."""


def _request_ok(condition: bool, problem: str) -> None:
    if not condition:
        raise SmokeError(problem)


def run_http_smoke(base_url: str) -> int:
    """Выполнить реальные HTTP-запросы к уже запущенному серверу.

    Не обращается к публичным адресам: разрешён только loopback.
    Не пишет отчёты и не создаёт untracked-файлы.
    """
    import httpx

    parsed = urlsplit(base_url)
    if (parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
            or parsed.username is not None or parsed.password is not None
            or parsed.path not in ("", "/") or parsed.query or parsed.fragment):
        raise SmokeError("smoke принимает только http://127.0.0.1:<port> (loopback)")
    try:
        with httpx.Client(base_url=base_url, timeout=5.0, trust_env=False) as client:
            health = client.get("/health")
            _request_ok(health.status_code == 200, "/health: HTTP != 200")
            h = health.json()
            _request_ok(h["mode"] == "synthetic_demo" and h["documents"] == 12
                        and h["training_examples"] == 24, "/health: неверный backend")

            openapi = client.get("/openapi.json")
            _request_ok(openapi.status_code == 200 and
                        all("post" in openapi.json()["paths"].get(path, {})
                            for path in ("/classify", "/search", "/ask")),
                        "OpenAPI: отсутствуют POST-маршруты")

            classify = client.post("/classify", json={"text": "Python enumerate start индекс"})
            _request_ok(classify.status_code == 200 and classify.json()["label"] == "python"
                        and classify.json()["backend"] == "demo_tfidf_logreg", "/classify: неверный результат")

            search = client.post("/search", json={"query": "SQL HAVING GROUP BY", "top_k": 2})
            _request_ok(search.status_code == 200 and
                        search.json()["hits"][0]["doc_id"] == "sql_02", "/search: неверный BM25")

            ask = client.post("/ask", json={"question": "Python range правая граница"})
            _request_ok(ask.status_code == 200 and ask.json()["mode"] == "extractive_demo"
                        and len(ask.json()["evidence"]) == 1
                        and ask.json()["answer"] == ask.json()["evidence"][0]["quote"],
                        "/ask: нет дословной evidence")

            empty = client.post("/ask", json={"question": "xyzunknowntoken93654"})
            _request_ok(empty.status_code == 200 and empty.json()["mode"] == "no_evidence"
                        and empty.json()["answer"] is None, "/ask: отсутствует no_evidence")

            invalid = client.post("/search", json={"query": "Linux pwd", "top_k": 0})
            _request_ok(invalid.status_code == 422, "/search: отсутствует Pydantic 422")

            unknown = client.get("/no_such_route")
            _request_ok(unknown.status_code == 404, "Неизвестный маршрут: нет 404")
    except (httpx.HTTPError, ValueError, KeyError, IndexError, TypeError) as exc:
        raise SmokeError(f"Ошибка HTTP / схемы: {exc}") from exc
    return 8


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="F5.4: локальный демо-FastAPI, не production")
    subs = parser.add_subparsers(dest="command", required=True)
    server = subs.add_parser("serve", help="Запустить локальный HTTP-сервер")
    server.add_argument("--host", default="127.0.0.1")
    server.add_argument("--port", type=int, default=8000)
    smoke = subs.add_parser("smoke", help="Проверить реальный HTTP-сервер (8 сценариев)")
    smoke.add_argument("--url", default="http://127.0.0.1:8000")
    args = parser.parse_args(argv)
    if args.command == "serve":
        if args.host not in {"127.0.0.1", "localhost", "::1"} or not 1 <= args.port <= 65535:
            parser.error("Для учебного сервера разрешён только loopback и порт 1..65535")
        try:
            app = create_app(Path(__file__).resolve().parents[1])
        except (APIFixtureError, OSError, ValueError) as exc:
            print(f"F5.4 STARTUP FAIL: {exc}", file=sys.stderr)
            return 2
        import uvicorn
        uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
        return 0
    try:
        n = run_http_smoke(args.url)
    except SmokeError as exc:
        print(f"F5.4 HTTP FAIL: {exc}", file=sys.stderr)
        return 2
    print(f"F5.4 HTTP PASS: {n}/{n} checks; backend=synthetic_demo; real TCP HTTP; LLM=NOT EXECUTED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
