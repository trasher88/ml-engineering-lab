"""F5.4: типизированный FastAPI-контракт над синтетическими CPU-бэкендами.

Публичные маршруты не запускают F1/F2 training, pretrained F3 encoder или Qwen.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from .demo_service import DemoEngine, load_demo


InputText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=500)]


class StrictInput(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ClassifyRequest(StrictInput):
    text: InputText


class ClassifyResponse(BaseModel):
    label: str | None
    score: float | None = Field(default=None, ge=0.0, le=1.0)
    status: str
    backend: str
    fixture_id: str


class SearchRequest(StrictInput):
    query: InputText
    top_k: int = Field(default=3, ge=1, le=5, strict=True)


class SearchHit(BaseModel):
    doc_id: str
    source_id: str
    label: str
    score: float = Field(gt=0)
    text: str


class SearchResponse(BaseModel):
    backend: str
    fixture_id: str
    hits: list[SearchHit]


class AskRequest(StrictInput):
    question: InputText


class Evidence(BaseModel):
    doc_id: str
    source_id: str
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    quote: str
    text_sha256: str


class AskResponse(BaseModel):
    mode: str
    answer: str | None
    evidence: list[Evidence]
    backend: str
    fixture_id: str


class HealthResponse(BaseModel):
    status: str
    mode: str
    fixture_id: str
    fixture_sha256: str
    documents: int
    training_examples: int


def create_app(root: Path | None = None) -> FastAPI:
    """Построить готовое demo-приложение; ошибки конфигурации не скрывать.

    Точное содержимое fixture подтверждается SHA-256 при создании приложения.
    Никаких пользовательских путей и загрузок весов в HTTP-запросах нет.
    """
    repo_root = root if root is not None else Path(__file__).resolve().parents[2]
    cfg, fixture = load_demo(repo_root)
    engine = DemoEngine(cfg, fixture)
    app = FastAPI(
        title="ML Engineering Lab — F5.4 Demo API",
        version="0.1.0",
        description=(
            "Офлайн-контракт HTTP поверх синтетических учебных данных: "
            "TF-IDF+LogReg, BM25 и дословный extractive answer. "
            "Не является F1/F3 trained inference, Qwen или production сервисом."
        ),
        docs_url="/docs", redoc_url=None,
    )

    @app.get("/health", response_model=HealthResponse, tags=["service"])
    def health() -> dict[str, Any]:
        return {"status": "ready", "mode": cfg["mode"], "fixture_id": engine.fixture_id,
                "fixture_sha256": engine.fixture_sha256, "documents": len(engine.docs),
                "training_examples": engine.training_examples}

    @app.post("/classify", response_model=ClassifyResponse, tags=["demo"])
    def classify(payload: ClassifyRequest) -> dict[str, Any]:
        """Классификация ТОЛЬКО по 24 synthetic-обучающим примерам."""
        return engine.classify(payload.text)

    @app.post("/search", response_model=SearchResponse, tags=["demo"])
    def search(payload: SearchRequest) -> dict[str, Any]:
        """BM25 top-k по 12 разрешённым синтетическим документам."""
        matches = engine.search(payload.query, payload.top_k)
        return {
            "backend": "demo_bm25_exact", "fixture_id": engine.fixture_id,
            "hits": [
                {"doc_id": hit.doc.doc_id, "source_id": hit.doc.source_id,
                 "label": hit.doc.label, "score": hit.score, "text": hit.doc.text}
                for hit in matches
            ],
        }

    @app.post("/ask", response_model=AskResponse, tags=["demo"])
    def ask(payload: AskRequest) -> dict[str, Any]:
        """Ответ не генерируется LLM: только дословное предложение из BM25-hit."""
        return engine.ask(payload.question)

    return app
