"""F5.4: небольшие реальные вычисления поверх проверенного synthetic corpus.

ВНИМАНИЕ: модели здесь обучены только на демонстрационных записях.
Они НЕ являются моделями F1/F2, а индекс НЕ является frozen F3.6 benchmark.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import re
from typing import Any


class APIFixtureError(ValueError):
    """Невалидный demo-набор, схема или идентичность версии данных."""


_WORD = re.compile(r"(?u)\b\w+\b")
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_ID = re.compile(r"[a-z][a-z0-9_]{1,79}\Z")
_MAX_BYTES = 64_000


def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Не допускать тихой замены полей при дублировании JSON-ключей."""
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise APIFixtureError(f"Повторён JSON-ключ: {key}")
        result[key] = value
    return result


def _invalid_constant(value: str) -> None:
    raise APIFixtureError(f"Недопустимая JSON-константа: {value}")


def tokens(text: str) -> list[str]:
    """Нормализовать строки для лексического retrieval (без стемминга)."""
    return _WORD.findall(text.casefold())


def load_demo(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Загрузить и проверить фиксированную конфигурацию/fixture без network I/O.

    Нельзя передать пользовательский путь через HTTP; пути фиксированы и
    проверены на выход через symlink, traversal и несоответствие SHA-256.
    """
    root = root.resolve()
    cfg_path = (root / "configs/f5_04_api.json").resolve()
    if not cfg_path.is_relative_to(root):
        raise APIFixtureError("Конфигурация находится вне репозитория")
    try:
        if cfg_path.stat().st_size > _MAX_BYTES:
            raise APIFixtureError("config слишком большой")
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"),
                         object_pairs_hook=_no_duplicate_keys, parse_constant=_invalid_constant)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise APIFixtureError(f"Не удалось прочитать конфигурацию F5.4: {exc}") from exc
    expected = {
        "schema_version", "mode", "fixture_path", "fixture_sha256", "expected_labels",
        "expected_training_examples", "expected_documents", "bm25_k1", "bm25_b", "max_top_k",
    }
    if not isinstance(cfg, dict) or set(cfg) != expected:
        raise APIFixtureError("Неправильная схема конфигурации")
    if (type(cfg["schema_version"]) is not int or cfg["schema_version"] != 1
            or cfg["mode"] != "synthetic_demo"
            or cfg["fixture_path"] != "tests/fixtures/f5_04_api_demo_v1.json"
            or not isinstance(cfg["fixture_sha256"], str) or not _SHA.fullmatch(cfg["fixture_sha256"]) 
            or cfg["expected_labels"] != ["linux", "ml", "python", "sql"]
            or type(cfg["expected_training_examples"]) is not int or cfg["expected_training_examples"] != 24
            or type(cfg["expected_documents"]) is not int or cfg["expected_documents"] != 12
            or type(cfg["max_top_k"]) is not int or cfg["max_top_k"] != 5
            or type(cfg["bm25_k1"]) not in (int, float) or float(cfg["bm25_k1"]) != 1.5
            or type(cfg["bm25_b"]) not in (int, float) or float(cfg["bm25_b"]) != 0.75):
        raise APIFixtureError("Неподдерживаемый протокол F5.4")
    fpath = (root / cfg["fixture_path"]).resolve()
    if not fpath.is_relative_to(root):
        raise APIFixtureError("Fixture находится вне репозитория")
    try:
        if fpath.stat().st_size > _MAX_BYTES:
            raise APIFixtureError("Fixture слишком большая")
        raw = fpath.read_bytes()
        if sha256(raw).hexdigest() != cfg["fixture_sha256"]:
            raise APIFixtureError("Fixture SHA-256 не совпадает с конфигурацией")
        fixture = json.loads(raw, object_pairs_hook=_no_duplicate_keys,
                             parse_constant=_invalid_constant)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise APIFixtureError(f"Не удалось прочитать fixture F5.4: {exc}") from exc
    if not isinstance(fixture, dict) or set(fixture) != {"schema_version", "fixture_id", "labels", "training_examples", "documents"}:
        raise APIFixtureError("Неправильная схема demo fixture")
    if (type(fixture["schema_version"]) is not int or fixture["schema_version"] != 1
            or fixture["fixture_id"] != "f5_04_api_synthetic_v1" or fixture["labels"] != cfg["expected_labels"]):
        raise APIFixtureError("Неверный ID или метки demo fixture")
    examples = fixture["training_examples"]
    docs = fixture["documents"]
    if (not isinstance(examples, list) or len(examples) != 24 or
            not isinstance(docs, list) or len(docs) != 12):
        raise APIFixtureError("Неверное количество обучающих записей/документов")
    counts: Counter[str] = Counter()
    for example in examples:
        if not isinstance(example, dict) or set(example) != {"label", "text"}:
            raise APIFixtureError("Неправильная запись обучения")
        label, value = example["label"], example["text"]
        if label not in cfg["expected_labels"] or not isinstance(value, str) or not 3 <= len(value) <= 500:
            raise APIFixtureError("Неверный label или text обучения")
        counts[label] += 1
    if counts != Counter(dict.fromkeys(cfg["expected_labels"], 6)):
        raise APIFixtureError("Демонстрационная обучающая выборка несбалансирована")
    ids: set[str] = set()
    for doc in docs:
        if not isinstance(doc, dict) or set(doc) != {"doc_id", "source_id", "label", "text"}:
            raise APIFixtureError("Неправильная запись документа")
        doc_id, source_id = doc["doc_id"], doc["source_id"]
        if (not isinstance(doc_id, str) or not _ID.fullmatch(doc_id) or doc_id in ids
                or not isinstance(source_id, str) or not source_id.startswith("synthetic://")
                or len(source_id) > 150 or doc["label"] not in cfg["expected_labels"]
                or not isinstance(doc["text"], str) or not 20 <= len(doc["text"]) <= 800):
            raise APIFixtureError("Неверный synthetic документ")
        ids.add(doc_id)
    if len(set(d["source_id"] for d in docs)) != 12:
        raise APIFixtureError("Повторён source_id")
    return cfg, fixture


@dataclass(frozen=True)
class Doc:
    doc_id: str
    source_id: str
    label: str
    text: str


@dataclass(frozen=True)
class Hit:
    doc: Doc
    score: float


class DemoEngine:
    """Неизменяемая конфигурация + детерминированный классификатор/BM25.

    Создаётся один раз при `create_app`, не скачивает веса и не сохраняет
    личные данные или тексты запросов на диск.
    """

    def __init__(self, cfg: dict[str, Any], fixture: dict[str, Any]):
        # Импорт выполнять только при старте приложения, не при import модуля.
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression

        examples = fixture["training_examples"]
        self.vectorizer = TfidfVectorizer(
            lowercase=True, token_pattern=r"(?u)\b\w+\b", ngram_range=(1, 2), sublinear_tf=False,
        )
        features = self.vectorizer.fit_transform([x["text"] for x in examples])
        self.classifier = LogisticRegression(max_iter=300, random_state=42, solver="lbfgs")
        self.classifier.fit(features, [x["label"] for x in examples])
        self.docs = tuple(Doc(**x) for x in fixture["documents"])
        self.doc_tokens = tuple(Counter(tokens(d.text)) for d in self.docs)
        self.lengths = tuple(sum(freq.values()) for freq in self.doc_tokens)
        self.average_length = sum(self.lengths) / len(self.docs)
        self.df = Counter(t for counter in self.doc_tokens for t in counter)
        self.k1 = float(cfg["bm25_k1"])
        self.b = float(cfg["bm25_b"])
        self.max_top_k = int(cfg["max_top_k"])
        self.fixture_sha256 = cfg["fixture_sha256"]
        self.fixture_id = fixture["fixture_id"]
        self.training_examples = len(examples)

    def classify(self, text: str) -> dict[str, Any]:
        """Выполнить настоящую TF-IDF + LogisticRegression на demo-корпусе."""
        vector = self.vectorizer.transform([text])
        # При отсутствии слов из словаря demo нет оснований выдавать класс.
        if vector.nnz == 0:
            return {"label": None, "score": None, "status": "no_features",
                    "backend": "demo_tfidf_logreg", "fixture_id": self.fixture_id}
        probabilities = self.classifier.predict_proba(vector)[0]
        selected = int(probabilities.argmax())
        return {"label": str(self.classifier.classes_[selected]), "score": float(probabilities[selected]),
                "status": "predicted", "backend": "demo_tfidf_logreg", "fixture_id": self.fixture_id}

    def search(self, query: str, top_k: int) -> list[Hit]:
        """BM25 без dense encoder, без использования qrels для ранжирования."""
        query_terms = set(tokens(query))
        n = len(self.docs)
        hits: list[Hit] = []
        for doc, counter, length in zip(self.docs, self.doc_tokens, self.lengths, strict=True):
            score = 0.0
            for word in query_terms:
                term_freq = counter[word]
                if not term_freq:
                    continue
                doc_freq = self.df[word]
                idf = math.log1p((n - doc_freq + .5) / (doc_freq + .5))
                norm = term_freq + self.k1 * (1 - self.b + self.b * length / self.average_length)
                score += idf * (term_freq * (self.k1 + 1)) / norm
            if score > 0:
                hits.append(Hit(doc, score))
        hits.sort(key=lambda hit: (-hit.score, hit.doc.doc_id))
        return hits[:top_k]

    def ask(self, question: str) -> dict[str, Any]:
        """Выдать дословное предложение и проверяемый символьный диапазон.

        Режим не вызывает LLM. Ответ строится только из сохранённого текста
        выбранного документа. Отсутствие BM25 hits — отсутствие доказательств.
        """
        hits = self.search(question, 1)
        if not hits:
            return {"mode": "no_evidence", "answer": None, "evidence": [],
                    "backend": "demo_extractive_bm25", "fixture_id": self.fixture_id}
        doc = hits[0].doc
        qwords = set(tokens(question))
        sentences = [m for m in re.finditer(r"[^.!?\n]+[.!?]?", doc.text) if m.group().strip()]
        selected = max(sentences, key=lambda m: (len(set(tokens(m.group())) & qwords), -m.start()))
        start, end = selected.span()
        # В ответ не включать leading/trailing whitespace из соседних предложений.
        raw = doc.text[start:end]
        left = len(raw) - len(raw.lstrip())
        right = len(raw.rstrip())
        start += left
        end = selected.start() + right
        quote = doc.text[start:end]
        assert quote and doc.text[start:end] == quote
        return {
            "mode": "extractive_demo", "answer": quote,
            "evidence": [{"doc_id": doc.doc_id, "source_id": doc.source_id,
                          "start": start, "end": end, "quote": quote,
                          "text_sha256": sha256(doc.text.encode("utf-8")).hexdigest()}],
            "backend": "demo_extractive_bm25", "fixture_id": self.fixture_id,
        }
