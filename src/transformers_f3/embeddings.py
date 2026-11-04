"""F3.2: преобразование текста в embeddings предобученного Transformer Encoder.

Здесь F2-подход masked mean pooling применяется уже не к статическим
векторам nn.Embedding, а к контекстным представлениям token-level encoder.
Реальную модель загружает отдельный модуль model_cache: inference всегда CPU
и не обращается к сети. В unit-тестах используется детерминированный двойник.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F


_SHA_PATTERN = re.compile(r"[0-9a-f]{40}\Z")
_MODEL_ID_PATTERN = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+\Z")


@dataclass(frozen=True)
class ModelSpec:
    """Неизменяемый контракт модели, версии, лицензии и pooling.

    Аргументы:
        model_id: Имя репозитория модели в пространстве Hugging Face.
        revision: Полный 40-символьный SHA коммита весов и токенизатора.
        license: Ожидаемый идентификатор лицензии в metadata model card.
        embedding_dim: Размерность sentence embedding после pooling.
        max_length: Максимальная длина токенизированной последовательности.
        pooling: Способ объединения контекстных токенов; только masked_mean.
        normalize: Применять ли L2-нормировку к sentence embedding.
    """

    model_id: str
    revision: str
    license: str
    embedding_dim: int
    max_length: int
    pooling: str = "masked_mean"
    normalize: bool = True

    def __post_init__(self) -> None:
        """Отклонить небезопасные и неоднозначные параметры до загрузки."""
        if (
            not isinstance(self.model_id, str)
            or not _MODEL_ID_PATTERN.fullmatch(self.model_id)
            or any(segment in (".", "..") for segment in self.model_id.split("/"))
        ):
            raise ValueError("model_id должен иметь вид organization/model")
        if not isinstance(self.revision, str) or not _SHA_PATTERN.fullmatch(self.revision):
            raise ValueError("revision должен содержать ровно 40 hex-символов Git SHA")
        if not isinstance(self.license, str) or not self.license.strip():
            raise ValueError("Необходимо явно определить лицензию модели")
        for name in ("embedding_dim", "max_length"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} должен быть положительным целым числом")
        if self.pooling != "masked_mean":
            raise ValueError("F3.2 поддерживает только masked_mean pooling")
        if type(self.normalize) is not bool:
            raise TypeError("normalize должен быть bool")

    @classmethod
    def from_json(cls, path: str | Path) -> "ModelSpec":
        """Прочитать pinned-конфигурацию; лишние или пропущенные поля — ошибка.

        Аргументы:
            path: Путь к JSON-конфигурации (только метаданные, без весов).
        Результат:
            Валидированный ModelSpec.
        """
        with Path(path).open("r", encoding="utf-8") as file:
            data = json.load(file)
        if not isinstance(data, dict):
            raise ValueError("Конфигурация должна содержать JSON object")
        expected = {"model_id", "revision", "license", "embedding_dim", "max_length", "pooling", "normalize"}
        if set(data) != expected:
            raise ValueError(f"Поля конфигурации должны совпадать с {sorted(expected)}")
        return cls(**data)

    def to_dict(self) -> dict[str, str | int | bool]:
        """Представить конфигурацию в стабильном порядке для manifest."""
        return {
            "model_id": self.model_id,
            "revision": self.revision,
            "license": self.license,
            "embedding_dim": self.embedding_dim,
            "max_length": self.max_length,
            "pooling": self.pooling,
            "normalize": self.normalize,
        }


def masked_mean_pooling(token_embeddings: Tensor, attention_mask: Tensor) -> Tensor:
    """Усреднить только реальные контекстные токены по длине последовательности.

    Аргументы:
        token_embeddings: Вещественные контекстные векторы [B, L, D].
        attention_mask: Булева либо 0/1-маска настоящих токенов [B, L].
    Результат:
        Sentence embeddings формы [B, D]; без дополнительной нормализации.

    В отличие от `mean(dim=1)`, padding не влияет на знаменатель.
    Пустые последовательности не допускаются, чтобы не скрывать ошибки данных.
    """
    if not isinstance(token_embeddings, Tensor) or not isinstance(attention_mask, Tensor):
        raise TypeError("Оба аргумента должны быть torch.Tensor")
    if token_embeddings.ndim != 3 or any(v == 0 for v in token_embeddings.shape):
        raise ValueError("token_embeddings должен иметь непустую форму [B, L, D]")
    if not token_embeddings.is_floating_point():
        raise TypeError("token_embeddings должен быть вещественным")
    batch, length, _ = token_embeddings.shape
    if attention_mask.shape != (batch, length):
        raise ValueError("attention_mask должен иметь форму [B, L]")
    if attention_mask.device != token_embeddings.device:
        raise ValueError("mask и embeddings должны находиться на одном устройстве")
    if attention_mask.dtype != torch.bool:
        if attention_mask.dtype not in (torch.int32, torch.int64, torch.uint8, torch.int16):
            raise TypeError("mask должен быть bool или целочисленной 0/1-маской")
        if not bool(torch.all((attention_mask == 0) | (attention_mask == 1))):
            raise ValueError("attention_mask допускает только значения 0 и 1")
    valid = attention_mask.to(dtype=torch.bool)
    counts = valid.sum(dim=1, keepdim=True)
    if bool(torch.any(counts == 0)):
        raise ValueError("Нельзя вычислить embedding текста из одних padding-токенов")
    if not bool(torch.isfinite(token_embeddings).all()):
        raise ValueError("token_embeddings содержит NaN или Inf")
    masked = token_embeddings * valid.unsqueeze(-1).to(dtype=token_embeddings.dtype)
    return masked.sum(dim=1) / counts.to(dtype=token_embeddings.dtype)


def l2_normalize(embeddings: Tensor) -> Tensor:
    """Нормировать непустые ненулевые векторы по L2 для cosine similarity.

    Аргументы:
        embeddings: Матрица текстовых представлений [B, D].
    Результат:
        Матрица той же формы, у каждой строки евклидова норма равна 1.
    """
    if not isinstance(embeddings, Tensor):
        raise TypeError("embeddings должен быть torch.Tensor")
    if embeddings.ndim != 2 or any(size <= 0 for size in embeddings.shape):
        raise ValueError("embeddings должен иметь непустую форму [B, D]")
    if not embeddings.is_floating_point():
        raise TypeError("embeddings должен быть вещественным")
    if not bool(torch.isfinite(embeddings).all()):
        raise ValueError("embeddings содержит NaN или Inf")
    norms = torch.linalg.vector_norm(embeddings, ord=2, dim=1)
    if bool(torch.any(norms == 0)):
        raise ValueError("Нулевой embedding нельзя L2-нормировать")
    return F.normalize(embeddings, p=2, dim=1)


class PretrainedTextEncoder:
    """CPU inference поверх уже загруженного токенизатора и encoder.

    Аргументы:
        spec: Зафиксированная версия модели и схема embedding.
        tokenizer: Callable Hugging Face tokenizer (можно подменить в тестах).
        model: nn.Module, возвращающий last_hidden_state формы [B,L,D].

    В конструкторе нет сетевых обращений и обучения. eval() выключает dropout;
    inference_mode() исключает запись графа градиентов во время encode().
    """

    def __init__(self, spec: ModelSpec, tokenizer: Any, model: nn.Module) -> None:
        if not isinstance(spec, ModelSpec):
            raise TypeError("spec должен быть ModelSpec")
        if not callable(tokenizer):
            raise TypeError("tokenizer должен быть callable")
        if not isinstance(model, nn.Module):
            raise TypeError("model должен быть torch.nn.Module")
        self.spec = spec
        self.tokenizer = tokenizer
        self.model = model.to("cpu").eval()

    @torch.inference_mode()
    def encode(self, texts: Sequence[str], *, batch_size: int = 16) -> Tensor:
        """Получить contextual sentence embeddings без обучения и сети.

        Аргументы:
            texts: Непустые строки технических вопросов или документов.
            batch_size: Максимальное число текстов за один forward pass.
        Результат:
            Float32-матрица CPU [N, embedding_dim], исходный порядок сохранён.
            Для пустого списка возвращается матрица [0, embedding_dim].
        """
        if not isinstance(texts, (list, tuple)):
            raise TypeError("texts должен быть списком или кортежем строк")
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
            raise ValueError("batch_size должен быть положительным целым числом")
        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("Каждый текст должен быть непустой строкой")
        if not texts:
            return torch.empty((0, self.spec.embedding_dim), dtype=torch.float32)

        chunks: list[Tensor] = []
        for offset in range(0, len(texts), batch_size):
            batch = list(texts[offset:offset + batch_size])
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.spec.max_length,
                return_tensors="pt",
            )
            if not isinstance(encoded, dict) and not hasattr(encoded, "items"):
                raise TypeError("tokenizer должен вернуть словарь тензоров")
            encoded = dict(encoded)
            if "input_ids" not in encoded or "attention_mask" not in encoded:
                raise ValueError("tokenizer должен вернуть input_ids и attention_mask")
            if any(not isinstance(value, Tensor) for value in encoded.values()):
                raise TypeError("tokenizer может возвращать только torch.Tensor")
            if encoded["input_ids"].ndim != 2 or encoded["input_ids"].shape[0] != len(batch):
                raise ValueError("input_ids должен иметь форму [B, L]")
            if encoded["attention_mask"].shape != encoded["input_ids"].shape:
                raise ValueError("attention_mask и input_ids должны иметь одну форму")
            if encoded["input_ids"].dtype not in (torch.int32, torch.int64):
                raise TypeError("input_ids должен содержать целочисленные идентификаторы")
            result = self.model(**{key: value.to("cpu") for key, value in encoded.items()})
            hidden = getattr(result, "last_hidden_state", None)
            if hidden is None and isinstance(result, (tuple, list)) and result:
                hidden = result[0]
            if not isinstance(hidden, Tensor):
                raise TypeError("model должен вернуть last_hidden_state Tensor")
            if hidden.ndim != 3 or hidden.shape != (*encoded["input_ids"].shape, self.spec.embedding_dim):
                raise ValueError("Неверная размерность last_hidden_state")
            pooled = masked_mean_pooling(hidden, encoded["attention_mask"])
            if self.spec.normalize:
                pooled = l2_normalize(pooled)
            chunks.append(pooled.to(dtype=torch.float32, device="cpu"))
        return torch.cat(chunks, dim=0)
