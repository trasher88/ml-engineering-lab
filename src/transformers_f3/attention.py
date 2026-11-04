"""Одноголовый Self-Attention: проекции Q/K/V, маски и взвешенная сумма.

В F2 токены объединялись в один вектор с помощью masked mean pooling.
Здесь каждый токен вычисляет отдельное контекстное представление, учитывая
все разрешённые позиции той же последовательности.

Реализация специально использует явные матричные операции PyTorch вместо
готового слоя MultiheadAttention, чтобы их можно было проверить по формулам.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


def _validate_attention_inputs(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    key_padding_mask: Tensor | None,
    query_padding_mask: Tensor | None,
) -> None:
    """Проверить согласованность форм и типов входных тензоров.

    Аргументы:
        query: Запросы формы [B, Lq, Dk].
        key: Ключи формы [B, Lk, Dk].
        value: Значения формы [B, Lk, Dv].
        key_padding_mask: True для настоящих ключей, форма [B, Lk].
        query_padding_mask: True для настоящих запросов, форма [B, Lq].

    Результат:
        None. При нарушении инвариантов возникает ValueError или TypeError.
    """
    for name, tensor in (("query", query), ("key", key), ("value", value)):
        if not isinstance(tensor, Tensor):
            raise TypeError(f"{name} должен быть torch.Tensor")
        if tensor.ndim != 3:
            raise ValueError(f"{name} должен иметь форму [B, L, D]")
        if not tensor.is_floating_point():
            raise TypeError(f"{name} должен иметь вещественный dtype")
        if any(size == 0 for size in tensor.shape):
            raise ValueError(f"{name}: нулевые размерности не допускаются")

    batch, query_length, key_dim = query.shape
    key_batch, key_length, other_key_dim = key.shape
    value_batch, value_length, _ = value.shape
    if batch != key_batch or batch != value_batch:
        raise ValueError("Размерность batch у query, key и value должна совпадать")
    if key_dim != other_key_dim:
        raise ValueError("Последняя размерность query и key должна совпадать")
    if key_length != value_length:
        raise ValueError("Число позиций key и value должно совпадать")
    if query.dtype != key.dtype or query.dtype != value.dtype:
        raise TypeError("query, key и value должны иметь одинаковый dtype")
    if query.device != key.device or query.device != value.device:
        raise ValueError("query, key и value должны находиться на одном устройстве")

    for name, mask, length in (
        ("key_padding_mask", key_padding_mask, key_length),
        ("query_padding_mask", query_padding_mask, query_length),
    ):
        if mask is None:
            continue
        if not isinstance(mask, Tensor):
            raise TypeError(f"{name} должен быть torch.Tensor")
        if mask.dtype != torch.bool:
            raise TypeError(f"{name} должен иметь dtype torch.bool")
        if mask.shape != (batch, length):
            raise ValueError(f"{name} должен иметь форму [{batch}, {length}]")
        if mask.device != query.device:
            raise ValueError(f"{name} должен находиться на том же устройстве")

    # Softmax над полностью замаскированной строкой ключей не определён:
    # если все ключи отсутствуют, нельзя построить распределение внимания.
    if key_padding_mask is not None and not torch.all(key_padding_mask.any(dim=1)):
        raise ValueError("В каждом элементе batch должен быть хотя бы один настоящий ключ")


def scaled_dot_product_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    *,
    key_padding_mask: Tensor | None = None,
    query_padding_mask: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Вычислить scaled dot-product attention с явной нормализацией.

    Аргументы:
        query: Q с формой [B, Lq, Dk].
        key: K с формой [B, Lk, Dk].
        value: V с формой [B, Lk, Dv].
        key_padding_mask: True означает разрешённый ключ, форма [B, Lk].
        query_padding_mask: True означает реальный запрос, форма [B, Lq].

    Результат:
        Пара (context, weights). Context имеет форму [B, Lq, Dv],
        weights — [B, Lq, Lk]. Для padding-запросов обе строки обнуляются.

    Ограничение:
        Должен существовать хотя бы один допустимый ключ на каждый batch.
        Эта функция не реализует causal mask или несколько attention heads.
    """
    _validate_attention_inputs(query, key, value, key_padding_mask, query_padding_mask)

    # Масштабирование sqrt(Dk) ограничивает рост скалярных произведений
    # при увеличении размерности пространства запросов и ключей.
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.shape[-1])

    if key_padding_mask is not None:
        # -inf превращается в нулевую вероятность после Softmax.
        # Проверка выше гарантирует хотя бы один незамаскированный ключ.
        scores = scores.masked_fill(~key_padding_mask.unsqueeze(1), float("-inf"))

    weights = torch.softmax(scores, dim=-1)
    context = torch.matmul(weights, value)

    if query_padding_mask is not None:
        # Запрет участия padding-запросов в результате — отдельное правило:
        # маска ключей сама по себе не обнуляет выход padding-токена.
        valid_queries = query_padding_mask.unsqueeze(-1)
        weights = weights.masked_fill(~valid_queries, 0.0)
        context = context.masked_fill(~valid_queries, 0.0)

    return context, weights


class SelfAttention(nn.Module):
    """Обучаемый одноголовый Self-Attention для входа [B, L, Dmodel].

    Для каждого токена строятся три независимые линейные проекции Q/K/V.
    Padding запрещён и в качестве ключа, и в качестве выходного запроса.
    Выходы не объединяются по длине последовательности: [B, L, Dv].

    Аргументы:
        d_model: Число признаков входного токена.
        d_k: Размерность запросов/ключей; по умолчанию d_model.
        d_v: Размерность значений; по умолчанию d_model.
    """

    def __init__(self, d_model: int, *, d_k: int | None = None, d_v: int | None = None) -> None:
        super().__init__()
        if not isinstance(d_model, int) or isinstance(d_model, bool) or d_model <= 0:
            raise ValueError("d_model должен быть положительным целым числом")
        if d_k is None:
            d_k = d_model
        if d_v is None:
            d_v = d_model
        for name, size in (("d_k", d_k), ("d_v", d_v)):
            if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
                raise ValueError(f"{name} должен быть положительным целым числом")

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        # Без bias легче сопоставить вычисления с Q=XWq, K=XWk, V=XWv.
        self.query_projection = nn.Linear(d_model, d_k, bias=False)
        self.key_projection = nn.Linear(d_model, d_k, bias=False)
        self.value_projection = nn.Linear(d_model, d_v, bias=False)

    def forward(
        self,
        embeddings: Tensor,
        padding_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Преобразовать токены в контекстные векторы и веса внимания.

        Аргументы:
            embeddings: Векторизованные токены [B, L, Dmodel].
            padding_mask: True для настоящих токенов, форма [B, L].

        Результат:
            (context [B, L, Dv], weights [B, L, L]).
            Строки padding-токенов в обоих результатах равны нулю.
        """
        if not isinstance(embeddings, Tensor):
            raise TypeError("embeddings должен быть torch.Tensor")
        if embeddings.ndim != 3 or embeddings.shape[-1] != self.d_model:
            raise ValueError("embeddings должен иметь форму [B, L, d_model]")
        if not embeddings.is_floating_point():
            raise TypeError("embeddings должен иметь вещественный dtype")
        if embeddings.dtype != self.query_projection.weight.dtype:
            raise TypeError("dtype embeddings должен совпадать с dtype параметров модели")
        if embeddings.device != self.query_projection.weight.device:
            raise ValueError("embeddings и параметры модели должны быть на одном устройстве")

        query = self.query_projection(embeddings)
        key = self.key_projection(embeddings)
        value = self.value_projection(embeddings)
        return scaled_dot_product_attention(
            query,
            key,
            value,
            key_padding_mask=padding_mask,
            query_padding_mask=padding_mask,
        )
