"""F2.3: небольшой классификатор текста на Embedding и masked mean pooling.

Модуль принимает батчи F2.2 и возвращает logits классов. Здесь нет
функции потерь, оптимизатора, обучения, сохранения весов или оценки качества.
"""

from __future__ import annotations

import torch
from torch import nn

from src.pytorch_f2.data_pipeline import PAD_ID, SPECIAL_TOKENS


class ModelInputError(ValueError):
    """Ошибка параметров модели либо несоответствие батча контракту F2.2."""


def masked_mean_pool(embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Усреднить только настоящие токены в каждой строке батча.

    Args:
        embeddings: Векторы токенов формы [B, L, D], вещественный тип.
        attention_mask: Булева маска формы [B, L]; True означает токен.
    Returns:
        Векторы текстов формы [B, D], с исключённым влиянием padding.
    Raises:
        ModelInputError: Пустая последовательность, неверные формы или типы.

    BOS/EOS включены в усреднение: F2.2 выставляет им True в маске.
    Функция не нормализует векторы по длине всего padded-батча.
    """
    if not isinstance(embeddings, torch.Tensor) or embeddings.ndim != 3:
        raise ModelInputError("embeddings must be a 3D tensor [B, L, D]")
    if not embeddings.is_floating_point() or any(size == 0 for size in embeddings.shape):
        raise ModelInputError("embeddings must be a nonempty floating-point tensor")
    if not isinstance(attention_mask, torch.Tensor) or attention_mask.dtype != torch.bool:
        raise ModelInputError("attention_mask must be a boolean tensor")
    if attention_mask.shape != embeddings.shape[:2]:
        raise ModelInputError("attention_mask shape must match embeddings [B, L]")
    if embeddings.device != attention_mask.device:
        raise ModelInputError("embeddings and attention_mask must use one device")
    if not bool(attention_mask.any(dim=1).all()):
        raise ModelInputError("each sequence must have at least one valid token")

    # Отсекаем заполнители *до* суммирования, независимо от их значений.
    visible = embeddings.masked_fill(~attention_mask.unsqueeze(-1), 0.0)
    counts = attention_mask.sum(dim=1).unsqueeze(-1)
    return visible.sum(dim=1) / counts


class EmbeddingClassifier(nn.Module):
    """Классификатор «Embedding → masked mean → Linear → ReLU → Linear».

    Параметры определяют размер словаря, классов и двух скрытых пространств.
    Полученные оценки — logits, не вероятности; softmax здесь не нужен.
    Модель не читает данные F1 и не изменяет словарь или разделение выборок.
    """

    def __init__(self, *, vocab_size: int, num_classes: int,
                 embedding_dim: int = 48, hidden_dim: int = 48) -> None:
        """Создать обучаемые слои без выполнения обучения или выбора модели."""
        super().__init__()
        for name, value, minimum in (
            ("vocab_size", vocab_size, len(SPECIAL_TOKENS)),
            ("num_classes", num_classes, 2),
            ("embedding_dim", embedding_dim, 1),
            ("hidden_dim", hidden_dim, 1),
        ):
            if type(value) is not int or value < minimum:
                raise ModelInputError(f"{name} must be an integer >= {minimum}")

        self.vocab_size = vocab_size
        self.num_classes = num_classes
        self.padding_idx = PAD_ID
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=PAD_ID)
        self.hidden = nn.Linear(embedding_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.output = nn.Linear(hidden_dim, num_classes)

    def forward(self, input_ids: torch.Tensor,
                attention_mask: torch.Tensor) -> torch.Tensor:
        """Получить logits для каждого текста одного батча F2.2.

        Args:
            input_ids: Тензор torch.long [B, L] с нулевым ID padding.
            attention_mask: Тензор torch.bool [B, L], True для BOS/слов/EOS.
        Returns:
            Вещественный тензор logits [B, num_classes].
        Raises:
            ModelInputError: Нарушен контракт формы, маски или ID токенов.
        """
        if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2:
            raise ModelInputError("input_ids must be a 2D tensor [B, L]")
        if input_ids.dtype != torch.long:
            raise ModelInputError("input_ids must use torch.long")
        if input_ids.shape[0] < 1 or input_ids.shape[1] < 2:
            raise ModelInputError("input_ids must have B >= 1 and L >= 2")
        if not isinstance(attention_mask, torch.Tensor) or attention_mask.dtype != torch.bool:
            raise ModelInputError("attention_mask must use torch.bool")
        if attention_mask.shape != input_ids.shape:
            raise ModelInputError("attention_mask must match input_ids shape")
        if attention_mask.device != input_ids.device:
            raise ModelInputError("input_ids and attention_mask must use one device")
        if not bool(attention_mask.any(dim=1).all()):
            raise ModelInputError("each question must contain valid tokens")
        if not torch.equal(input_ids.eq(self.padding_idx), ~attention_mask):
            raise ModelInputError("PAD positions must exactly match inverted attention_mask")
        # F2.2 добавляет PAD только справа: запрещаем «дыры» в маске.
        if bool((~attention_mask[:, :-1] & attention_mask[:, 1:]).any()):
            raise ModelInputError("padding must be right-aligned")
        if bool(((input_ids < 0) | (input_ids >= self.vocab_size)).any()):
            raise ModelInputError("token id is outside the configured vocabulary")

        token_vectors = self.embedding(input_ids)             # [B, L, D]
        question_vectors = masked_mean_pool(token_vectors, attention_mask)  # [B, D]
        hidden_vectors = self.activation(self.hidden(question_vectors))      # [B, H]
        return self.output(hidden_vectors)                    # [B, C], без softmax
