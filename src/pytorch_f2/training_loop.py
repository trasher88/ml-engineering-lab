"""F2.4: воспроизводимый CPU-цикл обучения EmbeddingClassifier только на train.

Обеспечивает CrossEntropyLoss, AdamW, обновление параметров и агрегаты loss.
Не реализует validation, checkpointing, выбор лучшей эпохи или test-оценку.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.pytorch_f2.embedding_classifier import EmbeddingClassifier


class TrainingError(ValueError):
    """Некорректный батч, параметры обучения или числовое состояние оптимизации."""


@dataclass(frozen=True)
class EpochResult:
    """Итог одной эпохи train без показателей validation или test."""

    epoch: int
    mean_loss: float
    samples: int
    batches: int
    nonzero_gradients: bool
    parameters_updated: bool


def _validate_batch(batch: Mapping[str, Any], *, num_classes: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Проверить форму/метки батча F2.2 до вычисления CrossEntropyLoss.

    В частности, не позволяем скаляру или индексу класса вне диапазона
    пройти в функцию потерь; такие ошибки должны быть обнаружены сразу.
    """
    if not isinstance(batch, Mapping) or not all(
        key in batch for key in ("input_ids", "attention_mask", "labels")
    ):
        raise TrainingError("expected F2.2 batch with input_ids, attention_mask and labels")
    ids, mask, labels = batch["input_ids"], batch["attention_mask"], batch["labels"]
    if not isinstance(ids, torch.Tensor) or ids.dtype != torch.long or ids.ndim != 2:
        raise TrainingError("input_ids must be a 2D torch.long tensor")
    if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool or mask.shape != ids.shape:
        raise TrainingError("attention_mask must be torch.bool with input_ids shape")
    if (not isinstance(labels, torch.Tensor) or labels.dtype != torch.long
            or labels.ndim != 1 or labels.shape[0] != ids.shape[0]):
        raise TrainingError("labels must be a 1D torch.long tensor of batch size")
    if ids.device.type != "cpu" or mask.device.type != "cpu" or labels.device.type != "cpu":
        raise TrainingError("F2.4 supports CPU batches only")
    if ids.shape[0] == 0 or bool(((labels < 0) | (labels >= num_classes)).any()):
        raise TrainingError("labels must be in [0, num_classes)")
    return ids, mask, labels


def _validate_hyperparameters(*, epochs: int, learning_rate: float,
                              weight_decay: float) -> None:
    """Отклонить некорректные параметры вместо молчаливого обучения с ними."""
    if type(epochs) is not int or epochs < 1:
        raise TrainingError("epochs must be a positive integer")
    for key, val, positive in (("learning_rate", learning_rate, True),
                               ("weight_decay", weight_decay, False)):
        if type(val) not in (float, int) or not math.isfinite(val) or (val <= 0 if positive else val < 0):
            raise TrainingError(f"{key} must be {'positive' if positive else 'nonnegative'} and finite")


def train_one_epoch(model: EmbeddingClassifier, loader: DataLoader, *,
                    optimizer: torch.optim.Optimizer, epoch: int = 1) -> EpochResult:
    """Обучить модель на одной эпохе с CE и обновлением AdamW-параметров.

    Args:
        model: Модель F2.3, находящаяся на CPU.
        loader: Загрузчик F2.2, созданный только из train Dataset.
        optimizer: Созданный оптимизатор, состояние которого сохраняется между эпохами.
        epoch: Порядковый номер эпохи для отчёта.
    Returns:
        Взвешенная по числу примеров средняя потеря и инварианты обновления.

    Эта функция не содержит validation-логики и не считывает held-out выборки.
    """
    if not isinstance(model, EmbeddingClassifier) or not isinstance(loader, DataLoader):
        raise TrainingError("expected EmbeddingClassifier and F2.2 DataLoader")
    if not isinstance(optimizer, torch.optim.Optimizer):
        raise TrainingError("expected torch optimizer")
    if type(epoch) is not int or epoch < 1:
        raise TrainingError("epoch must be positive")
    if any(p.device.type != "cpu" for p in model.parameters()):
        raise TrainingError("F2.4 supports CPU models only")
    model_parameter_ids = {id(p) for p in model.parameters() if p.requires_grad}
    optimized_ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
    if not model_parameter_ids or model_parameter_ids != optimized_ids:
        raise TrainingError("optimizer must own all trainable model parameters")

    model.train()
    criterion = nn.CrossEntropyLoss(reduction="mean")
    # Снимок параметров берём один раз за эпоху: нет постоянного копирования
    # крупной embedding-матрицы на каждом батче.
    before = [p.detach().clone() for p in model.parameters()]
    total_loss, samples, batches = 0.0, 0, 0
    nonzero_gradients = False

    for batch in loader:
        ids, mask, labels = _validate_batch(batch, num_classes=model.num_classes)
        optimizer.zero_grad(set_to_none=True)
        # Логиты [B, C] передаются в CrossEntropyLoss без предварительного softmax.
        logits = model(ids, mask)
        loss = criterion(logits, labels)
        if not bool(torch.isfinite(loss)):
            raise TrainingError("non-finite training loss")
        loss.backward()
        for parameter in model.parameters():
            if parameter.grad is not None:
                if not bool(torch.isfinite(parameter.grad).all()):
                    raise TrainingError("non-finite gradients")
                if bool(torch.count_nonzero(parameter.grad)):
                    nonzero_gradients = True
        optimizer.step()
        # CrossEntropyLoss(mean) усредняет в пределах батча: для всей эпохи
        # учитываем его вес по количеству примеров (последний батч короче).
        batch_size = int(labels.shape[0])
        total_loss += float(loss.detach()) * batch_size
        samples += batch_size
        batches += 1

    if samples == 0:
        raise TrainingError("empty training loader")
    parameters_updated = any(
        not torch.equal(parameter.detach(), old)
        for parameter, old in zip(model.parameters(), before)
    )
    return EpochResult(epoch=epoch, mean_loss=total_loss / samples,
                       samples=samples, batches=batches,
                       nonzero_gradients=nonzero_gradients,
                       parameters_updated=parameters_updated)


def train_epochs(model: EmbeddingClassifier, loader: DataLoader, *,
                 epochs: int = 2, learning_rate: float = 0.003,
                 weight_decay: float = 0.01) -> list[EpochResult]:
    """Выполнить заданное число эпох, сохраняя состояние AdamW между ними.

    Каждая эпоха использует тот же train DataLoader. После последней эпохи
    модель содержит обновлённые веса, но не сохраняется на диск.
    """
    _validate_hyperparameters(epochs=epochs, learning_rate=learning_rate,
                              weight_decay=weight_decay)
    if not isinstance(model, EmbeddingClassifier) or not isinstance(loader, DataLoader):
        raise TrainingError("expected EmbeddingClassifier and F2.2 DataLoader")
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate),
                                  weight_decay=float(weight_decay))
    results = [train_one_epoch(model, loader, optimizer=optimizer, epoch=i)
               for i in range(1, epochs + 1)]
    if any(not item.nonzero_gradients or not item.parameters_updated for item in results):
        raise TrainingError("training did not produce gradients and parameter updates")
    return results
