"""F2.5: независимая validation-оценка и безопасное хранение лучшей модели.

Оцениваются только val-батчи F2.2; веса на validation не обновляются.
Лучшая эпоха определяется по Macro F1. Test здесь не используется.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
import pickle
from pathlib import Path
import tempfile
from typing import Any, Mapping

import torch
from torch import nn
from torch.utils.data import DataLoader, SequentialSampler

from src.pytorch_f2.data_pipeline import QuestionDataset, Vocabulary
from src.pytorch_f2.embedding_classifier import EmbeddingClassifier
from src.pytorch_f2.training_loop import (
    EpochResult, TrainingError, _validate_batch, _validate_hyperparameters, train_one_epoch,
)


class ValidationError(ValueError):
    """Нарушен контракт validation, checkpoint или восстановленного состояния."""


@dataclass(frozen=True)
class ValidationResult:
    """Метрики всего val-набора, агрегированные по отдельным примерам."""

    mean_loss: float
    accuracy: float
    macro_f1: float
    samples: int
    confusion_matrix: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class EpochValidation:
    """Обучение одной эпохи и оценка на независимой validation-выборке."""

    train: EpochResult
    validation: ValidationResult


@dataclass(frozen=True)
class FitResult:
    """История validation и номер лучшей эпохи, выбранной без test."""

    history: tuple[EpochValidation, ...]
    best_epoch: int
    best_macro_f1: float
    checkpoint_path: Path


def _require_split(loader: DataLoader, split: str) -> QuestionDataset:
    """Отклонить загрузчик, который читает чужую выборку или теряет примеры.

    Проверка меток split у всех записей предотвращает случайную передачу
    train/test в validation, включая ошибочно переименованный DataLoader.
    """
    if not isinstance(loader, DataLoader) or not isinstance(loader.dataset, QuestionDataset):
        raise ValidationError("expected F2.2 QuestionDataset DataLoader")
    if not loader.dataset.rows or any(row.get("split") != split for row in loader.dataset.rows):
        raise ValidationError(f"expected {split} split exclusively")
    if split == "val" and (not isinstance(loader.sampler, SequentialSampler) or loader.drop_last):
        raise ValidationError("validation loader must be sequential and complete")
    return loader.dataset


def _scores_from_confusion(matrix: torch.Tensor) -> tuple[float, float]:
    """Вычислить Accuracy и Macro F1 с учётом каждого известного класса.

    Классы с нулевым знаменателем имеют F1=0 (zero_division=0).
    Поддерживается даже val-набор, в котором часть классов отсутствует.
    """
    cm = matrix.to(dtype=torch.float64)
    true_positive = cm.diagonal()
    actual = cm.sum(dim=1)
    predicted = cm.sum(dim=0)
    denominator = actual + predicted
    f1 = torch.where(denominator > 0, 2 * true_positive / denominator.clamp(min=1), 0.0)
    return float(true_positive.sum() / cm.sum()), float(f1.mean())


def evaluate_validation(model: EmbeddingClassifier, loader: DataLoader) -> ValidationResult:
    """Вычислить loss, accuracy, macro F1 на val без обратного прохода.

    Изменение режима model.eval() действует только внутри функции:
    исходное состояние training восстанавливается даже при исключении.
    Градиенты и параметры модели остаются неизменными.
    """
    if not isinstance(model, EmbeddingClassifier):
        raise ValidationError("expected EmbeddingClassifier")
    dataset = _require_split(loader, "val")
    if any(parameter.device.type != "cpu" for parameter in model.parameters()):
        raise ValidationError("CPU model required")
    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_samples = 0
    confusion = torch.zeros((model.num_classes, model.num_classes), dtype=torch.int64)
    try:
        with torch.inference_mode():
            for batch in loader:
                ids, mask, labels = _validate_batch(batch, num_classes=model.num_classes)
                logits = model(ids, mask)
                if not bool(torch.isfinite(logits).all()):
                    raise ValidationError("non-finite validation logits")
                batch_loss = nn.functional.cross_entropy(logits, labels, reduction="sum")
                if not bool(torch.isfinite(batch_loss)):
                    raise ValidationError("non-finite validation loss")
                total_loss += float(batch_loss)
                total_samples += labels.numel()
                predictions = logits.argmax(dim=1)
                encoded = labels * model.num_classes + predictions
                confusion += torch.bincount(encoded, minlength=model.num_classes ** 2).reshape_as(confusion)
    finally:
        model.train(was_training)
    if total_samples != len(dataset) or total_samples == 0:
        raise ValidationError("validation loader did not consume the entire split")
    accuracy, macro_f1 = _scores_from_confusion(confusion)
    return ValidationResult(
        mean_loss=total_loss / total_samples,
        accuracy=accuracy,
        macro_f1=macro_f1,
        samples=total_samples,
        confusion_matrix=tuple(tuple(int(v) for v in row) for row in confusion.tolist()),
    )


def vocabulary_fingerprint(vocab: Vocabulary) -> str:
    """Хешировать train-словарь без сохранения токенов в checkpoint.

    Так проверяется совместимость ID токенов при восстановлении, даже если
    число элементов словаря осталось тем же, а их порядок изменился.
    """
    if not isinstance(vocab, Vocabulary):
        raise ValidationError("expected F2.2 Vocabulary")
    serialized = json.dumps(vocab.token_to_id, ensure_ascii=False, sort_keys=True,
                            separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _validate_metadata(vocab: Vocabulary, label_to_id: Mapping[str, int],
                       max_length: int, split_counts: Mapping[str, int], seed: int) -> None:
    """Убедиться, что сохраняемое описание соответствует подготовленным данным."""
    if not isinstance(label_to_id, Mapping) or not label_to_id or (
        set(label_to_id.values()) != set(range(len(label_to_id)))
    ) or any(not isinstance(k, str) or type(v) is not int for k, v in label_to_id.items()):
        raise ValidationError("invalid class mapping")
    if type(max_length) is not int or max_length < 2:
        raise ValidationError("max_length must be >= 2")
    if type(seed) is not int or seed < 0:
        raise ValidationError("seed must be nonnegative integer")
    if not isinstance(split_counts, Mapping) or set(split_counts) != {"train", "val", "test"} or any(
        type(value) is not int or value < 1 for value in split_counts.values()
    ):
        raise ValidationError("invalid split counts")
    vocabulary_fingerprint(vocab)


def _save_checkpoint(path: Path, model: EmbeddingClassifier, *, metadata: dict[str, Any]) -> None:
    """Атомарно записать только CPU state_dict и простые метаданные.

    Файл временно создаётся в том же каталоге: os.replace предотвращает
    использование частично записанного best-checkpoint после сбоя записи.
    """
    if path.suffix != ".pt":
        raise ValidationError("checkpoint path must have .pt extension")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 1,
        "metadata": metadata,
        "model_state": {name: value.detach().cpu().clone()
                        for name, value in model.state_dict().items()},
    }
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", prefix=".f2_5_", suffix=".tmp",
                                         dir=path.parent, delete=False) as handle:
            temp_path = Path(handle.name)
            torch.save(payload, handle)
        os.replace(temp_path, path)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def fit_with_validation(model: EmbeddingClassifier, train_loader: DataLoader,
                        val_loader: DataLoader, *, vocab: Vocabulary,
                        label_to_id: Mapping[str, int], max_length: int,
                        split_counts: Mapping[str, int], seed: int,
                        checkpoint_path: Path, epochs: int = 3,
                        learning_rate: float = 0.003,
                        weight_decay: float = 0.01) -> FitResult:
    """Обучать на train; после каждой эпохи выбирать лучший val Macro F1.

    При равенстве Macro F1 сохраняется первая достигнутая лучшая эпоха.
    Test-набор не передаётся функции, поэтому не влияет на решение.
    Оптимизатор создаётся единожды и хранит моменты AdamW между эпохами.
    """
    if not isinstance(model, EmbeddingClassifier):
        raise ValidationError("expected EmbeddingClassifier")
    _validate_hyperparameters(epochs=epochs, learning_rate=learning_rate,
                              weight_decay=weight_decay)
    _validate_metadata(vocab, label_to_id, max_length, split_counts, seed)
    train_dataset = _require_split(train_loader, "train")
    val_dataset = _require_split(val_loader, "val")
    if train_dataset.vocab.token_to_id != vocab.token_to_id or val_dataset.vocab.token_to_id != vocab.token_to_id:
        raise ValidationError("dataset vocab mismatch")
    if train_dataset.label_to_id != dict(label_to_id) or val_dataset.label_to_id != dict(label_to_id):
        raise ValidationError("dataset label mapping mismatch")
    if train_dataset.max_length != max_length or val_dataset.max_length != max_length:
        raise ValidationError("dataset max_length mismatch")
    if len(train_dataset) != split_counts["train"] or len(val_dataset) != split_counts["val"]:
        raise ValidationError("split count mismatch")
    if model.vocab_size != len(vocab) or model.num_classes != len(label_to_id):
        raise ValidationError("model dimensions do not match F2.2 data")
    if any(parameter.device.type != "cpu" for parameter in model.parameters()):
        raise ValidationError("CPU model required")
    if not isinstance(checkpoint_path, (str, Path)):
        raise ValidationError("checkpoint_path must be a path")
    path = Path(checkpoint_path)
    if path.suffix != ".pt":
        raise ValidationError("checkpoint path must have .pt extension")
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate),
                                  weight_decay=float(weight_decay))
    history: list[EpochValidation] = []
    best_epoch, best_score = 0, -1.0
    for epoch in range(1, epochs + 1):
        train_result = train_one_epoch(model, train_loader, optimizer=optimizer, epoch=epoch)
        if not train_result.nonzero_gradients or not train_result.parameters_updated:
            raise TrainingError("training did not update model parameters")
        validation = evaluate_validation(model, val_loader)
        history.append(EpochValidation(train=train_result, validation=validation))
        if validation.macro_f1 > best_score:
            best_score = validation.macro_f1
            best_epoch = epoch
            metadata = {
                "selection_metric": "val_macro_f1",
                "tie_break": "earliest_epoch",
                "best_epoch": best_epoch,
                "val_macro_f1": best_score,
                "val_accuracy": validation.accuracy,
                "val_mean_loss": validation.mean_loss,
                "train_samples": split_counts["train"],
                "val_samples": validation.samples,
                "split_counts": dict(split_counts),
                "vocab_fingerprint": vocabulary_fingerprint(vocab),
                "label_to_id": dict(label_to_id),
                "max_length": max_length,
                "seed": seed,
                "learning_rate": float(learning_rate),
                "weight_decay": float(weight_decay),
                "planned_epochs": epochs,
                "torch_version": str(torch.__version__),
                "model": {
                    "vocab_size": model.vocab_size,
                    "num_classes": model.num_classes,
                    "embedding_dim": model.embedding.embedding_dim,
                    "hidden_dim": model.hidden.out_features,
                },
            }
            # Верификация выполняется на той же эпохе, пока веса model
            # ещё соответствуют только что сохранённому checkpoint.
            sample_batch = next(iter(val_loader))
            model.eval()
            with torch.inference_mode():
                before_save_logits = model(sample_batch["input_ids"], sample_batch["attention_mask"])
            # Сначала записываем атомарно, затем проверяем сам файл, а не объект в памяти.
            _save_checkpoint(path, model, metadata=metadata)
            recovered, _ = restore_best_checkpoint(
                path, vocab=vocab, label_to_id=label_to_id, max_length=max_length,
            )
            with torch.inference_mode():
                after_save_logits = recovered(sample_batch["input_ids"], sample_batch["attention_mask"])
            model.train()
            if not torch.equal(before_save_logits, after_save_logits):
                raise ValidationError("checkpoint changed forward predictions")
    return FitResult(history=tuple(history), best_epoch=best_epoch,
                     best_macro_f1=best_score, checkpoint_path=path)


def restore_best_checkpoint(path: Path, *, vocab: Vocabulary,
                            label_to_id: Mapping[str, int], max_length: int) -> tuple[EmbeddingClassifier, dict[str, Any]]:
    """Загрузить лучшие веса с CPU и проверить совместимость словаря и меток.

    weights_only=True ограничивает десериализацию простыми типами и тензорами.
    Для неизвестных/недоверенных checkpoint-файлов сохраняются обычные риски
    использования ресурсов и целостности данных: загружать свои файлы.
    """
    try:
        payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, ValueError, EOFError, pickle.UnpicklingError) as exc:
        raise ValidationError("checkpoint cannot be loaded") from exc
    if not isinstance(payload, dict) or set(payload) != {"format_version", "metadata", "model_state"} or payload["format_version"] != 1:
        raise ValidationError("unknown checkpoint format")
    metadata, weights = payload["metadata"], payload["model_state"]
    if not isinstance(metadata, dict) or not isinstance(weights, dict):
        raise ValidationError("invalid checkpoint payload")
    if metadata.get("vocab_fingerprint") != vocabulary_fingerprint(vocab):
        raise ValidationError("checkpoint vocabulary mismatch")
    if metadata.get("label_to_id") != dict(label_to_id):
        raise ValidationError("checkpoint class mapping mismatch")
    if metadata.get("max_length") != max_length:
        raise ValidationError("checkpoint sequence length mismatch")
    params = metadata.get("model")
    if not isinstance(params, dict) or set(params) != {"vocab_size", "num_classes", "embedding_dim", "hidden_dim"}:
        raise ValidationError("missing model architecture")
    if params["vocab_size"] != len(vocab) or params["num_classes"] != len(label_to_id):
        raise ValidationError("checkpoint shape mismatch")
    if any(not isinstance(x, torch.Tensor) or not torch.isfinite(x).all() for x in weights.values()):
        raise ValidationError("invalid checkpoint tensors")
    try:
        # Конструктор инициализирует случайные веса перед load_state_dict.
        # Изолируем RNG, чтобы восстановление не влияло на будущие эксперименты.
        with torch.random.fork_rng(devices=[]):
            model = EmbeddingClassifier(**params).cpu()
        model.load_state_dict(weights, strict=True)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ValidationError("checkpoint weights do not match architecture") from exc
    model.eval()
    return model, metadata
