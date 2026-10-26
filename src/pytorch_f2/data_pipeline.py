"""F2.2: токенизация технических вопросов, словарь train и батчи PyTorch.

Модуль использует уже разделённые данные F1.6 и не изменяет исходные записи.
Тексты применяются только в оперативной памяти; артефакты датасета не создаются.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import torch
from torch.utils.data import DataLoader, Dataset

from src.common.io import ROOT, read_jsonl

SPECIAL_TOKENS = ("<PAD>", "<UNK>", "<BOS>", "<EOS>")
PAD_ID, UNK_ID, BOS_ID, EOS_ID = range(4)
_SPLITS = ("train", "val", "test")
_TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)


class DataPipelineError(ValueError):
    """Нарушен контракт входных выборок или параметры преобразования текста."""


def tokenize(text: str) -> list[str]:
    """Разделить непустой текст на Unicode-слова и отдельные знаки пунктуации.

    Регистр нормализуется через casefold. Токенизатор детерминирован и не
    зависит от обучающих данных; он не является subword/BPE-токенизатором.
    """
    if not isinstance(text, str) or not text.strip():
        raise DataPipelineError("text must be a nonempty string")
    return _TOKEN_PATTERN.findall(text.casefold())


def validate_splits(splits: Mapping[str, Sequence[dict[str, Any]]]) -> dict[str, int]:
    """Проверить F1.6 split, уникальные ID, группы и известные train-метки.

    Здесь дополнительно проверяется граница между выборками непосредственно
    перед построением словаря: при ошибке группа или запись не должна попасть
    в несколько частей, даже если локальные файлы были заменены вручную.
    """
    if not isinstance(splits, Mapping) or set(splits) != set(_SPLITS):
        raise DataPipelineError("expected exactly train, val, test splits")

    seen_ids: set[str] = set()
    group_sets: dict[str, set[str | int]] = {}
    train_labels: set[str] = set()
    counts: dict[str, int] = {}
    for name in _SPLITS:
        rows = splits[name]
        if not isinstance(rows, (list, tuple)) or not rows:
            raise DataPipelineError(f"{name} must be a nonempty sequence")
        group_ids: set[str | int] = set()
        for index, row in enumerate(rows):
            if not isinstance(row, dict) or any(
                not isinstance(row.get(key), str) or not row[key].strip()
                for key in ("id", "context", "domain")
            ):
                raise DataPipelineError(f"invalid required field in {name}[{index}]")
            if row.get("split") != name:
                raise DataPipelineError(f"incorrect split marker in {name}[{index}]")
            group_id = row.get("group_id")
            # Bool — подкласс int, но булевы значения не могут быть ID группы.
            if type(group_id) not in (int, str) or not str(group_id):
                raise DataPipelineError(f"invalid group_id in {name}[{index}]")
            if row["id"] in seen_ids:
                raise DataPipelineError("duplicate record id across splits")
            seen_ids.add(row["id"])
            group_ids.add(group_id)
            if name == "train":
                train_labels.add(row["domain"])
        counts[name] = len(rows)
        group_sets[name] = group_ids

    if len(train_labels) < 2:
        raise DataPipelineError("train needs at least two distinct classes")
    for name in ("val", "test"):
        if any(row["domain"] not in train_labels for row in splits[name]):
            raise DataPipelineError(f"unseen label in {name}")
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        if group_sets[left] & group_sets[right]:
            raise DataPipelineError(f"group leakage between {left} and {right}")
    return counts


@dataclass(frozen=True)
class Vocabulary:
    """Словарь индексов, построенный только по обучающим вопросам.

    Первые четыре индекса зарезервированы. Любой новый токен из val/test
    преобразуется в UNK, а не добавляется в словарь.
    """

    token_to_id: dict[str, int]

    def __post_init__(self) -> None:
        """Проверить фиксированные ID и непрерывность словаря."""
        if not isinstance(self.token_to_id, dict):
            raise DataPipelineError("vocabulary must be a dict")
        if any(self.token_to_id.get(token) != index
               for index, token in enumerate(SPECIAL_TOKENS)):
            raise DataPipelineError("invalid special token IDs")
        if set(self.token_to_id.values()) != set(range(len(self.token_to_id))):
            raise DataPipelineError("vocabulary indices must be unique and contiguous")

    def __len__(self) -> int:
        """Вернуть число токенов вместе со специальными."""
        return len(self.token_to_id)

    def encode(self, text: str, *, max_length: int) -> list[int]:
        """Получить BOS + токены + EOS, обрезая только середину.

        Args:
            text: Вопрос из любой выборки или новый текст.
            max_length: Максимальная полная длина, включая BOS и EOS (>=2).
        Returns:
            Список целочисленных ID без padding; длина не больше max_length.
        """
        if type(max_length) is not int or max_length < 2:
            raise DataPipelineError("max_length must be an integer >= 2")
        terms = tokenize(text)[:max_length - 2]
        return [BOS_ID, *(self.token_to_id.get(term, UNK_ID) for term in terms), EOS_ID]


def build_vocab(train_rows: Sequence[dict[str, Any]], *, max_vocab_size: int = 12000,
                min_freq: int = 1) -> Vocabulary:
    """Построить частотный словарь строго по train, без просмотра val/test.

    Частоты упорядочиваются по убыванию, равные частоты — по строке токена;
    одинаковые входные записи дают одинаковую нумерацию независимо от порядка.
    """
    if type(max_vocab_size) is not int or max_vocab_size < len(SPECIAL_TOKENS):
        raise DataPipelineError("max_vocab_size is too small")
    if type(min_freq) is not int or min_freq < 1:
        raise DataPipelineError("min_freq must be a positive integer")
    if not isinstance(train_rows, (list, tuple)) or not train_rows:
        raise DataPipelineError("train rows must be nonempty")
    frequencies: Counter[str] = Counter()
    for index, row in enumerate(train_rows):
        if not isinstance(row, dict) or "context" not in row:
            raise DataPipelineError(f"missing context in train[{index}]")
        frequencies.update(tokenize(row["context"]))

    ordering = sorted(
        ((token, count) for token, count in frequencies.items() if count >= min_freq),
        key=lambda pair: (-pair[1], pair[0]),
    )
    mapping = {token: index for index, token in enumerate(SPECIAL_TOKENS)}
    for token, _ in ordering:
        if len(mapping) >= max_vocab_size:
            break
        # Не переопределяем служебные ID, даже если такой текст встретился.
        if token not in mapping:
            mapping[token] = len(mapping)
    return Vocabulary(mapping)


class QuestionDataset(Dataset):
    """Отображать записи одного split в ID токенов и числовые метки.

    Возвращает последовательности переменной длины. Padding выполняется не
    здесь, а в collate_fn при формировании батча.
    """

    def __init__(self, rows: Sequence[dict[str, Any]], *, vocab: Vocabulary,
                 label_to_id: Mapping[str, int], max_length: int):
        """Проверить метки и параметры без изменения исходных словарей."""
        if not isinstance(rows, (list, tuple)) or not rows:
            raise DataPipelineError("dataset rows must be a nonempty sequence")
        if not isinstance(vocab, Vocabulary):
            raise DataPipelineError("vocab must be a Vocabulary")
        if not isinstance(label_to_id, Mapping) or not label_to_id:
            raise DataPipelineError("label mapping must be nonempty")
        if set(label_to_id.values()) != set(range(len(label_to_id))):
            raise DataPipelineError("label indices must be contiguous and unique")
        if type(max_length) is not int or max_length < 2:
            raise DataPipelineError("max_length must be an integer >= 2")
        self.rows = rows
        self.vocab = vocab
        self.label_to_id = dict(label_to_id)
        self.max_length = max_length
        for index, row in enumerate(rows):
            if not isinstance(row, dict) or not isinstance(row.get("id"), str):
                raise DataPipelineError(f"invalid id in row {index}")
            if row.get("domain") not in self.label_to_id:
                raise DataPipelineError(f"unknown class in row {index}")
            tokenize(row.get("context"))

    def __len__(self) -> int:
        """Вернуть число записей данного split."""
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Вернуть один пример с целочисленными тензорами без padding."""
        row = self.rows[index]
        return {
            "input_ids": torch.tensor(
                self.vocab.encode(row["context"], max_length=self.max_length), dtype=torch.long
            ),
            "label": torch.tensor(self.label_to_id[row["domain"]], dtype=torch.long),
            "sample_id": row["id"],
        }


def collate_questions(samples: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Выровнять последовательности в батче вправо PAD и построить маску.

    attention_mask=True только для действительных токенов, в том числе BOS
    и EOS. Такой формат пригодится в F2.3 для masked pooling.
    """
    if not samples:
        raise DataPipelineError("cannot collate an empty batch")
    lengths = [int(sample["input_ids"].numel()) for sample in samples]
    if any(length < 2 for length in lengths):
        raise DataPipelineError("each encoded question must contain BOS and EOS")
    width = max(lengths)
    ids = torch.full((len(samples), width), PAD_ID, dtype=torch.long)
    mask = torch.zeros((len(samples), width), dtype=torch.bool)
    labels = torch.empty(len(samples), dtype=torch.long)
    sample_ids: list[str] = []
    for i, (sample, length) in enumerate(zip(samples, lengths)):
        sequence = sample["input_ids"]
        if not isinstance(sequence, torch.Tensor) or sequence.ndim != 1 or sequence.dtype != torch.long:
            raise DataPipelineError("input_ids must be 1D torch.long tensors")
        if sequence[0].item() != BOS_ID or sequence[-1].item() != EOS_ID:
            raise DataPipelineError("encoded sequence must start with BOS and end with EOS")
        if (not isinstance(sample["label"], torch.Tensor) or
                sample["label"].ndim != 0 or sample["label"].dtype != torch.long):
            raise DataPipelineError("label must be a scalar torch.long tensor")
        ids[i, :length] = sequence
        mask[i, :length] = True
        labels[i] = sample["label"]
        sample_ids.append(sample["sample_id"])
    return {"input_ids": ids, "attention_mask": mask, "labels": labels,
            "sample_ids": sample_ids}


@dataclass
class PreparedData:
    """Контейнер общих отображений и трёх Dataset на неизменном split F1."""

    vocab: Vocabulary
    label_to_id: dict[str, int]
    datasets: dict[str, QuestionDataset]
    counts: dict[str, int]


def prepare_data(splits: Mapping[str, Sequence[dict[str, Any]]], *,
                 max_vocab_size: int = 12000, max_length: int = 128) -> PreparedData:
    """Проверить разбиение и использовать train как единственный источник fit."""
    counts = validate_splits(splits)
    vocab = build_vocab(splits["train"], max_vocab_size=max_vocab_size)
    labels = sorted({row["domain"] for row in splits["train"]})
    label_to_id = {label: index for index, label in enumerate(labels)}
    datasets = {name: QuestionDataset(
        splits[name], vocab=vocab, label_to_id=label_to_id, max_length=max_length
    ) for name in _SPLITS}
    return PreparedData(vocab, label_to_id, datasets, counts)


def make_loader(dataset: QuestionDataset, *, batch_size: int = 32,
                shuffle: bool = False, seed: int = 42) -> DataLoader:
    """Создать CPU DataLoader с воспроизводимым порядком при заданном seed.

    Установка seed только в локальном torch.Generator не изменяет глобальный
    генератор случайных чисел; num_workers=0 упрощает Windows-совместимость.
    """
    if not isinstance(dataset, QuestionDataset):
        raise DataPipelineError("expected QuestionDataset")
    if type(batch_size) is not int or batch_size < 1:
        raise DataPipelineError("batch_size must be a positive integer")
    if type(seed) is not int or seed < 0:
        raise DataPipelineError("seed must be a nonnegative integer")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, collate_fn=collate_questions, generator=generator)


def read_f1_splits(data_dir: Path = ROOT / "data/processed") -> dict[str, list[dict]]:
    """Прочитать только существующие train/val/test JSONL из F1.6.

    Функция не инициирует новый split, скачивание, перезапись или экспорт
    текстовых примеров за пределы локальной памяти.
    """
    root = Path(data_dir)
    paths = {name: root / f"{name}.jsonl" for name in _SPLITS}
    for name, path in paths.items():
        if not path.is_file():
            raise DataPipelineError(f"missing prepared F1 split: {path}")
    try:
        splits = {name: read_jsonl(path) for name, path in paths.items()}
    except (ValueError, TypeError, OSError) as exc:
        raise DataPipelineError("cannot parse F1 JSONL split") from exc
    validate_splits(splits)
    return splits
