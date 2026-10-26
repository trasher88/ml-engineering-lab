"""F2.2: синтетические тесты токенизации, словаря и батчей PyTorch.

Проверяется отсутствие утечек F1, детерминизм, типы, формы, маски и ошибки.
Ни один тест не читает и не публикует реальные Q&A из локального корпуса.
"""

from __future__ import annotations

import copy
import json

import pytest
import torch

from scripts.pytorch_batches import run, synthetic_splits
from src.pytorch_f2.data_pipeline import (
    BOS_ID, EOS_ID, PAD_ID, SPECIAL_TOKENS, UNK_ID, DataPipelineError,
    QuestionDataset, Vocabulary, build_vocab, collate_questions, make_loader,
    prepare_data, read_f1_splits, tokenize, validate_splits,
)


def fixtures():
    """Вернуть новые независимые объекты для каждого теста."""
    return synthetic_splits()


def test_unicode_tokenization_and_casefold():
    """Слова Unicode и пунктуация различаются, регистр не создаёт новых токенов."""
    assert tokenize("PyTorch, ЛИНУКС!") == ["pytorch", ",", "линукс", "!"]


@pytest.mark.parametrize("value", ["", "   ", None, 12])
def test_reject_empty_or_nontext(value):
    """Пустые и нестроковые вопросы не формируют фиктивные примеры."""
    with pytest.raises(DataPipelineError, match="nonempty"):
        tokenize(value)


def test_special_token_indices_are_stable():
    """Служебные ID неизменны и занимают первые четыре позиции словаря."""
    vocab = build_vocab(fixtures()["train"])
    assert [vocab.token_to_id[t] for t in SPECIAL_TOKENS] == [0, 1, 2, 3]


def test_train_only_vocab_does_not_leak_val_test_tokens():
    """Слова из validation/test не добавляются в словарь при fit."""
    prepared = prepare_data(fixtures())
    assert "unseenlinux" not in prepared.vocab.token_to_id
    assert "unseensystem" not in prepared.vocab.token_to_id
    assert prepared.datasets["val"][0]["input_ids"].tolist()[1] == UNK_ID
    assert prepared.datasets["test"][0]["input_ids"].tolist()[1] == UNK_ID


def test_vocab_frequency_ranking_and_determinism():
    """При одинаковых частотах порядок лексикографический, а не случайный."""
    train = fixtures()["train"]
    a = build_vocab(train)
    b = build_vocab(list(reversed(train)))
    assert a.token_to_id == b.token_to_id
    assert a.token_to_id["linux"] < a.token_to_id["kernel"]


def test_vocab_limit_and_min_freq():
    """Ограничение словаря сохраняет служебные ID, редкие слова идут в UNK."""
    rows = fixtures()["train"]
    small = build_vocab(rows, max_vocab_size=5)
    assert len(small) == 5
    assert UNK_ID in small.encode("linux kernel", max_length=12)
    frequent = build_vocab(rows, min_freq=3)
    assert "kernel" not in frequent.token_to_id


@pytest.mark.parametrize("size", [0, 1, 3, True, 1.5])
def test_bad_vocab_size_fails(size):
    """max_vocab_size должен быть целым и вмещать специальные токены."""
    with pytest.raises(DataPipelineError, match="max_vocab_size"):
        build_vocab(fixtures()["train"], max_vocab_size=size)


@pytest.mark.parametrize("frequency", [0, -2, True, 1.5])
def test_bad_min_frequency_fails(frequency):
    """Частотный порог не допускает нуля, bool и дробного значения."""
    with pytest.raises(DataPipelineError, match="min_freq"):
        build_vocab(fixtures()["train"], min_freq=frequency)


def test_encode_truncation_preserves_bos_and_eos():
    """Длинный вопрос ограничивается max_length без потери краевых токенов."""
    vocab = build_vocab(fixtures()["train"])
    ids = vocab.encode("one two three four five", max_length=4)
    assert len(ids) == 4
    assert ids[0] == BOS_ID and ids[-1] == EOS_ID
    assert vocab.encode("anything", max_length=2) == [BOS_ID, EOS_ID]


@pytest.mark.parametrize("max_length", [0, 1, -1, True, 2.5])
def test_invalid_max_length_fails(max_length):
    """Длина должна вмещать два специальных токена и быть целой."""
    with pytest.raises(DataPipelineError, match="max_length"):
        build_vocab(fixtures()["train"]).encode("valid text", max_length=max_length)


def test_dataset_lengths_and_long_scalar_labels():
    """Dataset возвращает один вопрос и скаляр класса в torch.long."""
    prepared = prepare_data(fixtures())
    assert {name: len(ds) for name, ds in prepared.datasets.items()} == prepared.counts
    sample = prepared.datasets["train"][0]
    assert sample["input_ids"].dtype == torch.long
    assert sample["label"].dtype == torch.long
    assert sample["label"].shape == torch.Size([])
    assert sample["sample_id"] == "synthetic-0"


def test_label_ids_are_alphabetic_and_shared_between_splits():
    """Классы нумеруются по train; val/test получают ту же таблицу."""
    prepared = prepare_data(fixtures())
    assert prepared.label_to_id == {"linux": 0, "systems": 1}
    assert prepared.datasets["val"][0]["label"].item() == 0
    assert prepared.datasets["test"][0]["label"].item() == 1


def test_collation_padding_attention_mask_and_shapes():
    """Проверить динамический padding и маску только по фактической длине."""
    prepared = prepare_data(fixtures())
    batch = collate_questions([
        prepared.datasets["train"][0], prepared.datasets["train"][1],
    ])
    ids, mask, labels = batch["input_ids"], batch["attention_mask"], batch["labels"]
    assert ids.shape == mask.shape == (2, 5)
    assert labels.shape == (2,)
    assert ids.dtype == labels.dtype == torch.long
    assert mask.dtype == torch.bool
    assert ids.device.type == "cpu"
    assert torch.equal(ids == PAD_ID, ~mask)
    assert all(ids[i, mask[i]][0].item() == BOS_ID for i in range(2))
    assert all(ids[i, mask[i]][-1].item() == EOS_ID for i in range(2))


def test_collation_real_padding_with_unequal_lengths():
    """На более коротком вопросе возникают нули ID и False в маске."""
    prepared = prepare_data(fixtures())
    short = prepared.datasets["train"][0]
    long = prepared.datasets["train"][1]
    # Искусственно укоротим один кодированный пример, не меняя источник Dataset.
    short = {**short, "input_ids": torch.tensor([BOS_ID, EOS_ID])}
    batch = collate_questions([short, long])
    assert batch["input_ids"][0, 2:].tolist() == [PAD_ID] * 3
    assert batch["attention_mask"][0].tolist() == [True, True, False, False, False]


def test_reject_empty_batch():
    """Нельзя создать корректный тензор из пустого набора примеров."""
    with pytest.raises(DataPipelineError, match="empty batch"):
        collate_questions([])


def test_dataloader_last_partial_batch():
    """Последняя неполная пачка остаётся доступной без потери записей."""
    prepared = prepare_data(fixtures())
    loader = make_loader(prepared.datasets["train"], batch_size=3)
    batches = list(loader)
    assert [b["labels"].shape[0] for b in batches] == [3, 1]


def test_dataloader_seed_reproduces_shuffle_without_global_seed():
    """Независимые DataLoader с одинаковым seed дают один порядок ID."""
    prepared = prepare_data(fixtures())
    def ids(seed: int) -> list[str]:
        """Собрать только идентификаторы из независимого прохода DataLoader."""
        return [sid for batch in make_loader(prepared.datasets["train"],
               batch_size=2, shuffle=True, seed=seed) for sid in batch["sample_ids"]]
    assert ids(21) == ids(21)
    assert sorted(ids(21)) == ["synthetic-0", "synthetic-1", "synthetic-2", "synthetic-3"]


def test_no_source_mutation_and_repeatable_preparation():
    """Подготовка не меняет входные строки и повторяется побайтно по отображениям."""
    split = fixtures()
    original = copy.deepcopy(split)
    first = prepare_data(split)
    second = prepare_data(split)
    assert split == original
    assert first.vocab.token_to_id == second.vocab.token_to_id
    assert first.label_to_id == second.label_to_id


@pytest.mark.parametrize("violation, match", [
    ("wrong_split", "split marker"),
    ("duplicate_id", "duplicate record"),
    ("group_overlap", "group leakage"),
    ("unseen_class", "unseen label"),
    ("empty_context", "required field"),
])
def test_split_contract_rejects_leakage_and_invalid_data(violation, match):
    """Ошибки F1.6 блокируются до fit словаря и кодирования выборок."""
    rows = fixtures()
    if violation == "wrong_split":
        rows["val"][0]["split"] = "test"
    elif violation == "duplicate_id":
        rows["test"][0]["id"] = rows["train"][0]["id"]
    elif violation == "group_overlap":
        rows["val"][0]["group_id"] = rows["train"][0]["group_id"]
    elif violation == "unseen_class":
        rows["test"][0]["domain"] = "unknown"
    elif violation == "empty_context":
        rows["train"][0]["context"] = ""
    with pytest.raises(DataPipelineError, match=match):
        prepare_data(rows)


def test_reject_missing_or_extra_splits():
    """Нельзя незаметно пропустить test или подмешать новую выборку."""
    rows = fixtures()
    del rows["test"]
    with pytest.raises(DataPipelineError, match="exactly"):
        validate_splits(rows)
    rows = fixtures()
    rows["other"] = rows["val"]
    with pytest.raises(DataPipelineError, match="exactly"):
        validate_splits(rows)


def test_reject_unseen_label_in_dataset():
    """Классы validation/test не расширяют train-таблицу меток."""
    prepared = prepare_data(fixtures())
    with pytest.raises(DataPipelineError, match="unknown class"):
        QuestionDataset([{"id": "s", "context": "safe text", "domain": "unseen"}],
                        vocab=prepared.vocab, label_to_id=prepared.label_to_id,
                        max_length=20)


def test_read_existing_f1_split_paths_and_preserve_order(tmp_path):
    """Локальная загрузка совместима с контрактом F1.6, без нового split."""
    rows = fixtures()
    for name, values in rows.items():
        (tmp_path / f"{name}.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in values),
            encoding="utf-8",
        )
    loaded = read_f1_splits(tmp_path)
    assert loaded == rows


def test_missing_f1_files_raise_clear_error(tmp_path):
    """Отсутствующие локальные данные не заменяются скачанными или тестовыми."""
    with pytest.raises(DataPipelineError, match="missing prepared"):
        read_f1_splits(tmp_path)


def test_synthetic_cli_is_text_free_and_deterministic():
    """Smoke test выдаёт только агрегаты, без исходных текстов и ID."""
    result = run(synthetic=True)
    assert result == run(synthetic=True)
    assert result["split_counts"] == {"train": 4, "val": 1, "test": 1}
    assert result["batch_invariants_pass"]
    output = json.dumps(result, ensure_ascii=False)
    assert "synthetic-" not in output and "linux kernel" not in output


def test_synthetic_cli_expectation_gate():
    """Ошибочный ожидаемый размер split завершает подготовку с ошибкой."""
    with pytest.raises(DataPipelineError, match="unexpected train size"):
        run(synthetic=True, expect_train=100)


def test_vocab_rejects_corrupt_token_mapping():
    """Недопустимо изменять ID PAD/UNK/BOS/EOS или создавать пропуски."""
    with pytest.raises(DataPipelineError, match="special"):
        Vocabulary({"<PAD>": 1, "<UNK>": 0, "<BOS>": 2, "<EOS>": 3})
    with pytest.raises(DataPipelineError, match="contiguous"):
        Vocabulary({"<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3, "foo": 7})


@pytest.mark.parametrize("batch_size", [0, -1, True, 1.2])
def test_loader_rejects_invalid_batch_size(batch_size):
    """DataLoader не принимает дробный, булев или неположительный batch_size."""
    dataset = prepare_data(fixtures()).datasets["train"]
    with pytest.raises(DataPipelineError, match="batch_size"):
        make_loader(dataset, batch_size=batch_size)
