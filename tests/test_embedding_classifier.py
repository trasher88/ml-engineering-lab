"""F2.3: синтетические тесты архитектуры, формы logits и masked pooling.

Ни один тест не читает F1 Q&A или обучает модель. Проверяются крайние
случаи padding, значения градиентов, воспроизводимость и интеграция с F2.2.
"""

from __future__ import annotations

import json

import pytest
import torch

from scripts.pytorch_batches import synthetic_splits
from scripts.pytorch_classifier import run
from src.pytorch_f2.data_pipeline import PAD_ID, make_loader, prepare_data
from src.pytorch_f2.embedding_classifier import (
    EmbeddingClassifier, ModelInputError, masked_mean_pool,
)


def model(*, vocab_size: int = 12, num_classes: int = 3,
          embedding_dim: int = 4, hidden_dim: int = 5) -> EmbeddingClassifier:
    """Создать небольшую независимую модель для одного синтетического теста."""
    return EmbeddingClassifier(vocab_size=vocab_size, num_classes=num_classes,
                               embedding_dim=embedding_dim, hidden_dim=hidden_dim)


def inputs() -> tuple[torch.Tensor, torch.Tensor]:
    """Вернуть два примера разной длины с правым заполнением PAD."""
    ids = torch.tensor([[2, 4, 3, 0], [2, 5, 6, 3]], dtype=torch.long)
    mask = ids != PAD_ID
    return ids, mask


def test_embedding_classifier_is_nn_module_with_expected_layers():
    """Модель использует стандартные обучаемые слои PyTorch, без потерь."""
    net = model()
    assert isinstance(net, torch.nn.Module)
    assert isinstance(net.embedding, torch.nn.Embedding)
    assert isinstance(net.hidden, torch.nn.Linear)
    assert isinstance(net.activation, torch.nn.ReLU)
    assert isinstance(net.output, torch.nn.Linear)
    assert net.embedding.padding_idx == PAD_ID
    assert net.embedding.weight.shape == (12, 4)
    assert net.hidden.weight.shape == (5, 4)
    assert net.output.weight.shape == (3, 5)


def test_parameter_count_is_exact():
    """Число параметров равно сумме embedding, весов и bias двух Linear."""
    net = model()
    expected = 12 * 4 + (4 * 5 + 5) + (5 * 3 + 3)
    assert sum(p.numel() for p in net.parameters()) == expected


def test_forward_shape_dtype_and_device():
    """Два текста должны породить по одному вектору logits размера C."""
    ids, mask = inputs()
    logits = model()(ids, mask)
    assert logits.shape == (2, 3)
    assert logits.dtype == torch.float32
    assert logits.device.type == "cpu"
    assert bool(torch.isfinite(logits).all())


def test_single_row_returns_matrix_not_vector():
    """При batch_size=1 ось B сохраняется, что требуется для обучения."""
    ids, mask = inputs()
    assert model()(ids[:1], mask[:1]).shape == (1, 3)


def test_model_returns_raw_logits_not_probabilities():
    """Softmax нельзя добавлять до CrossEntropyLoss следующего этапа."""
    net = model()
    with torch.no_grad():
        net.output.weight.zero_()
        net.output.bias.fill_(2.0)
    logits = net(*inputs())
    assert torch.equal(logits, torch.full((2, 3), 2.0))
    assert not torch.allclose(logits.sum(dim=1), torch.ones(2))


def test_pooling_ignores_masked_vectors_exactly():
    """Числовой пример: усреднение [2,4] исключает огромный padding."""
    vectors = torch.tensor([[[2.0, 4.0], [4.0, 6.0], [1000.0, 1000.0]]])
    mask = torch.tensor([[True, True, False]])
    assert torch.equal(masked_mean_pool(vectors, mask), torch.tensor([[3.0, 5.0]]))


def test_pooling_denominator_uses_true_tokens_not_padded_length():
    """Усредняем по двум валидным позициям, а не по длине L=4."""
    vectors = torch.tensor([[[2.0], [6.0], [0.0], [0.0]]])
    mask = torch.tensor([[True, True, False, False]])
    assert masked_mean_pool(vectors, mask).item() == 4.0


def test_pooling_preserves_float64_for_numerical_checks():
    """Отдельная операция pooling не понижает точность float64."""
    vectors = torch.tensor([[[3.0], [5.0]]], dtype=torch.float64)
    pooled = masked_mean_pool(vectors, torch.tensor([[True, True]]))
    assert pooled.dtype == torch.float64 and pooled.item() == 4.0


def test_pooling_preserves_gradient_on_unmasked_tokens():
    """Автодифференцирование проходит через среднее, но не через PAD."""
    vectors = torch.tensor([[[1.0], [3.0], [99.0]]], requires_grad=True)
    mask = torch.tensor([[True, True, False]])
    masked_mean_pool(vectors, mask).sum().backward()
    assert torch.equal(vectors.grad.flatten(), torch.tensor([0.5, 0.5, 0.0]))


def test_known_weights_yield_expected_logits():
    """Численно проследить Embedding → mean → ReLU → два logits."""
    net = model(vocab_size=8, num_classes=2, embedding_dim=2, hidden_dim=2)
    with torch.no_grad():
        net.embedding.weight.zero_()
        net.embedding.weight[2] = torch.tensor([2.0, 0.0])
        net.embedding.weight[3] = torch.tensor([0.0, 2.0])
        net.embedding.weight[4] = torch.tensor([4.0, 0.0])
        net.hidden.weight.copy_(torch.eye(2))
        net.hidden.bias.zero_()
        net.output.weight.copy_(torch.eye(2))
        net.output.bias.zero_()
    ids = torch.tensor([[2, 3, 0], [2, 4, 3]])
    mask = ids != PAD_ID
    expected = torch.tensor([[1.0, 1.0], [2.0, 2.0 / 3.0]])
    assert torch.allclose(net(ids, mask), expected, atol=1e-6)


def test_extra_padding_has_no_effect_on_logits():
    """Одинаковый вопрос не зависит от длины самого длинного текста батча."""
    net = model().eval()
    ids, mask = inputs()
    longer_ids = torch.nn.functional.pad(ids, (0, 7), value=PAD_ID)
    longer_mask = torch.nn.functional.pad(mask, (0, 7), value=False)
    with torch.no_grad():
        assert torch.allclose(net(ids, mask), net(longer_ids, longer_mask), atol=1e-7)


def test_padding_embedding_value_is_explicitly_masked():
    """Даже искусственная ненулевая PAD-строка не влияет на forward."""
    net = model().eval()
    ids, mask = inputs()
    with torch.no_grad():
        before = net(ids, mask)
        net.embedding.weight[PAD_ID].fill_(10000.0)
        after = net(ids, mask)
    assert torch.allclose(before, after, atol=1e-7)


def test_batch_composition_does_not_change_first_prediction():
    """Другие тексты в батче не должны влиять на вектор данного вопроса."""
    net = model().eval()
    ids, mask = inputs()
    with torch.no_grad():
        single = net(ids[:1, :3], mask[:1, :3])
        together = net(ids, mask)[:1]
    assert torch.allclose(single, together, atol=1e-7)


def test_embedding_padding_row_receives_zero_gradient():
    """PyTorch исключает PAD_ID из обучения таблицы nn.Embedding."""
    net = model()
    with torch.no_grad():
        net.hidden.weight.fill_(0.5)
        net.hidden.bias.fill_(1.0)
        net.output.weight.fill_(0.5)
        net.output.bias.zero_()
    logits = net(*inputs())
    logits.sum().backward()
    assert net.embedding.weight.grad is not None
    assert torch.all(net.embedding.weight.grad[PAD_ID] == 0)
    assert torch.any(net.embedding.weight.grad[2] != 0)
    assert net.hidden.weight.grad is not None
    assert net.output.weight.grad is not None


def test_model_integrates_with_f2_2_dataset_and_loader():
    """Берём настоящий формат батча из F2.2, без адаптеров и перезаписи F1."""
    prepared = prepare_data(synthetic_splits())
    batch = next(iter(make_loader(prepared.datasets["train"], batch_size=3, seed=42)))
    net = EmbeddingClassifier(vocab_size=len(prepared.vocab),
                              num_classes=len(prepared.label_to_id))
    logits = net(batch["input_ids"], batch["attention_mask"])
    assert logits.shape == (3, 2)
    assert batch["labels"].shape == (3,)


def test_model_accepts_special_only_sequence():
    """Пустой по смысловым словам текст с BOS/EOS корректен для модели."""
    ids = torch.tensor([[2, 3]])
    mask = torch.tensor([[True, True]])
    assert model()(ids, mask).shape == (1, 3)


@pytest.mark.parametrize("kwargs", [
    {"vocab_size": 3}, {"vocab_size": 0}, {"vocab_size": True},
    {"num_classes": 1}, {"num_classes": 0}, {"num_classes": False},
    {"embedding_dim": 0}, {"embedding_dim": 1.5}, {"embedding_dim": True},
    {"hidden_dim": -1}, {"hidden_dim": 2.0}, {"hidden_dim": False},
])
def test_invalid_constructor_parameters_fail(kwargs):
    """Размеры слоёв обязаны быть целыми положительными числами, кроме bool."""
    with pytest.raises(ModelInputError):
        model(**kwargs)


@pytest.mark.parametrize("broken", [
    torch.tensor([2, 3]),
    torch.empty(0, 2, dtype=torch.long),
    torch.empty(1, 0, dtype=torch.long),
    torch.ones(2, 3, 4, dtype=torch.long),
    torch.tensor([[2.0, 3.0]]),
    None,
])
def test_invalid_input_id_shapes_or_dtype_fail(broken):
    """Embedding ожидает матрицу целочисленных ID размера [B, L]."""
    ids, mask = inputs()
    with pytest.raises(ModelInputError):
        model()(broken, mask)


@pytest.mark.parametrize("bad_mask", [
    torch.ones(2, 4, dtype=torch.long),
    torch.ones(1, 4, dtype=torch.bool),
    torch.ones(2, 2, 4, dtype=torch.bool),
    None,
])
def test_invalid_mask_dtype_or_shape_fail(bad_mask):
    """Маска обязана иметь bool dtype и форму [B, L] без broadcasting."""
    ids, _ = inputs()
    with pytest.raises(ModelInputError):
        model()(ids, bad_mask)


def test_mismatched_padding_and_mask_fail():
    """Нельзя включать нулевой PAD_ID в настоящий текст или скрывать слово."""
    ids, mask = inputs()
    mask[0, -1] = True
    with pytest.raises(ModelInputError, match="PAD positions"):
        model()(ids, mask)


def test_mask_hole_before_real_token_fails():
    """Маска должна отражать правосторонний padding F2.2 без пробелов."""
    ids = torch.tensor([[2, 0, 3]], dtype=torch.long)
    mask = torch.tensor([[True, False, True]])
    with pytest.raises(ModelInputError, match="right-aligned"):
        model()(ids, mask)


@pytest.mark.parametrize("bad_id", [-1, 12, 100])
def test_out_of_vocabulary_indices_fail(bad_id):
    """Недопустимый ID не должен превращаться в незаметную ошибку Embedding."""
    ids, mask = inputs()
    ids[0, 1] = bad_id
    with pytest.raises(ModelInputError, match="outside"):
        model()(ids, mask)


def test_all_padding_row_fails():
    """При нулевом числе реальных токенов среднее не определено."""
    ids = torch.zeros(1, 2, dtype=torch.long)
    mask = torch.zeros(1, 2, dtype=torch.bool)
    with pytest.raises(ModelInputError, match="valid tokens"):
        model()(ids, mask)


@pytest.mark.parametrize("vectors,masks", [
    (torch.ones(1, 2), torch.ones(1, 2, dtype=torch.bool)),
    (torch.ones(1, 2, 3, dtype=torch.int64), torch.ones(1, 2, dtype=torch.bool)),
    (torch.ones(1, 2, 3), torch.ones(1, 3, dtype=torch.bool)),
    (torch.ones(1, 2, 3), torch.ones(1, 2, dtype=torch.int64)),
    (torch.ones(1, 2, 3), torch.zeros(1, 2, dtype=torch.bool)),
])
def test_masked_pool_rejects_invalid_inputs(vectors, masks):
    """Ошибки размера, типа и пустой строки перехватываются до деления."""
    with pytest.raises(ModelInputError):
        masked_mean_pool(vectors, masks)


def test_initialization_is_deterministic_with_local_rng():
    """Одинаковый seed даёт идентичный forward без изменения глобального RNG."""
    ids, mask = inputs()
    original = torch.random.get_rng_state().clone()
    results = []
    for _ in range(2):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(42)
            results.append(model()(ids, mask).detach())
    assert torch.equal(results[0], results[1])
    assert torch.equal(original, torch.random.get_rng_state())


def test_cli_synthetic_smoke_contract():
    """CLI выдаёт только агрегаты формы, параметров и статуса без обучения."""
    report = run(synthetic=True)
    assert report["split_counts"] == {"train": 4, "val": 1, "test": 1}
    assert report["batch_shape"] == [4, 5]
    assert report["logits_shape"] == [4, 2]
    assert report["parameter_count"] > 0
    assert report["padding_invariant"] is True
    assert report["repeatable_forward"] is True
    assert report["forward_invariants_pass"] is True
    assert report["trained"] is False
    assert "questions" not in report and "sample_ids" not in report


def test_cli_repeat_is_deterministic():
    """Синтетический отчёт воспроизводим при повторном запуске."""
    assert run(synthetic=True) == run(synthetic=True)


def test_cli_rejects_mismatched_expected_counts():
    """Случайно подменённый split выявляется до построения выводов."""
    with pytest.raises(ModelInputError, match="unexpected train"):
        run(synthetic=True, expect_train=1824)


def test_cli_fails_if_local_config_is_absent(tmp_path):
    """Локальный запуск не подменяет отсутствующую конфигурацию фикстурами."""
    with pytest.raises(ModelInputError, match="missing F1 configuration"):
        run(root=tmp_path)


def test_cli_local_mode_uses_existing_split_and_config(tmp_path):
    """Проверить путь локального запуска только на синтетических JSONL F1."""
    config_dir = tmp_path / "configs" / "local_cpu"
    config_dir.mkdir(parents=True)
    config_dir.joinpath("lab.yaml").write_text(
        "seed: 42\nneural:\n  vocab_size: 100\n  max_length: 8\n"
        "  batch_size: 3\n  embedding_dim: 3\n  hidden_dim: 4\n",
        encoding="utf-8",
    )
    data_dir = tmp_path / "data" / "processed"
    data_dir.mkdir(parents=True)
    for name, rows in synthetic_splits().items():
        (data_dir / f"{name}.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )
    report = run(root=tmp_path, expect_train=4, expect_val=1, expect_test=1)
    assert report["source"] == "local_f1_split"
    assert report["batch_shape"] == [3, 5]
    assert report["logits_shape"] == [3, 2]
    assert report["embedding_dim"] == 3
    assert report["hidden_dim"] == 4
    assert report["parameter_count"] == 11 * 3 + (3 * 4 + 4) + (4 * 2 + 2)
    assert report["trained"] is False


def test_cli_rejects_missing_neural_parameter(tmp_path):
    """Неполная конфигурация не должна молча заменяться случайным значением."""
    config_dir = tmp_path / "configs" / "local_cpu"
    config_dir.mkdir(parents=True)
    config_dir.joinpath("lab.yaml").write_text(
        "seed: 42\nneural:\n  vocab_size: 12000\n",
        encoding="utf-8",
    )
    with pytest.raises(ModelInputError, match="missing classifier parameter"):
        run(root=tmp_path)
