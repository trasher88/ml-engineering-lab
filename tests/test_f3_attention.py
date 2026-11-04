"""Регрессионные тесты F3.1: формулы attention, masking и autograd.

Каждый тест использует только синтетические числовые тензоры. Проверки
позиций padding защищают от влияния фиктивных токенов на результаты.
"""

from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

# Импорт src-layout не зависит от editable install или текущего каталога pytest.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from transformers_f3 import SelfAttention, scaled_dot_product_attention  # noqa: E402


def _sample_tensors() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Подготовить маленькие Q/K/V с проверяемыми целыми значениями."""
    query = torch.tensor([[[1., 0.], [0., 1.]]], dtype=torch.float64)
    key = torch.tensor([[[1., 0.], [0., 1.], [1., 1.]]], dtype=torch.float64)
    value = torch.tensor([[[2., 0.], [0., 4.], [3., 3.]]], dtype=torch.float64)
    return query, key, value


def test_numeric_example_matches_manual_softmax_and_weighted_sum() -> None:
    """Сравнить с ручным Softmax: [1/sqrt(2), 0] и двумя V."""
    q = torch.tensor([[[1.0, 0.0]]], dtype=torch.float64)
    k = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]], dtype=torch.float64)
    v = torch.tensor([[[2.0, 0.0], [0.0, 4.0]]], dtype=torch.float64)
    context, weights = scaled_dot_product_attention(q, k, v)
    first = math.exp(1.0 / math.sqrt(2.0))
    p1 = first / (first + 1.0)
    p2 = 1.0 - p1
    torch.testing.assert_close(weights[0, 0], torch.tensor([p1, p2], dtype=q.dtype))
    torch.testing.assert_close(context[0, 0], torch.tensor([2*p1, 4*p2], dtype=q.dtype))


def test_attention_shapes_allow_different_query_key_and_value_lengths() -> None:
    """Длина Lq отличается от Lk, а ширина Dv не обязана равняться Dk."""
    q = torch.ones(2, 3, 4)
    k = torch.ones(2, 5, 4)
    v = torch.ones(2, 5, 7)
    context, weights = scaled_dot_product_attention(q, k, v)
    assert context.shape == (2, 3, 7)
    assert weights.shape == (2, 3, 5)
    assert context.dtype == torch.float32
    assert context.device.type == "cpu"


def test_reference_matches_pytorch_functional_sdpa() -> None:
    """Ручные матричные операции сопоставимы со штатным эталоном PyTorch."""
    q, k, v = _sample_tensors()
    context, _ = scaled_dot_product_attention(q, k, v)
    expected = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
    torch.testing.assert_close(context, expected, atol=1e-12, rtol=1e-12)


def test_each_unmasked_attention_row_sums_to_one() -> None:
    """Softmax создаёт нормированные веса вдоль оси ключей."""
    q, k, v = _sample_tensors()
    _, weights = scaled_dot_product_attention(q, k, v)
    torch.testing.assert_close(weights.sum(-1), torch.ones(1, 2, dtype=q.dtype))
    assert bool((weights >= 0).all())


def test_key_padding_mask_has_exact_zero_weights() -> None:
    """Padding-ключ не получает долю внимания у настоящих запросов."""
    q, k, v = _sample_tensors()
    mask = torch.tensor([[True, True, False]])
    _, weights = scaled_dot_product_attention(q, k, v, key_padding_mask=mask)
    assert bool((weights[:, :, 2] == 0).all())
    torch.testing.assert_close(weights[:, :, :2].sum(-1), torch.ones(1, 2, dtype=q.dtype))


def test_changing_masked_key_and_value_does_not_change_valid_outputs() -> None:
    """Замена фиктивных токенов не меняет результаты для настоящих позиций."""
    q, k, v = _sample_tensors()
    mask = torch.tensor([[True, True, False]])
    original, original_weights = scaled_dot_product_attention(q, k, v, key_padding_mask=mask)
    changed_key, changed_value = k.clone(), v.clone()
    changed_key[:, -1] = 1e5
    changed_value[:, -1] = -1e5
    new, new_weights = scaled_dot_product_attention(
        q, changed_key, changed_value, key_padding_mask=mask
    )
    torch.testing.assert_close(original, new, atol=0, rtol=0)
    torch.testing.assert_close(original_weights, new_weights, atol=0, rtol=0)


def test_query_padding_zeroes_context_and_entire_weight_rows() -> None:
    """Отдельная маска запросов обнуляет выход фиктивной позиции."""
    q, k, v = _sample_tensors()
    query_mask = torch.tensor([[True, False]])
    context, weights = scaled_dot_product_attention(q, k, v, query_padding_mask=query_mask)
    assert torch.count_nonzero(context[:, 1]) == 0
    assert torch.count_nonzero(weights[:, 1]) == 0
    assert weights[0, 0].sum().item() == pytest.approx(1.0)


def test_different_batch_masks_are_applied_independently() -> None:
    """Маска одного элемента batch не меняет матрицу внимания другого."""
    q, k, v = _sample_tensors()
    mask = torch.tensor([[True, False, False], [True, True, False]])
    context, weights = scaled_dot_product_attention(q.repeat(2, 1, 1), k.repeat(2, 1, 1), v.repeat(2, 1, 1), key_padding_mask=mask)
    torch.testing.assert_close(weights[0, :, 0], torch.ones(2, dtype=q.dtype))
    assert bool((weights[0, :, 1:] == 0).all())
    assert bool((weights[1, :, 1] > 0).all())
    assert context.shape == (2, 2, 2)


def test_reference_gradients_match_pytorch_autograd() -> None:
    """Градиенты по Q/K/V совпадают с PyTorch SDPA для той же функции."""
    q, k, v = (t.clone().requires_grad_() for t in _sample_tensors())
    context, _ = scaled_dot_product_attention(q, k, v)
    context.square().sum().backward()
    manual_grads = tuple(t.grad.detach().clone() for t in (q, k, v))
    q2, k2, v2 = (t.detach().clone().requires_grad_() for t in (q, k, v))
    F.scaled_dot_product_attention(q2, k2, v2, dropout_p=0.0).square().sum().backward()
    for got, expected in zip(manual_grads, (q2.grad, k2.grad, v2.grad)):
        torch.testing.assert_close(got, expected, atol=1e-12, rtol=1e-12)


def test_masked_key_and_value_receive_no_gradient() -> None:
    """Не допускаем обратного распространения от потерь к padding-ключу."""
    q, k, v = (t.clone().requires_grad_() for t in _sample_tensors())
    mask = torch.tensor([[True, True, False]])
    context, _ = scaled_dot_product_attention(q, k, v, key_padding_mask=mask)
    context.sum().backward()
    assert torch.count_nonzero(k.grad[:, -1]) == 0
    assert torch.count_nonzero(v.grad[:, -1]) == 0


def test_self_attention_shapes_and_trainable_parameters() -> None:
    """Проекции Q/K/V — параметры nn.Module, выход сохраняет длину L."""
    model = SelfAttention(4, d_k=3, d_v=5)
    out, weights = model(torch.ones(2, 6, 4))
    assert out.shape == (2, 6, 5)
    assert weights.shape == (2, 6, 6)
    assert sum(p.numel() for p in model.parameters()) == 4 * (3 + 3 + 5)
    assert all(p.requires_grad for p in model.parameters())


def test_self_attention_identity_projections_match_direct_formula() -> None:
    """При единичных Wq/Wk/Wv модуль вычисляет Q=K=V=X."""
    model = SelfAttention(2).double()
    with torch.no_grad():
        for projection in (model.query_projection, model.key_projection, model.value_projection):
            projection.weight.copy_(torch.eye(2, dtype=torch.float64))
    x = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]], dtype=torch.float64)
    actual = model(x)
    expected = scaled_dot_product_attention(x, x, x)
    for result, target in zip(actual, expected):
        torch.testing.assert_close(result, target)


def test_self_attention_padding_rows_are_zero() -> None:
    """Padding-токен не участвует как ключ и не получает выходной вектор."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(9)
        model = SelfAttention(4)
    x = torch.randn(2, 4, 4)
    mask = torch.tensor([[True, True, False, False], [True, True, True, False]])
    out, weights = model(x, mask)
    for batch in range(2):
        invalid = ~mask[batch]
        assert bool((weights[batch, :, invalid] == 0).all())
        assert bool((weights[batch, invalid, :] == 0).all())
        assert bool((out[batch, invalid, :] == 0).all())
        torch.testing.assert_close(
            weights[batch, mask[batch]].sum(-1),
            torch.ones(mask[batch].sum().item()),
        )


def test_self_attention_padding_values_cannot_leak_into_real_tokens() -> None:
    """Независимость от padding защищает корректность батчирования F2.2."""
    model = SelfAttention(3)
    x = torch.tensor([[[1., 0., 0.], [0., 1., 0.], [0., 0., 0.]]])
    mask = torch.tensor([[True, True, False]])
    clean, _ = model(x, mask)
    x[:, -1, :] = torch.tensor([10000., -10000., 9999.])
    changed, _ = model(x, mask)
    torch.testing.assert_close(clean[:, :2], changed[:, :2], atol=0, rtol=0)


def test_padding_token_receives_zero_embedding_gradient() -> None:
    """Градиент к фиктивным входным токенам должен быть точно нулевым."""
    model = SelfAttention(3)
    x = torch.randn(1, 3, 3, requires_grad=True)
    mask = torch.tensor([[True, True, False]])
    out, _ = model(x, mask)
    out.sum().backward()
    assert torch.count_nonzero(x.grad[0, -1]) == 0
    assert bool(torch.isfinite(x.grad).all())


def test_optimizer_step_changes_attention_weights() -> None:
    """Самостоятельный модуль участвует в training loop с autograd и AdamW."""
    model = SelfAttention(3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    x = torch.tensor([[[1., 0., 0.], [0., 1., 0.]]])
    target = torch.zeros(1, 2, 3)
    before = model.query_projection.weight.detach().clone()
    optimizer.zero_grad()
    output, _ = model(x)
    ((output - target) ** 2).sum().backward()
    assert model.query_projection.weight.grad is not None
    assert bool(torch.isfinite(model.query_projection.weight.grad).all())
    optimizer.step()
    assert not torch.equal(before, model.query_projection.weight.detach())


def test_seed_controls_initialization_and_forward() -> None:
    """Повторение начального seed воспроизводит все параметры и выходы."""
    x = torch.arange(12, dtype=torch.float32).reshape(1, 3, 4)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(77)
        first = SelfAttention(4)
        torch.manual_seed(77)
        second = SelfAttention(4)
    for left, right in zip(first.parameters(), second.parameters()):
        torch.testing.assert_close(left, right, atol=0, rtol=0)
    for left, right in zip(first(x), second(x)):
        torch.testing.assert_close(left, right, atol=0, rtol=0)


def test_forward_does_not_mutate_inputs_or_masks() -> None:
    """Attention — чистое преобразование входов и не меняет маску F2.2."""
    q, k, v = _sample_tensors()
    mask = torch.tensor([[True, True, False]])
    originals = [t.clone() for t in (q, k, v, mask)]
    scaled_dot_product_attention(q, k, v, key_padding_mask=mask)
    for current, old in zip((q, k, v, mask), originals):
        assert torch.equal(current, old)


@pytest.mark.parametrize("invalid", [0, -1, 0.5, True, "4"])
def test_self_attention_rejects_invalid_d_model(invalid: object) -> None:
    """Размерность признаков должна быть строго положительным целым числом."""
    with pytest.raises(ValueError, match="d_model"):
        SelfAttention(invalid)  # type: ignore[arg-type]


@pytest.mark.parametrize("parameter", ["d_k", "d_v"])
@pytest.mark.parametrize("invalid", [0, -1, 1.5, False])
def test_self_attention_rejects_invalid_head_dimensions(parameter: str, invalid: object) -> None:
    """Проекции с нулевой, дробной или логической шириной недопустимы."""
    with pytest.raises(ValueError, match=parameter):
        SelfAttention(4, **{parameter: invalid})  # type: ignore[arg-type]


@pytest.mark.parametrize("invalid_shape", [(2, 3), (2, 3, 4, 1), (2, 3, 5)])
def test_self_attention_rejects_wrong_embedding_shape(invalid_shape: tuple[int, ...]) -> None:
    """Forward принимает только [B, L, d_model], иначе ошибка до matmul."""
    with pytest.raises(ValueError, match="embeddings"):
        SelfAttention(4)(torch.ones(invalid_shape))


def test_self_attention_rejects_integer_input() -> None:
    """Перед attention требуется вещественный embedding, а не token id."""
    with pytest.raises(TypeError, match="вещественный dtype"):
        SelfAttention(2)(torch.ones(1, 2, 2, dtype=torch.int64))


def test_self_attention_rejects_wrong_float_dtype() -> None:
    """Несовпадение dtype модели и данных выявляется явно."""
    with pytest.raises(TypeError, match="dtype"):
        SelfAttention(2)(torch.ones(1, 2, 2, dtype=torch.float64))


@pytest.mark.parametrize("name", ["query", "key", "value"])
def test_attention_rejects_non_tensor_input(name: str) -> None:
    """При передаче не-тензора выдаётся понятная ошибка интерфейса."""
    q, k, v = _sample_tensors()
    args = {"query": q, "key": k, "value": v}
    args[name] = 123  # type: ignore[assignment]
    with pytest.raises(TypeError, match=name):
        scaled_dot_product_attention(**args)


@pytest.mark.parametrize("name", ["query", "key", "value"])
def test_attention_rejects_non_float_input(name: str) -> None:
    """Сырые идентификаторы токенов не могут участвовать в матричном Softmax."""
    q, k, v = _sample_tensors()
    args = {"query": q, "key": k, "value": v}
    args[name] = args[name].long()
    with pytest.raises(TypeError, match="dtype"):
        scaled_dot_product_attention(**args)


@pytest.mark.parametrize("name", ["query", "key", "value"])
def test_attention_rejects_wrong_rank(name: str) -> None:
    """Одна из осей B, L, D обязательна даже при batch=1."""
    q, k, v = _sample_tensors()
    args = {"query": q, "key": k, "value": v}
    args[name] = args[name][0]
    with pytest.raises(ValueError, match="форму"):
        scaled_dot_product_attention(**args)


@pytest.mark.parametrize("name", ["query", "key", "value"])
def test_attention_rejects_empty_tensor_dimensions(name: str) -> None:
    """Вход без токенов или без признаков не создаёт валидного внимания."""
    q, k, v = _sample_tensors()
    args = {"query": q, "key": k, "value": v}
    args[name] = args[name][:, :0, :]
    with pytest.raises(ValueError, match="нулевые размерности"):
        scaled_dot_product_attention(**args)


def test_attention_rejects_mismatched_batch() -> None:
    """Несогласованный batch запрещён до перемножения матриц."""
    q, k, v = _sample_tensors()
    with pytest.raises(ValueError, match="batch"):
        scaled_dot_product_attention(q.repeat(2, 1, 1), k, v)


def test_attention_rejects_mismatched_qk_dimensions() -> None:
    """Q K^T требует равенства ширины запросов и ключей."""
    q, k, v = _sample_tensors()
    with pytest.raises(ValueError, match="query и key"):
        scaled_dot_product_attention(q, torch.ones(1, 3, 4, dtype=q.dtype), v)


def test_attention_rejects_mismatched_kv_lengths() -> None:
    """Каждому ключу должен соответствовать ровно один вектор V."""
    q, k, v = _sample_tensors()
    with pytest.raises(ValueError, match="key и value"):
        scaled_dot_product_attention(q, k, v[:, :-1])


def test_attention_rejects_mismatched_float_dtypes() -> None:
    """Неявное смешение float32 и float64 в attention запрещено."""
    q, k, v = _sample_tensors()
    with pytest.raises(TypeError, match="одинаковый dtype"):
        scaled_dot_product_attention(q.float(), k, v)


@pytest.mark.parametrize("mask_name", ["key_padding_mask", "query_padding_mask"])
def test_attention_rejects_mask_wrong_dtype(mask_name: str) -> None:
    """Mask имеет булеву семантику True=настоящий токен."""
    q, k, v = _sample_tensors()
    with pytest.raises(TypeError, match="torch.bool"):
        scaled_dot_product_attention(q, k, v, **{mask_name: torch.ones(1, 3, dtype=torch.long)})


@pytest.mark.parametrize("mask_name", ["key_padding_mask", "query_padding_mask"])
def test_attention_rejects_mask_wrong_shape(mask_name: str) -> None:
    """Нельзя смешивать последовательность ключей и запросов в маске."""
    q, k, v = _sample_tensors()
    with pytest.raises(ValueError, match="форму"):
        scaled_dot_product_attention(q, k, v, **{mask_name: torch.ones(1, 10, dtype=torch.bool)})


def test_attention_rejects_all_keys_masked() -> None:
    """Softmax полностью из -inf дал бы NaN: заранее выдаём ошибку."""
    q, k, v = _sample_tensors()
    with pytest.raises(ValueError, match="хотя бы один"):
        scaled_dot_product_attention(q, k, v, key_padding_mask=torch.zeros(1, 3, dtype=torch.bool))


def test_attention_rejects_one_empty_sample_in_batch() -> None:
    """Проверка всех элементов batch, а не только первого."""
    q, k, v = _sample_tensors()
    mask = torch.tensor([[True, False, False], [False, False, False]])
    with pytest.raises(ValueError, match="хотя бы один"):
        scaled_dot_product_attention(q.repeat(2, 1, 1), k.repeat(2, 1, 1), v.repeat(2, 1, 1), key_padding_mask=mask)


def test_demo_cli_smoke() -> None:
    """Демо запускается как отдельный скрипт без загрузки чужих данных."""
    script = PROJECT_ROOT / "scripts" / "attention_demo.py"
    completed = subprocess.run(
        [sys.executable, str(script)], capture_output=True, text=True, check=True, timeout=30,
        cwd=PROJECT_ROOT,
    )
    assert "sum(weights): 1.000000" in completed.stdout
    assert "output shape: (1, 4, 4)" in completed.stdout
    assert "padding key has zero weight: True" in completed.stdout
    assert "padding query has zero output: True" in completed.stdout


def test_self_attention_is_permutation_equivariant_without_positions() -> None:
    """Без positional encoding перестановка токенов переставляет выходы."""
    model = SelfAttention(3)
    x = torch.tensor([[[1., 0., 0.], [0., 1., 0.], [0., 0., 2.]]])
    permutation = torch.tensor([2, 0, 1])
    context, weights = model(x)
    context_perm, weights_perm = model(x[:, permutation])
    torch.testing.assert_close(context_perm, context[:, permutation])
    torch.testing.assert_close(weights_perm, weights[:, permutation][:, :, permutation])


def test_padded_and_unpadded_sequences_match_for_real_tokens() -> None:
    """Выравнивание в batch не изменяет контекст исходных токенов."""
    model = SelfAttention(4)
    real_tokens = torch.tensor([[[1., 0., 0., 0.], [0., 1., 0., 0.]]])
    padded = torch.cat((real_tokens, torch.full((1, 3, 4), 300.0)), dim=1)
    mask = torch.tensor([[True, True, False, False, False]])
    original, original_weights = model(real_tokens)
    result, weights = model(padded, mask)
    torch.testing.assert_close(original, result[:, :2])
    torch.testing.assert_close(original_weights, weights[:, :2, :2])


def test_all_queries_masked_returns_zero_if_keys_exist() -> None:
    """Запросы могут быть фиктивными; допустимые ключи нужны для Softmax."""
    q, k, v = _sample_tensors()
    mask = torch.zeros(1, 2, dtype=torch.bool)
    context, weights = scaled_dot_product_attention(q, k, v, query_padding_mask=mask)
    assert torch.count_nonzero(context) == 0
    assert torch.count_nonzero(weights) == 0
    assert bool(torch.isfinite(context).all())
