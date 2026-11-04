"""F3.2: офлайн-тесты pooling, нормировки и CPU-инференса embeddings.

В каждом сценарии используются только фиктивные токены, собственная
необученная nn.Embedding и подмена tokenizer; датасет и модель не скачиваются.
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from transformers_f3.embeddings import (  # noqa: E402
    ModelSpec, PretrainedTextEncoder, l2_normalize, masked_mean_pooling,
)


@pytest.fixture
def spec() -> ModelSpec:
    """Задать корректную, но полностью синтетическую версию encoder."""
    return ModelSpec("synthetic/tiny", "a" * 40, "synthetic", 4, 12)


class FakeTokenizer:
    """Токенизатор по идентификаторам без чтения ресурсов и сети."""

    def __init__(self, *, invalid: str = "") -> None:
        self.invalid = invalid
        self.calls: list[dict[str, object]] = []

    def __call__(self, texts: list[str], **kwargs: object) -> dict[str, torch.Tensor]:
        """Вернуть тензоры с масками для проверки padding."""
        self.calls.append(dict(kwargs))
        ids = [[len(token) % 6 + 1 for token in text.split()] for text in texts]
        max_len = max(map(len, ids))
        input_ids = torch.zeros(len(ids), max_len, dtype=torch.long)
        attention_mask = torch.zeros_like(input_ids)
        for i, row in enumerate(ids):
            input_ids[i, :len(row)] = torch.tensor(row)
            attention_mask[i, :len(row)] = 1
        data: dict[str, torch.Tensor] = {"input_ids": input_ids, "attention_mask": attention_mask}
        if self.invalid == "no_mask":
            del data["attention_mask"]
        elif self.invalid == "bad_shape":
            data["attention_mask"] = attention_mask[:, :-1]
        elif self.invalid == "bad_ids":
            data["input_ids"] = input_ids.float()
        return data


class FakeModel(nn.Module):
    """Необученный embedding-lookup с фиксированными числовыми значениями."""

    def __init__(self, *, invalid: str = "") -> None:
        super().__init__()
        self.invalid = invalid
        self.table = nn.Embedding(8, 4)
        with torch.no_grad():
            self.table.weight.copy_(torch.arange(32, dtype=torch.float32).reshape(8, 4) + 1)
        self.training_during_forward: list[bool] = []
        self.grad_during_forward: list[bool] = []

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> SimpleNamespace:
        """Эмулировать последний hidden state Transformer без contextual attention."""
        self.training_during_forward.append(self.training)
        self.grad_during_forward.append(torch.is_grad_enabled())
        hidden = self.table(input_ids)
        if self.invalid == "bad_hidden":
            hidden = hidden[:, :, :3]
        if self.invalid == "not_tensor":
            hidden = None
        return SimpleNamespace(last_hidden_state=hidden)


def test_model_spec_from_repository_config() -> None:
    """Фактическая публичная конфигурация фиксирует 40-символьную revision."""
    config = Path(__file__).resolve().parents[1] / "configs" / "f3_02_model.json"
    loaded = ModelSpec.from_json(config)
    assert loaded.model_id == "sentence-transformers/paraphrase-MiniLM-L3-v2"
    assert loaded.revision == "4ca70771034acceecb2e72475f72050fcdde4ddc"
    assert loaded.embedding_dim == 384
    assert loaded.max_length == 128
    assert loaded.license == "apache-2.0"
    assert loaded.pooling == "masked_mean"
    assert loaded.normalize is True


@pytest.mark.parametrize("kwargs,exception", [
    ({"model_id": "../bad"}, ValueError),
    ({"revision": "main"}, ValueError),
    ({"revision": "x" * 40}, ValueError),
    ({"license": ""}, ValueError),
    ({"embedding_dim": 0}, ValueError),
    ({"max_length": -1}, ValueError),
    ({"max_length": True}, ValueError),
    ({"embedding_dim": 1.2}, ValueError),
    ({"pooling": "cls"}, ValueError),
    ({"normalize": 1}, TypeError),
])
def test_invalid_model_spec(spec: ModelSpec, kwargs: dict, exception: type[Exception]) -> None:
    """Не допускаем непинованную версию, неверную размерность и неизвестный pooling."""
    data = spec.to_dict()
    data.update(kwargs)
    with pytest.raises(exception):
        ModelSpec(**data)


@pytest.mark.parametrize("data", [[], {}, {"model_id": "foo/bar"}, {"unknown": True}])
def test_invalid_config_file(tmp_path: Path, data: object) -> None:
    """Конфигурация со сдвигом схемы не должна незаметно менять эксперимент."""
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError):
        ModelSpec.from_json(path)


def test_spec_round_trip(spec: ModelSpec, tmp_path: Path) -> None:
    """Все поля pinned-конфигурации сохраняются при сериализации."""
    path = tmp_path / "config.json"
    path.write_text(json.dumps(spec.to_dict()), encoding="utf-8")
    assert ModelSpec.from_json(path) == spec


def test_masked_mean_matches_hand_calculation() -> None:
    """Первый токен и второй усредняются, padding с большим значением исключается."""
    hidden = torch.tensor([[[2., 4.], [6., 8.], [900., 900.]]])
    mask = torch.tensor([[1, 1, 0]])
    result = masked_mean_pooling(hidden, mask)
    torch.testing.assert_close(result, torch.tensor([[4., 6.]]))


def test_masked_mean_independent_per_sample() -> None:
    """Знаменатель считается отдельно для каждого текста внутри batch."""
    hidden = torch.tensor([[[2., 2.], [4., 4.]], [[10., 10.], [100., 100.]]])
    mask = torch.tensor([[1, 1], [1, 0]], dtype=torch.bool)
    torch.testing.assert_close(masked_mean_pooling(hidden, mask), torch.tensor([[3., 3.], [10., 10.]]))


def test_masked_mean_gradients() -> None:
    """Градиент не протекает в скрытый embedding padding-позиции."""
    hidden = torch.tensor([[[1., 2.], [3., 4.], [5., 6.]]], requires_grad=True)
    result = masked_mean_pooling(hidden, torch.tensor([[True, True, False]])).sum()
    result.backward()
    torch.testing.assert_close(hidden.grad, torch.tensor([[[.5, .5], [.5, .5], [0., 0.]]]))


@pytest.mark.parametrize("mask", [torch.tensor([[0, 0]]), torch.tensor([[False, False]])])
def test_pooling_rejects_only_padding(mask: torch.Tensor) -> None:
    """Все нулевые маски означают отсутствие текста, деление запрещено."""
    with pytest.raises(ValueError, match="padding"):
        masked_mean_pooling(torch.ones(1, 2, 2), mask)


@pytest.mark.parametrize("hidden,mask,exception", [
    (torch.zeros((0, 2, 2)), torch.zeros((0, 2)), ValueError),
    (torch.ones(1, 2), torch.ones(1, 2), ValueError),
    (torch.ones(1, 2, 2).long(), torch.ones(1, 2), TypeError),
    (torch.ones(1, 2, 2), torch.ones(1, 1), ValueError),
    (torch.ones(1, 2, 2), torch.ones(1, 2).float(), TypeError),
    (torch.ones(1, 2, 2), torch.tensor([[1, 2]]), ValueError),
    (torch.full((1, 2, 2), float("nan")), torch.ones(1, 2, dtype=torch.long), ValueError),
    (torch.full((1, 2, 2), float("inf")), torch.ones(1, 2, dtype=torch.long), ValueError),
])
def test_pooling_rejects_bad_inputs(hidden: torch.Tensor, mask: torch.Tensor, exception: type[Exception]) -> None:
    """Формы, dtype, NaN и некорректные masks не маскируются ошибкой знаменателя."""
    with pytest.raises(exception):
        masked_mean_pooling(hidden, mask)


def test_normalization_manual_reference() -> None:
    """Для вектора [3,4] нормированный результат точно [0.6,0.8]."""
    torch.testing.assert_close(l2_normalize(torch.tensor([[3., 4.]])), torch.tensor([[.6, .8]]))


@pytest.mark.parametrize("bad", [
    torch.tensor([[0., 0.]]), torch.tensor([[float("nan"), 1.]]),
    torch.tensor([[float("inf"), 1.]]), torch.ones(1, 2, dtype=torch.long),
    torch.zeros(0, 2), torch.ones(2),
])
def test_normalize_rejects_bad_inputs(bad: torch.Tensor) -> None:
    """Отклоняем NaN, Inf, пустые и нулевые векторы; silent epsilon запрещён."""
    with pytest.raises((TypeError, ValueError)):
        l2_normalize(bad)


def test_encoder_output_shape_norm_dtype_cpu(spec: ModelSpec) -> None:
    """Прямой проход выдаёт [N,D] на CPU и L2=1 без градиентов."""
    tokenizer, model = FakeTokenizer(), FakeModel()
    encoder = PretrainedTextEncoder(spec, tokenizer, model)
    result = encoder.encode(["sql index", "linux", "kernel error"], batch_size=2)
    assert result.shape == (3, 4)
    assert result.dtype == torch.float32 and result.device.type == "cpu"
    torch.testing.assert_close(result.norm(dim=1), torch.ones(3), atol=1e-6, rtol=1e-6)
    assert model.training_during_forward == [False, False]
    assert model.grad_during_forward == [False, False]
    assert len(tokenizer.calls) == 2
    assert tokenizer.calls[0]["max_length"] == spec.max_length
    assert tokenizer.calls[0]["truncation"] is True


def test_encoder_manual_pooling_reference(spec: ModelSpec) -> None:
    """Результат encoder совпадает с ручным masked mean на таблице embedding."""
    model, tokenizer = FakeModel(), FakeTokenizer()
    encoder = PretrainedTextEncoder(spec, tokenizer, model)
    values = encoder.encode(["aa bbb"], batch_size=1)
    expected = (model.table.weight[3] + model.table.weight[4]) / 2
    expected = expected / expected.norm()
    torch.testing.assert_close(values[0], expected, atol=1e-6, rtol=1e-6)


def test_encoder_no_normalization(spec: ModelSpec) -> None:
    """Режим normalize=False оставляет значения среднего без L2-деления."""
    params = spec.to_dict()
    params["normalize"] = False
    encoder = PretrainedTextEncoder(ModelSpec(**params), FakeTokenizer(), FakeModel())
    result = encoder.encode(["aa bbb"])
    assert not torch.allclose(result.norm(dim=1), torch.ones(1))


def test_encoder_determinism_across_batch_sizes(spec: ModelSpec) -> None:
    """Разная разбивка на batches не влияет на значения и порядок строк."""
    encoder = PretrainedTextEncoder(spec, FakeTokenizer(), FakeModel())
    texts = ["alpha", "x y", "linux kernel errors", "database"]
    reference = encoder.encode(texts, batch_size=1)
    for size in (2, 3, 4, 9):
        torch.testing.assert_close(encoder.encode(texts, batch_size=size), reference)


def test_encoder_empty_texts_no_model_call(spec: ModelSpec) -> None:
    """Пустая коллекция является допустимым N=0 и не вызывает forward."""
    tok, model = FakeTokenizer(), FakeModel()
    result = PretrainedTextEncoder(spec, tok, model).encode([])
    assert result.shape == (0, 4) and result.dtype == torch.float32
    assert not tok.calls and not model.training_during_forward


@pytest.mark.parametrize("texts", ["str", None, [""], ["   "], ["valid", None], [123]])
def test_encoder_rejects_invalid_text(texts: object, spec: ModelSpec) -> None:
    """Пустые и нестроковые входы запрещены до токенизации."""
    with pytest.raises((TypeError, ValueError)):
        PretrainedTextEncoder(spec, FakeTokenizer(), FakeModel()).encode(texts)


@pytest.mark.parametrize("batch_size", [0, -1, True, 1.0])
def test_encoder_rejects_invalid_batch_size(batch_size: object, spec: ModelSpec) -> None:
    """Отклоняем неверный batch_size даже на пустом списке."""
    with pytest.raises(ValueError):
        PretrainedTextEncoder(spec, FakeTokenizer(), FakeModel()).encode([], batch_size=batch_size)


@pytest.mark.parametrize("bad", ["no_mask", "bad_shape", "bad_ids"])
def test_encoder_rejects_bad_tokenizer_output(bad: str, spec: ModelSpec) -> None:
    """Ошибки токенизатора должны выявляться до применения encoder."""
    with pytest.raises((TypeError, ValueError)):
        PretrainedTextEncoder(spec, FakeTokenizer(invalid=bad), FakeModel()).encode(["hello test"])


@pytest.mark.parametrize("bad", ["bad_hidden", "not_tensor"])
def test_encoder_rejects_bad_model_output(bad: str, spec: ModelSpec) -> None:
    """Не допускаем mismatch hidden_size и непредусмотренный формат выхода."""
    with pytest.raises((TypeError, ValueError)):
        PretrainedTextEncoder(spec, FakeTokenizer(), FakeModel(invalid=bad)).encode(["hello test"])


def test_encoder_requires_module_and_tokenizer(spec: ModelSpec) -> None:
    """Интерфейс модели должен быть nn.Module и получать callable tokenizer."""
    with pytest.raises(TypeError):
        PretrainedTextEncoder(spec, None, FakeModel())
    with pytest.raises(TypeError):
        PretrainedTextEncoder(spec, FakeTokenizer(), object())


def test_demo_synthetic_runs_without_transformers_or_network() -> None:
    """CLI smoke запускается в голом окружении torch, без HF-модели."""
    demo = Path(__file__).resolve().parents[1] / "scripts" / "embedding_demo.py"
    result = subprocess.run([sys.executable, str(demo)], capture_output=True, text=True, check=True)
    assert "Режим: synthetic" in result.stdout
    assert "embeddings shape: (3, 4)" in result.stdout
    assert "repeat deterministic: True" in result.stdout
    assert "pretrained" in result.stdout


def test_demo_download_without_real_is_rejected() -> None:
    """Нельзя включать скачивание модели случайно без выбора real режима."""
    demo = Path(__file__).resolve().parents[1] / "scripts" / "embedding_demo.py"
    result = subprocess.run([sys.executable, str(demo), "--download"], capture_output=True, text=True)
    assert result.returncode != 0
    assert "--download" in result.stderr
