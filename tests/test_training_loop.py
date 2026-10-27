"""F2.4: синтетические проверки train loop, обновления весов и инвариантов.

Ни один тест не обучает модель на F1 Q&A, validation или test.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from scripts.pytorch_batches import synthetic_splits
from scripts.pytorch_training import run
from src.pytorch_f2.data_pipeline import make_loader, prepare_data
from src.pytorch_f2.embedding_classifier import EmbeddingClassifier
from src.pytorch_f2.training_loop import (
    TrainingError, _validate_batch, train_epochs, train_one_epoch,
)


def prepared():
    """Создать три разделённых синтетических Dataset со словарём только train."""
    return prepare_data(synthetic_splits(), max_vocab_size=100, max_length=16)


def network(dataset=None):
    """Создать маленький классификатор по словарю синтетической выборки."""
    source = dataset or prepared()
    return EmbeddingClassifier(vocab_size=len(source.vocab),
                               num_classes=len(source.label_to_id),
                               embedding_dim=5, hidden_dim=6)


def train_loader(data=None, batch_size=3, shuffle=False):
    """Подготовить единственный источник примеров, разрешённый для fit."""
    source = data or prepared()
    return make_loader(source.datasets["train"], batch_size=batch_size,
                       shuffle=shuffle, seed=42)


def test_one_epoch_consumes_all_samples_including_short_last_batch():
    """Последний батч не отбрасывается, средняя потеря конечна."""
    torch.manual_seed(3)
    net = network()
    result = train_one_epoch(net, train_loader(batch_size=3),
                             optimizer=torch.optim.AdamW(net.parameters(), lr=0.01))
    assert result.epoch == 1
    assert result.samples == 4
    assert result.batches == 2
    assert math.isfinite(result.mean_loss) and result.mean_loss > 0
    assert result.nonzero_gradients and result.parameters_updated


def test_weights_change_and_gradients_are_not_all_zero():
    """Связываем backward с действительным обновлением обучаемых параметров."""
    torch.manual_seed(11)
    net = network()
    before = [p.detach().clone() for p in net.parameters()]
    results = train_epochs(net, train_loader(), epochs=2, learning_rate=0.02)
    assert all(r.nonzero_gradients and r.parameters_updated for r in results)
    assert any(not torch.equal(old, p.detach()) for old, p in zip(before, net.parameters()))
    assert any(p.grad is not None and torch.count_nonzero(p.grad) > 0
               for p in net.parameters())


def test_embedding_padding_row_keeps_zero_value_after_adamw():
    """Веса PAD с padding_idx=0 не обучаются и остаются нулями."""
    net = network()
    assert torch.count_nonzero(net.embedding.weight[0]) == 0
    train_epochs(net, train_loader(), epochs=2, learning_rate=0.03)
    assert torch.count_nonzero(net.embedding.weight[0]) == 0


def test_multi_epoch_training_preserves_order_and_counts():
    """Эпохи нумеруются по порядку, каждая обрабатывает весь train."""
    net = network()
    history = train_epochs(net, train_loader(batch_size=2), epochs=3, learning_rate=0.01)
    assert [h.epoch for h in history] == [1, 2, 3]
    assert [h.samples for h in history] == [4, 4, 4]
    assert [h.batches for h in history] == [2, 2, 2]


def test_epoch_loss_is_weighted_by_sample_count():
    """Средний loss для B=3+1 равен взвешенному, а не среднему по батчам."""
    torch.manual_seed(5)
    net = network()
    # lr задан очень малым, чтобы обновление не влияло заметно на подсчёт.
    loader = train_loader(batch_size=3)
    loss_fn = torch.nn.CrossEntropyLoss()
    expected_losses = []
    with torch.no_grad():
        for b in loader:
            expected_losses.append((float(loss_fn(net(b["input_ids"], b["attention_mask"]),
                                                 b["labels"])), len(b["labels"])))
    result = train_one_epoch(net, train_loader(batch_size=3),
                             optimizer=torch.optim.AdamW(net.parameters(), lr=1e-10,
                                                         weight_decay=0.0))
    expected = sum(loss * size for loss, size in expected_losses) / 4
    assert result.mean_loss == pytest.approx(expected, abs=1e-6)


def test_reproducible_training_given_seed_and_data():
    """Одинаковые состояния RNG и train-данных дают равные losses и веса."""
    def fit():
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(234)
            net = network()
            history = train_epochs(net, train_loader(shuffle=True), epochs=3,
                                   learning_rate=0.03)
            return [r.mean_loss for r in history], [p.detach().clone() for p in net.parameters()]
    losses1, weights1 = fit()
    losses2, weights2 = fit()
    assert losses1 == losses2
    assert all(torch.equal(a, b) for a, b in zip(weights1, weights2))


def test_loss_decreases_on_small_synthetic_training_problem():
    """Длинный учебный smoke показывает способность оптимизатора снижать loss."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        net = network()
        history = train_epochs(net, train_loader(batch_size=4), epochs=12,
                               learning_rate=0.03)
    assert history[-1].mean_loss < history[0].mean_loss


def test_training_sets_model_to_train_mode():
    """Даже после eval() цикл переводит модель в train() перед обновлением."""
    net = network().eval()
    train_epochs(net, train_loader(), epochs=1)
    assert net.training is True


def test_train_epochs_does_not_iterate_validation_or_test(monkeypatch):
    """Ни одна held-out запись не попадает в обучение или расчёт loss."""
    data = prepared()
    def forbidden(_):
        raise AssertionError("validation/test dataset was accessed")
    monkeypatch.setattr(data.datasets["val"], "__getitem__", forbidden)
    monkeypatch.setattr(data.datasets["test"], "__getitem__", forbidden)
    history = train_epochs(network(data), train_loader(data), epochs=2)
    assert sum(h.samples for h in history) == 2 * data.counts["train"]


def test_train_does_not_modify_prepared_label_and_vocab_mapping():
    """Optimizer воздействует на веса модели, но не на словарь F2.2."""
    data = prepared()
    before_vocab = dict(data.vocab.token_to_id)
    before_labels = dict(data.label_to_id)
    train_epochs(network(data), train_loader(data), epochs=2)
    assert data.vocab.token_to_id == before_vocab
    assert data.label_to_id == before_labels


def test_cli_synthetic_smoke_has_expected_aggregates():
    """CLI проверяет обучение, а не accuracy или внешний test-набор."""
    report = run(synthetic=True, epochs=2, expect_train=4, expect_val=1, expect_test=1)
    assert report["source"] == "synthetic"
    assert report["split_counts"] == {"train": 4, "val": 1, "test": 1}
    assert report["trained_split"] == "train"
    assert report["class_count"] == 2
    assert report["epoch_samples"] == [4, 4]
    assert report["epoch_batches"] == [1, 1]
    assert len(report["loss_by_epoch"]) == 2
    assert report["train_invariants_pass"] is True
    assert report["parameters_updated"] is True
    assert report["nonzero_gradients"] is True
    assert report["validation_performed"] is False
    assert report["checkpoint_saved"] is False


def test_cli_synthetic_repeatable():
    """Повторное создание модели с одним seed воспроизводит историю потерь."""
    assert run(synthetic=True, epochs=2) == run(synthetic=True, epochs=2)


def test_cli_preserves_external_global_torch_rng():
    """Независимый smoke не сбрасывает глобальное RNG состояние вызывающего кода."""
    state = torch.random.get_rng_state().clone()
    run(synthetic=True, epochs=2)
    assert torch.equal(torch.random.get_rng_state(), state)


def test_cli_rejects_incorrect_expected_train_size():
    """Неверный ожидаемый объём train не скрывается изменением корпуса."""
    with pytest.raises(TrainingError, match="unexpected train"):
        run(synthetic=True, expect_train=100)


@pytest.mark.parametrize("epochs", [0, -1, 1.5, "2", True])
def test_invalid_epochs_rejected(epochs):
    """Параметр числа эпох строго положительное целое, не bool/float/str."""
    with pytest.raises(TrainingError, match="epochs must"):
        train_epochs(network(), train_loader(), epochs=epochs)


@pytest.mark.parametrize("lr", [0, -1, float("nan"), float("inf"), "0.1", True])
def test_invalid_learning_rate_rejected(lr):
    """Обучение с нулевой, бесконечной или некорректной скоростью запрещено."""
    with pytest.raises(TrainingError, match="learning_rate must"):
        train_epochs(network(), train_loader(), learning_rate=lr)


@pytest.mark.parametrize("weight_decay", [-0.1, float("nan"), float("inf"), True])
def test_invalid_weight_decay_rejected(weight_decay):
    """Коэффициент регуляризации должен быть конечным неотрицательным числом."""
    with pytest.raises(TrainingError, match="weight_decay must"):
        train_epochs(network(), train_loader(), weight_decay=weight_decay)


def test_invalid_epoch_index_rejected():
    """Номер эпохи в результате не может быть нулевым."""
    net = network()
    with pytest.raises(TrainingError, match="epoch must"):
        train_one_epoch(net, train_loader(), optimizer=torch.optim.AdamW(net.parameters()), epoch=0)


def test_training_rejects_empty_loader():
    """Пустой train loader не должен приводить к делению на нуль."""
    net = network()
    empty = torch.utils.data.DataLoader([], batch_size=2)
    with pytest.raises(TrainingError, match="empty training loader"):
        train_one_epoch(net, empty, optimizer=torch.optim.AdamW(net.parameters()))


@pytest.mark.parametrize("field", ["input_ids", "attention_mask", "labels"])
def test_batch_rejects_missing_fields(field):
    """Запрещены неполные батчи, несовместимые с F2.2 контрактом."""
    b = dict(next(iter(train_loader())))
    b.pop(field)
    with pytest.raises(TrainingError, match="expected F2.2 batch"):
        _validate_batch(b, num_classes=2)


def test_batch_rejects_invalid_label_shape():
    """CrossEntropyLoss принимает одномерные метки размера батча."""
    b = dict(next(iter(train_loader())))
    b["labels"] = b["labels"].unsqueeze(1)
    with pytest.raises(TrainingError, match="labels must"):
        _validate_batch(b, num_classes=2)


def test_batch_rejects_invalid_label_dtype():
    """Индекс класса не должен быть вещественным тензором."""
    b = dict(next(iter(train_loader())))
    b["labels"] = b["labels"].float()
    with pytest.raises(TrainingError, match="labels must"):
        _validate_batch(b, num_classes=2)


def test_batch_rejects_label_outside_classes():
    """Недопустимый класс обнаруживается до операции CrossEntropyLoss."""
    b = dict(next(iter(train_loader())))
    b["labels"][0] = 2
    with pytest.raises(TrainingError, match="labels must"):
        _validate_batch(b, num_classes=2)


def test_batch_rejects_wrong_mask_dtype():
    """Маска из целых чисел не заменяет булеву маску F2.2."""
    b = dict(next(iter(train_loader())))
    b["attention_mask"] = b["attention_mask"].long()
    with pytest.raises(TrainingError, match="attention_mask must"):
        _validate_batch(b, num_classes=2)


def test_batch_rejects_wrong_id_dtype():
    """Embedding layer требует индексы torch.long от DataLoader."""
    b = dict(next(iter(train_loader())))
    b["input_ids"] = b["input_ids"].float()
    with pytest.raises(TrainingError, match="input_ids must"):
        _validate_batch(b, num_classes=2)


def test_batch_rejects_inconsistent_mask_dimensions():
    """Нельзя обучать модель с маской другой формы, чем входы."""
    b = dict(next(iter(train_loader())))
    b["attention_mask"] = b["attention_mask"][:, :-1]
    with pytest.raises(TrainingError, match="attention_mask must"):
        _validate_batch(b, num_classes=2)


def test_cli_rejects_missing_local_config(tmp_path):
    """Локальный режим не подменяет отсутствующий конфиг тестовой синтетикой."""
    with pytest.raises(TrainingError, match="missing F1 configuration"):
        run(root=tmp_path)


def test_cli_local_uses_only_train_of_existing_synthetic_split(tmp_path):
    """Проверить реальный путь JSONL/YAML без приватного корпуса и без test fit."""
    config_dir = tmp_path / "configs" / "local_cpu"
    config_dir.mkdir(parents=True)
    (config_dir / "lab.yaml").write_text(
        "seed: 42\nneural:\n  vocab_size: 100\n  max_length: 12\n"
        "  batch_size: 3\n  embedding_dim: 5\n  hidden_dim: 6\n  learning_rate: 0.01\n",
        encoding="utf-8",
    )
    path = tmp_path / "data" / "processed"
    path.mkdir(parents=True)
    for name, rows in synthetic_splits().items():
        (path / f"{name}.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
        )
    report = run(root=tmp_path, epochs=2, expect_train=4, expect_val=1, expect_test=1)
    assert report["source"] == "local_f1_train"
    assert report["epoch_samples"] == [4, 4]
    assert report["epoch_batches"] == [2, 2]
    assert report["validation_performed"] is False


def test_cli_returns_failure_exit_for_bad_synthetic_params():
    """Командная проверка должна вернуть exit code 1 при некорректных эпохах."""
    outcome = subprocess.run(
        [sys.executable, "-m", "scripts.pytorch_training", "--synthetic", "--epochs", "0", "--check"],
        capture_output=True, text=True, timeout=30,
    )
    assert outcome.returncode == 1
    assert "F2.4 Training Loop CHECK: FAIL" in outcome.stdout


def test_cli_returns_success_exit_for_smoke():
    """Положительный сценарий должен быть пригоден для GitHub Actions."""
    outcome = subprocess.run(
        [sys.executable, "-m", "scripts.pytorch_training", "--synthetic", "--check"],
        capture_output=True, text=True, timeout=30,
    )
    assert outcome.returncode == 0, outcome.stderr + outcome.stdout
    assert "F2.4 Training Loop CHECK: PASS" in outcome.stdout


def test_each_training_batch_performs_zero_grad_and_optimizer_step(monkeypatch):
    """Ровно один цикл очистки и обновления на батч, даже неполный."""
    net = network()
    optimizer = torch.optim.AdamW(net.parameters(), lr=0.01)
    counts = {"zero": 0, "step": 0}
    original_zero = optimizer.zero_grad
    original_step = optimizer.step

    def record_zero(*, set_to_none=False):
        """Зафиксировать запрошенный режим сброса предыдущих градиентов."""
        assert set_to_none is True
        counts["zero"] += 1
        return original_zero(set_to_none=set_to_none)

    def record_step():
        """Отследить одно обновление параметров для каждого батча."""
        counts["step"] += 1
        return original_step()

    monkeypatch.setattr(optimizer, "zero_grad", record_zero)
    monkeypatch.setattr(optimizer, "step", record_step)
    train_one_epoch(net, train_loader(batch_size=3), optimizer=optimizer)
    assert counts == {"zero": 2, "step": 2}


def test_train_rejects_optimizer_for_another_model():
    """Нельзя молча заявить обучение, оптимизируя чужие параметры."""
    net = network()
    another = network()
    optimizer = torch.optim.AdamW(another.parameters())
    with pytest.raises(TrainingError, match="optimizer must own"):
        train_one_epoch(net, train_loader(), optimizer=optimizer)


def test_train_rejects_nonfinite_loss():
    """При NaN-весах training loop прекращает расчёт, не выдавая ложный PASS."""
    net = network()
    with torch.no_grad():
        net.output.bias.fill_(float("nan"))
    with pytest.raises(TrainingError, match="non-finite training loss"):
        train_one_epoch(net, train_loader(),
                        optimizer=torch.optim.AdamW(net.parameters()))


def test_train_rejects_nonfinite_gradients():
    """Оптимизатор не должен применять бесконечные градиенты."""
    net = network()
    hook = net.output.bias.register_hook(lambda grad: torch.full_like(grad, float("inf")))
    try:
        with pytest.raises(TrainingError, match="non-finite gradients"):
            train_one_epoch(net, train_loader(),
                            optimizer=torch.optim.AdamW(net.parameters()))
    finally:
        hook.remove()
