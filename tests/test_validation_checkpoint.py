"""F2.5: синтетические тесты val, Macro F1 и безопасного checkpointing.

Проверки не используют реальные вопросы F1 и никогда не оценивают test.
"""

from __future__ import annotations

import math
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from scripts.pytorch_batches import synthetic_splits
from scripts.pytorch_validation import run
from src.pytorch_f2.data_pipeline import Vocabulary, make_loader, prepare_data
from src.pytorch_f2.embedding_classifier import EmbeddingClassifier
from src.pytorch_f2.validation_checkpoint import (
    ValidationError, _scores_from_confusion, _save_checkpoint,
    evaluate_validation, fit_with_validation, restore_best_checkpoint,
    vocabulary_fingerprint,
)


def prepared():
    """Создать независимые группы train/val/test с единым train-словарём."""
    return prepare_data(synthetic_splits(), max_vocab_size=100, max_length=16)


def model(data=None):
    """Небольшая детерминируемая модель для проверки числовых инвариантов."""
    p = data or prepared()
    return EmbeddingClassifier(vocab_size=len(p.vocab), num_classes=len(p.label_to_id),
                               embedding_dim=5, hidden_dim=6)


def loader(data, split="val", *, batch_size=2, shuffle=False):
    """Сконструировать DataLoader по признаку раздела без пересмешивания."""
    return make_loader(data.datasets[split], batch_size=batch_size,
                       shuffle=shuffle, seed=42)


def fit(tmp_path, *, epochs=3, data=None):
    """Обучить только на train, сохранить checkpoint в изолированную папку."""
    d = data or prepared()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        net = model(d)
        history = fit_with_validation(
            net, loader(d, "train", shuffle=True), loader(d),
            vocab=d.vocab, label_to_id=d.label_to_id, max_length=16,
            split_counts=d.counts, seed=42, checkpoint_path=tmp_path / "best.pt",
            epochs=epochs, learning_rate=0.03,
        )
    return d, net, history


@pytest.mark.parametrize(
    "matrix,accuracy,macro_f1",
    [
        ([[5, 0], [0, 5]], 1.0, 1.0),
        ([[5, 0], [5, 0]], 0.5, 1 / 3),
        ([[1, 2], [3, 4]], 0.5, (2 / 7 + 8 / 13) / 2),
        ([[0, 0], [0, 3]], 1.0, 0.5),
        ([[0, 2], [2, 0]], 0.0, 0.0),
    ],
)
def test_metrics_formula(matrix, accuracy, macro_f1):
    """Macro F1 усредняет F1 классов, а не совпадает с accuracy или micro F1."""
    acc, f1 = _scores_from_confusion(torch.tensor(matrix, dtype=torch.int64))
    assert acc == pytest.approx(accuracy)
    assert f1 == pytest.approx(macro_f1)


def test_validation_aggregates_one_heldout_record():
    """Validation использует ровно одну удержанную запись и её правильный класс."""
    data = prepared()
    result = evaluate_validation(model(data), loader(data))
    assert result.samples == 1
    assert math.isfinite(result.mean_loss) and result.mean_loss > 0
    assert 0 <= result.accuracy <= 1
    assert 0 <= result.macro_f1 <= 1
    assert sum(sum(row) for row in result.confusion_matrix) == 1
    assert len(result.confusion_matrix) == 2


def test_validation_no_parameter_or_gradient_change():
    """В val-проходе нет backward и optimizer.step; состояние не меняется."""
    data = prepared()
    net = model(data)
    before = [p.detach().clone() for p in net.parameters()]
    result = evaluate_validation(net, loader(data))
    assert result.samples == 1
    assert all(torch.equal(a, b) for a, b in zip(before, net.parameters()))
    assert all(p.grad is None for p in net.parameters())


@pytest.mark.parametrize("mode", [True, False])
def test_validation_restores_original_train_eval_mode(mode):
    """Оценка не оставляет вызывающую модель в неожиданном режиме."""
    data = prepared()
    net = model(data).train(mode)
    evaluate_validation(net, loader(data))
    assert net.training is mode


def test_validation_preserves_existing_gradients():
    """При проверке val уже накопленные train-градиенты не должны пропасть."""
    data = prepared()
    net = model(data)
    batch = next(iter(loader(data, "train")))
    net(batch["input_ids"], batch["attention_mask"]).sum().backward()
    gradients = [None if p.grad is None else p.grad.clone() for p in net.parameters()]
    evaluate_validation(net, loader(data))
    for p, grad in zip(net.parameters(), gradients):
        assert (p.grad is None and grad is None) or torch.equal(p.grad, grad)


@pytest.mark.parametrize("split", ["train", "test"])
def test_validation_rejects_train_and_test(split):
    """Нельзя случайно выбрать модель по train или по независимому test."""
    d = prepared()
    with pytest.raises(ValidationError, match="expected val split"):
        evaluate_validation(model(d), loader(d, split))


def test_validation_rejects_shuffled_val():
    """Проверка должна охватывать все held-out примеры без перемешивания."""
    d = prepared()
    with pytest.raises(ValidationError, match="sequential"):
        evaluate_validation(model(d), loader(d, shuffle=True))


def test_validation_rejects_mixed_split_rows():
    """Подмена split в одной записи обнаруживается до оценки."""
    d = prepared()
    d.datasets["val"].rows[0]["split"] = "test"
    with pytest.raises(ValidationError, match="expected val split"):
        evaluate_validation(model(d), loader(d))


def test_validation_weighted_loss_agrees_with_manual_sum():
    """Взвешивание validation loss должно учитывать размер каждого батча."""
    d = prepared()
    # Добавляем второй и третий val-примеры с отличающимися длинами.
    val_rows = [dict(d.datasets["val"].rows[0])]
    for i, question in enumerate(("linux service", "system service memory"), start=30):
        val_rows.append({"id": f"val-{i}", "context": question,
                         "domain": "linux" if i == 30 else "systems", "group_id": i, "split": "val"})
    d.datasets["val"] = type(d.datasets["val"])(
        val_rows, vocab=d.vocab, label_to_id=d.label_to_id, max_length=16,
    )
    net = model(d)
    val_loader = loader(d, batch_size=2)
    expected_sum = 0.0
    with torch.inference_mode():
        for batch in val_loader:
            logits = net(batch["input_ids"], batch["attention_mask"])
            expected_sum += float(torch.nn.functional.cross_entropy(
                logits, batch["labels"], reduction="sum"))
    result = evaluate_validation(net, val_loader)
    assert result.samples == 3
    assert result.mean_loss == pytest.approx(expected_sum / 3)


def test_fingerprint_stable_under_dict_ordering():
    """Порядок обхода Python-словаря не должен менять идентичность vocabulary."""
    v = prepared().vocab
    reversed_mapping = dict(reversed(list(v.token_to_id.items())))
    assert vocabulary_fingerprint(v) == vocabulary_fingerprint(Vocabulary(reversed_mapping))


def test_fingerprint_changes_when_same_size_vocab_reindexed():
    """Словарь с иным распределением ID несовместим даже при той же длине."""
    v = prepared().vocab
    mapping = dict(v.token_to_id)
    tokens = [name for name, index in mapping.items() if index >= 4]
    mapping[tokens[0]], mapping[tokens[1]] = mapping[tokens[1]], mapping[tokens[0]]
    assert vocabulary_fingerprint(v) != vocabulary_fingerprint(Vocabulary(mapping))


def test_fitting_creates_checkpoint_with_best_metric(tmp_path):
    """Вес сохраняется по лучшей validation Macro F1, а не train loss."""
    data, net, result = fit(tmp_path, epochs=3)
    assert result.checkpoint_path.is_file()
    earliest_best = next(i for i, row in enumerate(result.history, start=1)
                         if row.validation.macro_f1 == result.best_macro_f1)
    assert result.best_epoch == earliest_best
    assert len(result.history) == 3
    assert result.best_macro_f1 == max(r.validation.macro_f1 for r in result.history)
    assert all(r.train.samples == data.counts["train"] for r in result.history)
    restored, metadata = restore_best_checkpoint(result.checkpoint_path,
                        vocab=data.vocab, label_to_id=data.label_to_id, max_length=16)
    assert metadata["selection_metric"] == "val_macro_f1"
    assert metadata["tie_break"] == "earliest_epoch"
    assert metadata["best_epoch"] == result.best_epoch
    assert restored.training is False


def test_checkpoint_metadata_has_reproducibility_fields(tmp_path):
    """Checkpoint объясняет архитектуру, train-словарь и критерий выбора."""
    d, _, result = fit(tmp_path, epochs=2)
    _, info = restore_best_checkpoint(result.checkpoint_path,
                                      vocab=d.vocab, label_to_id=d.label_to_id, max_length=16)
    assert info["split_counts"] == {"train": 4, "val": 1, "test": 1}
    assert info["seed"] == 42
    assert info["planned_epochs"] == 2
    assert info["model"]["num_classes"] == 2
    assert info["max_length"] == 16
    assert info["label_to_id"] == d.label_to_id
    assert info["vocab_fingerprint"] == vocabulary_fingerprint(d.vocab)
    assert isinstance(info["torch_version"], str)
    assert "test_accuracy" not in info and "test_f1" not in info


def test_checkpoint_weights_are_saved_on_cpu_and_are_plain_tensors(tmp_path):
    """На диск попадает state_dict, не сериализованный экземпляр nn.Module."""
    _, _, result = fit(tmp_path)
    payload = torch.load(result.checkpoint_path, map_location="cpu", weights_only=True)
    assert set(payload) == {"format_version", "metadata", "model_state"}
    assert all(isinstance(t, torch.Tensor) and t.device.type == "cpu"
               for t in payload["model_state"].values())
    assert all("context" not in key and "sample_id" not in key for key in payload)


def test_checkpoint_predictions_equal_at_saved_best_epoch(tmp_path):
    """Проверяется реальный round-trip весов, не лишь успешность torch.load."""
    d = prepared()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        net = model(d)
        fit_with_validation(net, loader(d, "train", shuffle=True), loader(d),
            vocab=d.vocab, label_to_id=d.label_to_id, max_length=16,
            split_counts=d.counts, seed=42, checkpoint_path=tmp_path / "one.pt",
            epochs=1, learning_rate=0.03)
    recovered, _ = restore_best_checkpoint(tmp_path / "one.pt", vocab=d.vocab,
                                           label_to_id=d.label_to_id, max_length=16)
    batch = next(iter(loader(d)))
    with torch.inference_mode():
        before = net(batch["input_ids"], batch["attention_mask"])
        after = recovered(batch["input_ids"], batch["attention_mask"])
    assert torch.equal(before, after)
    assert torch.equal(before.argmax(dim=1), after.argmax(dim=1))


def test_restored_metrics_match_saved_epoch_not_last_epoch(tmp_path):
    """Загруженная оценка совпадает с выбранной эпохой, даже если позже было обучение."""
    d, _, result = fit(tmp_path, epochs=3)
    recovered, info = restore_best_checkpoint(result.checkpoint_path,
                                              vocab=d.vocab, label_to_id=d.label_to_id,
                                              max_length=16)
    score = evaluate_validation(recovered, loader(d))
    assert score.macro_f1 == pytest.approx(info["val_macro_f1"])
    assert score.accuracy == pytest.approx(info["val_accuracy"])
    assert score.mean_loss == pytest.approx(info["val_mean_loss"], abs=1e-8)


def test_no_test_dataset_access_during_fit(tmp_path, monkeypatch):
    """Test не открывается в training/validation даже для промежуточных метрик."""
    d = prepared()
    def forbidden(*args, **kwargs):
        raise AssertionError("test dataset accessed")
    monkeypatch.setattr(d.datasets["test"], "__getitem__", forbidden)
    fit(tmp_path, data=d)


def test_train_and_val_vocabulary_unchanged_after_fit(tmp_path):
    """Обучение и checkpointing не изменяют отображения F2.2."""
    d = prepared()
    previous = (dict(d.vocab.token_to_id), dict(d.label_to_id))
    fit(tmp_path, data=d)
    assert (d.vocab.token_to_id, d.label_to_id) == previous


def test_fit_determinism_for_same_seed(tmp_path):
    """Одинаковый seed на одном CPU воспроизводит обучение и выбор эпохи."""
    a, _, result_a = fit(tmp_path / "a")
    b, _, result_b = fit(tmp_path / "b")
    assert result_a.best_epoch == result_b.best_epoch
    assert result_a.best_macro_f1 == result_b.best_macro_f1
    assert [e.validation.mean_loss for e in result_a.history] == [
        e.validation.mean_loss for e in result_b.history]
    aa, _ = restore_best_checkpoint(result_a.checkpoint_path,
                                    vocab=a.vocab, label_to_id=a.label_to_id, max_length=16)
    bb, _ = restore_best_checkpoint(result_b.checkpoint_path,
                                    vocab=b.vocab, label_to_id=b.label_to_id, max_length=16)
    assert all(torch.equal(aa.state_dict()[key], bb.state_dict()[key]) for key in aa.state_dict())


@pytest.mark.parametrize("epochs", [0, -1, True, 2.5, "2"])
def test_invalid_number_of_epochs_rejected(tmp_path, epochs):
    """Не запускаем выбор checkpoint при неверном числе эпох."""
    d = prepared()
    with pytest.raises(ValueError, match="epochs must"):
        fit_with_validation(model(d), loader(d, "train"), loader(d),
            vocab=d.vocab, label_to_id=d.label_to_id, max_length=16,
            split_counts=d.counts, seed=42, checkpoint_path=tmp_path / "best.pt",
            epochs=epochs)


@pytest.mark.parametrize("max_length", [0, 1, True, 3.5])
def test_invalid_sequence_length_rejected(tmp_path, max_length):
    """Метаданные и кодировщик должны соблюдать один предел длины."""
    d = prepared()
    with pytest.raises(ValidationError, match="max_length"):
        fit_with_validation(model(d), loader(d, "train"), loader(d),
            vocab=d.vocab, label_to_id=d.label_to_id, max_length=max_length,
            split_counts=d.counts, seed=42, checkpoint_path=tmp_path / "best.pt")


def test_validation_loader_is_not_allowed_as_train(tmp_path):
    """Передачу validation вместо train в цикл оптимизации отклоняем."""
    d = prepared()
    with pytest.raises(ValidationError, match="expected train split"):
        fit_with_validation(model(d), loader(d), loader(d),
            vocab=d.vocab, label_to_id=d.label_to_id, max_length=16,
            split_counts=d.counts, seed=42, checkpoint_path=tmp_path / "best.pt")


def test_test_loader_is_not_allowed_as_validation(tmp_path):
    """Выбор по test запрещён явно на уровне fit API."""
    d = prepared()
    with pytest.raises(ValidationError, match="expected val split"):
        fit_with_validation(model(d), loader(d, "train"), loader(d, "test"),
            vocab=d.vocab, label_to_id=d.label_to_id, max_length=16,
            split_counts=d.counts, seed=42, checkpoint_path=tmp_path / "best.pt")


def test_mismatched_data_mapping_rejected(tmp_path):
    """Индекс метки должен совпадать с mapping, собранным исключительно по train."""
    d = prepared()
    mapping = {"linux": 1, "systems": 0}
    with pytest.raises(ValidationError, match="label mapping mismatch"):
        fit_with_validation(model(d), loader(d, "train"), loader(d),
            vocab=d.vocab, label_to_id=mapping, max_length=16,
            split_counts=d.counts, seed=42, checkpoint_path=tmp_path / "best.pt")


def test_mismatched_split_counts_rejected(tmp_path):
    """Файл метаданных не должен приписывать experiment чужой объём train."""
    d = prepared()
    with pytest.raises(ValidationError, match="split count mismatch"):
        fit_with_validation(model(d), loader(d, "train"), loader(d),
            vocab=d.vocab, label_to_id=d.label_to_id, max_length=16,
            split_counts={"train": 999, "val": 1, "test": 1}, seed=42,
            checkpoint_path=tmp_path / "best.pt")


def test_restore_rejects_mismatched_vocabulary(tmp_path):
    """При перестановке ID нельзя использовать веса embedding старого словаря."""
    d, _, result = fit(tmp_path)
    changed = dict(d.vocab.token_to_id)
    tokens = [token for token in changed if changed[token] >= 4]
    changed[tokens[0]], changed[tokens[1]] = changed[tokens[1]], changed[tokens[0]]
    with pytest.raises(ValidationError, match="vocabulary mismatch"):
        restore_best_checkpoint(result.checkpoint_path, vocab=Vocabulary(changed),
                                label_to_id=d.label_to_id, max_length=16)


def test_restore_rejects_mismatched_labels(tmp_path):
    """Если поменять индексы классов, логиты перестают быть сопоставимыми."""
    d, _, result = fit(tmp_path)
    with pytest.raises(ValidationError, match="class mapping mismatch"):
        restore_best_checkpoint(result.checkpoint_path, vocab=d.vocab,
                                label_to_id={"linux": 1, "systems": 0}, max_length=16)


def test_restore_rejects_mismatched_max_length(tmp_path):
    """Отклоняем восстановление на другом варианте обрезания вопроса."""
    d, _, result = fit(tmp_path)
    with pytest.raises(ValidationError, match="sequence length mismatch"):
        restore_best_checkpoint(result.checkpoint_path, vocab=d.vocab,
                                label_to_id=d.label_to_id, max_length=128)


def test_restore_rejects_corrupted_checkpoint(tmp_path):
    """Неполный файл не должен восприниматься как рабочая модель."""
    d = prepared()
    path = tmp_path / "invalid.pt"
    path.write_bytes(b"not a torch checkpoint")
    with pytest.raises(ValidationError, match="checkpoint cannot be loaded"):
        restore_best_checkpoint(path, vocab=d.vocab,
                                label_to_id=d.label_to_id, max_length=16)


@pytest.mark.parametrize("extension", [".txt", ".json", ".pth"])
def test_save_rejects_nonstandard_checkpoint_path(tmp_path, extension):
    """Для корректного .gitignore ограничиваем расширение артефакта .pt."""
    with pytest.raises(ValidationError, match=".pt extension"):
        _save_checkpoint(tmp_path / ("best" + extension), model(), metadata={})


def test_restore_rejects_missing_model_weight(tmp_path):
    """Строгая загрузка не допускает частично сохранённую архитектуру."""
    d, _, result = fit(tmp_path)
    payload = torch.load(result.checkpoint_path, weights_only=True)
    payload["model_state"].pop("output.bias")
    torch.save(payload, result.checkpoint_path)
    with pytest.raises(ValidationError, match="weights do not match"):
        restore_best_checkpoint(result.checkpoint_path, vocab=d.vocab,
                                label_to_id=d.label_to_id, max_length=16)


def test_restore_rejects_nan_weights(tmp_path):
    """Вес NaN повреждает logits, поэтому checkpoint отклоняется заранее."""
    d, _, result = fit(tmp_path)
    payload = torch.load(result.checkpoint_path, weights_only=True)
    payload["model_state"]["output.bias"][0] = float("nan")
    torch.save(payload, result.checkpoint_path)
    with pytest.raises(ValidationError, match="invalid checkpoint tensors"):
        restore_best_checkpoint(result.checkpoint_path, vocab=d.vocab,
                                label_to_id=d.label_to_id, max_length=16)


def test_restore_rejects_bad_format(tmp_path):
    """Не разрешаем произвольные вложенные форматы вместо известной схемы."""
    d, _, result = fit(tmp_path)
    torch.save({"format_version": 999, "metadata": {}, "model_state": {}}, result.checkpoint_path)
    with pytest.raises(ValidationError, match="unknown checkpoint format"):
        restore_best_checkpoint(result.checkpoint_path, vocab=d.vocab,
                                label_to_id=d.label_to_id, max_length=16)


def test_synthetic_cli_smoke_and_boundaries():
    """Отчёт показывает val-выбор, но не выдаёт метрики или записи test."""
    report = run(synthetic=True, epochs=3, expect_train=4, expect_val=1, expect_test=1)
    assert report["split_counts"] == {"train": 4, "val": 1, "test": 1}
    assert report["selection_split"] == "val"
    assert report["trained_split"] == "train"
    assert report["test_evaluated"] is False
    assert report["selection_metric"] == "val_macro_f1"
    assert report["checkpoint_saved"] is True
    assert report["checkpoint_reload_match"] is True
    assert report["validation_invariants_pass"] is True
    assert report["checkpoint_location"] == "temporary"
    assert len(report["history"]) == 3
    assert report["best_epoch"] == 1


def test_cli_outputs_no_raw_questions_or_record_ids():
    """Публичный stdout содержит только агрегаты, а не лицензируемый текст."""
    result = subprocess.run([sys.executable, "-m", "scripts.pytorch_validation", "--synthetic", "--check"],
                            capture_output=True, text=True, check=True)
    assert "F2.5 Validation/Checkpoint CHECK: PASS" in result.stdout
    assert "synthetic-0" not in result.stdout
    assert "linux service command" not in result.stdout
    assert '"test_evaluated": false' in result.stdout


def test_cli_reproducible_summary():
    """Два CPU smoke с одинаковым seed дают равные метрики и лучшую эпоху."""
    assert run(synthetic=True, epochs=3) == run(synthetic=True, epochs=3)


def test_cli_does_not_modify_global_torch_rng():
    """Демонстрация не сбрасывает torch RNG вызывающего приложения."""
    state = torch.random.get_rng_state().clone()
    run(synthetic=True, epochs=2)
    assert torch.equal(state, torch.random.get_rng_state())


def test_synthetic_checkpoint_is_temporary(tmp_path):
    """Без явного пути синтетический эксперимент не создаёт файл в проекте."""
    root = tmp_path / "project"
    report = run(synthetic=True, root=root)
    assert report["checkpoint_location"] == "temporary"
    assert not root.exists()


def test_explicit_local_checkpoint_is_created(tmp_path):
    """Пользовательский checkpoint сохраняется, если указан явный путь."""
    path = tmp_path / "checkpoints" / "experiment.pt"
    report = run(synthetic=True, epochs=2, checkpoint_path=path)
    assert path.is_file() and report["checkpoint_saved"]
    assert report["checkpoint_reload_match"]


def test_cli_rejects_wrong_expected_split():
    """Несовпадение фактического числа записей F1 нельзя игнорировать."""
    with pytest.raises(ValidationError, match="unexpected val"):
        run(synthetic=True, expect_val=500)


def test_local_f1_interface_with_synthetic_jsonl_files(tmp_path):
    """Проверить реальный маршрут чтения F1 JSONL/YAML без публикации корпуса."""
    import json
    from scripts.pytorch_batches import synthetic_splits

    data_dir = tmp_path / "data" / "processed"
    config_dir = tmp_path / "configs" / "local_cpu"
    data_dir.mkdir(parents=True)
    config_dir.mkdir(parents=True)
    for name, rows in synthetic_splits().items():
        (data_dir / f"{name}.jsonl").write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
            encoding="utf-8",
        )
    (config_dir / "lab.yaml").write_text(
        "seed: 42\nneural:\n  vocab_size: 100\n  max_length: 16\n"
        "  batch_size: 2\n  embedding_dim: 5\n  hidden_dim: 6\n"
        "  learning_rate: 0.03\n", encoding="utf-8",
    )
    result = run(synthetic=False, root=tmp_path, epochs=2,
                 expect_train=4, expect_val=1, expect_test=1)
    assert result["source"] == "local_f1_split"
    assert result["checkpoint_saved"] and result["checkpoint_reload_match"]
    assert result["test_evaluated"] is False
    assert (tmp_path / "checkpoints" / "f2_5_best.pt").is_file()


def test_checkpoint_restore_does_not_consume_global_torch_rng(tmp_path):
    """Загрузка весов не должна менять будущую случайную инициализацию моделей."""
    d, _, result = fit(tmp_path)
    state = torch.random.get_rng_state().clone()
    restore_best_checkpoint(result.checkpoint_path, vocab=d.vocab,
                            label_to_id=d.label_to_id, max_length=16)
    assert torch.equal(state, torch.random.get_rng_state())
