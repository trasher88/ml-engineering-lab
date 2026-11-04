"""Демонстрация F3.1: расчёт Self-Attention на синтетических векторах.

Первый опыт проверяет результат вручную. Второй показывает, что padding
не участвует в attention и не влияет на векторы настоящих токенов.
Никакие датасеты, checkpoints и предобученные модели не загружаются.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

# Демонстрация запускается из scripts/ без предварительной установки пакета.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from transformers_f3 import SelfAttention, scaled_dot_product_attention  # noqa: E402


def run_demo() -> None:
    """Вывести числовой эталон, размеры результатов и контроль padding.

    Аргументы:
        Нет.

    Результат:
        Текст в консоли; файлы и внешние ресурсы не создаются.
    """
    query = torch.tensor([[[1.0, 0.0]]], dtype=torch.float64)
    key = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]], dtype=torch.float64)
    value = torch.tensor([[[2.0, 0.0], [0.0, 4.0]]], dtype=torch.float64)
    context, weights = scaled_dot_product_attention(query, key, value)
    print("F3.1 — числовой пример scaled dot-product attention")
    print(f"weights: {weights[0, 0].tolist()}")
    print(f"context: {context[0, 0].tolist()}")
    print(f"sum(weights): {weights[0, 0].sum().item():.6f}")

    # Создание модели при фиксированном seed — только для воспроизводимого
    # сравнения одинаковых операций, а не для получения смысловых весов.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(31)
        model = SelfAttention(d_model=4)
    embeddings = torch.tensor(
        [[[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [999., 999., 999., 999.]]]
    )
    mask = torch.tensor([[True, True, True, False]])
    output, attention_weights = model(embeddings, mask)
    print("F3.1 — обучаемый single-head Self-Attention (без обучения)")
    print(f"output shape: {tuple(output.shape)}")
    print(f"attention shape: {tuple(attention_weights.shape)}")
    print(f"веса внимания первого токена: {attention_weights[0, 0].tolist()}")
    print(f"padding key has zero weight: {bool((attention_weights[:, :, -1] == 0).all())}")
    print(f"padding query has zero output: {bool((output[:, -1, :] == 0).all())}")
    print("Внимание не обучено; величины не измеряют семантическую релевантность.")


if __name__ == "__main__":
    run_demo()
