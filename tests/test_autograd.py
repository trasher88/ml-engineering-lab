"""Автономные unit- и smoke-тесты F2.1 на синтетических тензорах CPU.

Подготовленные тексты, обученные модели и артефакты предыдущих фаз не нужны.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from src.pytorch_f2.autograd_demo import (
    analytic_derivative,
    gradient_audit,
    numerical_derivative,
    polynomial,
    run_demo,
    tensor_examples,
)


# Проверяем, что простые операции сохраняют ожидаемые формы и значения.
def test_tensor_operations_shapes_and_values():
    """Сложение вектора и матричное произведение должны быть точными на CPU."""
    result = tensor_examples()
    assert result["vector_shape"] == [3]
    assert result["vector_dtype"] == "torch.float64"
    assert result["vector_device"] == "cpu"
    assert result["shifted_vector"] == [3.0, 4.0, 5.0]
    assert result["matrix_shape"] == [2, 2]
    assert result["matrix_product_shape"] == [2, 1]
    assert result["matrix_product"] == [4.0, 10.0]


# Для x=2 в учебном примере f(x)=17, а точная производная равна 14.
def test_polynomial_known_value_and_derivative():
    """Зафиксировать базовый числовой инвариант для визуальной демонстрации."""
    assert polynomial(2.0) == 17.0
    assert analytic_derivative(2.0) == 14.0


# Сверяем аналитику, центральную разность и backprop в нескольких точках.
@pytest.mark.parametrize("x", [-3.0, -1.0, 0.0, 2.0, 4.0])
def test_three_derivative_methods_agree(x):
    """Autograd должен совпадать с аналитикой, разность — в пределах погрешности."""
    audit = gradient_audit(x)
    expected = 6.0 * x + 2.0
    assert audit["value"] == pytest.approx(polynomial(x))
    assert audit["autograd_gradient"] == pytest.approx(expected, abs=1e-12)
    assert audit["numerical_gradient"] == pytest.approx(expected, rel=1e-6, abs=1e-6)
    assert audit["gradient_error"] < 1e-6


# У листового x градиент до backward отсутствует, а y имеет вычислительный граф.
def test_computational_graph_and_grad_flag():
    """Отслеживание вычислительного графа должно быть включено явно."""
    audit = gradient_audit()
    assert audit["requires_grad"] is True
    assert audit["has_computational_graph"] is True
    assert audit["gradient_initially_none"] is True


# Проверяем, что второй backward прибавляет 2*x, а zero_ сбрасывает сумму.
def test_gradient_accumulation_and_zeroing():
    """После нового backward x² значение .grad должно стать равным 2*x."""
    audit = gradient_audit(2.0)
    assert audit["accumulated_gradient"] == pytest.approx(18.0)
    assert audit["gradient_after_zero"] == 0.0
    assert audit["gradient_after_new_backward"] == pytest.approx(4.0)


# Оба подхода должны разорвать/не строить граф именно для полученных значений.
def test_detach_and_no_grad_do_not_track_operations():
    """detach и no_grad должны создавать результаты без отслеживания градиента."""
    audit = gradient_audit()
    assert audit["detached_requires_grad"] is False
    assert audit["no_grad_requires_grad"] is False


# Отличие от float32: для сравнения конечных разностей важна точность double.
def test_double_precision_and_cpu_only():
    """Демонстрация должна обходиться без CUDA и возвращать конечные числа."""
    result = run_demo()
    assert result["tensors"]["vector_dtype"] == str(torch.float64)
    assert result["tensors"]["vector_device"] == "cpu"
    assert math.isfinite(result["autograd"]["autograd_gradient"])


# Некорректные параметры нельзя молча принимать: NaN разрушит аудит производной.
@pytest.mark.parametrize("x,step", [(float("nan"), 1e-4), (float("inf"), 1e-4), (2.0, 0.0), (2.0, -1.0), (2.0, float("nan"))])
def test_invalid_numerical_inputs_are_rejected(x, step):
    """Численная производная обязана отказывать на неконечных и неверных входах."""
    with pytest.raises(ValueError, match="конечным"):
        numerical_derivative(x, step=step)


# Центральная разность не должна скрывать потерю точности при очень больших x.
def test_reject_step_too_small_for_float64():
    """При x=1e16 значение x+1e-4 совпадёт с x из-за округления."""
    with pytest.raises(ValueError, match="слишком мал"):
        numerical_derivative(1e16)


# Проверяем запуск как публичного модуля, а не только прямой вызов функций.
def test_cli_smoke_check():
    """Команда python -m должна возвращать код 0 и валидный JSON при --check."""
    root = Path(__file__).resolve().parents[1]
    process = subprocess.run(
        [sys.executable, "-m", "src.pytorch_f2.autograd_demo", "--check"],
        cwd=root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    assert process.returncode == 0, process.stderr
    marker = "\nF2.1 Autograd CHECK: PASS"
    assert marker in process.stdout
    payload = json.loads(process.stdout.split(marker)[0])
    assert payload["autograd"]["autograd_gradient"] == 14.0


# Каждый вызов независим: вычислительный граф не должен переживать один запуск.
def test_deterministic_repeatability():
    """Результаты нескольких запусков должны совпадать без seed и случайности."""
    assert run_demo() == run_demo()
