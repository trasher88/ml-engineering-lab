"""F2.1: воспроизводимая CPU-демонстрация тензоров и автоматического дифференцирования.

На синтетической функции проверяются прямые вычисления, вычислительный граф,
аналитическая и численная производные, накопление/сброс градиента и отключение
отслеживания через detach() и torch.no_grad(). Реальные данные F1 не требуются.
"""

from __future__ import annotations

import argparse
import json
import math
from typing import Any

import torch


def polynomial(x: float) -> float:
    """Вычислить 3*x² + 2*x + 1 по обычной числовой арифметике."""
    return 3.0 * x * x + 2.0 * x + 1.0


def analytic_derivative(x: float) -> float:
    """Вернуть точную производную многочлена: 6*x + 2."""
    return 6.0 * x + 2.0


def numerical_derivative(x: float, *, step: float = 1e-4) -> float:
    """Оценить производную центральной конечной разностью.

    Args:
        x: Точка измерения производной.
        step: Положительный шаг разности (не должен быть слишком мал).

    Returns:
        Приближённое значение производной.

    Raises:
        ValueError: Вход не конечен или шаг не положителен/не конечен.

    Ограничение: погрешность зависит от шага и округления float64.
    """
    if not math.isfinite(x) or not math.isfinite(step) or step <= 0:
        raise ValueError("x должен быть конечным, а step — конечным и положительным")
    if x + step == x or x - step == x:
        raise ValueError("step слишком мал относительно x для арифметики float64")
    approximation = (polynomial(x + step) - polynomial(x - step)) / (2.0 * step)
    if not math.isfinite(approximation):
        raise ValueError("Численное вычисление производной привело к нечисловому результату")
    return approximation


def tensor_examples() -> dict[str, Any]:
    """Показать формы вектора и матрицы, сложение и матричное умножение на CPU."""
    vector = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64, device="cpu")
    shifted = vector + 2.0
    matrix = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64)
    column = torch.tensor([[2.0], [1.0]], dtype=torch.float64)
    product = matrix @ column
    return {
        "vector_shape": list(vector.shape),
        "vector_dtype": str(vector.dtype),
        "vector_device": vector.device.type,
        "shifted_vector": shifted.tolist(),
        "matrix_shape": list(matrix.shape),
        "matrix_product_shape": list(product.shape),
        "matrix_product": product.flatten().tolist(),
    }


def gradient_audit(x_value: float = 2.0, *, step: float = 1e-4) -> dict[str, Any]:
    """Вычислить производную тремя способами и показать особенности Autograd.

    Args:
        x_value: Конечное вещественное значение входного скаляра.
        step: Шаг центральной конечной разности.

    Returns:
        Словарь только со скалярными значениями и флагами для печати/тестов.

    Тензор double обеспечивает надёжное сопоставление с численной производной;
    расчёты не используют CUDA, случайность, датасеты и состояние модели.
    """
    numeric = numerical_derivative(x_value, step=step)
    x = torch.tensor(x_value, dtype=torch.float64, device="cpu", requires_grad=True)
    y = 3.0 * x.square() + 2.0 * x + 1.0
    # grad_fn означает, что результат получен из отслеживаемых операций.
    has_graph = y.grad_fn is not None
    # До backward() у листового тензора градиента нет.
    grad_initially_none = x.grad is None
    y.backward()
    pytorch_grad = float(x.grad.item())

    # PyTorch добавляет новые производные к .grad, а не заменяет прежние.
    x.square().backward()
    accumulated_grad = float(x.grad.item())
    x.grad.zero_()
    zeroed_grad = float(x.grad.item())
    # Для нового backward создаём НОВЫЙ граф; старый уже освобождён.
    x.square().backward()
    grad_after_new_backward = float(x.grad.item())

    # detach() исключает полученный тензор из графа, не меняя исходный x.
    detached = (x * 5.0).detach()
    with torch.no_grad():
        untracked = x * 5.0

    return {
        "x": x_value,
        "value": float(y.item()),
        "analytic_gradient": analytic_derivative(x_value),
        "numerical_gradient": numeric,
        "autograd_gradient": pytorch_grad,
        "gradient_error": abs(pytorch_grad - numeric),
        "requires_grad": x.requires_grad,
        "has_computational_graph": has_graph,
        "gradient_initially_none": grad_initially_none,
        "accumulated_gradient": accumulated_grad,
        "gradient_after_zero": zeroed_grad,
        "gradient_after_new_backward": grad_after_new_backward,
        "detached_requires_grad": detached.requires_grad,
        "no_grad_requires_grad": untracked.requires_grad,
    }


def run_demo(x_value: float = 2.0, *, step: float = 1e-4) -> dict[str, Any]:
    """Собрать детерминированный результат для CLI и автоматизированной проверки."""
    return {"tensors": tensor_examples(), "autograd": gradient_audit(x_value, step=step)}


def main() -> int:
    """Запустить демонстрацию и вернуть ненулевой код при провале сверки градиентов."""
    parser = argparse.ArgumentParser(description="F2.1: тензоры и Autograd на CPU")
    parser.add_argument("--x", type=float, default=2.0, help="точка производной")
    parser.add_argument("--step", type=float, default=1e-4, help="шаг конечной разности")
    parser.add_argument("--check", action="store_true", help="проверить совпадение градиентов")
    args = parser.parse_args()
    try:
        result = run_demo(args.x, step=args.step)
    except ValueError as exc:
        parser.error(str(exc))

    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    if args.check:
        audit = result["autograd"]
        exact = math.isclose(
            audit["autograd_gradient"], audit["analytic_gradient"], rel_tol=1e-12, abs_tol=1e-12
        )
        approximate = math.isclose(
            audit["numerical_gradient"], audit["analytic_gradient"], rel_tol=1e-6, abs_tol=1e-6
        )
        print("F2.1 Autograd CHECK: PASS" if exact and approximate else "F2.1 Autograd CHECK: FAIL")
        return 0 if exact and approximate else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
