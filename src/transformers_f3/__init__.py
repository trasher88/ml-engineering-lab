"""Учебные компоненты Transformers для третьей фазы ML Engineering Lab.

F3.1 содержит собственную реализацию одноголового Self-Attention.
Она не использует предобученные модели или внешний корпус документов.
"""

from .attention import SelfAttention, scaled_dot_product_attention

__all__ = ["SelfAttention", "scaled_dot_product_attention"]
