"""F3.2: офлайн-демонстрация embeddings и отдельный pretrained CPU-режим.

По умолчанию запускается небольшой синтетический encoder без зависимостей,
сети и сохранения файлов. Реальный pretrained encoder доступен только с --real;
флаг --download отдельно разрешает первичную загрузку pinned revision.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from types import SimpleNamespace

import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from transformers_f3.embeddings import ModelSpec, PretrainedTextEncoder  # noqa: E402
from transformers_f3.model_cache import default_cache_dir, load_pretrained_encoder  # noqa: E402


class TinyTokenizer:
    """Детерминированный учебный токенизатор: только для синтетического smoke.

    Не выдаёт языковой модели знания или предметную семантику.
    """

    def __call__(self, texts: list[str], **_: object) -> dict[str, torch.Tensor]:
        """Преобразовать слова в ID и добавить нули до общей длины batch."""
        sequences = [[len(word) % 7 + 1 for word in text.lower().split()] for text in texts]
        length = max(map(len, sequences))
        input_ids = torch.zeros((len(sequences), length), dtype=torch.long)
        mask = torch.zeros_like(input_ids)
        for idx, ids in enumerate(sequences):
            input_ids[idx, : len(ids)] = torch.tensor(ids)
            mask[idx, : len(ids)] = 1
        return {"input_ids": input_ids, "attention_mask": mask}


class TinyEncoder(nn.Module):
    """Фиксированные обученные не были веса: только симуляция форм и pooling."""

    def __init__(self) -> None:
        super().__init__()
        self.table = nn.Embedding(8, 4, padding_idx=0)
        with torch.no_grad():
            self.table.weight.copy_(torch.tensor([
                [0.0, 0.0, 0.0, 0.0],
                [1.0, 0.1, 0.2, 0.0],
                [0.2, 1.0, 0.0, 0.3],
                [0.3, 0.2, 1.0, 0.1],
                [0.5, 0.1, 0.1, 1.0],
                [0.1, 0.8, 0.7, 0.2],
                [0.9, 0.4, 0.2, 0.3],
                [0.2, 0.3, 0.5, 0.8],
            ], dtype=torch.float32))

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> SimpleNamespace:
        """Вернуть [B, L, 4] — фиксированные векторы токенов."""
        return SimpleNamespace(last_hidden_state=self.table(input_ids))


def run_demo(*, real: bool = False, download: bool = False, cache_dir: Path | None = None) -> None:
    """Проверить shape, L2-норму и детерминированность на CPU.

    Аргументы:
        real: Использовать Hugging Face pretrained encoder вместо учебного.
        download: Явно разрешить первое скачивание модели (только при real).
        cache_dir: Расположение пользовательского кеша модели.
    Результат:
        Печать числовых характеристик; пользовательские тексты не сохраняются.
    """
    if download and not real:
        raise ValueError("--download используется только вместе с --real")
    if real:
        spec = ModelSpec.from_json(PROJECT_ROOT / "configs" / "f3_02_model.json")
        root = cache_dir if cache_dir is not None else default_cache_dir()
        manifest_path = root.parent / "manifest.json"
        encoder, manifest = load_pretrained_encoder(
            spec, cache_dir=root, allow_download=download, manifest_path=manifest_path
        )
        print("Режим: pretrained encoder (реальный, фиксированный revision)")
        print(f"Model: {spec.model_id}@{spec.revision}")
        print(f"License: {spec.license}")
        print(f"Manifest: {manifest_path}")
        print(f"Artifacts hashed: {len(manifest['sha256'])}")
    else:
        # Ревизия валидна по формату, но синтетический encoder не принадлежит Hub.
        spec = ModelSpec("examples/tiny-synthetic", "0" * 40, "synthetic", 4, 8)
        encoder = PretrainedTextEncoder(spec, TinyTokenizer(), TinyEncoder())
        print("Режим: synthetic (не является pretrained и не измеряет релевантность)")

    texts = ["postgresql slow query", "postgresql index", "linux kernel memory"]
    vectors = encoder.encode(texts, batch_size=2)
    print(f"embeddings shape: {tuple(vectors.shape)}")
    print(f"dtype: {vectors.dtype}; device: {vectors.device}")
    print(f"L2 norms: {[round(float(v), 6) for v in vectors.norm(dim=1)]}")
    print(f"repeat deterministic: {bool(torch.allclose(vectors, encoder.encode(texts, batch_size=1), atol=1e-6))}")
    similarity = float(torch.dot(vectors[0], vectors[1]))
    print(f"cosine(first, second): {similarity:.6f}")
    print("Cosine similarity здесь не вероятность релевантности и не retrieval-метрика.")


def main() -> None:
    """Обработать CLI-флаги; сеть запрещена без явного --download."""
    parser = argparse.ArgumentParser(description="F3.2 — Pretrained Embeddings, Versions & Cache")
    parser.add_argument("--real", action="store_true", help="Использовать pretrained модель из локального кеша")
    parser.add_argument("--download", action="store_true", help="Явно разрешить загрузку pinned модели")
    parser.add_argument("--cache-dir", type=Path, help="Пользовательский каталог кеша, вне репозитория")
    args = parser.parse_args()
    run_demo(real=args.real, download=args.download, cache_dir=args.cache_dir)


if __name__ == "__main__":
    main()
