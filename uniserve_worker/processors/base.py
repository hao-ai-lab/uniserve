"""Multimodal processor base types."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import torch
from PIL import Image

__all__ = [
    'MultimodalDataItem',
    'MultimodalProcessor',
]


@dataclass(frozen=True)
class MultimodalDataItem:
    modality: str
    tensor: torch.Tensor
    mm_hash: int | None = None
    metadata: dict[str, Any] | None = None


@runtime_checkable
class MultimodalProcessor(Protocol):
    """Structural contract for per-model multimodal preprocessors.

    Processors are matched to models via :attr:`model_architectures` (the
    registry) and expose the concrete preprocessing helpers their model calls
    directly (e.g. ``vae_tensor``/``vit_tensor``). There is no uniform async
    entry point: models invoke the helpers by name, so the surface here is the
    set of members the registry and serving path depend on. Conformance is
    structural, so processors satisfy it without explicit inheritance.
    """

    model_architectures: tuple[str, ...]

    def vit_tensor(self, image: Image.Image) -> torch.Tensor: ...

    def vae_tensor(self, image: Image.Image) -> torch.Tensor: ...
