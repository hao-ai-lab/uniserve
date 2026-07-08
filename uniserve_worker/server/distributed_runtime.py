"""Explicit distributed runtime initialization."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

__all__ = [
    "DistributedRuntime",
    "MeshContext",
]


@dataclass(frozen=True)
class MeshContext:
    mesh: Any
    device: torch.device


class DistributedRuntime:
    """Adapter seam for torch distributed and CUDA mesh setup."""

    def initialize(self, **kwargs: Any) -> MeshContext:
        from .distributed import build_device_mesh

        mesh = build_device_mesh(**kwargs)
        return MeshContext(mesh=mesh, device=getattr(mesh, "device", torch.device(kwargs.get("device", "cpu"))))

    def shutdown(self) -> None:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
