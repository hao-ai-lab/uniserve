"""Live model weights and their installation version."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import torch
from torch import nn

__all__ = ["WeightSet"]


@dataclass(frozen=True, slots=True)
class WeightSet:
    """A read-only view of one installed generation of live module tensors."""

    version: int
    tensors: Mapping[str, torch.Tensor]

    def __post_init__(self) -> None:
        if self.version < 0:
            raise ValueError("weight version cannot be negative")
        if any(
            not isinstance(name, str)
            or not name
            or not isinstance(tensor, torch.Tensor)
            or tensor.is_meta
            for name, tensor in self.tensors.items()
        ):
            raise ValueError("live weight tensors must be named materialized tensors")
        object.__setattr__(self, "tensors", MappingProxyType(dict(self.tensors)))

    @classmethod
    def from_module(
        cls,
        module: nn.Module,
        *,
        version: int = 0,
    ) -> "WeightSet":
        return cls(
            version=int(version),
            tensors=_live_tensors(module),
        )


def _live_tensors(module: nn.Module) -> dict[str, torch.Tensor]:
    return {
        **{
            name: parameter.detach()
            for name, parameter in module.named_parameters()
            if not parameter.is_meta
        },
        **{name: buffer.detach() for name, buffer in module.named_buffers() if not buffer.is_meta},
    }
