"""Immutable model-weight snapshots selected by execution plans."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import torch
from torch import nn

__all__ = ["WeightSet", "module_weight_digest"]


@dataclass(frozen=True, slots=True)
class WeightSet:
    """One immutable parameter/buffer mapping for stateless module execution."""

    digest: str
    version: int
    tensors: Mapping[str, torch.Tensor]

    def __post_init__(self) -> None:
        object.__setattr__(self, "tensors", MappingProxyType(dict(self.tensors)))

    @classmethod
    def from_module(
        cls,
        module: nn.Module,
        *,
        digest: str | None = None,
    ) -> "WeightSet":
        tensors = {
            **{name: parameter.detach() for name, parameter in module.named_parameters()},
            **{name: buffer.detach() for name, buffer in module.named_buffers()},
        }
        return cls(
            digest=digest or module_weight_digest(module),
            version=0,
            tensors=tensors,
        )


def module_weight_digest(module: nn.Module) -> str:
    """Hash tensor names, metadata, and bytes for an in-memory weight graph."""

    digest = hashlib.sha256(b"uniserve-weight-set\0")
    tensors = {
        **dict(module.named_parameters()),
        **dict(module.named_buffers()),
    }
    for name in sorted(tensors):
        tensor = tensors[name].detach()
        encoded_name = name.encode("utf-8")
        digest.update(len(encoded_name).to_bytes(8, "little"))
        digest.update(encoded_name)
        dtype = str(tensor.dtype).encode("ascii")
        digest.update(len(dtype).to_bytes(2, "little"))
        digest.update(dtype)
        digest.update(len(tensor.shape).to_bytes(2, "little"))
        for dimension in tensor.shape:
            digest.update(int(dimension).to_bytes(8, "little"))
        raw = tensor.contiguous().cpu().reshape(-1).view(torch.uint8).numpy()
        digest.update(raw.tobytes())
    return digest.hexdigest()
