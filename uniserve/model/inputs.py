"""Borrowed numerical model inputs with no request or execution ownership."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Generic, Literal, TypeVar

import torch

from uniserve.nn.attention import AttentionInput, DenseInput
from uniserve.nn.routing import RouteSpan

SizeT = TypeVar("SizeT")


@dataclass(frozen=True, slots=True)
class EntryPoint:
    method: str
    stage: Literal["all", "first", "last"] = "all"
    groups: tuple[str, ...] = ()

    def __post_init__(self):
        if not self.method or self.stage not in {"all", "first", "last"}:
            raise ValueError(
                "entry points require a method and a valid pipeline participation stage"
            )
        if len(set(self.groups)) != len(self.groups) or any(not name for name in self.groups):
            raise ValueError("entry-point communication axes must be distinct nonempty names")


@dataclass(frozen=True, slots=True)
class EmbeddingReplacement:
    values: torch.Tensor
    mask: torch.Tensor


@dataclass(frozen=True, slots=True)
class TextInput:
    input_ids: torch.Tensor
    positions: torch.Tensor
    attention: AttentionInput
    embeddings: EmbeddingReplacement | None = None
    routes: tuple[RouteSpan, ...] = ()

    @property
    def batch_size(self) -> int:
        if isinstance(self.attention, DenseInput):
            return self.input_ids.shape[0] if self.input_ids.ndim == 2 else 1
        return self.attention.queries.batch_size


@dataclass(frozen=True, slots=True)
class VisionInput:
    images: tuple[torch.Tensor, ...]
    grids: tuple[torch.Tensor | None, ...]
    grid_shapes: tuple[tuple[int, int] | None, ...]

    def __post_init__(self):
        if len(self.images) != len(self.grids) or len(self.images) != len(self.grid_shapes):
            raise ValueError("vision tensors and grid descriptions must align by sample")

    @property
    def batch_size(self) -> int:
        return len(self.images)


@dataclass(frozen=True, slots=True)
class LatentInput:
    tensor: torch.Tensor
    timestep: torch.Tensor


@dataclass(frozen=True)
class DenoiserInput(Generic[SizeT]):
    latents: Mapping[str, tuple[LatentInput, ...]]
    sizes: tuple[SizeT, ...]
    step_index: int

    def __post_init__(self):
        if type(self.step_index) is not int or self.step_index < 0:
            raise ValueError("denoiser step index must be a nonnegative integer")
        if any(not name or len(values) != len(self.sizes) for name, values in self.latents.items()):
            raise ValueError("latent modalities must align with the numerical sample sizes")
        object.__setattr__(self, "latents", MappingProxyType(dict(self.latents)))

    @property
    def batch_size(self) -> int:
        return len(self.sizes)


@dataclass(frozen=True, slots=True)
class TextSize:
    num_tokens: int
    batch_size: int

    def __post_init__(self):
        if any(type(value) is not int or value < 0 for value in (self.num_tokens, self.batch_size)):
            raise ValueError("text sizes require nonnegative token and sequence counts")
