"""Shared encoder batching over ordinary numerical modules."""

from __future__ import annotations

from typing import Any, Literal, Protocol, cast

import torch

from uniserve.model.batch import EncodeBatch, TensorOutput
from uniserve.model.tensors import TensorViews

EncodeKind = Literal["text", "vision", "latent", "conditioning"]


class LatentEncoder(Protocol):
    def encode(self, pixels: torch.Tensor) -> torch.Tensor: ...


class _FeatureEncoder(Protocol):
    downsample_factor: int

    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...


class _LatentModules(Protocol):
    @property
    def latent_encoder(self) -> LatentEncoder: ...


class _EncoderModules(Protocol):
    text_encoder: torch.nn.Module | None
    conditioner: torch.nn.Module | None

    @property
    def vision_encoder(self) -> _FeatureEncoder: ...


class EncoderMixin:
    """Encode aligned rows using the declared kind and numerical representation.

    Text and conditioning modules consume each variable-length row. Vision uses
    uniform image tensors or packed patch tensors with explicit grids.
    VAE inputs use ``encode`` to preserve posterior sampling semantics. Modules
    and borrowed inputs must already reside on the participating device.
    """

    max_vit_grid_tokens: int = 0
    encoder_kinds: frozenset[EncodeKind] = frozenset({"latent"})

    def encode(
        self,
        kind: EncodeKind,
        batch: EncodeBatch,
        *,
        constants: TensorViews,
        scratch: TensorViews,
    ) -> TensorOutput:
        if kind not in self.encoder_kinds:
            raise ValueError(f"unsupported encoder kind {kind!r}")
        modules = cast(_EncoderModules, self)
        if kind in {"text", "conditioning"}:
            encoder = modules.text_encoder if kind == "text" else modules.conditioner
            if encoder is None:
                raise ValueError(f"this model partition does not participate in {kind} encoding")
            return TensorOutput({"conditioning": tuple(encoder(value) for value in batch.values)})
        if kind == "latent":
            latents = cast(_LatentModules, self).latent_encoder.encode(torch.stack(batch.values))
            if latents.shape[0] != len(batch.values):
                raise ValueError("latent encoder must preserve its input batch dimension")
            return TensorOutput({"latents": tuple(latents.unbind(0))})

        if all(value.ndim == 3 for value in batch.values):
            features = modules.vision_encoder(torch.stack(batch.values))
            if features.shape[0] != len(batch.values):
                raise ValueError("vision encoder must preserve its input batch dimension")
            return TensorOutput({"features": tuple(features.unbind(0))})
        if (
            not batch.grids
            or not batch.grid_shapes
            or any(grid is None for grid in batch.grids)
            or any(shape is None for shape in batch.grid_shapes)
        ):
            raise TypeError("packed vision encoding requires patch grids")
        grids = torch.cat(tuple(grid for grid in batch.grids if grid is not None), dim=0)
        grid_shapes = tuple(shape for shape in batch.grid_shapes if shape is not None)
        features = modules.vision_encoder(
            torch.cat(batch.values, dim=0), grids, grid_shapes=grid_shapes
        )
        # Dense spatial downsampling preserves one contiguous feature interval
        # per image. The host-known ratio avoids reading device grids for sizes.
        factor = modules.vision_encoder.downsample_factor
        counts = tuple(int(value.shape[0]) // (factor * factor) for value in batch.values)
        if sum(counts) != int(features.shape[0]):
            raise ValueError("vision encoder output does not align with input rows")
        return TensorOutput({"features": tuple(features.split(counts))})
