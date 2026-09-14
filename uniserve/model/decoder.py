"""Numerical media decoding composed from shared decoder modules."""

from __future__ import annotations

from typing import Literal, Protocol, cast

import torch

from uniserve.model.batch import DecodeBatch, TensorOutput
from uniserve.model.media import ImageSize
from uniserve.model.tensors import TensorViews
from uniserve.tensors import ImageRange, OutputLayout

DecodeKind = Literal["image", "video", "audio"]


class ImageDecoder(Protocol):
    """A numerical image decoder, including its output interval."""

    value_range: ImageRange

    def decode(self, latents: torch.Tensor, height: int, width: int) -> torch.Tensor: ...


class _ImageModules(Protocol):
    @property
    def image_decoder(self) -> ImageDecoder: ...


class DecoderMixin:
    """Decode homogeneous latent rows through an ordinary decoder module.

    The image implementation batches a common geometry and preserves both
    input order and the decoder's numerical range. Video and audio models
    implement their windowed mathematical computation through the same entry.
    """

    decoder_kinds: frozenset[DecodeKind] = frozenset({"image"})

    def decode(
        self,
        kind: DecodeKind,
        batch: DecodeBatch[ImageSize],
        *,
        constants: TensorViews,
        scratch: TensorViews,
    ) -> TensorOutput:
        if kind != "image":
            raise ValueError(f"unsupported decoder kind {kind!r}")
        if batch.windows or any(not isinstance(size, ImageSize) for size in batch.sizes):
            raise ValueError("image decode requires image sizes without temporal windows")
        geometry = {(shape.height, shape.width) for shape in batch.sizes}
        if len(geometry) != 1:
            raise ValueError("image decode requires one output geometry per numerical batch")
        height, width = next(iter(geometry))
        decoder = cast(_ImageModules, self).image_decoder
        pixels = decoder.decode(torch.stack(batch.latents), height, width)
        if pixels.ndim != 4 or tuple(pixels.shape) != (len(batch.latents), 3, height, width):
            raise ValueError("image decoder output must have shape [batch, 3, height, width]")
        values = tuple(pixels.unbind(0))
        layout = OutputLayout(
            shape=(3, height, width), dtype=pixels.dtype, value_range=decoder.value_range
        )
        return TensorOutput({"image": values}, {"image": (layout,) * len(values)})
