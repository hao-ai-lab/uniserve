"""Numerical media decoding composed from shared decoder modules."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Protocol, cast

import torch

from .batch import DecodeBatch, TensorOutput
from .components import Call
from .geometry import MediaShape, Shape, TensorOutputLayout
from .resources import TensorNeeds
from .tensors import ImageRange, TensorViews

if TYPE_CHECKING:
    from ..nn.vae.decoder import LatentDecoder
    from .model import Model

DecodeKind = Literal["image", "video", "audio"]


class ImageDecoder(Protocol):
    """A numerical image decoder, including its output interval."""

    value_range: ImageRange

    def tensor_specs(self, call: Call, shape: MediaShape) -> TensorNeeds: ...

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

    video_decoder: LatentDecoder | None
    audio_decoder: LatentDecoder | None

    decoder_kinds: frozenset[DecodeKind] = frozenset({"image"})

    def tensor_specs(self, call: Call, shape: Shape) -> TensorNeeds:
        """Declare the composed image decoder's raster and arithmetic representation."""

        if call is not Call.DECODE_IMAGE:
            return cast("Model", super()).tensor_specs(call, shape)
        if not isinstance(shape, MediaShape) or shape.frames != 1:
            raise ValueError("image decoding requires a single-frame media shape")
        return cast(_ImageModules, self).image_decoder.tensor_specs(call, shape)

    def decode(
        self,
        kind: DecodeKind,
        batch: DecodeBatch,
        *,
        constants: TensorViews,
        scratch: TensorViews,
    ) -> TensorOutput:
        if kind != "image":
            raise ValueError(f"unsupported decoder kind {kind!r}")
        if batch.windows or any(shape.frames != 1 for shape in batch.shapes):
            raise ValueError("image decode requires single-frame inputs without temporal windows")
        geometry = {(shape.height, shape.width) for shape in batch.shapes}
        if len(geometry) != 1:
            raise ValueError("image decode requires one output geometry per numerical batch")
        height, width = next(iter(geometry))
        decoder = cast(_ImageModules, self).image_decoder
        pixels = decoder.decode(torch.stack(batch.latents), height, width)
        if pixels.ndim != 4 or tuple(pixels.shape) != (len(batch.latents), 3, height, width):
            raise ValueError("image decoder output must have shape [batch, 3, height, width]")
        values = tuple(pixels.unbind(0))
        layout = TensorOutputLayout(shape=(3, height, width), value_range=decoder.value_range)
        return TensorOutput({"image": values}, {"image": (layout,) * len(values)})
