"""Shared latent normalization with externally bound decoder execution."""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar

import torch
from torch import nn

DecoderCall = Callable[[torch.Tensor], torch.Tensor]
_decoders: ContextVar[Mapping["LatentDecoder", DecoderCall] | None] = ContextVar(
    "latent_decoders", default=None
)


@contextmanager
def decoder_scope(bindings: Mapping[LatentDecoder, DecoderCall] | None) -> Iterator[None]:
    """Borrow native decoder calls supplied by the current execution owner.

    An explicit mapping must cover every invoked latent decoder. None selects
    ordinary numerical execution, including the operation that a runtime
    captures. The scope retains no bindings after host enqueue completes;
    execution owners retain graphs, storage, and readers until GPU completion.
    """

    token = _decoders.set(bindings)
    try:
        yield
    finally:
        _decoders.reset(token)


class LatentDecoder(nn.Module):
    """Restore channel statistics in float32 before native reconstruction.

    Concrete decoders supply their learned reconstruction, temporal crop, and
    output representation. Normalization precedes any model-specific autocast.
    Native execution can be bound by a caller without passing an execution
    owner into the model or changing the module's parameter namespace.
    """

    vae: nn.Module
    latent_shape: tuple[int | None, ...]
    latents_mean: torch.Tensor
    latents_std: torch.Tensor

    @property
    def device(self) -> torch.device:
        return next(self.vae.parameters()).device

    @torch.inference_mode()
    def forward(self, normalized_latents: torch.Tensor) -> torch.Tensor:
        expected = self.latent_shape
        if normalized_latents.ndim != len(expected) or any(
            size is not None and size != actual
            for size, actual in zip(expected, normalized_latents.shape, strict=True)
        ):
            raise ValueError(f"decoder latent shape must match {expected}")
        bindings = _decoders.get()
        if bindings is not None:
            operation = bindings.get(self)
            if operation is None:
                raise RuntimeError("native decoder is missing from the explicit execution binding")
            return operation(normalized_latents)
        latents = normalized_latents.to(device=self.device, dtype=torch.float32)
        latents = latents * self.latents_std + self.latents_mean
        return self._reconstruct(latents)

    @abstractmethod
    def _reconstruct(self, latents: torch.Tensor) -> torch.Tensor:
        """Reconstruct already denormalized latents in the decoder's native units."""
