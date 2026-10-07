"""Latent normalization conventions of latent encoders and decoders.

A codec's reference maps its native latent to the normalized latent a
generative model consumes, and back. ``LatentNormalization`` declares that
pair of maps; ``LatentEncoder`` and ``LatentDecoder`` call it, so one encoding
and decoding pipeline serves every convention. Each subclass reproduces its
reference's operation order and precision exactly, because the normalized
latent a model was trained on depends on both.
"""

from __future__ import annotations

import math

import torch
from torch import nn


class LatentNormalization(nn.Module):
    """Map native codec latents to normalized latents and back.

    ``normalize`` follows the posterior of an encoder; ``denormalize``
    precedes a decoder network. Both take and return tensors and keep any
    constants they hold as non-persistent state.
    """

    def normalize(self, latents: torch.Tensor) -> torch.Tensor:
        """Return the normalized latent of a native latent."""
        raise NotImplementedError

    def denormalize(self, latents: torch.Tensor) -> torch.Tensor:
        """Return the native latent of a normalized latent."""
        raise NotImplementedError


class ChannelStatistics(LatentNormalization):
    """Standardize every latent channel by its mean and deviation in FP32.

    ``normalize`` first rounds the native latent to ``latent_dtype``, the
    representation in which the codec's reference keeps its latents, then
    computes ``(latent - mean) / std`` in FP32; ``denormalize`` computes
    ``latent * std + mean`` in FP32. Mean and standard deviation are real,
    finite tensors of one shape with positive deviations that broadcast
    against the latent. They register as non-persistent buffers, so they
    retain real values during meta construction and loading moves their
    backing with the codec.
    """

    mean: torch.Tensor
    std: torch.Tensor

    def __init__(
        self,
        *,
        mean: torch.Tensor,
        std: torch.Tensor,
        latent_dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        if mean.is_meta or std.is_meta or mean.shape != std.shape:
            raise ValueError("latent statistics require real matching tensors")
        if (
            not bool(torch.isfinite(mean).all())
            or not bool(torch.isfinite(std).all())
            or not bool((std > 0).all())
        ):
            raise ValueError(
                "latent statistics must be finite with positive standard "
                "deviations"
            )
        if not latent_dtype.is_floating_point:
            raise ValueError("latent representation must be floating point")
        self.latent_dtype = latent_dtype
        self.register_buffer("mean", mean.float(), persistent=False)
        self.register_buffer("std", std.float(), persistent=False)

    def _require_device(self, latents: torch.Tensor) -> None:
        if (
            latents.device != self.mean.device
            or self.std.device != latents.device
        ):
            raise ValueError(
                "latent values and normalization statistics must share "
                "their device"
            )

    def normalize(self, latents: torch.Tensor) -> torch.Tensor:
        self._require_device(latents)
        native = latents.to(self.latent_dtype).float()
        return (native - self.mean) / self.std

    def denormalize(self, latents: torch.Tensor) -> torch.Tensor:
        self._require_device(latents)
        return latents.float() * self.std + self.mean


class ScaleShift(LatentNormalization):
    """Shift then scale the whole latent by two scalars in its own dtype.

    ``normalize`` computes ``scale * (latent - shift)`` and ``denormalize``
    computes ``latent / scale + shift``, each in the dtype of its input, as
    the latent diffusion codecs that define ``scale_factor`` and
    ``shift_factor`` evaluate them.
    """

    def __init__(self, *, scale: float, shift: float):
        super().__init__()
        if not math.isfinite(scale) or scale <= 0 or not math.isfinite(shift):
            raise ValueError(
                "latent scale must be positive and finite, with a finite shift"
            )
        self.scale, self.shift = scale, shift

    def normalize(self, latents: torch.Tensor) -> torch.Tensor:
        return self.scale * (latents - self.shift)

    def denormalize(self, latents: torch.Tensor) -> torch.Tensor:
        return latents / self.scale + self.shift
