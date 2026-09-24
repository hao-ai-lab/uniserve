"""Immutable numerical configuration of the stub model."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Config:
    """Numerical configuration of the stub model.

    Attributes:
        patch_size: Pixel side of one square patch, shared by the vision
            encoder, latent codec, denoiser and image decoder. The worker
            builds ``Model`` with the default, which ``image_processor``'s
            fixed 16-pixel transforms assume.
    """

    patch_size: int = 16

    def __post_init__(self):
        if type(self.patch_size) is not int or self.patch_size < 1:
            raise ValueError(
                "simulation patches must have a positive pixel width"
            )
