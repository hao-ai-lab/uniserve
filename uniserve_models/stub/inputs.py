"""Borrowed image denoising inputs of the stub model."""

from __future__ import annotations

from dataclasses import dataclass

from uniserve.media import image
from uniserve.model import DenoiserInput as NumericalDenoiserInput
from uniserve.nn.attention import AttentionBatch


@dataclass(frozen=True)
class DenoiserInput(NumericalDenoiserInput[image.Config]):
    """Latents and image sizes plus the attention input for cache publication."""  # noqa: E501

    # ``Denoiser.forward`` rejects a dense batch or one without host query
    # lengths, and writes zero K/V at the cache layer's table entry's
    # ``write_indices`` when it carries them.
    attention: AttentionBatch
