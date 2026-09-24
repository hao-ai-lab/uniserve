"""Borrowed image denoising inputs of the stub model."""

from __future__ import annotations

from dataclasses import dataclass

from uniserve.media import image
from uniserve.model import DenoiserInput as NumericalDenoiserInput
from uniserve.nn.attention import AttentionInput


@dataclass(frozen=True)
class DenoiserInput(NumericalDenoiserInput[image.Config]):
    """Latents and image sizes plus the attention input for cache publication."""  # noqa: E501

    # ``Denoiser.forward`` rejects a dense input or one without host query
    # lengths, and writes zero K/V at ``write_indices`` when the input is
    # paged or segmented and carries them.
    attention: AttentionInput
