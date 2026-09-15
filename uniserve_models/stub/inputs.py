"""Borrowed image denoising inputs for the serving simulator."""

from __future__ import annotations

from dataclasses import dataclass

from uniserve.media import image
from uniserve.model import DenoiserInput as NumericalDenoiserInput
from uniserve.nn.attention import AttentionInput


@dataclass(frozen=True)
class DenoiserInput(NumericalDenoiserInput[image.Config]):
    attention: AttentionInput
