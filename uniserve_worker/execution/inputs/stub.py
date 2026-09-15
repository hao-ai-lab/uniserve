"""Construct the deterministic simulator's numerical image input."""

from uniserve.model import LatentInput
from uniserve_models.stub import DenoiserInput

from .image import ImageBuilder


class StubBuilder(ImageBuilder):
    """Construct the simulator's denoising inputs with its framing tokens."""

    framing = 2
    max_tokens = 1024

    def bind(
        self, *, samples, sizes, timesteps, positions, attention, step_index
    ):
        """Assemble one denoising step's typed input from resident tensors."""
        return DenoiserInput(
            latents={
                "image": tuple(
                    LatentInput(value, time)
                    for value, time in zip(samples, timesteps, strict=True)
                )
            },
            sizes=sizes,
            step_index=step_index,
            attention=attention,
        )
