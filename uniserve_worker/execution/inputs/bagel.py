"""Construct BAGEL's framed image input at the worker boundary."""

from uniserve.model import LatentInput
from uniserve.processing import BranchSource
from uniserve_models.bagel import DenoiserInput

from .image import ImageBuilder


class BagelBuilder(ImageBuilder):
    """Construct BAGEL's denoising inputs with its two framing tokens."""

    framing = 2
    image_unconditional = BranchSource.CONDITIONING

    @property
    def max_tokens(self):
        return self.denoiser.config.max_latent_size**2

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
            positions=positions,
            sequence_lengths=tuple(
                self.sequence_length(size) for size in sizes
            ),
            attention=attention,
        )
