"""Worker construction of image trajectory noise and numerical coordinates."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from uniserve.diffusion import Branch, normal_noise
from uniserve.media import image
from uniserve.model import ImageDenoiser
from uniserve_models.processing import BranchSource


class ImageInputs:
    """Borrow image mathematics while constructing inputs for admitted requests.

    Coordinates include any framing tokens; sample storage contains only
    patches. Prefix selection and the language-position advance describe how
    the worker inserts a generated image into a continuing conversation.
    """

    framing = 0
    rope_advance = 2
    image_unconditional = BranchSource.START

    def __init__(self, denoiser: ImageDenoiser):
        self.denoiser = denoiser

    def sequence_length(self, size: image.Config) -> int:
        return self.denoiser.latent_shape("image", size)[0] + self.framing

    def positions(self, size: image.Config, temporal: int, *, device) -> torch.Tensor:
        """Construct temporal/height/width coordinates in model sequence order."""

        stride = self.denoiser.downsample
        height, width = size.height // stride, size.width // stride
        count = self.sequence_length(size)
        result = torch.zeros((3, count), dtype=torch.int64, device=device)
        result[0].fill_(temporal)
        interior = result[:, 1:-1] if self.framing else result
        interior[1].copy_(torch.arange(height, device=device).repeat_interleave(width))
        interior[2].copy_(torch.arange(width, device=device).repeat(height))
        return result

    def branch_source(self, branch: Branch) -> BranchSource:
        if branch is Branch.CONDITIONED:
            return BranchSource.CONDITIONING
        if branch is Branch.TEXT_UNCONDITIONAL:
            return BranchSource.NEGATIVE_OR_START
        if branch is Branch.IMAGE_UNCONDITIONAL:
            return self.image_unconditional
        raise ValueError("unknown image guidance branch")

    @torch.inference_mode()
    def initialize(self, size: image.Config, *, seed: int, out: torch.Tensor) -> None:
        """Draw on the trajectory device and preserve the model's native RNG order."""

        if out.shape != self.denoiser.latent_shape("image", size):
            raise ValueError("image trajectory storage must have its canonical patch shape")
        noise = torch.empty(
            (1, *self.denoiser.noise_shape("image", size)), dtype=out.dtype, device=out.device
        )
        normal_noise((seed,), out=(noise,))
        self.denoiser.prepare_latents(
            (size,),
            noise={"image": noise},
            state={"image": out.unsqueeze(0)},
            constants={},
            workspace={},
        )


@dataclass(frozen=True, slots=True)
class DecodeInput:
    """Arguments the worker supplies to ImageDecoder.decode."""

    latents: tuple[torch.Tensor, ...]
    sizes: tuple[image.Config, ...]
