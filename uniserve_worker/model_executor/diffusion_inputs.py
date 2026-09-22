"""Worker construction of image trajectory noise and numerical coordinates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from uniserve.diffusion import Branch, normal_noise
from uniserve.media import image
from uniserve.model import ImageDenoiser, LatentInput
from uniserve.processing import BranchSource, FlowPrompt
from uniserve_worker.errors import invalid_descriptor
from uniserve_worker.model_executor.input_batch import AttentionRow


class ImageBuilder:
    """Borrow image mathematics while constructing inputs for admitted requests.

    Coordinates include any framing tokens; sample storage contains only
    patches. Prefix selection and the language-position advance describe how
    the worker inserts a generated image into a continuing conversation.
    """

    rope_advance = 2

    def __init__(self, denoiser: ImageDenoiser):
        self.denoiser = denoiser

    @property
    def framing(self) -> int:
        """Framing tokens the network places around a generated image."""
        return self.denoiser.framing_tokens

    @property
    def max_tokens(self) -> int:
        """Longest image sequence the network accepts."""
        return self.denoiser.max_sequence_tokens

    def sequence_length(self, size: image.Config) -> int:
        return self.denoiser.latent_shape("image", size)[0] + self.framing

    def bind(
        self, *, samples, sizes, timesteps, positions, attention, step_index
    ):
        """Delegate typed input construction to the network that owns it."""
        return self.denoiser.bind_inputs(
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

    def positions(
        self, size: image.Config, temporal: int, *, device
    ) -> torch.Tensor:
        """Construct temporal/height/width coordinates in model sequence.

        order.
        """
        stride = self.denoiser.downsample
        height, width = size.height // stride, size.width // stride
        count = self.sequence_length(size)

        # [3, count]: one (temporal, height, width) coordinate per sequence
        # position; framing tokens keep zero h/w coordinates at the edges.
        result = torch.zeros((3, count), dtype=torch.int64, device=device)
        result[0].fill_(temporal)
        interior = result[:, 1:-1] if self.framing else result
        interior[1].copy_(
            torch.arange(height, device=device).repeat_interleave(width)
        )
        interior[2].copy_(torch.arange(width, device=device).repeat(height))
        return result

    def branch_source(self, branch: Branch) -> BranchSource:
        """Map a guidance branch to the prefix it conditions on."""
        if branch is Branch.CONDITIONED:
            return BranchSource.CONDITIONING
        if branch is Branch.TEXT_UNCONDITIONAL:
            return BranchSource.NEGATIVE_OR_START
        if branch is Branch.IMAGE_UNCONDITIONAL:
            return self.denoiser.image_unconditional
        raise ValueError("unknown image guidance branch")

    @torch.inference_mode()
    def initialize(
        self, size: image.Config, *, seed: int, out: torch.Tensor
    ) -> None:
        """Draw on the trajectory device and preserve the model's native RNG.

        order.
        """
        if out.shape != self.denoiser.latent_shape("image", size):
            raise ValueError(
                "image trajectory storage must have its canonical patch shape"
            )

        noise = torch.empty(
            (1, *self.denoiser.noise_shape("image", size)),
            dtype=out.dtype,
            device=out.device,
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


@dataclass(frozen=True, slots=True, kw_only=True)
class DiffusionRow(AttentionRow):
    """Latent sample, solver time and spatial extent of an image sequence."""

    timestep: torch.Tensor
    latent: torch.Tensor
    image_tokens: int
    image_height: int
    image_width: int

    @property
    def query_tokens(self) -> int:
        return self.image_tokens


def resolve_prefix(
    prompt: FlowPrompt | None,
    source: BranchSource,
    *,
    image_prompt: str,
    negative_prompt: str,
    negative_token_ids: tuple[int, ...],
    tokenizer: Any | None,
) -> tuple[tuple[int, ...], bool]:
    """Resolve a branch prefix and detect empty positive-prompt conditioning."""
    if source is BranchSource.CONDITIONING and not image_prompt.strip():
        return (), True
    if source is BranchSource.NEGATIVE_OR_START and negative_token_ids:
        return negative_token_ids, False

    if prompt is None:
        if source is BranchSource.CONDITIONING:
            raise invalid_descriptor(
                "this model does not accept a generation prompt override"
            )
        return (), False

    if source is BranchSource.CONDITIONING:
        text = image_prompt.strip()
        conditioned = True
    elif source is BranchSource.NEGATIVE_OR_START:
        text = negative_prompt.strip()
        conditioned = False
    else:
        text = ""
        conditioned = False
    return prompt.encode(tokenizer, text=text, conditioned=conditioned), False
