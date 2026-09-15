"""Deterministic image prediction and diffusion schedules."""

from __future__ import annotations

import torch

from uniserve.diffusion import AdditiveGuidance, EulerSolver, NoiseScale, make_schedule
from uniserve.model import (
    ImageDenoiser,
    TransformerDecoder,
)
from uniserve.nn.attention import PagedInput, SegmentedInput
from uniserve.tensors import OutputLayout, TensorOutput

from .config import Config
from .inputs import DenoiserInput


class Denoiser(ImageDenoiser):
    def __init__(self, config: Config, backbone: TransformerDecoder):
        super().__init__(
            patch_size=config.patch_size,
            latent_channels=3,
            downsample=config.patch_size,
            noise_scale=NoiseScale(1.0, "constant", 1.0, 1.0),
            prediction_dtype=torch.bfloat16,
            solver=EulerSolver(),
        )
        self.backbone = backbone

    def make_schedules(self, steps, *, shift, device):
        return {
            "image": make_schedule(
                steps,
                shift=1.0 if shift is None else shift,
                direction="ascending",
                shift_domain="time",
                device=device,
            )
        }

    def make_guidance(self, *, text_scale, image_scale, interval, renorm, renorm_min):
        return AdditiveGuidance(text_scale, image_scale, interval, renorm, renorm_min)

    def forward(self, inputs: DenoiserInput, *, state, constants, workspace):
        if set(inputs.latents) != {"image"}:
            raise ValueError("simulation predicts the image latent modality")
        # Latent-feature prefill and image prediction share the scalar cache
        # layer. Read-only attention inputs leave the prefix untouched.
        count = inputs.attention.queries.num_tokens
        reference = inputs.latents["image"][0].tensor
        values = reference.new_zeros((count, 1, 1))
        if (
            isinstance(inputs.attention, (PagedInput, SegmentedInput))
            and inputs.attention.write_indices is not None
        ):
            self.backbone.layers["0"].attention.update_cache(
                values, values, indices=inputs.attention.write_indices
            )
        outputs = []
        for latent, size in zip(inputs.latents["image"], inputs.sizes, strict=True):
            shape = self.latent_shape("image", size)
            if tuple(latent.tensor.shape) != shape:
                raise ValueError("simulation latents must match their canonical image patches")
            outputs.append(
                TensorOutput(
                    torch.zeros_like(latent.tensor, dtype=self.prediction_dtype),
                    OutputLayout(shape, self.prediction_dtype, tuple(slice(0, n) for n in shape)),
                )
            )
        return {"image": tuple(outputs)}
