"""Deterministic image prediction and diffusion schedules.

``Denoiser`` predicts exactly zero for every image sample while using the
library's real schedule, guidance and Euler solver. A zero prediction leaves
each sample unchanged through every solver step, and the noise scale is the
constant 1, so a generated image decodes to its initial normal noise draw.
"""

from __future__ import annotations

import torch

from uniserve.diffusion import (
    AdditiveGuidance,
    EulerSolver,
    NoiseScale,
    make_schedule,
)
from uniserve.model import (
    ImageDenoiser,
    TransformerDecoder,
)
from uniserve.nn.attention import Attention
from uniserve.tensors import OutputLayout, TensorOutput

from .config import Config
from .inputs import DenoiserInput


class Denoiser(ImageDenoiser):
    """Zero-valued image prediction over the real diffusion solver machinery.

    Latents are canonical patch rows of ``patch_size**2 * 3`` values, one row
    per ``patch_size`` pixel square: the layout ``Model.latent_encoder``
    produces and ``Model.image_decoder`` consumes.
    """

    framing_tokens = 2

    @property
    def max_sequence_tokens(self) -> int:
        return 1024

    def bind_inputs(
        self,
        *,
        latents,
        sizes,
        step,
        positions,
        sequence_lengths,
        attention,
    ) -> DenoiserInput:
        """Assemble one denoising step's typed input from resident tensors.

        ``positions`` and ``sequence_lengths`` are accepted for the
        ``ImageDenoiser`` contract and unused: the zero prediction depends on
        neither.
        """
        return DenoiserInput(
            latents=latents,
            sizes=sizes,
            step=step,
            attention=attention,
        )

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

    def make_guidance(
        self, *, text_scale, image_scale, interval, renorm, renorm_min
    ):
        return AdditiveGuidance(
            text_scale, image_scale, interval, renorm, renorm_min
        )

    def forward(self, inputs: DenoiserInput, *, state, constants, workspace):
        if set(inputs.latents) != {"image"}:
            raise ValueError("simulation predicts the image latent modality")

        # The host query token total sizes the K/V rows written below; dense
        # inputs and inputs without host lengths cannot supply it.
        queries = inputs.attention.queries
        if queries is None or queries.num_tokens is None:
            raise ValueError(
                "simulation denoising requires packed host query lengths"
            )
        # [query tokens, kv heads, head dim] for the one-head, width-one cache.
        reference = inputs.latents["image"][0].tensor
        values = reference.new_zeros((queries.num_tokens, 1, 1))

        # The step's zero K/V publish through the backbone's single attention
        # layer, the cache ``model._Layer`` writes for text tokens. The
        # layer's table entry supplies the addresses; an entry without write
        # addresses leaves the cache untouched.
        cache = self.backbone.layers["0"].attention
        if not isinstance(cache, Attention):
            raise ValueError(
                "simulation backbone must publish through its scalar "
                "attention cache layer"
            )
        cache.update_cache(values, values, inputs.attention)

        # The prediction is exactly zero; only the shape must match each
        # latent's canonical patch grid.
        outputs = []
        for latent, size in zip(
            inputs.latents["image"], inputs.sizes, strict=True
        ):
            shape = self.latent_shape("image", size)
            if tuple(latent.tensor.shape) != shape:
                raise ValueError(
                    "simulation latents must match their canonical "
                    "image patches"
                )
            outputs.append(
                TensorOutput(
                    torch.zeros_like(
                        latent.tensor, dtype=self.prediction_dtype
                    ),
                    OutputLayout(
                        shape,
                        self.prediction_dtype,
                        tuple(slice(0, n) for n in shape),
                    ),
                )
            )
        return {"image": tuple(outputs)}
