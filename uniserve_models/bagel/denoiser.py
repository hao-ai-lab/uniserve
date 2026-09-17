"""BAGEL latent prediction, guidance and diffusion schedules."""

from __future__ import annotations

import torch

from uniserve.diffusion import (
    EulerSolver,
    NestedGuidance,
    NoiseScale,
    Renorm,
    make_schedule,
)
from uniserve.media import image
from uniserve.model import ImageDenoiser
from uniserve.nn.linear import Linear
from uniserve.nn.routing import RouteSpan
from uniserve.nn.timestep import TimestepEmbedding
from uniserve.nn.vision import PositionEmbedding
from uniserve.processing import BranchSource
from uniserve.tensors import OutputLayout, TensorOutput

from .config import Config
from .inputs import DenoiserInput
from .transformer import Transformer


class Denoiser(ImageDenoiser[DenoiserInput]):
    """Predict image-latent velocity through the shared text/flow backbone."""

    # BAGEL frames each generated image with its start and end marker tokens,
    # and its image-unconditional branch reads the conditioning prefix.
    framing_tokens = 2
    image_unconditional = BranchSource.CONDITIONING

    def __init__(self, config: Config, backbone: Transformer):
        super().__init__(
            patch_size=config.latent_patch_size,
            latent_channels=config.vae.latent_channels,
            downsample=config.vae.downsample * config.latent_patch_size,
            noise_scale=NoiseScale(1.0, "constant", 1.0, 1.0),
            prediction_dtype=torch.bfloat16,
            solver=EulerSolver("velocity"),
        )
        self.config, self.backbone = config, backbone
        width = config.latent_patch_size**2 * config.vae.latent_channels
        self.input = Linear(width, config.text.hidden_size)
        self.time_embedding = TimestepEmbedding(config.text.hidden_size)
        self.position = PositionEmbedding(
            (config.max_latent_size,) * 2, config.text.hidden_size
        )
        self.prediction = Linear(config.text.hidden_size, width)
        # Derived marker IDs are numerical constants. Explicit CPU construction
        # survives meta initialization; loading places their borrowed buffer.
        self.register_buffer(
            "markers",
            torch.tensor(
                (config.start_of_image_id, config.end_of_image_id),
                dtype=torch.long,
                device="cpu",
            ),
            persistent=False,
        )

    @property
    def max_sequence_tokens(self) -> int:
        return self.config.max_latent_size**2

    def bind_inputs(
        self,
        *,
        latents,
        sizes,
        step_index,
        positions,
        sequence_lengths,
        attention,
    ) -> DenoiserInput:
        """Assemble one denoising step's typed input from resident tensors."""
        return DenoiserInput(
            latents=latents,
            sizes=sizes,
            step_index=step_index,
            positions=positions,
            sequence_lengths=sequence_lengths,
            attention=attention,
        )

    def noise_shape(self, modality: str, size: image.Config):
        # BAGEL draws directly in canonical patch/channel order.
        return self.latent_shape(modality, size)

    def make_schedules(self, steps, *, shift, device):
        return {
            "image": make_schedule(
                steps,
                shift=self.config.timestep_shift if shift is None else shift,
                direction="descending",
                shift_domain="time",
                device=device,
            )
        }

    def make_guidance(
        self,
        *,
        text_scale: float,
        image_scale: float,
        interval: tuple[float, float],
        renorm: Renorm,
        renorm_min: float,
    ):
        return NestedGuidance(
            text_scale, image_scale, interval, renorm, renorm_min
        )

    def forward(self, inputs: DenoiserInput, *, state, constants, workspace):
        if set(inputs.latents) != {"image"}:
            raise ValueError("BAGEL predicts the image latent modality")

        group = self.mesh.get_group("pp" if "pp" in self.mesh.axes else ())
        chunks, routes = [], []
        cursor = 0
        for latent, size, positions, count in zip(
            inputs.latents["image"],
            inputs.sizes,
            inputs.positions,
            inputs.sequence_lengths,
            strict=True,
        ):
            shape = self.latent_shape("image", size)
            if latent.tensor.shape != shape or count != shape[0] + 2:
                raise ValueError(
                    "BAGEL latents must cover their framed image sequence"
                )

            if group.rank == 0:
                # Frame latent features with embedded start/end-of-image
                # markers; the marker rows route as text, the interior rows
                # as flow.
                marker = self.backbone.embed_input_ids(self.markers).to(
                    torch.bfloat16
                )
                coordinates = (
                    positions[1, 1:-1] * self.config.max_latent_size
                    + positions[2, 1:-1]
                )
                features = self.input(
                    latent.tensor.to(torch.bfloat16)
                )  # [latents, hidden]
                features = (
                    features
                    + self.time_embedding(
                        latent.timestep.reshape(1).expand(shape[0])
                    )
                    + self.position(coordinates)
                ).to(torch.bfloat16)
                chunks.append(torch.cat((marker[:1], features, marker[1:])))

            routes.extend(
                (
                    RouteSpan("text", cursor, 1),
                    RouteSpan("flow", cursor + 1, shape[0]),
                    RouteSpan("text", cursor + count - 1, 1),
                )
            )
            cursor += count

        if not inputs.batch_size:
            return {"image": ()}

        hidden = self.backbone(
            torch.cat(chunks) if group.rank == 0 else None,
            torch.cat(inputs.positions, dim=1),
            inputs.attention,
            routes=tuple(routes),
        )
        if group.rank != group.size - 1:
            return {"image": (None,) * inputs.batch_size}

        outputs = []
        for hidden_row, size in zip(
            hidden.split(inputs.sequence_lengths), inputs.sizes, strict=True
        ):
            # Strip the framing markers before predicting per-latent patches.
            prediction = self.prediction(hidden_row[1:-1].to(torch.bfloat16))
            shape = self.latent_shape("image", size)
            outputs.append(
                TensorOutput(
                    prediction,
                    OutputLayout(
                        shape,
                        prediction.dtype,
                        tuple(slice(0, n) for n in shape),
                    ),
                )
            )
        return {"image": tuple(outputs)}
