"""SenseNova U1 image prediction, guidance and diffusion schedules."""

from __future__ import annotations

import torch
from torch import nn

from uniserve.diffusion import (
    AdditiveGuidance,
    EulerSolver,
    Renorm,
    make_schedule,
)
from uniserve.model import ImageDenoiser
from uniserve.nn.functional import unpatchify
from uniserve.nn.linear import Linear
from uniserve.nn.routing import RouteSpan
from uniserve.nn.timestep import TimestepEmbedding
from uniserve.tensors import OutputLayout, TensorOutput

from . import flow, vision
from .config import Config
from .inputs import DenoiserInput, ImageConditioning
from .transformer import Transformer


class Denoiser(ImageDenoiser[DenoiserInput]):
    """Predict image velocity from pixels through the shared text/flow backbone."""  # noqa: E501

    @property
    def max_sequence_tokens(self) -> int:
        return self.config.max_image_seq_len

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
        """Assemble one denoising step's typed input and per-step conditioning.

        This tower consumes the current image, which must be rebuilt after every
        solver update and must stay distinct from a trajectory's conditioning
        prefix in the K/V cache.
        """
        images = []
        for latent, size in zip(latents["image"], sizes, strict=True):
            sample = latent.value
            pixels = unpatchify(
                sample.unsqueeze(0),
                size,
                patch_size=self.patch_size,
                channels=self.latent_channels,
            )
            patch = self.config.vision.patch_size
            grid = torch.tensor(
                [[size.height // patch, size.width // patch]],
                device=sample.device,
                dtype=torch.int64,
            )
            scale = sample.new_tensor([self.noise_scale.scale(sample.shape[0])])
            images.append(ImageConditioning(pixels, grid, scale))

        return DenoiserInput(
            latents=latents,
            sizes=sizes,
            step_index=step_index,
            positions=positions,
            sequence_lengths=sequence_lengths,
            attention=attention,
            images=tuple(images),
        )

    def __init__(self, config: Config, backbone: Transformer):
        stride = config.vision.patch_size * round(
            1 / config.vision.downsample_ratio
        )
        super().__init__(
            patch_size=stride,
            latent_channels=3,
            downsample=stride,
            noise_scale=config.flow.noise,
            prediction_dtype=torch.float32,
            solver=EulerSolver("velocity"),
        )
        self.config, self.backbone = config, backbone
        self.input = vision.Encoder(config.vision)
        self.time_embedding = TimestepEmbedding(config.text.hidden_size)
        self.noise_embedding = (
            TimestepEmbedding(config.text.hidden_size)
            if config.flow.add_noise_scale_embedding
            else None
        )

        # Three checkpoint head variants: a convolutional pixel decoder, a deep
        # adaptive time head, or a shallow two-layer MLP, exactly one of which
        # is active per checkpoint.
        if config.flow.use_pixel_head:
            head = nn.Identity()
            decoder = flow.Decoder(
                config.text.hidden_size, final_upscale=stride // 4
            )
        elif config.flow.head.num_layers > 2:
            head = flow.Head(
                config.flow.head,
                input_size=config.text.hidden_size,
                output_size=3 * stride**2,
            )
            decoder = nn.Identity()
        else:
            head = nn.Sequential(
                Linear(config.text.hidden_size, config.flow.head.hidden_size),
                nn.GELU(),
                Linear(config.flow.head.hidden_size, 3 * stride**2),
            )
            decoder = nn.Identity()
        self.prediction = flow.Velocity(head, decoder, patch_size=stride)

    def make_schedules(self, steps, *, shift, device):
        return {
            "image": make_schedule(
                steps,
                shift=1.0 if shift is None else shift,
                direction="ascending",
                shift_domain="sigma",
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
        return AdditiveGuidance(
            text_scale, image_scale, interval, renorm, renorm_min
        )

    def forward(self, inputs: DenoiserInput, *, state, constants, workspace):
        if set(inputs.latents) != {"image"}:
            raise ValueError("SenseNova predicts the image latent modality")
        if not inputs.batch_size:
            return {"image": ()}

        for latent, size, conditioning, count in zip(
            inputs.latents["image"],
            inputs.sizes,
            inputs.images,
            inputs.sequence_lengths,
            strict=True,
        ):
            shape = self.latent_shape("image", size)
            if (
                latent.tensor.shape != shape
                or shape[0] != count
                or conditioning.pixels.shape[-2:] != (size.height, size.width)
            ):
                raise ValueError(
                    "SenseNova samples and conditioning must cover "
                    "their declared image dimensions"
                )

        pipeline = self.mesh.get_group("pp" if "pp" in self.mesh.axes else ())
        hidden = None
        if pipeline.rank == 0:
            patch = self.config.vision.patch_size
            shapes = tuple(
                (size.height // patch, size.width // patch)
                for size in inputs.sizes
            )
            # [1, 3, H, W] pixels -> [patches, 3*patch*patch] rows per image.
            pixels = torch.cat(
                tuple(
                    value.pixels.reshape(1, 3, height, patch, width, patch)
                    .permute(0, 2, 4, 1, 3, 5)
                    .reshape(-1, 3 * patch**2)
                    for value, (height, width) in zip(
                        inputs.images, shapes, strict=True
                    )
                )
            )
            hidden = self.input(
                pixels,
                torch.cat(tuple(value.grid for value in inputs.images)),
                shapes,
            )
            times = torch.cat(
                tuple(
                    latent.timestep.reshape(1).expand(count)
                    for latent, count in zip(
                        inputs.latents["image"],
                        inputs.sequence_lengths,
                        strict=True,
                    )
                )
            )
            hidden = hidden + self.time_embedding(times)
            if self.noise_embedding is not None:
                scales = torch.cat(
                    tuple(
                        value.noise_scale.reshape(1).expand(count)
                        for value, count in zip(
                            inputs.images, inputs.sequence_lengths, strict=True
                        )
                    )
                )
                hidden = hidden + self.noise_embedding(
                    scales / self.noise_scale.maximum
                )

        hidden = self.backbone(
            hidden,
            torch.cat(inputs.positions, dim=1),
            inputs.attention,
            routes=(RouteSpan("flow", 0, sum(inputs.sequence_lengths)),),
        )
        if pipeline.rank != pipeline.size - 1:
            return {"image": (None,) * inputs.batch_size}

        outputs = []
        for features, latent, size, conditioning in zip(
            hidden.split(inputs.sequence_lengths),
            inputs.latents["image"],
            inputs.sizes,
            inputs.images,
            strict=True,
        ):
            from uniserve.nn.functional import unpatchify

            # Conditioning is the network input; the solver's current sample
            # is the origin of the velocity equation, even when they differ.
            sample = unpatchify(
                latent.tensor, size, patch_size=self.patch_size, channels=3
            ).unsqueeze(0)
            velocity = self.prediction(
                features,
                latent.timestep,
                ImageConditioning(
                    sample, conditioning.grid, conditioning.noise_scale
                ),
            )
            shape = self.latent_shape("image", size)
            outputs.append(
                TensorOutput(
                    velocity,
                    OutputLayout(
                        shape, velocity.dtype, tuple(slice(0, n) for n in shape)
                    ),
                )
            )
        return {"image": tuple(outputs)}
