"""SenseNova U1 image prediction, guidance and diffusion schedules.

SenseNova denoises in pixel space. A sample is one image stored as canonical
patch rows ``[tokens, 3 * stride**2]``, where ``stride`` is the vision patch
size times the vision downsampling factor, so every row is one backbone token
and holds that token's ``stride x stride`` RGB block in pixel, then channel
order (``uniserve.nn.functional.patchify``). There is no VAE: the model's
``ImageDecoder`` only unfolds the patch rows back into pixels.

Each step re-encodes the current sample through this module's own
``vision.Encoder``, adds timestep (and optionally noise-scale) embeddings, and
runs every image token through the ``flow`` experts of the backbone shared
with the text model. The image tokens attend to their guidance branch's
prefix in the K/V cache. A velocity head then turns the backbone output into
the solver's velocity (``flow.Velocity``).
"""

from __future__ import annotations

import torch
from torch import nn

from uniserve.diffusion import (
    AdditiveGuidance,
    EulerSolver,
    LinearGrid,
    Renorm,
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
        # ``framing_tokens`` keeps the base default of zero, so this bounds
        # the image's patch tokens alone; the opening ``<img>`` token belongs
        # to the prompt prefix (``processing.flow_prompt``).
        return self.config.max_image_seq_len

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
        """Assemble one denoising step's typed input and per-step conditioning.

        The network input is the current solver sample itself, so the image
        conditioning is rebuilt from ``latents`` on every step: each sample is
        unpatchified to ``[1, 3, H, W]`` pixels in the sample dtype, paired
        with its int64 vision patch grid ``[[H // patch, W // patch]]`` and
        with the resolution-dependent noise scale that ``prepare_latents``
        applied to its initial draw. The image tokens only read the
        trajectory's prefix in the K/V cache; the worker's denoising rows
        (``flow_rows``) do not write them into it.
        """
        images = []
        for latent, size in zip(latents["image"], sizes, strict=True):
            sample = latent.tensor
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
            step=step,
            positions=positions,
            sequence_lengths=sequence_lengths,
            attention=attention,
            images=tuple(images),
        )

    def __init__(self, config: Config, backbone: Transformer):
        # Output pixels per backbone token on each spatial axis. Samples are
        # pixel-space patches, so the patch size and downsampling both equal
        # this stride and a token row carries 3 * stride**2 values.
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
            # Network time rises from pure noise (0) to the clean image (1),
            # the direction ``flow.Velocity``'s ``1 - t`` denominator
            # assumes. A requested shift applies in the sigma domain; by
            # default the grid is unshifted.
            grid=LinearGrid(1.0, direction="ascending", shift_domain="sigma"),
        )
        # ``backbone`` is the same instance as ``Model.text.backbone``; its
        # weights load once, through the backbone mapping of
        # ``weights.checkpoint_mappings``. The input encoder has its own
        # generation weights, distinct from ``Model.vision_encoder``.
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
        # is active per checkpoint. ``use_pixel_head`` takes precedence over
        # the head depth. The decoder's two internal 2x shuffles account for
        # a factor of four of the stride; ``final_upscale`` supplies the rest.
        if config.flow.use_pixel_head:
            head: nn.Module = nn.Identity()
            decoder: nn.Module = flow.Decoder(
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
        """Predict each image's velocity for one homogeneous denoising batch.

        Returns ``{"image": outputs}`` with one entry per sample: a
        ``TensorOutput`` holding the FP32 velocity in the sample's canonical
        ``[tokens, 3 * stride**2]`` shape on the last pipeline stage, and
        ``None`` on every other stage, where the worker's
        ``DiffusionRunner.batch_forward`` receives the prediction by broadcast
        from the last stage. ``state``, ``constants`` and ``workspace`` are
        unused.

        Raises:
            ValueError: Among others, if the latents are not exactly the
                ``image`` modality, an image size holds no latent patch, or a
                sample, its declared token count or its conditioning pixels
                disagree with the sample's image size. The backbone and
                velocity head raise their own shape errors.
        """
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

        # Only the first pipeline stage builds input embeddings; later stages
        # pass None and the backbone receives hidden states from the previous
        # stage. A mesh without a ``pp`` axis selects the rank-local group.
        pipeline = self.mesh.get_group("pp" if "pp" in self.mesh.axes else ())
        hidden = None
        if pipeline.rank == 0:
            patch = self.config.vision.patch_size
            shapes = tuple(
                (size.height // patch, size.width // patch)
                for size in inputs.sizes
            )
            # [1, 3, H, W] pixels -> [(H/patch)*(W/patch), 3*patch*patch]
            # flattened CHW vision patches in raster order, the packed layout
            # vision.Encoder accepts. These are vision patches, not backbone
            # tokens: the encoder's dense reduction merges them into tokens.
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

            # Every token of an image receives its sample's timestep and, when
            # the checkpoint enables it, its noise scale normalized by the
            # configured maximum.
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

        # Every packed image token takes the flow experts; text tokens exist
        # only as the cached prefix the shared attention reads.
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

            # The conditioning pixels feed the network, while the velocity's
            # origin is always the solver sample ``latent.tensor``.
            # ``bind_inputs`` derives both from the same sample, but a
            # directly constructed input may supply different conditioning.
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
