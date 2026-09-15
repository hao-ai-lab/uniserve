"""H3's native latent packing, fixed schedules and sparse denoising computation."""  # noqa: E501

from __future__ import annotations

import math
from collections.abc import Mapping

import torch

from uniserve.diffusion import CleanSampleEulerSolver, Schedule
from uniserve.model import Denoiser as BaseDenoiser
from uniserve.nn import RotaryEmbedding
from uniserve.nn.attention import vsa
from uniserve.tensors import BufferConfig, OutputLayout, TensorOutput

from .conditioning import Conditioner
from .config import DiffusionConfig, TransformerConfig
from .inputs import AttentionInput, DenoiserInput, DenoiserSize
from .packing import (
    AUDIO_TAG,
    Packing,
    audio_latent_frames,
    build_packing,
    patchify_video,
    video_latent_frames,
)
from .transformer import Transformer


def schedules(
    config: DiffusionConfig, *, device: torch.device | str
) -> Mapping[str, Schedule]:
    """Materialize sigma before subtracting it from one in FP32.

    The analytical coordinates retain the unrounded trained ladder. Its four
    evaluations exclude the clean endpoint, which the solver still consumes.
    """
    result = {}
    for name, shift in (
        ("video", config.video_shift),
        ("audio", config.audio_shift),
    ):
        sigmas = tuple(
            shift
            * (value / config.time_scale)
            / (1 + (shift - 1) * (value / config.time_scale))
            for value in (*config.ladder, 0)
        )
        sigma = torch.tensor(sigmas, dtype=torch.float32, device=device)
        result[name] = Schedule(
            1.0 - sigma, sigma, tuple(1.0 - value for value in sigmas)
        )
    return result


class Denoiser(BaseDenoiser[DenoiserInput, DenoiserSize]):
    """Predict video/audio velocities from explicit latent and text tensors.

    Native random draws and their contiguous sample destinations use CPU FP32;
    callers transfer prepared samples into device views before evaluation.
    Text features are the output of ``conditioner.encode``. Packed padding is
    numerical attention input and never represents a second computation batch.
    """

    def __init__(self, config: TransformerConfig, diffusion: DiffusionConfig):
        super().__init__(
            modalities=("video", "audio"),
            prediction_dtype=torch.float32,
            solver=CleanSampleEulerSolver(),
        )
        self.config, self.diffusion = config, diffusion
        self.transformer = Transformer(config)
        self.conditioner = Conditioner(config)
        self.rotary = RotaryEmbedding(
            2 * config.rope_frequency_dim, theta=config.rope_theta
        )

    def _sequence_group(self):
        layer = next(iter(self.transformer.layers.values()))
        distribution = layer.attention.projection.projections[
            "q"
        ].input_distribution
        return distribution.mesh.get_group(distribution.shard_axes(0))

    def _packing(self, size: DenoiserSize) -> Packing:
        return build_packing(
            num_text_tokens=size.num_text_tokens,
            num_frames=size.num_frames,
            token_multiple=64 * math.lcm(4, self._sequence_group().size),
        )

    def _token_slice(self, packing: Packing) -> slice:
        group = self._sequence_group()
        width = packing.padded_tokens // group.size
        return slice(group.rank * width, (group.rank + 1) * width)

    def latent_shape(
        self, modality: str, size: DenoiserSize
    ) -> tuple[int, ...]:
        if modality == "video":
            # The 48x84 latent raster yields 24x42 tokens per frame at
            # patch 2x2.
            return video_latent_frames(
                size.num_frames
            ) * 24 * 42, self.config.video_channels * 4
        if modality == "audio":
            return 2 * audio_latent_frames(
                size.num_frames
            ), self.config.audio_channels
        raise ValueError(f"unknown H3 latent modality {modality!r}")

    def noise_shape(self, modality: str, size: DenoiserSize) -> tuple[int, ...]:
        if modality == "video":
            # Native draws keep the [C, T, 48, 84] latent layout before
            # patching.
            return (
                1,
                self.config.video_channels,
                video_latent_frames(size.num_frames),
                48,
                84,
            )
        return self.latent_shape(modality, size)

    def make_schedules(
        self, steps: int, *, shift: float | None, device
    ) -> Mapping[str, Schedule]:
        if (
            type(steps) is not int
            or steps != len(self.diffusion.ladder)
            or shift is not None
        ):
            raise ValueError(
                "H3 requires four evaluations with its trained modality shifts"
            )
        return schedules(self.diffusion, device=device)

    def output_layout(self, size: DenoiserSize) -> Mapping[str, OutputLayout]:
        packing = self._packing(size)
        interval = self._token_slice(packing)
        result = {}
        for name, indices in (
            ("video", packing.video_indices),
            ("audio", packing.audio_indices),
        ):
            start = int(torch.searchsorted(indices, interval.start))
            stop = int(torch.searchsorted(indices, interval.stop))
            shape = self.latent_shape(name, size)
            result[name] = OutputLayout(
                shape,
                self.prediction_dtype,
                local_slice=(slice(start, stop), slice(0, shape[1])),
                variable_axes=(0,),
            )
        return result

    def state_buffers(self, size: DenoiserSize) -> Mapping[str, BufferConfig]:
        """Describe one sample's local canonical storage on the caller's device."""  # noqa: E501
        return {
            name: BufferConfig(
                tuple(part.stop - part.start for part in layout.local_slice),
                layout.dtype,
            )
            for name, layout in self.output_layout(size).items()
        }

    def _metadata(self, size: DenoiserSize) -> dict[str, torch.Tensor]:
        packing = self._packing(size)
        interval = self._token_slice(packing)
        values = {}
        for name, indices in (
            ("text", packing.text_indices),
            ("video", packing.video_indices),
            ("audio", packing.audio_indices),
        ):
            selected = (indices >= interval.start) & (indices < interval.stop)
            values[f"local_{name}_indices"] = indices[selected] - interval.start
            if name == "text":
                values["global_text_indices"] = indices[selected]
            else:
                raster = (
                    packing.video_raster_indices
                    if name == "video"
                    else torch.arange(indices.numel(), device="cpu")
                )
                values[f"{name}_indices"] = raster[selected]

        # Modulation rows are addressed by token tag, with audio offset past
        # the video/text rows of the packed per-layer table.
        tags = packing.token_tags[interval]
        values["modulation_indices"] = (tags == AUDIO_TAG).long() * 3 + tags

        values.update(
            tile_valid_sizes=packing.tile_valid_sizes,
            prefix_key_indices=torch.arange(
                packing.prefix_tiles, dtype=torch.int32, device="cpu"
            ),
            dense_key_indices=torch.arange(
                packing.prefix_tiles + packing.video_tiles,
                dtype=torch.int32,
                device="cpu",
            ),
            prefix_count=torch.tensor(
                packing.prefix_tiles, dtype=torch.int32, device="cpu"
            ),
        )

        cosine, sine = self.rotary(
            packing.position_ids,
            dtype=torch.float32,
            sequence_length=packing.padded_tokens,
        )
        values["cos"], values["sin"] = cosine.flatten(1), sine.flatten(1)
        return values

    def constant_buffers(
        self, size: DenoiserSize
    ) -> Mapping[str, BufferConfig]:
        return {
            name: BufferConfig(
                tuple(value.shape),
                value.dtype,
                host=name in {"video_indices", "audio_indices"},
            )
            for name, value in self._metadata(size).items()
        }

    @torch.inference_mode()
    def prepare_constants(
        self, size: DenoiserSize, *, out: Mapping[str, torch.Tensor]
    ) -> None:
        values = self._metadata(size)
        if values.keys() != out.keys():
            raise ValueError(
                "H3 constant views must cover every declared numerical field"
            )
        for name, value in values.items():
            target = out[name]
            if target.shape != value.shape or target.dtype != value.dtype:
                raise ValueError(
                    f"H3 constant {name!r} has incompatible shape or dtype"
                )
            if (
                name in {"video_indices", "audio_indices"}
                and target.device.type != "cpu"
            ):
                raise ValueError(
                    "H3 native draw indices require CPU representation"
                )

        for name, value in values.items():
            out[name].copy_(value)

    def workspace_buffers(
        self, size: DenoiserSize
    ) -> Mapping[str, BufferConfig]:
        packing = self._packing(size)
        interval = self._token_slice(packing)
        attention = next(iter(self.transformer.layers.values())).attention
        distribution = attention.projection.projections["q"].output_distribution
        context = distribution.mesh.get_group(distribution.shard_axes(0))
        return {
            "hidden": BufferConfig(
                (interval.stop - interval.start, self.config.hidden_size),
                torch.bfloat16,
            ),
            # Adjacent layers coexist while completed residual chunks feed the
            # next projection. Two numerical scratch sets keep their key pools
            # and fine-attention products independent until final consumption.
            **{
                f"attention.{slot}.{name}": config
                for slot in range(min(2, len(self.transformer.layers)))
                for name, config in attention.workspace_buffers(
                    packing.padded_tokens,
                    packing.padded_tokens // context.size,
                    dtype=torch.bfloat16,
                ).items()
            },
        }

    @torch.inference_mode()
    def prepare_latents(
        self, sizes, *, noise, state, constants, workspace
    ) -> None:
        if len(sizes) != 1 or tuple(noise) != self.modalities:
            raise ValueError(
                "H3 preparation requires one ordered video/audio sample"
            )
        size = sizes[0]
        for name in self.modalities:
            source, target, indices = (
                noise[name],
                state[name],
                constants[f"{name}_indices"],
            )
            if (
                tuple(source.shape) != (1, *self.noise_shape(name, size))
                or tuple(target.shape)
                != (1, indices.numel(), self.latent_shape(name, size)[1])
                or source.device.type != "cpu"
                or target.device.type != "cpu"
                or indices.device.type != "cpu"
                or source.dtype != torch.float32
                or target.dtype != torch.float32
                or indices.dtype != torch.int64
            ):
                raise ValueError(
                    "H3 initialization requires complete native CPU FP32 "
                    "draws and local sample views"
                )
        video = patchify_video(noise["video"][0])[0]
        for name, source in (("video", video), ("audio", noise["audio"][0])):
            torch.index_select(
                source, 0, constants[f"{name}_indices"], out=state[name][0]
            )

    @torch.inference_mode()
    def forward(self, inputs: DenoiserInput, *, state, constants, workspace):
        if inputs.batch_size != 1 or not 0 <= inputs.step_index < len(
            self.diffusion.ladder
        ):
            raise ValueError(
                "H3 denoising requires one sample on its four-evaluation ladder"
            )
        size = inputs.sizes[0]
        packing = self._packing(size)
        interval = self._token_slice(packing)
        attention = AttentionInput(
            packing,
            interval,
            self._sequence_group(),
            vsa.Input(
                packing.padded_tokens,
                packing.prefix_tiles,
                packing.video_tiles,
                packing.prefix_tiles + packing.video_tiles,
                constants["tile_valid_sizes"],
                constants["prefix_key_indices"],
                constants["dense_key_indices"],
                constants["prefix_count"],
            ),
            constants["local_text_indices"],
            constants["local_video_indices"],
            constants["local_audio_indices"],
        )
        hidden = workspace["hidden"]
        if (
            hidden.shape
            != (interval.stop - interval.start, self.config.hidden_size)
            or hidden.dtype != torch.bfloat16
        ):
            raise ValueError(
                "H3 hidden workspace must match the local BF16 token shard"
            )

        pipeline = self.mesh.get_group("pp" if "pp" in self.mesh.axes else ())
        if pipeline.rank == 0:
            text = inputs.text_features[0]
            if text.shape != (size.num_text_tokens, self.config.hidden_size):
                raise ValueError(
                    "H3 text features must be refined tokens "
                    "with the declared width"
                )

            # Scatter text and projected latents into the packed token rows;
            # padding rows stay zero.
            hidden.zero_()
            hidden.index_copy_(
                0,
                attention.local_text_indices,
                text.index_select(0, constants["global_text_indices"]).to(
                    hidden.dtype
                ),
            )
            for name, projection, indices in (
                (
                    "video",
                    self.transformer.video_input,
                    attention.local_video_indices,
                ),
                (
                    "audio",
                    self.transformer.audio_input,
                    attention.local_audio_indices,
                ),
            ):
                sample = inputs.latents[name][0].tensor
                if (
                    sample.shape
                    != (indices.numel(), self.latent_shape(name, size)[1])
                    or sample.dtype != torch.float32
                ):
                    raise ValueError(
                        "H3 latent input must supply its local canonical "
                        "FP32 sample"
                    )
                hidden.index_copy_(
                    0,
                    indices,
                    projection(sample, output_dtype=torch.float32).to(
                        hidden.dtype
                    ),
                )

        predictions = self.transformer(
            hidden,
            attention,
            step_index=inputs.step_index,
            constants=constants,
            workspace=workspace,
        )
        if not predictions:
            return dict.fromkeys(self.modalities, (None,))

        layouts = self.output_layout(size)
        return {
            name: (TensorOutput(value, layouts[name]),)
            for name, value in zip(self.modalities, predictions, strict=True)
        }
