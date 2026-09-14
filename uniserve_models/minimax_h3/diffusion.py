"""H3 denoising, native noise packing and conditioning composition."""

from __future__ import annotations

from dataclasses import fields
from typing import cast

import torch
from torch import nn

from uniserve.distributed.mesh import DeviceMesh
from uniserve.distributed.parallel import ParallelConfig
from uniserve.model.batch import DiffusionBatch, TensorOutput
from uniserve.model.diffusion import DiffusionMixin
from uniserve.model.encoder import EncoderMixin
from uniserve.model.limits import ModelLimits
from uniserve.model.media import VideoSize
from uniserve.model.tensors import TensorViews
from uniserve.nn.diffusion.integrator import CleanSampleEulerSolver
from uniserve.nn.diffusion.schedule import DiffusionSchedule
from uniserve.nn.parallel_pipeline import LayerPipeline
from uniserve.nn.video_attention import VideoAttention
from uniserve.tensors import BufferConfig, OutputLayout, TensorRegion
from uniserve_models.minimax_h3.config import H3DiffusionConfig, H3TransformerConfig
from uniserve_models.minimax_h3.layout import (
    H3Layout,
    state_buffers,
    validate_frames,
    workspace_buffers,
)
from uniserve_models.minimax_h3.packing import (
    audio_latent_frames,
    patchify_video,
    video_latent_frames,
)
from uniserve_models.minimax_h3.transformer import (
    MODALITIES,
    MiniMaxH3Transformer,
    build_transformer_metadata,
)


class Denoiser(EncoderMixin, DiffusionMixin[VideoSize], nn.Module):
    """Compose first-stage conditioning and the partitioned H3 transformer.

    Nonresident components retain numerical configuration for global output
    sizing. They neither allocate weights nor supply executable tensor resources.
    """

    encoder_kinds = frozenset({"conditioning"})
    generation = None
    modalities: tuple[str, ...] = ("video", "audio")
    solver: CleanSampleEulerSolver

    @property
    def prediction_dtype(self) -> torch.dtype:
        return torch.float32

    def __init__(
        self,
        config: H3TransformerConfig,
        diffusion: H3DiffusionConfig,
        *,
        transformer: MiniMaxH3Transformer | None,
        conditioner: nn.Module | None,
        parallel: ParallelConfig,
        mesh: DeviceMesh | None,
        limits: ModelLimits,
    ) -> None:
        super().__init__()
        self.solver = CleanSampleEulerSolver()
        self.config = config
        self.diffusion_config = diffusion
        self.transformer = transformer
        self.conditioner = conditioner
        self.parallel = parallel
        self.mesh = mesh
        self.limits = limits
        if (transformer is not None and transformer.pipeline.first) != (conditioner is not None):
            raise ValueError("conditioning must belong to the denoiser input stage")

    @property
    def num_inference_steps(self) -> int:
        return len(self.diffusion_config.ladder)

    def output_layout(self, size: VideoSize) -> dict[str, OutputLayout]:
        layout = self.layout(size)
        outputs = {}
        for name, indices, count, width in (
            ("video", layout.packed.video_indices, layout.local_video_rows, 96),
            ("audio", layout.packed.audio_indices, layout.local_audio_rows, 32),
        ):
            region = None
            if count:
                start = int(torch.searchsorted(indices, layout.local_start))
                region = TensorRegion((start, 0), (count, width))
            outputs[name] = OutputLayout(
                (int(indices.numel()), width), torch.float32, region=region, variable_axes=(0,)
            )
        return outputs

    def state_buffers(self, size: VideoSize) -> dict[str, BufferConfig]:
        if self.transformer is None:
            raise ValueError("nonresident denoiser has no executable state")
        return state_buffers(self.layout(size))

    def workspace_buffers(self, size: VideoSize) -> dict[str, BufferConfig]:
        model = self.transformer
        if model is None:
            raise ValueError("nonresident denoiser has no executable workspace")
        return workspace_buffers(
            self.layout(size),
            block_params_shape=(
                len(model.pipeline.layers),
                2,
                MODALITIES * 6 * self.config.hidden_size,
            ),
            final_params_shape=(2, 2 * self.config.hidden_size) if model.pipeline.last else (0,),
            attention=cast(VideoAttention, next(iter(model.transformer_blocks.values())).attn),
        )

    def constant_buffers(self, size: VideoSize) -> dict[str, BufferConfig]:
        return {
            name: BufferConfig(
                tuple(value.shape), value.dtype, host=name in {"video_indices", "audio_indices"}
            )
            for name, value in self._diffusion_metadata(self.layout(size)).items()
        }

    @torch.inference_mode()
    def prepare_constants(self, size: VideoSize, *, out: TensorViews) -> None:
        values = self._diffusion_metadata(self.layout(size))
        if values.keys() != out.keys():
            raise ValueError("H3 constants must cover their numerical fields")
        for name, value in values.items():
            target = out[name]
            if target.shape != value.shape or target.dtype != value.dtype:
                raise ValueError(f"H3 constant {name!r} has incompatible shape or dtype")
            if name in {"video_indices", "audio_indices"} and target.device.type != "cpu":
                raise ValueError("H3 native noise indices require CPU representation")
        for name, value in values.items():
            out[name].copy_(value)

    def layout(self, shape: VideoSize) -> H3Layout:
        """Resolve one legal numerical shape within the model's logical partition."""

        audio_frames = audio_latent_frames(shape.frames)
        text_rows = ((shape.prompt_tokens + 63) // 64) * 64
        if shape.frames > self.limits.video_frames or not 0 < text_rows <= self.limits.text_tokens:
            raise ValueError("H3 diffusion exceeds its numerical frame or text limits")
        return H3Layout.build(
            self.parallel,
            self.mesh,
            frames=shape.frames,
            text_rows=text_rows,
            audio_frames=audio_frames,
        )

    def _diffusion_metadata(self, layout: H3Layout) -> dict[str, torch.Tensor]:
        """Derive immutable indices and base positions for a packed page geometry."""

        packed = layout.packed
        values = {
            "video_indices": layout.local_video_raster_indices,
            "audio_indices": layout.local_audio_raster_indices,
            "tile_valid_sizes": packed.tile_valid_sizes,
            "prefix_key_indices": torch.arange(
                packed.prefix_tiles, dtype=torch.int32, device="cpu"
            ),
            "dense_key_indices": torch.arange(
                packed.prefix_tiles + packed.video_tiles, dtype=torch.int32, device="cpu"
            ),
            "prefix_count": torch.tensor(packed.prefix_tiles, dtype=torch.int32, device="cpu"),
        }
        if self.transformer is not None:
            metadata = build_transformer_metadata(layout, torch.device("cpu"))
            values.update({field.name: getattr(metadata, field.name) for field in fields(metadata)})
        return values

    @property
    def diffusion_pipeline(self) -> LayerPipeline | None:
        """Expose denoiser partitioning for the caller's numerical feedback."""

        return None if self.transformer is None else self.transformer.pipeline

    def forward_diffusion(
        self,
        batch: DiffusionBatch[VideoSize],
        *,
        state: TensorViews,
        constants: TensorViews,
        scratch: TensorViews,
    ) -> TensorOutput:
        """Predict ordered video/audio velocities without updating latent state."""

        if batch.row_count != 1 or tuple(batch.latents) != ("video", "audio"):
            raise ValueError("H3 diffusion requires one video/audio sequence")
        step = batch.ladder_index
        if step is None or not 0 <= step < self.num_inference_steps:
            raise ValueError("H3 denoise step is outside the four-evaluation ladder")
        if self.transformer is None:
            raise ValueError("H3 diffusion requires a resident denoiser")
        predictions = self.transformer(batch, state=state, constants=constants, scratch=scratch)
        return TensorOutput(
            {
                name: (None if predictions is None else predictions[index],)
                for index, name in enumerate(batch.latents)
            }
        )

    def validate_schedule(self, schedule: DiffusionSchedule) -> None:
        super().validate_schedule(schedule)
        if schedule.sigmas[0].numel() != self.num_inference_steps + 1:
            raise ValueError("H3 requires its four-evaluation trained ladder")

    def latent_shape(self, name: str, size: VideoSize) -> tuple[int, ...]:
        """Global patch rows before sequence sharding."""

        validate_frames(size.frames)
        if name == "video":
            return (video_latent_frames(size.frames) * 24 * 42, 96)
        if name == "audio":
            return (2 * audio_latent_frames(size.frames), 32)
        raise ValueError(f"unknown H3 latent modality {name!r}")

    def noise_shape(self, name: str, size: VideoSize) -> tuple[int, ...]:
        """Full native draws preserve video NCTHW and packed stereo audio order."""

        validate_frames(size.frames)
        if name == "video":
            return (1, 24, video_latent_frames(size.frames), 48, 84)
        if name == "audio":
            return (2 * audio_latent_frames(size.frames), 32)
        raise ValueError(f"unknown H3 noise modality {name!r}")

    @torch.inference_mode()
    def prepare_latents(
        self,
        batch: DiffusionBatch[VideoSize],
        *,
        noise: TensorViews,
        state: TensorViews,
        constants: TensorViews,
        scratch: TensorViews,
    ) -> None:
        """Pack full CPU noise and select the declared logical sequence shard.

        Noise and state views have a leading logical batch row. Video normal
        draws retain native NCTHW order; audio follows in channel-major row
        order. Modality outputs are CPU shard views which the caller delivers
        to device state. Prompt validity and RoPE are written directly into
        the supplied device state, using borrowed preparation scratch.
        """

        if tuple(batch.latents) != ("video", "audio"):
            raise ValueError("H3 preparation requires ordered video and audio modalities")
        for row, shape in enumerate(batch.sizes):
            for name in self.modalities:
                source, target = noise[name], state[name]
                indices = constants[f"{name}_indices"]
                if (
                    tuple(source.shape) != (batch.row_count, *self.noise_shape(name, shape))
                    or tuple(target.shape)
                    != (batch.row_count, indices.numel(), self.latent_shape(name, shape)[-1])
                    or source.device.type != "cpu"
                    or target.device.type != "cpu"
                    or source.dtype != torch.float32
                    or target.dtype != torch.float32
                    or indices.device.type != "cpu"
                    or indices.dtype != torch.int64
                ):
                    raise ValueError("H3 initialization views disagree with native CPU geometry")
            if self.transformer is not None:
                # Valid prompt rows and temporal offsets vary within one cached
                # page layout. They belong to this request's numerical state.
                text_rows = ((shape.prompt_tokens + 63) // 64) * 64
                if shape.prompt_tokens < 1 or text_rows != state["text_condition"].shape[-2]:
                    raise ValueError("H3 prompt length disagrees with its prepared text pages")
                valid = state["tile_valid_sizes"][row]
                valid.copy_(constants["tile_valid_sizes"])
                valid[: text_rows // 64].zero_()
                full, remaining = divmod(shape.prompt_tokens, 64)
                valid[:full].fill_(64)
                if remaining:
                    valid[full].fill_(remaining)
                positions = scratch["rotary_positions"]
                positions.copy_(constants["positions"])
                positions[text_rows:, 0].add_(shape.prompt_tokens - text_rows)
                self.transformer.rope.forward_into(
                    positions,
                    state["rotary_cosine"][row],
                    state["rotary_sine"][row],
                    scratch["rotary_frequencies"],
                )
            video = patchify_video(noise["video"][row])[0]
            for name, source in (("video", video), ("audio", noise["audio"][row])):
                torch.index_select(source, 0, constants[f"{name}_indices"], out=state[name][row])
