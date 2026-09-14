"""Concrete MiniMax H3 numerical composition and fixed-profile mathematics."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import torch

from uniserve.distributed.mesh import DeviceMesh
from uniserve.distributed.parallel import ParallelConfig
from uniserve.model.batch import DecodeBatch, DiffusionBatch, EncodeBatch, TensorOutput
from uniserve.model.components import ComponentCall
from uniserve.model.decoder import DecodeKind, DecoderMixin
from uniserve.model.diffusion import DiffusionMixin
from uniserve.model.encoder import EncodeKind, EncoderMixin
from uniserve.model.limits import ModelLimits
from uniserve.model.media import DecodeWindow, VideoInfo, VideoSize
from uniserve.model.model import Model
from uniserve.model.tensors import TensorViews
from uniserve.model.video import VideoMixin
from uniserve.nn.diffusion.integrator import CleanSampleEulerSolver
from uniserve.nn.diffusion.schedule import DiffusionSchedule
from uniserve.nn.layer import LayerConfig
from uniserve.nn.parallel_pipeline import LayerPipeline
from uniserve_models.minimax_h3.config import H3Config
from uniserve_models.minimax_h3.decoding import AudioDecoder, VideoDecoder
from uniserve_models.minimax_h3.diffusion import Denoiser
from uniserve_models.minimax_h3.encoder import TextEncoder
from uniserve_models.minimax_h3.layout import MIN_H3_FRAMES
from uniserve_models.minimax_h3.output import VideoOutput
from uniserve_models.minimax_h3.weights import build_components

if TYPE_CHECKING:
    from uniserve.loading.component import CheckpointComponent

__all__ = ["MiniMaxH3Model"]


class MiniMaxH3Model(EncoderMixin, DiffusionMixin[VideoSize], DecoderMixin, VideoMixin, Model):
    """Compose text conditioning, four denoiser evaluations, and video/audio recovery.

    Logical components declare mathematical participation. Request progress,
    physical placement, media encoding, and output publication belong to the
    caller's execution infrastructure.
    """

    denoiser: Denoiser
    video_decoder: VideoDecoder
    audio_decoder: AudioDecoder

    config: H3Config

    @property
    def num_inference_steps(self) -> int:
        return len(self.config.diffusion.ladder)

    min_frames = MIN_H3_FRAMES
    text_alignment = 64
    minimum_cuda_capability = (9, 0)
    architecture = "MiniMaxH3Transformer3DModel"
    decoder_kinds: frozenset[DecodeKind] = frozenset({"video", "audio"})
    encoder_kinds: frozenset[EncodeKind] = frozenset({"text", "conditioning"})

    @classmethod
    def component_calls(cls, config: object) -> tuple[ComponentCall, ...]:
        """Declare actual numerical methods and their mathematical participation."""

        return (
            ComponentCall("text_encoder", "encode:text", groups=("tp",)),
            ComponentCall("denoiser", "encode:conditioning", stage="first", groups=("tp", "sp")),
            ComponentCall(
                "denoiser", "forward_diffusion", groups=("tp", "sp", "pp", "cp", "ulysses")
            ),
            ComponentCall("video_decoder", "decode:video"),
            ComponentCall("audio_decoder", "decode:audio"),
            ComponentCall("video_output", "postprocess_video"),
        )

    @classmethod
    def validate_parallel(cls, config: Any, parallel: Mapping[str, ParallelConfig]) -> None:
        """Validate head partitions and supported numerical parallel algorithms."""

        super().validate_parallel(config, parallel)
        if not parallel:
            raise ValueError("H3 requires at least one numerical component")
        for name, geometry in parallel.items():
            if name in {"video_decoder", "audio_decoder", "video_output"}:
                if geometry.world_size != 1:
                    raise ValueError(f"H3 {name} requires a local numerical computation")
            elif name == "text_encoder":
                if geometry.pipeline_parallel_size != 1 or geometry.sequence_parallel_size != 1:
                    raise ValueError("H3 text encoder supports direct tensor parallelism")
                if any(width % geometry.tensor_parallel_size for width in (64, 25600)):
                    raise ValueError("H3 encoder TP must divide query heads and MLP width")
            elif name == "denoiser":
                if geometry.pipeline_parallel_size > 50:
                    raise ValueError("H3 pipeline stages cannot exceed its 50 transformer layers")
                if geometry.sequence_parallel.kind not in {
                    "local",
                    "ulysses",
                    "allgather",
                    "ring",
                    "hybrid",
                    "attention2d",
                }:
                    raise ValueError("H3 sequence attention requires global sparse selection")
                tensor = geometry.tensor_parallel_size
                ulysses = dict(geometry.dimensions)["ulysses"]
                if 56 % (tensor * ulysses) or 5376 % tensor or 14336 % tensor:
                    raise ValueError(
                        "H3 TP × Ulysses must divide heads; TP must divide hidden and MLP widths"
                    )

    @property
    def diffusion_pipeline(self) -> LayerPipeline | None:
        """Expose the composed denoiser's mathematical feedback partition."""

        return self.denoiser.diffusion_pipeline

    def forward_diffusion(
        self,
        batch: DiffusionBatch[VideoSize],
        *,
        state: TensorViews,
        constants: TensorViews,
        scratch: TensorViews,
    ) -> TensorOutput:
        return self.denoiser.forward_diffusion(
            batch, state=state, constants=constants, scratch=scratch
        )

    def __init__(
        self,
        config: H3Config,
        *,
        parallel: Mapping[str, ParallelConfig],
        meshes: Mapping[str, DeviceMesh],
        layers: Mapping[str, LayerConfig],
        limits: ModelLimits,
    ) -> None:
        """Compose local numerical modules before the caller loads their parameters."""

        super().__init__(config)
        components, layout, self._checkpoints = build_components(
            config, parallel=parallel, meshes=meshes, layers=layers, limits=limits
        )
        resident_mesh = next(iter(meshes.values()), None)
        device = torch.device("cpu") if resident_mesh is None else resident_mesh.local_device
        denoiser_parallel = parallel.get("denoiser", ParallelConfig())
        mesh = meshes.get("denoiser")
        self.device = device
        self.layout = layout
        self.denoiser = Denoiser(
            config.denoiser,
            config.diffusion,
            transformer=components.transformer,
            conditioner=components.conditioner,
            parallel=denoiser_parallel,
            mesh=mesh,
            limits=ModelLimits(
                text_tokens=int(layout.packed.text_indices.numel()),
                video_frames=layout.frame_count,
            ),
        )
        self.text_encoder = TextEncoder(
            config.text_encoder,
            components.encoder,
            max_tokens=int(layout.packed.text_indices.numel()),
        )
        self.video_output = VideoOutput(
            width=config.video_decoder.width,
            height=config.video_decoder.height,
            frame_rate=config.video_decoder.fps,
            audio_rate=config.audio_decoder.sampling_rate,
            frame_limit=layout.frame_count,
        )
        self.video_decoder = VideoDecoder(config.video_decoder, components.video_vae)
        self.audio_decoder = AudioDecoder(config.audio_decoder, components.audio_vae)
        self.output_capacity = self.video_output.output_capacity
        self.decode_frame_capacity = self.video_output.decode_frame_capacity

    def checkpoint_components(self) -> tuple[CheckpointComponent, ...]:
        """Declare resident checkpoint namespaces and fixed-schedule precomputation."""

        return self._checkpoints

    @property
    def modalities(self) -> tuple[str, ...]:
        return self.denoiser.modalities

    @property
    def solver(self) -> CleanSampleEulerSolver:
        return self.denoiser.solver

    @property
    def prediction_dtype(self) -> torch.dtype:
        return self.denoiser.prediction_dtype

    def validate_schedule(self, schedule: DiffusionSchedule) -> None:
        self.denoiser.validate_schedule(schedule)

    def latent_shape(self, name: str, size: VideoSize) -> tuple[int, ...]:
        return self.denoiser.latent_shape(name, size)

    def noise_shape(self, name: str, size: VideoSize) -> tuple[int, ...]:
        return self.denoiser.noise_shape(name, size)

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
        self.denoiser.prepare_latents(
            batch, noise=noise, state=state, constants=constants, scratch=scratch
        )

    def video_info(self, frames: int) -> VideoInfo:
        return self.video_output.video_info(frames)

    def decode_windows(self, video: VideoInfo) -> tuple[DecodeWindow, ...]:
        return self.video_output.decode_windows(video)

    def postprocess_video(
        self,
        segments: tuple[torch.Tensor, ...],
        windows: tuple[DecodeWindow, ...],
        *,
        state: TensorViews,
        constants: TensorViews,
        scratch: TensorViews,
    ) -> TensorOutput:
        return self.video_output.postprocess_video(
            segments, windows, state=state, constants=constants, scratch=scratch
        )

    def encode(
        self, kind: EncodeKind, batch: EncodeBatch, *, constants: TensorViews, scratch: TensorViews
    ) -> TensorOutput:
        if kind == "text":
            return self.text_encoder.encode(kind, batch, constants=constants, scratch=scratch)
        if kind == "conditioning":
            return self.denoiser.encode(kind, batch, constants=constants, scratch=scratch)
        raise ValueError(f"unsupported H3 encoder kind {kind!r}")

    def decode(
        self, kind: DecodeKind, batch: DecodeBatch, *, constants: TensorViews, scratch: TensorViews
    ) -> TensorOutput:
        size = batch.sizes[0]
        if any(value != size for value in batch.sizes):
            raise ValueError("H3 decoding requires one numerical size per batch")
        if kind == "video":
            if not isinstance(size, VideoSize):
                raise ValueError("video decoding requires a VideoSize")
            return self.video_decoder.decode(
                batch.latents, size, batch.windows, constants=constants, scratch=scratch
            )
        if kind == "audio":
            if batch.windows or type(size) is not int or size < 1:
                raise ValueError("audio decoding requires a positive sample count without windows")
            return self.audio_decoder.decode(batch.latents, samples=size, scratch=scratch)
        raise ValueError(f"unsupported H3 decoder kind {kind!r}")
