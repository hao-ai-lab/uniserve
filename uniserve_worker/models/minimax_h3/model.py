"""Concrete MiniMax H3 numerical composition and fixed-profile mathematics."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields
from typing import TYPE_CHECKING, Any, cast

import torch

from uniserve_worker.modeling.geometry import (
    DecodeWindow,
    MediaShape,
    Shape,
    TensorOutputLayout,
    TextShape,
    VideoShape,
)

from ...modeling.batch import DecodeBatch, DiffusionBatch, TensorOutput
from ...modeling.components import Call, CallSpec, ComponentSpec
from ...modeling.context import BuildContext
from ...modeling.decoder import DecodeKind, DecoderMixin
from ...modeling.diffusion import DiffusionMixin
from ...modeling.encoder import EncodeKind, EncoderMixin
from ...modeling.model import Model
from ...modeling.resources import TensorAlias, TensorNeeds, TensorSchema
from ...modeling.tensors import TensorViews
from ...modeling.video import VideoMixin
from ...nn.diffusion.schedule import ScheduleDirection, ScheduleShiftDomain
from ...nn.diffusion.spec import DiffusionSpec, ModalitySpec, ScheduleRule
from ...nn.parallel import ParallelConfig
from ...nn.parallel_pipeline import LayerPipeline
from .config import FASTH3_LADDER, FASTH3_SHIFTS, FASTH3_TIME_SCALE, H3TransformerConfig
from .encoder import H3TextEncoderConfig
from .layout import (
    MIN_H3_FRAMES,
    PROFILE_AUDIO_RATE,
    PROFILE_FPS,
    PROFILE_HEIGHT,
    PROFILE_WIDTH,
    H3Layout,
    compute_specs,
    reconstruction_unit_frames,
    state_specs,
    tensor_output_layout,
)
from .packing import (
    audio_latent_frames,
    build_packed_layout,
    patchify_video,
    unpatchify_video_into,
    video_latent_frames,
)
from .transformer import MODALITIES, MiniMaxH3Transformer, build_transformer_metadata
from .weights import build_components

if TYPE_CHECKING:
    from ...loader.component import CheckpointComponent
    from ...nn.video_attention import VideoAttention
    from .audio_vae import MiniMaxH3AudioVAE
    from .video_vae import MiniMaxH3VideoVAE

__all__ = ["MiniMaxH3Model"]


class MiniMaxH3Model(EncoderMixin, DiffusionMixin, DecoderMixin, VideoMixin, Model):
    """Compose text conditioning, four denoiser evaluations, and video/audio recovery.

    Logical components declare mathematical participation. Request progress,
    physical placement, media encoding, and output publication belong to the
    caller's execution infrastructure.
    """

    denoiser: MiniMaxH3Transformer | None
    video_decoder: MiniMaxH3VideoVAE | None
    audio_decoder: MiniMaxH3AudioVAE | None

    num_inference_steps = len(FASTH3_LADDER)
    min_frames = MIN_H3_FRAMES
    text_alignment = 64
    architecture = "MiniMaxH3Transformer3DModel"
    generation = None
    image_processor = None
    media_profile = "minimax_h3"
    decoder_kinds: frozenset[DecodeKind] = frozenset({"video", "audio"})
    encoder_kinds: frozenset[EncodeKind] = frozenset({"text", "conditioning"})

    @classmethod
    def components(cls, config: Any) -> tuple[ComponentSpec, ...]:
        return (
            ComponentSpec("text_encoder", (CallSpec(Call.ENCODE_TEXT, groups=("tp",)),)),
            ComponentSpec(
                "denoiser",
                (
                    CallSpec(Call.ENCODE_CONDITIONING, stage="first", groups=("tp", "sp")),
                    CallSpec(Call.DIFFUSION, groups=("tp", "sp", "pp", "cp", "ulysses")),
                ),
            ),
            ComponentSpec("video_decoder", (CallSpec(Call.DECODE_VIDEO),)),
            ComponentSpec("audio_decoder", (CallSpec(Call.DECODE_AUDIO),)),
            ComponentSpec("output", (CallSpec(Call.POSTPROCESS_VIDEO),)),
        )

    @classmethod
    def validate_parallel(cls, config: Any, parallel: Mapping[str, ParallelConfig]) -> None:
        """Validate head partitions and supported numerical parallel algorithms."""

        super().validate_parallel(config, parallel)
        if not parallel:
            raise ValueError("H3 requires at least one numerical component")
        for name, geometry in parallel.items():
            if name in {"video_decoder", "audio_decoder", "output"}:
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

    def tensor_specs(self, call: Call, shape: Shape) -> TensorNeeds:
        """Declare immutable diffusion metadata and native reconstruction tensors."""

        if call in {Call.ENCODE_TEXT, Call.ENCODE_CONDITIONING}:
            if not isinstance(shape, TextShape) or shape.rows != 1:
                raise ValueError("H3 text encoding requires one numerical token sequence")
            if not 1 <= shape.tokens <= self.text_max_tokens:
                raise ValueError("H3 text encoding length lies outside its numerical token limits")
            width = (
                H3TextEncoderConfig().hidden_size
                if call is Call.ENCODE_TEXT
                else H3TransformerConfig().hidden_size
            )
            return TensorNeeds(
                outputs={
                    "conditioning": TensorSchema(
                        (1, shape.tokens, width),
                        torch.bfloat16,
                        variable_axes=(1,),
                    )
                }
            )
        if not isinstance(shape, MediaShape):
            raise ValueError("H3 media computation requires raster and temporal geometry")
        if (shape.height, shape.width) != (PROFILE_HEIGHT, PROFILE_WIDTH):
            raise ValueError("H3 reconstruction requires its checkpoint raster geometry")
        reconstruction_unit_frames(shape.frames)
        if call is Call.DIFFUSION:
            layout = self._diffusion_layout(shape)
            values = self._diffusion_metadata(layout)
            constants = {
                name: TensorSchema(
                    tuple(value.shape),
                    value.dtype,
                    domain="host" if name in {"video_indices", "audio_indices"} else "device",
                )
                for name, value in values.items()
            }
            return TensorNeeds(
                constants=constants,
                outputs={
                    "video": TensorSchema(
                        (int(layout.packed.video_indices.numel()), 96),
                        torch.float32,
                        variable_axes=(0,),
                    ),
                    "audio": TensorSchema(
                        (int(layout.packed.audio_indices.numel()), 32),
                        torch.float32,
                        variable_axes=(0,),
                    ),
                },
                state=state_specs(layout),
                scratch=compute_specs(
                    layout,
                    self.denoiser_mesh,
                    block_params_shape=(
                        (
                            len(self.denoiser.pipeline.layers),
                            2,
                            MODALITIES * 6 * self.denoiser.config.hidden_size,
                        )
                        if self.denoiser is not None
                        else (0,)
                    ),
                    final_params_shape=(
                        (2, 2 * self.denoiser.config.hidden_size)
                        if self.denoiser is not None and self.denoiser.pipeline.last
                        else (0,)
                    ),
                    attention=(
                        cast(
                            "VideoAttention",
                            next(iter(self.denoiser.transformer_blocks.values())).attn,
                        )
                        if self.denoiser is not None
                        else None
                    ),
                ),
            )
        if call is Call.DECODE_VIDEO:
            return TensorNeeds(
                constants={
                    "video_raster_order": TensorSchema(
                        (video_latent_frames(shape.frames) * 24 * 42,), torch.int64
                    )
                },
                scratch={
                    "video_input": TensorSchema((1, 24, 7, 48, 84), torch.float32),
                    "reconstruction_rows": TensorSchema((7 * 24 * 42, 96), torch.float32),
                },
                outputs={
                    "video": TensorSchema(
                        (shape.unit_count or 1, 1, 3, 25, shape.height, shape.width),
                        torch.float16,
                        variable_axes=(0,),
                    )
                },
            )
        if call is Call.DECODE_AUDIO:
            return TensorNeeds(
                scratch={
                    "audio_latents": TensorSchema(
                        (2, 32, audio_latent_frames(shape.frames)), torch.float32
                    )
                },
                outputs={
                    "audio": TensorSchema(
                        (round(shape.frames * PROFILE_AUDIO_RATE / PROFILE_FPS), 2),
                        torch.int16,
                        variable_axes=(0,),
                    )
                },
            )
        if call is Call.POSTPROCESS_VIDEO:
            return TensorNeeds(
                constants={
                    name: TensorSchema((1, 3, 1, 1, 1), torch.float32)
                    for name in ("pixel_mean", "pixel_std")
                },
                state={
                    "video_overlap": TensorSchema(
                        (1, 3, 5, shape.height, shape.width), torch.float16
                    )
                },
                scratch={
                    "rgb_frames": TensorSchema(
                        (shape.frames, shape.height, shape.width, 3), torch.uint8
                    )
                },
                outputs={
                    "video": TensorSchema(
                        (shape.frames, shape.height, shape.width, 3),
                        torch.uint8,
                        variable_axes=(0,),
                        alias=TensorAlias("scratch", "rgb_frames"),
                    )
                },
            )
        raise ValueError(f"H3 tensor requirements are not declared for {call.value}")

    def _diffusion_layout(self, shape: MediaShape) -> H3Layout:
        """Resolve one legal numerical shape within the model's logical partition."""

        audio_frames = audio_latent_frames(shape.frames)
        text_rows = ((shape.prompt_tokens + 63) // 64) * 64
        if (
            (shape.height, shape.width) != (PROFILE_HEIGHT, PROFILE_WIDTH)
            or shape.frames > self.layout.frame_count
            or not 0 < text_rows <= self.text_max_tokens
            or shape.audio_frames not in {0, audio_frames}
        ):
            raise ValueError("H3 metadata requires supported raster, prompt, and audio geometry")
        return H3Layout.build(
            self.denoiser_parallel,
            self.denoiser_mesh,
            frames=shape.frames,
            text_rows=text_rows,
            audio_frames=audio_frames,
            postprocess=self.layout.postprocess,
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
        if self.denoiser is not None:
            metadata = build_transformer_metadata(layout, torch.device("cpu"))
            values.update({field.name: getattr(metadata, field.name) for field in fields(metadata)})
        return values

    @torch.inference_mode()
    def prepare_metadata(self, call: Call, shape: Shape, *, out: TensorViews) -> None:
        """Fill borrowed constants without retaining their backing or execution owner."""

        if isinstance(shape, TextShape):
            return super().prepare_metadata(call, shape, out=out)
        if call is Call.DIFFUSION:
            values = self._diffusion_metadata(self._diffusion_layout(shape))
        elif call is Call.DECODE_VIDEO:
            self.tensor_specs(call, shape)
            packed = build_packed_layout(
                text_rows=64,
                num_frames=shape.frames,
                audio_frames=audio_latent_frames(shape.frames),
            )
            values = {"video_raster_order": torch.argsort(packed.video_raster_indices)}
        elif call is Call.POSTPROCESS_VIDEO:
            values = {
                "pixel_mean": torch.tensor(
                    (0.485, 0.456, 0.406), dtype=torch.float32, device="cpu"
                ).view(1, 3, 1, 1, 1),
                "pixel_std": torch.tensor(
                    (0.229, 0.224, 0.225), dtype=torch.float32, device="cpu"
                ).view(1, 3, 1, 1, 1),
            }
        else:
            return super().prepare_metadata(call, shape, out=out)
        schemas = self.tensor_specs(call, shape).constants
        if out.keys() != schemas.keys():
            raise ValueError("H3 metadata views must cover the declared constants")
        for name, schema in schemas.items():
            target = out[name]
            if tuple(target.shape) != schema.shape or target.dtype != schema.dtype:
                raise ValueError(f"H3 metadata view {name!r} has an incompatible representation")
            if name in {"video_indices", "audio_indices"} and target.device.type != "cpu":
                raise ValueError("H3 native noise indices require CPU representation")
        for name, source in values.items():
            out[name].copy_(source)

    @property
    def diffusion_pipeline(self) -> LayerPipeline | None:
        """Expose denoiser partitioning for the caller's numerical feedback."""

        return None if self.denoiser is None else self.denoiser.pipeline

    def forward_diffusion(
        self,
        batch: DiffusionBatch,
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
        if self.denoiser is None:
            raise ValueError("H3 diffusion requires a resident denoiser")
        predictions = self.denoiser(batch, state=state, constants=constants, scratch=scratch)
        return TensorOutput(
            {
                name: (None if predictions is None else predictions[index],)
                for index, name in enumerate(batch.latents)
            }
        )

    def __init__(self, config: dict[str, Any], context: BuildContext) -> None:
        """Compose local numerical modules before the caller loads their parameters."""

        super().__init__(config)
        components, layout, self._checkpoints = build_components(config, context)
        schedule = context.schedule
        assert schedule is not None
        device = schedule.sigmas[0].device
        parallel = context.parallel.get("denoiser", ParallelConfig())
        mesh = context.meshes.get("denoiser")
        self.device = device
        self.denoiser_parallel = parallel
        self.denoiser_mesh = mesh
        self.layout = layout
        self.denoiser = components.transformer
        self.conditioner = components.conditioner
        if (self.denoiser is not None and self.denoiser.pipeline.first) != (
            self.conditioner is not None
        ):
            raise ValueError("conditioning modules must belong to the denoiser input stage")
        self.text_encoder = components.encoder
        self.text_max_tokens = int(layout.packed.text_indices.numel())
        maximum = MediaShape(
            PROFILE_HEIGHT,
            PROFILE_WIDTH,
            frames=layout.frame_count,
            prompt_tokens=self.text_max_tokens,
            unit_count=layout.video_reconstruction_units,
        )
        self.output_shapes = {
            "text_encoder": (Call.ENCODE_TEXT, TextShape(self.text_max_tokens)),
            "denoiser": (Call.DIFFUSION, maximum),
            "video_decoder": (Call.DECODE_VIDEO, maximum),
            "audio_decoder": (Call.DECODE_AUDIO, maximum),
        }
        self.video_decoder = components.video_vae
        self.audio_decoder = components.audio_vae
        self.output_capacity = VideoShape(
            frame_count=layout.frame_count,
            unit_frames=layout.reconstruction_unit_frames,
            width=PROFILE_WIDTH,
            height=PROFILE_HEIGHT,
            frame_rate=PROFILE_FPS,
            audio_rate=PROFILE_AUDIO_RATE,
        )
        self.decode_frame_capacity = layout.frame_count

    def checkpoint_components(self) -> tuple[CheckpointComponent, ...]:
        """Declare resident checkpoint namespaces and fixed-schedule precomputation."""

        return self._checkpoints

    def diffusion_spec(self, shape: MediaShape, steps: int) -> DiffusionSpec:
        """Declare full native normal draws before sequence sharding and packing."""

        if steps != len(FASTH3_LADDER):
            raise ValueError("H3 requires its four-evaluation trained ladder")
        if (shape.height, shape.width) != (PROFILE_HEIGHT, PROFILE_WIDTH):
            raise ValueError("H3 diffusion requires its checkpoint raster geometry")
        reconstruction_unit_frames(shape.frames)
        video_frames = video_latent_frames(shape.frames)
        audio_frames = audio_latent_frames(shape.frames)
        modalities = []
        for name, latent, noise, shift in (
            (
                "video",
                (video_frames * 24 * 42, 96),
                (1, 24, video_frames, 48, 84),
                FASTH3_SHIFTS[0],
            ),
            ("audio", (2 * audio_frames, 32), (2 * audio_frames, 32), FASTH3_SHIFTS[1]),
        ):
            modalities.append(
                ModalitySpec(
                    name=name,
                    latent_shape=latent,
                    noise_shape=noise,
                    schedule=ScheduleRule(
                        ScheduleDirection.ASCENDING,
                        ScheduleShiftDomain.SIGMA,
                        shift,
                        timestep="one_minus_sigma",
                        ladder=FASTH3_LADDER,
                        scale=FASTH3_TIME_SCALE,
                    ),
                    prediction="velocity",
                    prediction_dtype=torch.float32,
                )
            )
        return DiffusionSpec(
            tuple(modalities), steps, None, 1, "clean_sample_euler", "cpu", "identity"
        )

    @torch.inference_mode()
    def prepare_latents(
        self,
        batch: DiffusionBatch,
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
        for row, shape in enumerate(batch.shapes):
            spec = self.diffusion_spec(shape, self.num_inference_steps)
            for modality in spec.modalities:
                source, target = noise[modality.name], state[modality.name]
                indices = constants[f"{modality.name}_indices"]
                if (
                    tuple(source.shape) != (batch.row_count, *modality.noise_shape)
                    or tuple(target.shape)
                    != (batch.row_count, indices.numel(), modality.latent_shape[-1])
                    or source.device.type != "cpu"
                    or target.device.type != "cpu"
                    or source.dtype != torch.float32
                    or target.dtype != torch.float32
                    or indices.device.type != "cpu"
                    or indices.dtype != torch.int64
                ):
                    raise ValueError("H3 initialization views disagree with native CPU geometry")
            if self.denoiser is not None:
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
                self.denoiser.rope.forward_into(
                    positions,
                    state["rotary_cosine"][row],
                    state["rotary_sine"][row],
                    scratch["rotary_frequencies"],
                )
            video = patchify_video(noise["video"][row])[0]
            for name, source in (("video", video), ("audio", noise["audio"][row])):
                torch.index_select(source, 0, constants[f"{name}_indices"], out=state[name][row])

    def output_layout(
        self,
        entry: str,
        output_index: int,
        *,
        frames: int | None,
        units: int | None,
        prompt_tokens: int,
    ) -> TensorOutputLayout | None:
        """Describe unique logical modality rows and temporal decoder results."""

        if frames is None:
            raise ValueError("H3 tensor results require media geometry")
        self.output_geometry(frames)
        layout = self.layout
        if entry == "denoiser":
            layout = self._diffusion_layout(
                MediaShape(
                    PROFILE_HEIGHT, PROFILE_WIDTH, frames=frames, prompt_tokens=prompt_tokens
                )
            )
        return tensor_output_layout(
            layout,
            entry,
            output_index,
            frames=frames,
            unit_count=units,
            prompt_tokens=prompt_tokens,
        )

    @torch.inference_mode()
    def decode(
        self,
        kind: DecodeKind,
        batch: DecodeBatch,
        *,
        constants: TensorViews,
        scratch: TensorViews,
    ) -> TensorOutput:
        """Reconstruct assigned windows or exact-duration interleaved stereo.

        Input rows use the denoiser's complete packed modality order. Numerical
        packing precedes the ordinary native decoder call; the caller may bind
        that public layer to a fixed capture. Physical rank selection and media
        publication do not enter this computation.
        """

        if kind not in self.decoder_kinds:
            raise ValueError(f"unsupported H3 decoder kind {kind!r}")
        geometries = {(shape.height, shape.width, shape.frames) for shape in batch.shapes}
        if len(geometries) != 1:
            raise ValueError("H3 decoding requires one media geometry per numerical batch")
        shape = batch.shapes[0]
        video_shape = self.output_geometry(shape.frames)
        if (shape.height, shape.width) != (video_shape.height, video_shape.width):
            raise ValueError("H3 decoding requires its checkpoint raster geometry")
        if kind == "video":
            if self.video_decoder is None:
                raise ValueError("this partition does not participate in video decoding")
            legal = self.decode_windows(video_shape)
            if len(batch.windows) != len(batch.latents) or any(
                window not in legal for window in batch.windows
            ):
                raise ValueError("video decoding requires explicitly assigned legal windows")
            rows = video_latent_frames(shape.frames) * 24 * 42
            if any(tuple(value.shape) != (rows, 96) for value in batch.latents):
                raise ValueError("video decoder requires complete final latent rows")
            values = []
            for latents, window in zip(batch.latents, batch.windows, strict=True):
                start = window.latent_start * 24 * 42
                indices = constants["video_raster_order"][start : start + 7 * 24 * 42]
                torch.index_select(latents, 0, indices, out=scratch["reconstruction_rows"])
                unpatchify_video_into(
                    scratch["reconstruction_rows"],
                    scratch["video_input"],
                    frames=7,
                    height=48,
                    width=84,
                )
                decoded = self.video_decoder(scratch["video_input"])
                value = decoded.unsqueeze(0)
                # A native execution binding can reuse one captured output.
                # Distinct rows of this numerical result must remain independent.
                values.append(value.clone() if len(batch.latents) > 1 else value)
        else:
            if self.audio_decoder is None:
                raise ValueError("this partition does not participate in audio decoding")
            if batch.windows:
                raise ValueError("audio decoding requires complete stereo without video windows")
            frames = audio_latent_frames(shape.frames)
            if any(tuple(value.shape) != (2 * frames, 32) for value in batch.latents):
                raise ValueError("audio decoder requires one complete stereo latent")
            samples = round(shape.frames * PROFILE_AUDIO_RATE / PROFILE_FPS)
            values = []
            for latents in batch.latents:
                scratch["audio_latents"].copy_(latents.view(2, frames, 32).permute(0, 2, 1))
                decoded = self.audio_decoder(scratch["audio_latents"])
                if decoded.shape[0] < samples:
                    raise RuntimeError("audio decoder returned less than the video duration")
                value = decoded[:samples]
                values.append(value.clone() if len(batch.latents) > 1 else value)
        return TensorOutput(
            {kind: tuple(values)},
            {kind: tuple(TensorOutputLayout(tuple(value.shape)) for value in values)},
        )

    def output_geometry(self, frames: int) -> VideoShape:
        """Describe the exact raster and sample timing required by the mathematics."""

        if frames > self.layout.frame_count:
            raise ValueError("H3 output exceeds configured frame capacity")
        return VideoShape(
            frame_count=frames,
            unit_frames=reconstruction_unit_frames(frames),
            width=PROFILE_WIDTH,
            height=PROFILE_HEIGHT,
            frame_rate=PROFILE_FPS,
            audio_rate=PROFILE_AUDIO_RATE,
        )

    def decode_windows(self, shape: VideoShape) -> tuple[DecodeWindow, ...]:
        """Map each H3 unit to seven latent frames and its 17/22 RGB frames."""

        if shape != self.output_geometry(shape.frame_count):
            raise ValueError("H3 decode windows require the configured raster and frame rates")
        return tuple(
            DecodeWindow(
                latent_start=unit * 5,
                latent_stop=unit * 5 + 7,
                frame_start=unit * 17,
                frame_stop=unit * 17 + frames,
                body_frames=17,
                overlap_frames=5,
                padding_frames=3,
                crop=(3, 0),
                final=unit + 1 == len(shape.unit_frames),
            )
            for unit, frames in enumerate(shape.unit_frames)
        )
