"""Concrete fixed-profile MiniMax H3 model and resident request state."""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import replace
from functools import partial
from typing import TYPE_CHECKING, Any

import torch

from uniserve_worker.nn.mesh import EntryBindings

from ...execution.batch import (
    DecodeRange,
    MediaGeometry,
    MediaTrack,
    OpCode,
)
from ...execution.bounded_storage import BoundedTensorStorage
from ...execution.denoising import DenoisingStep
from ...nn.diffusion.schedule import DiffusionSchedule
from ...nn.parallel_attention import AttentionContextGeometry, AttentionContextWorkspace
from ..runtime import ResourceGeometry, TensorOutputLayout
from ..video import (
    MediaExecutionPlan,
    MediaPlanRepeat,
    MediaPlanStage,
    VideoModel,
    VideoOutputGeometry,
)
from .config import FASTH3_LADDER
from .layout import (
    PROFILE_AUDIO_RATE,
    PROFILE_FPS,
    H3ComputeInputs,
    H3Layout,
    H3Tensors,
    bind_request_tensors,
    entry_output_schema,
    media_tensor_schema,
    reconstruction_unit_frames,
    request_tensor_schema,
    scratch_tensor_schema,
    tensor_output_layout,
    warmup_geometries,
)
from .packing import audio_latent_frames
from .presentation import H3MediaGeometry, image_presentation_tags
from .transformer import MiniMaxH3Transformer, build_transformer_metadata
from .video_vae import H3VideoAssembler
from .weights import H3Components, build_h3_checkpoint

if TYPE_CHECKING:
    from ...execution.model_runner import ModelRunner
    from ...loader.component import ModelBuildContext, ModelConstruction

__all__ = ["MiniMaxH3Model"]


class MiniMaxH3Model(VideoModel[H3ComputeInputs, H3Tensors]):
    """Fixed T2VA plan executed by the shared media scheduler.

    Encode text, prepare conditioning and seeded latents, then schedule one
    prediction/solver step per ladder position. Video temporal units and audio
    decode independently; the output entry joins both tracks into one MP4.
    Components below bind those entries to caller-owned tensors. The shared
    scheduler evaluates these dependencies while retaining admission and cancellation.
    """

    denoiser: MiniMaxH3Transformer | None
    video_assembler: H3VideoAssembler | None

    media_plan = MediaExecutionPlan(
        (
            MediaPlanStage("encode", OpCode.ENCODER_TEXT, "text_encoder"),
            MediaPlanStage(
                "prepare",
                OpCode.DIFFUSION_PREPARE,
                "denoiser",
                dependencies=("encode",),
                input_from="encode",
            ),
            MediaPlanStage(
                "denoise",
                OpCode.DIFFUSION_STEP,
                "denoiser",
                dependencies=("prepare",),
                repeat=MediaPlanRepeat.FIXED,
                count=len(FASTH3_LADDER),
            ),
            MediaPlanStage(
                "video_decode",
                OpCode.DIFFUSION_DECODE,
                "video_decoder",
                dependencies=("denoise",),
                input_from="denoise",
                repeat=MediaPlanRepeat.VIDEO_UNITS,
            ),
            MediaPlanStage(
                "audio_decode",
                OpCode.DIFFUSION_DECODE,
                "audio_decoder",
                dependencies=("denoise",),
                input_from="denoise",
            ),
            MediaPlanStage(
                "video_append",
                OpCode.MEDIA_APPEND,
                "output",
                dependencies=("denoise",),
                input_from="video_decode",
                repeat=MediaPlanRepeat.VIDEO_UNITS,
            ),
            MediaPlanStage(
                "audio_append",
                OpCode.MEDIA_APPEND,
                "output",
                dependencies=("denoise",),
                input_from="audio_decode",
            ),
            MediaPlanStage(
                "finalize",
                OpCode.DIFFUSION_FINALIZE,
                "output",
                dependencies=("video_append", "audio_append"),
            ),
        )
    )
    architecture = "MiniMaxH3Transformer3DModel"
    serving_dtype = "bfloat16"
    resource_geometry = ResourceGeometry(kv=False)
    supported_work = media_plan.operations
    generation = None
    image_processor = None
    tensorized_mixed = False
    media_profile = "minimax_h3"
    ordered_collective_execution = True

    def denoising_signature(
        self, tensors: H3Tensors, metadata: H3ComputeInputs
    ) -> tuple[int, int, int]:
        """Identify one packed H3 shape independently of its request slot."""

        del tensors
        return metadata.layout.shape_key

    def bind_denoising_step(
        self, tensors: H3Tensors, metadata: H3ComputeInputs, step: int, schedule: DiffusionSchedule
    ) -> DenoisingStep:
        """Compose the learned denoiser with the shared solver and pipeline feedback."""

        if not 0 <= step < self.denoise_steps:
            raise ValueError("H3 denoise step is outside the checkpoint ladder")
        assert self.denoiser is not None and metadata.scratch is not None
        assert metadata.transformer_metadata is not None
        return DenoisingStep(
            partial(self.denoiser, tensors, metadata.scratch, metadata.transformer_metadata, step),
            (tensors.video_rows, tensors.audio_rows),
            self.denoiser.pipeline.feedback,
            schedule,
            step,
        )

    def __init__(
        self,
        bindings: EntryBindings,
        components: H3Components,
        layout: H3Layout,
        *,
        denoise_steps: int = len(FASTH3_LADDER),
        presentation_processor=None,
    ) -> None:
        """Bind H3 model components to runtime-owned state, scratch, and device products."""

        super().__init__()
        self.media_plan = MediaExecutionPlan(
            tuple(
                replace(stage, count=denoise_steps)
                if stage.operation is OpCode.DIFFUSION_STEP
                else stage
                for stage in type(self).media_plan.stages
            )
        )

        self.bindings: EntryBindings = bindings
        self.owns_media_output = bindings.owns("output")
        self.device = bindings.process_group.device
        self.layout = layout
        self.presentation_processor = presentation_processor
        self.denoiser = components.transformer
        self.conditioner = components.conditioner
        if (self.denoiser is not None and self.denoiser.pipeline.first) != (
            self.conditioner is not None
        ):
            raise ValueError("conditioning modules must belong to the denoiser input stage")
        self.text_encoder = components.encoder
        self.text_max_tokens = int(layout.packed.text_indices.numel())
        self.entry_outputs = entry_output_schema(layout)
        self.video_decoder = components.video_vae
        self.image_vae = components.image_vae
        self.audio_decoder = components.audio_vae
        self.video_assembler = H3VideoAssembler(self.device) if self.owns_media_output else None
        for name, module in (
            ("denoiser", self.denoiser),
            ("text_encoder", self.text_encoder),
            ("video_decoder", self.video_decoder),
            ("audio_decoder", self.audio_decoder),
        ):
            if bindings.owns(name) != (module is not None):
                raise ValueError(f"H3 {name} materialization disagrees with assigned membership")
        self.scratch_schema = media_tensor_schema(layout, bindings)
        self.context_geometry = None
        if self.denoiser is not None:
            denoiser_mesh = bindings.meshes["denoiser"]
            self.scratch_schema.update(
                scratch_tensor_schema(
                    layout,
                    denoiser_mesh,
                    block_params_shape=tuple(self.denoiser.modulation_plan.blocks.shape[1:]),
                    final_params_shape=(
                        tuple(self.denoiser.modulation_plan.final.shape[1:])
                        if self.denoiser.modulation_plan.final is not None
                        else (0,)
                    ),
                    attention_workspace_dtype=torch.bfloat16
                    if self.denoiser.attention_linear_precision == "bf16"
                    else torch.uint8,
                )
            )
            if layout.sp_size > layout.ulysses_size:
                self.context_geometry = AttentionContextGeometry(
                    group=denoiser_mesh.get_group(
                        "cp_row" if layout.sequence_kind == "attention2d" else "cp"
                    ),
                    rows=layout.attention_rows * layout.context_col_size,
                    heads=56 // (layout.tp_size * layout.ulysses_size),
                    mapped=layout.sequence_kind != "allgather",
                    head_dim=128,
                    dtype=torch.bfloat16,
                    block_size=64,
                )

        self.output_capacity = VideoOutputGeometry(
            frame_count=layout.frame_count,
            unit_frames=layout.reconstruction_unit_frames,
            width=layout.width,
            height=layout.height,
            frame_rate=PROFILE_FPS,
            audio_rate=PROFILE_AUDIO_RATE,
        )
        self.decode_frame_capacity = layout.frame_count

        self.resource_geometry = ResourceGeometry(
            kv=False, request_tensors=request_tensor_schema(layout)
        )

    @classmethod
    def build_checkpoint(
        cls, config: dict[str, Any], context: ModelBuildContext
    ) -> ModelConstruction:
        """Declare H3 component construction and numerical checkpoint mappings."""

        return build_h3_checkpoint(config, context)

    def media_geometry(self, media) -> MediaGeometry:
        """Resolve reference rows identically on every entry owner before product allocation."""

        if not media.references:
            return media.geometry
        if self.presentation_processor is None:
            raise ValueError("this H3 checkpoint does not support references")
        if len(media.references) != 1:
            raise ValueError("H3 requires one image reference")
        reference = media.references[0]
        if reference.kind != "image" or reference.task != "reference" or reference.pixels is None:
            raise ValueError("H3 requires an image reference task")
        shape = tuple(dim.extent for dim in reference.pixels.shape_bound.dims)[1:3]
        tags = image_presentation_tags(
            self.presentation_processor, shape, media.geometry.prompt_tokens
        )
        return H3MediaGeometry(
            media.geometry.frame_count,
            media.geometry.video_units,
            media.geometry.prompt_tokens,
            media.geometry.denoise_steps,
            shape,
            tags,
        )

    def conditioning_rows(self, geometry: MediaGeometry) -> int:
        """Count all presentation rows, including label and merged vision spans."""

        return len(getattr(geometry, "presentation_tags", ())) or geometry.prompt_tokens

    def execution_key(self, geometry: MediaGeometry) -> tuple:
        """Validate admitted bounds and describe equivalent packed metadata."""

        tags = getattr(geometry, "presentation_tags", ())
        page_rows = ((len(tags) or geometry.prompt_tokens) + 63) // 64 * 64
        audio_frames = audio_latent_frames(geometry.frame_count)
        if (
            page_rows > int(self.layout.packed.text_indices.numel())
            or geometry.frame_count > self.layout.frame_count
            or audio_frames > self.layout.packed.audio_frames
        ):
            raise ValueError("H3 media geometry exceeds the configured model capacity")
        units = reconstruction_unit_frames(geometry.frame_count)
        if geometry.video_units != len(units) or geometry.denoise_steps != self.denoise_steps:
            raise ValueError("the H3 worker received invalid computation bounds")
        key = (geometry.frame_count, page_rows, audio_frames, self.layout.height, self.layout.width)
        shape = getattr(geometry, "reference_shape", None)
        if shape is None:
            return key
        if min(shape) < 32 or max(shape) > 4096 or any(size % 32 for size in shape):
            raise ValueError("H3 reference dimensions must be multiples of 32 within 4096")
        capacity_shape = self.layout.packed.reference_shape
        if capacity_shape is None or shape[0] * shape[1] > capacity_shape[0] * capacity_shape[1]:
            raise ValueError("H3 reference geometry exceeds the configured model capacity")
        return (*key, shape, tags)

    def build_execution(
        self,
        geometry: MediaGeometry,
        storage: BoundedTensorStorage,
        context: AttentionContextWorkspace | None,
    ) -> H3ComputeInputs:
        """Build immutable packed metadata for a validated public geometry cache key."""

        frames, text_rows, audio_frames, height, width = self.execution_key(geometry)[:5]
        layout = H3Layout.build(
            self.bindings,
            frames=frames,
            text_rows=text_rows,
            audio_frames=audio_frames,
            height=height,
            width=width,
            sparsity=self.layout.sparsity,
            attention_backend=self.layout.attention_backend,
            attention=self.layout.attention,
            video_dtype=self.layout.video_dtype,
            reference_shape=getattr(geometry, "reference_shape", None),
            presentation_tags=torch.tensor(geometry.presentation_tags, dtype=torch.long)
            if getattr(geometry, "presentation_tags", ())
            else None,
        )
        return H3ComputeInputs.bind(
            self.bindings,
            layout,
            storage,
            context,
            build_transformer_metadata(layout, self.device) if self.denoiser is not None else None,
            self.device,
        )

    def request_tensors(
        self, storage: BoundedTensorStorage, geometry: MediaGeometry, metadata: H3ComputeInputs
    ) -> H3Tensors:
        """Borrow mathematical inputs from the request's publicly owned tensor slot."""

        if self.execution_key(geometry) != metadata.layout.shape_key:
            raise ValueError("H3 tensor views disagree with their computation metadata")
        return bind_request_tensors(storage, metadata.layout)

    @torch.inference_mode()
    def prepare_tensors(
        self,
        slot: H3Tensors,
        execution: H3ComputeInputs,
        encoded: torch.Tensor | None,
        text_rows: int,
        *,
        presentation_tags: torch.Tensor | None = None,
        reference_image: torch.Tensor | None = None,
    ) -> None:
        """Install refined presentation and optional decoded image in a bound layout."""

        if presentation_tags is None and (
            self.denoiser is None or not self.denoiser.pipeline.first
        ):
            # Only the input stage consumes encoder products; later stages still
            # bind identical row tags and rotary geometry from shape-only planning.
            presentation_tags = execution.layout.packed.presentation_tags
        execution.prepare_tensors(
            slot,
            encoded,
            text_rows,
            self.denoiser,
            presentation_tags=presentation_tags,
            reference_image=reference_image,
            video_vae=self.image_vae,
        )

    @torch.inference_mode()
    def initialize_tensors(
        self, slot: H3Tensors, seed: int
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
        """Initialize owned rows with the checkpoint's physical-layout-independent noise."""

        return H3ComputeInputs.initialize_tensors(slot, seed)

    def output_layout(
        self,
        entry: str,
        output_index: int,
        media: MediaGeometry | None,
        decode: DecodeRange | None,
    ) -> TensorOutputLayout | None:
        """Describe unique logical modality rows and temporal decoder results."""

        if media is None:
            raise ValueError("H3 tensor results require media geometry")
        frames, text_rows, audio_frames, height, width = self.execution_key(media)[:5]
        return tensor_output_layout(
            self.bindings,
            entry,
            output_index,
            decode,
            frames=frames,
            text_rows=text_rows,
            prompt_tokens=len(getattr(media, "presentation_tags", ())) or media.prompt_tokens,
            reference_shape=getattr(media, "reference_shape", None),
            presentation_tags=torch.tensor(media.presentation_tags, dtype=torch.long)
            if getattr(media, "presentation_tags", ())
            else None,
            audio_frames=audio_frames,
            height=height,
            width=width,
        )

    def decoder_input(
        self,
        execution: H3ComputeInputs,
        latents: torch.Tensor,
        track: MediaTrack,
        cursor: int,
        max_units: int,
    ) -> torch.Tensor:
        """Pack the selected temporal window or stereo latent rows for its decoder."""

        if track is MediaTrack.VIDEO:
            if self.video_decoder is None:
                raise RuntimeError("video decode was routed to a rank without the decoder")
            rank = self.bindings.entries["video_decoder"].ranks.index(
                self.bindings.process_group.rank
            )
            return self.video_decoder.prepare_input(execution, latents, cursor, max_units, rank)
        if self.audio_decoder is None:
            raise RuntimeError("audio decode was routed to a rank without the decoder")
        return self.audio_decoder.prepare_input(execution, latents, cursor, max_units)

    def decoder_output(
        self, execution: H3ComputeInputs, value: torch.Tensor, track: MediaTrack
    ) -> torch.Tensor:
        """Expose decoded segments or the exact duration's interleaved PCM samples."""

        if track is MediaTrack.VIDEO:
            return value.unsqueeze(0)
        if self.audio_decoder is None:
            raise RuntimeError("audio decode was routed to a rank without the decoder")
        return self.audio_decoder.logical_output(execution, value)

    def assemble_video(
        self,
        slot: H3Tensors,
        execution: H3ComputeInputs,
        segments: torch.Tensor,
        start_unit: int,
        unit_count: int,
    ) -> torch.Tensor:
        """Blend overlapping decoded windows and apply the checkpoint pixel transform."""

        if self.video_assembler is None:
            raise RuntimeError("video assembly was routed to a rank without the output component")
        return self.video_assembler.assemble(slot, execution, segments, start_unit, unit_count)

    def output_geometry(self, geometry: MediaGeometry) -> VideoOutputGeometry:
        """Describe the exact raster and sample timing required by the mathematics."""

        self.execution_key(geometry)
        return VideoOutputGeometry(
            frame_count=geometry.frame_count,
            unit_frames=reconstruction_unit_frames(geometry.frame_count),
            width=self.layout.width,
            height=self.layout.height,
            frame_rate=PROFILE_FPS,
            audio_rate=PROFILE_AUDIO_RATE,
        )

    def bind_execution(self, runner: ModelRunner) -> None:
        """Assemble numerical owners with each component's actual collective mesh."""

        from ...execution.runners.denoise import DenoiseRunner

        def groups(entry: str, axes: tuple[str, ...] = ("tp", "sp", "pp")):
            mesh = self.bindings.meshes[entry]
            return tuple(mesh.get_group(axis) for axis in axes if mesh.size(axis) > 1)

        if self.video_decoder is not None:
            runner.bind_module(
                "video_decoder",
                self.video_decoder,
                inputs=(
                    torch.zeros(
                        (
                            1,
                            24,
                            7,
                            self.layout.packed.latent_height,
                            self.layout.packed.latent_width,
                        ),
                        dtype=torch.float32,
                        device=self.device,
                    ),
                ),
            )
        if self.text_encoder is not None:
            runner.bind_module(
                "text_encoder", self.text_encoder.numerical_entry, groups=groups("text_encoder")
            )
        if self.conditioner is not None:
            runner.bind_module(
                "conditioner", self.conditioner, groups=groups("denoiser", ("tp", "sp"))
            )
        if self.audio_decoder is not None:
            runner.bind_module("audio_decoder", self.audio_decoder, groups=groups("audio_decoder"))
        if self.denoiser is not None:
            runner.denoise = DenoiseRunner(
                self.bind_denoising_step,
                self.denoising_signature,
                device=self.device,
                backend=runner.graph_backend(shared_pool=True),
                groups=groups("denoiser"),
                capacity=runner.worker_config.max_request_pool_size,
            )

    @torch.inference_mode()
    def warmup_execution(
        self, runner: ModelRunner, storage: tuple[BoundedTensorStorage, ...]
    ) -> None:
        """Prepare representative denoising, pixel transform, and audio geometry."""

        scratch = runner.scratch
        assert scratch is not None
        if not storage:
            raise RuntimeError("H3 warmup requires declared request tensor storage")
        if self.denoiser is not None:
            if runner.schedule is None or runner.denoise is None:
                raise RuntimeError("denoiser warmup requires its execution owner and schedule")
            prepared: set[Hashable] = set()
            for geometry in warmup_geometries(self.layout, self.denoise_steps):
                key = self.execution_key(geometry)
                if key in prepared:
                    continue
                prepared.add(key)
                execution = runner.prepare_geometry(
                    key, lambda: self.build_execution(geometry, scratch, runner.context_workspace)
                )
                views = execution.prepare_warmup_slots(storage, self.denoiser)
                runner.denoise.warmup(views[0], execution, runner.schedule)
        if self.bindings.owns("output"):
            assert self.video_assembler is not None
            self.video_assembler.warmup(self.layout)
        if self.audio_decoder is not None:
            audio_latents = self.audio_decoder.warmup_input(self.layout.packed.audio_frames)
            runner.modules["audio_decoder"].warmup(audio_latents)
