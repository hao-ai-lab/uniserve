"""Concrete fixed-profile MiniMax H3 model and resident request state."""

from __future__ import annotations

from collections.abc import Hashable, Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from uniserve_worker.nn.mesh import EntryBindings

from ...execution.batch import (
    DecodeRange,
    DeviceDim,
    DType,
    MediaGeometry,
    MediaTrack,
    OpCode,
    ShapeBound,
    StaticDim,
    TensorSpec,
)
from ...execution.bounded_storage import BoundedTensorStorage, TensorSchema
from ...media.codec import video_segment_rgb
from ...nn.diffusion.schedule import DiffusionSchedule
from ...nn.parallel_attention import AttentionContextGeometry, AttentionContextWorkspace
from ...transfer.layout import TensorRegion
from ..runtime import ModuleWarmup, ResourceGeometry, TensorOutputLayout
from ..video import VideoModel, VideoOutputGeometry
from .encoder import H3TextEncoderConfig
from .packing import audio_latent_frames, patchify_video, unpatchify_video_into
from .state import (
    MIN_H3_FRAMES,
    PROFILE_AUDIO_RATE,
    PROFILE_FPS,
    PROFILE_HEIGHT,
    PROFILE_WIDTH,
    H3Layout,
    H3MediaScratch,
    H3Scratch,
    H3Tensors,
    bind_compute_tensors,
    bind_request_tensors,
    media_tensor_schema,
    reconstruction_unit_frames,
    request_tensor_schema,
    scratch_tensor_schema,
)
from .transformer import H3TransformerMetadata, MiniMaxH3Transformer, build_transformer_metadata
from .video_vae_decoder import MiniMaxH3VideoDecoder
from .weights import H3Components, build_h3_checkpoint

if TYPE_CHECKING:
    from ...loader.component import ModelBuildContext, ModelConstruction

__all__ = ["MiniMaxH3Model"]


@dataclass(frozen=True, slots=True)
class H3ComputeInputs:
    """Binds an H3 layout to scratch storage, transformer indices, and sparse-attention page metadata."""

    layout: H3Layout
    scratch: H3Scratch | None
    media: H3MediaScratch
    transformer_metadata: H3TransformerMetadata | None
    base_tile_valid_sizes: torch.Tensor
    prompt_prefix_indices: torch.Tensor
    prompt_dense_indices: torch.Tensor
    prompt_prefix_counts: torch.Tensor


class MiniMaxH3Model(VideoModel[H3ComputeInputs, H3Tensors]):
    """Numerical H3 entries over explicitly bound tensors and metadata."""

    denoiser: MiniMaxH3Transformer | None
    video_pixel_mean: torch.Tensor | None
    video_pixel_std: torch.Tensor | None

    architecture = "MiniMaxH3Transformer3DModel"
    serving_dtype = "bfloat16"
    resource_geometry = ResourceGeometry(kv=False)
    supported_work = frozenset(
        {
            OpCode.ENCODER_TEXT,
            OpCode.DIFFUSION_PREPARE,
            OpCode.DIFFUSION_STEP,
            OpCode.DIFFUSION_DECODE,
            OpCode.DIFFUSION_FINALIZE,
            OpCode.MEDIA_APPEND,
        }
    )
    generation = None
    image_processor = None
    tensorized_mixed = False
    media_profile = "minimax_h3"
    supports_weight_updates = False
    ordered_collective_execution = True

    def __init__(
        self,
        bindings: EntryBindings,
        components: H3Components,
        layout: H3Layout,
    ) -> None:
        """Bind H3 model components to runtime-owned state, scratch, and device products."""

        super().__init__()

        self.warmup_inputs = self._warmup_inputs
        self.bindings = bindings
        self.owns_media_output = bindings.owns("output")
        self.device = bindings.process_group.device
        self.layout = layout
        self.denoiser = components.transformer
        self.conditioner = components.conditioner
        if (self.denoiser is not None and self.denoiser.pipeline.first) != (
            self.conditioner is not None
        ):
            raise ValueError("conditioning modules must belong to the denoiser input stage")
        self.text_encoder = components.encoder
        self.text_max_tokens = int(layout.packed.text_indices.numel())
        self.entry_outputs = {
            "text_encoder": (
                TensorSpec(
                    "conditioning",
                    DType.BF16,
                    ShapeBound(
                        (
                            StaticDim(1),
                            DeviceDim(self.text_max_tokens),
                            StaticDim(H3TextEncoderConfig().hidden_size),
                        )
                    ),
                ),
            ),
        }
        self.entry_outputs.update(
            {
                "denoiser": (
                    TensorSpec(
                        "video_latents",
                        DType.F32,
                        ShapeBound(
                            (DeviceDim(int(layout.packed.video_indices.numel())), StaticDim(96))
                        ),
                    ),
                    TensorSpec(
                        "audio_latents",
                        DType.F32,
                        ShapeBound(
                            (DeviceDim(int(layout.packed.audio_indices.numel())), StaticDim(32))
                        ),
                    ),
                ),
                "video_decoder": (
                    TensorSpec(
                        "video_segments",
                        DType.F16,
                        ShapeBound(
                            (
                                DeviceDim(layout.video_reconstruction_units),
                                StaticDim(1),
                                StaticDim(3),
                                StaticDim(25),
                                StaticDim(PROFILE_HEIGHT),
                                StaticDim(PROFILE_WIDTH),
                            )
                        ),
                    ),
                ),
                "audio_decoder": (
                    TensorSpec(
                        "audio_samples",
                        DType.I16,
                        ShapeBound(
                            (
                                DeviceDim(
                                    round(layout.frame_count * PROFILE_AUDIO_RATE / PROFILE_FPS)
                                ),
                                StaticDim(2),
                            )
                        ),
                    ),
                ),
            }
        )
        self.video_decoder = components.video_vae
        self.capture_inputs = (
            {"video_decoder": (TensorSchema((1, 24, 7, 48, 84), torch.float32),)}
            if self.video_decoder is not None
            else {}
        )
        self.audio_decoder = components.audio_vae
        for name, module in (
            ("denoiser", self.denoiser),
            ("text_encoder", self.text_encoder),
            ("video_decoder", self.video_decoder),
            ("audio_decoder", self.audio_decoder),
        ):
            if bindings.owns(name) != (module is not None):
                raise ValueError(f"H3 {name} materialization disagrees with assigned membership")
        for name, values in (
            ("video_pixel_mean", (0.485, 0.456, 0.406)),
            ("video_pixel_std", (0.229, 0.224, 0.225)),
        ):
            self.register_buffer(
                name,
                torch.tensor(values, dtype=torch.float32, device=self.device).view(1, 3, 1, 1, 1)
                if bindings.owns("output")
                else None,
                persistent=False,
            )
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
            width=PROFILE_WIDTH,
            height=PROFILE_HEIGHT,
            frame_rate=PROFILE_FPS,
            audio_rate=PROFILE_AUDIO_RATE,
        )
        self.decode_frame_capacity = layout.frame_count

        self.resource_geometry = ResourceGeometry(
            kv=False, request_tensors=request_tensor_schema(layout)
        )

    def local_product_storage_bytes(self, *, max_unresolved_ops: int) -> int:
        """Bound retained decode batches and persistent conditioning/latent results."""

        if max_unresolved_ops < 1:
            raise ValueError("H3 product storage requires a positive execution window")
        rank = self.bindings.process_group.rank
        total = 0
        for entry, outputs in self.entry_outputs.items():
            stores_output = rank in self.bindings.output_ranks(entry)
            imports_output = (
                entry == "text_encoder"
                and rank in self.bindings.entries["denoiser"].ranks
            )
            if not stores_output and not imports_output:
                continue
            for output in outputs:
                size = output.max_bytes
                if entry == "video_decoder":
                    # Each unresolved decode can publish another complete rank
                    # group of segments. Their total is bounded by the video.
                    live_units = min(
                        self.layout.video_reconstruction_units,
                        max_unresolved_ops
                        * len(self.bindings.entries[entry].ranks)
                        * self.bindings.entries[entry].units_per_rank,
                    )
                    size = (
                        size
                        * live_units
                        // self.layout.video_reconstruction_units
                    )
                total += ((size + 255) // 256) * 256
        return total

    def _build_page_execution(
        self,
        layout: H3Layout,
        storage: BoundedTensorStorage,
        context: AttentionContextWorkspace | None,
    ) -> H3ComputeInputs:
        """Build shape-specific sparse-attention metadata and a bounded scratch view."""

        text_tiles = int(layout.packed.text_indices.numel()) // 64
        if text_tiles < 1:
            raise ValueError("H3 page execution requires at least one text page")
        prefix = torch.arange(layout.packed.prefix_tiles, dtype=torch.int32)
        dense = torch.arange(
            layout.packed.prefix_tiles + layout.packed.video_tiles,
            dtype=torch.int32,
        )
        scratch, media = bind_compute_tensors(
            storage, layout, self.bindings.meshes.get("denoiser"), context
        )
        return H3ComputeInputs(
            layout=layout,
            scratch=scratch,
            media=media,
            transformer_metadata=build_transformer_metadata(layout, self.device)
            if self.denoiser is not None
            else None,
            base_tile_valid_sizes=layout.packed.tile_valid_sizes.to(self.device),
            prompt_prefix_indices=prefix.to(self.device),
            prompt_dense_indices=dense.to(self.device),
            prompt_prefix_counts=torch.tensor(
                layout.packed.prefix_tiles,
                dtype=torch.int32,
                device=self.device,
            ),
        )

    @classmethod
    def build_checkpoint(
        cls, config: dict[str, Any], context: ModelBuildContext
    ) -> ModelConstruction:
        """Declare H3 component construction and numerical checkpoint mappings."""

        return build_h3_checkpoint(config, context)

    def execution_key(self, geometry: MediaGeometry) -> tuple[int, int, int]:
        """Validate admitted bounds and describe equivalent packed metadata."""

        page_rows = ((geometry.prompt_tokens + 63) // 64) * 64
        audio_frames = audio_latent_frames(geometry.frame_count)
        if (
            page_rows > int(self.layout.packed.text_indices.numel())
            or geometry.frame_count > self.layout.frame_count
            or audio_frames > self.layout.packed.audio_frames
        ):
            raise ValueError("H3 media geometry exceeds the configured model capacity")
        units = reconstruction_unit_frames(geometry.frame_count)
        if geometry.video_units != len(units) or geometry.denoise_steps != 4:
            raise ValueError("the H3 worker received invalid computation bounds")
        return geometry.frame_count, page_rows, audio_frames

    def build_execution(
        self,
        geometry: MediaGeometry,
        storage: BoundedTensorStorage,
        context: AttentionContextWorkspace | None,
    ) -> H3ComputeInputs:
        """Build immutable packed metadata for a validated public geometry cache key."""

        frames, text_rows, audio_frames = self.execution_key(geometry)
        layout = H3Layout.build(
            self.bindings,
            frames=frames,
            text_rows=text_rows,
            audio_frames=audio_frames,
        )
        return self._build_page_execution(layout, storage, context)

    def request_tensors(
        self, storage: BoundedTensorStorage, geometry: MediaGeometry, metadata: H3ComputeInputs
    ) -> H3Tensors:
        """Borrow mathematical inputs from the request's publicly owned tensor slot."""

        if self.execution_key(geometry) != metadata.layout.shape_key:
            raise ValueError("H3 tensor views disagree with their computation metadata")
        return bind_request_tensors(storage, metadata.layout)

    def _prepare_tile_metadata(
        self,
        execution: H3ComputeInputs,
        slot: H3Tensors,
        text_rows: int,
    ) -> None:
        """Copy shape-bound sparse tile validity and prefix indices into one state slot."""

        packed = execution.layout.packed
        valid = slot.tile_valid_sizes
        text_tiles = int(packed.text_indices.numel()) // 64
        full_tiles, remaining = divmod(int(text_rows), 64)
        valid.copy_(execution.base_tile_valid_sizes)
        valid[:text_tiles].zero_()
        if full_tiles:
            valid[:full_tiles].fill_(64)
        if remaining:
            valid[full_tiles].fill_(remaining)
        slot.prefix_key_indices.copy_(execution.prompt_prefix_indices)
        slot.dense_key_indices.copy_(execution.prompt_dense_indices)
        slot.prefix_count.copy_(execution.prompt_prefix_counts)

    def _prepare_rotary(
        self,
        execution: H3ComputeInputs,
        slot: H3Tensors,
        text_rows: int,
    ) -> None:
        """Build packed multimodal positions and write rotary tables into one state slot."""

        transformer = self.denoiser
        assert transformer is not None
        assert execution.scratch is not None and execution.transformer_metadata is not None
        positions = execution.scratch.rotary_positions
        positions.copy_(execution.transformer_metadata.positions)
        non_text_start = int(execution.layout.packed.text_indices.numel())
        positions[non_text_start:, 0].add_(
            int(text_rows) - int(execution.layout.packed.text_indices.numel())
        )
        transformer.rope.forward_into(
            positions,
            slot.rotary_cosine,
            slot.rotary_sine,
            execution.scratch.rotary_frequencies,
        )

    @torch.inference_mode()
    def prepare_tensors(
        self,
        slot: H3Tensors,
        execution: H3ComputeInputs,
        encoded: torch.Tensor | None,
        text_rows: int,
    ) -> None:
        """Install refined conditioning and shape metadata into the explicit request tensors."""

        # Text conditioning and all shape-dependent metadata are stable across
        # the four denoiser steps, so they are materialized once at admission.
        if self.denoiser is not None:
            if self.denoiser.pipeline.first:
                if encoded is None:
                    raise RuntimeError("denoiser input owner did not receive text conditioning")
                slot.text_condition.zero_()
                slot.text_condition[:, : encoded.shape[1]].copy_(encoded)
            self._prepare_tile_metadata(execution, slot, text_rows)
            self._prepare_rotary(execution, slot, text_rows)
        slot.video_overlap.zero_()

    @torch.inference_mode()
    def initialize_tensors(
        self, slot: H3Tensors, seed: int
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
        """Initialize owned rows with the checkpoint's physical-layout-independent noise."""

        if self.denoiser is None:
            return ()
        layout = slot.layout
        # Diffusers' CPU-generator path draws the full video tensor first and
        # then the audio rows. Reproducing that order preserves the public seed.
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        video_noise = slot.video_noise
        video_noise.normal_(generator=generator)
        raster_rows = patchify_video(video_noise)[0]
        tiled_rows = raster_rows.index_select(0, layout.packed.video_raster_indices)
        video_owned = (layout.packed.video_indices >= layout.local_start) & (
            layout.packed.video_indices < layout.local_end
        )
        video_source = slot.video_source
        torch.index_select(
            tiled_rows,
            0,
            video_owned.nonzero(as_tuple=False).flatten(),
            out=video_source,
        )
        audio_noise = slot.audio_noise
        audio_noise.normal_(generator=generator)
        audio_owned = (layout.packed.audio_indices >= layout.local_start) & (
            layout.packed.audio_indices < layout.local_end
        )
        audio_source = slot.audio_source
        torch.index_select(
            audio_noise,
            0,
            audio_owned.nonzero(as_tuple=False).flatten(),
            out=audio_source,
        )
        return ((slot.video_rows, video_source), (slot.audio_rows, audio_source))

    def output_layout(
        self,
        entry: str,
        output_index: int,
        media: MediaGeometry | None,
        decode: DecodeRange | None,
    ) -> TensorOutputLayout | None:
        """Describe unique logical modality rows and temporal decoder results."""

        if self.bindings.process_group.rank not in self.bindings.output_ranks(entry):
            return None
        if media is None:
            raise ValueError("H3 tensor results require media geometry")
        frames, text_rows, audio_frames = self.execution_key(media)
        if entry == "text_encoder":
            return TensorOutputLayout((1, media.prompt_tokens, H3TextEncoderConfig().hidden_size))
        if entry == "denoiser":
            layout = H3Layout.build(
                self.bindings, frames=frames, text_rows=text_rows, audio_frames=audio_frames
            )
            indices = (
                layout.packed.video_indices if output_index == 0 else layout.packed.audio_indices
            )
            count = layout.local_video_rows if output_index == 0 else layout.local_audio_rows
            if count == 0:
                return None
            width = 96 if output_index == 0 else 32
            start = int(torch.searchsorted(indices, layout.local_start))
            shape: tuple[int, ...] = (int(indices.numel()), width)
            return TensorOutputLayout(shape, TensorRegion((start, 0), (count, width)))
        if entry == "video_decoder":
            if decode is None or decode.track is not MediaTrack.VIDEO:
                raise ValueError("video reconstruction requires a temporal range")
            rank = self.bindings.entries[entry].ranks.index(self.bindings.process_group.rank)
            if rank >= decode.max_units:
                return None
            shape = (decode.max_units, 1, 3, 25, PROFILE_HEIGHT, PROFILE_WIDTH)
            return TensorOutputLayout(shape, TensorRegion((rank, 0, 0, 0, 0, 0), (1, *shape[1:])))
        if entry == "audio_decoder":
            return TensorOutputLayout((round(frames * PROFILE_AUDIO_RATE / PROFILE_FPS), 2))
        raise ValueError(f"H3 entry {entry!r} has no Tensor result")

    def decoder_input(
        self,
        execution: H3ComputeInputs,
        latents: torch.Tensor,
        track: MediaTrack,
        cursor: int,
        max_units: int,
    ) -> torch.Tensor:
        """Pack the selected temporal window or stereo latent rows for its decoder."""

        layout, scratch = execution.layout, execution.media
        if track is MediaTrack.VIDEO:
            rank = self.bindings.entries["video_decoder"].ranks.index(
                self.bindings.process_group.rank
            )
            if rank >= max_units or cursor + max_units > layout.video_reconstruction_units:
                raise ValueError("video decode exceeds its temporal range")
            if tuple(latents.shape) != (int(layout.packed.video_indices.numel()), 96):
                raise ValueError("video decoder requires complete final latent rows")
            start = (cursor + rank) * 5 * 24 * 42
            selected = scratch.video_raster_order[start : start + 7 * 24 * 42]
            torch.index_select(latents, 0, selected, out=scratch.reconstruction_rows)
            unpatchify_video_into(
                scratch.reconstruction_rows, scratch.video_input, frames=7, height=48, width=84
            )
            return scratch.video_input
        if (
            cursor != 0
            or max_units != 1
            or tuple(latents.shape) != (int(layout.packed.audio_indices.numel()), 32)
        ):
            raise ValueError("audio decoder requires one complete stereo latent")
        scratch.audio_latents.copy_(
            latents.view(2, layout.packed.audio_frames, 32).permute(0, 2, 1)
        )
        return scratch.audio_latents

    def decoder_output(
        self, execution: H3ComputeInputs, value: torch.Tensor, track: MediaTrack
    ) -> torch.Tensor:
        """Expose decoded segments or the exact duration's interleaved PCM samples."""

        if track is MediaTrack.VIDEO:
            return value.unsqueeze(0)
        samples = round(execution.layout.frame_count * PROFILE_AUDIO_RATE / PROFILE_FPS)
        if value.shape[0] < samples:
            raise RuntimeError("audio decoder returned less than the video duration")
        return value[:samples]

    def assemble_video(
        self,
        slot: H3Tensors,
        execution: H3ComputeInputs,
        segments: torch.Tensor,
        start_unit: int,
        unit_count: int,
    ) -> torch.Tensor:
        """Blend overlapping decoded windows and apply the checkpoint pixel transform."""

        layout, scratch = execution.layout, execution.media
        if tuple(segments.shape) != (unit_count, 1, 3, 25, PROFILE_HEIGHT, PROFILE_WIDTH):
            raise ValueError("video assembly requires complete decoded segments")
        if start_unit + unit_count > layout.video_reconstruction_units:
            raise ValueError("video assembly exceeds its temporal extent")
        overlap = None if start_unit == 0 else slot.video_overlap
        frame_start = 0
        for offset in range(unit_count):
            unit = start_unit + offset
            assert self.video_pixel_mean is not None and self.video_pixel_std is not None
            rgb, overlap = video_segment_rgb(
                segments[offset],
                overlap,
                body_frames=(
                    MiniMaxH3VideoDecoder.tokens_chunk_size
                    * MiniMaxH3VideoDecoder.temporal_compression_ratio
                    - MiniMaxH3VideoDecoder.frame_pre_padding
                ),
                overlap_frames=MiniMaxH3VideoDecoder.frame_overlap,
                padding_frames=MiniMaxH3VideoDecoder.frame_pre_padding,
                pixel_mean=self.video_pixel_mean,
                pixel_std=self.video_pixel_std,
                final_unit=unit + 1 == layout.video_reconstruction_units,
            )
            valid_frames = layout.reconstruction_unit_frames[unit]
            if int(rgb.shape[0]) != valid_frames:
                raise RuntimeError("H3 video decoder returned an unexpected frame count")
            scratch.rgb_round[frame_start : frame_start + valid_frames].copy_(rgb)
            frame_start += valid_frames
        assert overlap is not None
        slot.video_overlap.copy_(overlap)
        return scratch.rgb_round[:frame_start]

    def output_geometry(self, geometry: MediaGeometry) -> VideoOutputGeometry:
        """Describe the exact raster and sample timing required by the mathematics."""

        self.execution_key(geometry)
        return VideoOutputGeometry(
            frame_count=geometry.frame_count,
            unit_frames=reconstruction_unit_frames(geometry.frame_count),
            width=PROFILE_WIDTH,
            height=PROFILE_HEIGHT,
            frame_rate=PROFILE_FPS,
            audio_rate=PROFILE_AUDIO_RATE,
        )

    @torch.inference_mode()
    def _warmup_inputs(
        self,
        storage: tuple[BoundedTensorStorage, ...],
        scratch: BoundedTensorStorage | None,
        context_workspace: AttentionContextWorkspace | None,
        schedule: DiffusionSchedule | None,
    ) -> Iterator[ModuleWarmup]:
        """Supply representative numerical inputs over the declared startup storage."""

        assert scratch is not None
        if not storage:
            raise RuntimeError("H3 warmup requires declared request tensor storage")
        if self.denoiser is not None:
            if schedule is None:
                raise RuntimeError("denoiser warmup requires its diffusion schedule")
            max_rows = int(self.layout.packed.text_indices.numel())
            # Cover the minimum request, text-page residues and capacity shape.
            shapes = (
                (self.layout.frame_count, 65),
                (MIN_H3_FRAMES, 1),
                *((self.layout.frame_count, count) for count in (129, 193, 257)),
                (self.layout.frame_count, max_rows),
            )
            prepared: set[Hashable] = set()
            for frames, token_count in shapes:
                geometry = MediaGeometry(
                    frame_count=frames,
                    video_units=len(reconstruction_unit_frames(frames)),
                    prompt_tokens=token_count,
                    denoise_steps=4,
                )
                key = self.execution_key(geometry)
                if key in prepared:
                    continue
                prepared.add(key)
                execution = self.build_execution(geometry, scratch, context_workspace)
                page_rows = int(execution.layout.packed.text_indices.numel())
                views = tuple(
                    bind_request_tensors(tensors, execution.layout) for tensors in storage
                )
                for slot in views:
                    slot.text_condition.zero_()
                    slot.video_rows.zero_()
                    slot.audio_rows.zero_()
                    self._prepare_tile_metadata(execution, slot, page_rows)
                    self._prepare_rotary(execution, slot, page_rows)
                assert execution.scratch is not None and execution.transformer_metadata is not None
                yield ModuleWarmup(
                    "denoiser", (views[0], execution, 0, 1, schedule), (key, execution)
                )
        if self.bindings.owns("output"):
            assert self.video_pixel_mean is not None and self.video_pixel_std is not None
            segment = torch.zeros(
                (1, 3, 25, PROFILE_HEIGHT, PROFILE_WIDTH), dtype=torch.float16, device=self.device
            )
            video_segment_rgb(
                segment,
                None,
                body_frames=(
                    MiniMaxH3VideoDecoder.tokens_chunk_size
                    * MiniMaxH3VideoDecoder.temporal_compression_ratio
                    - MiniMaxH3VideoDecoder.frame_pre_padding
                ),
                overlap_frames=MiniMaxH3VideoDecoder.frame_overlap,
                padding_frames=MiniMaxH3VideoDecoder.frame_pre_padding,
                pixel_mean=self.video_pixel_mean,
                pixel_std=self.video_pixel_std,
                final_unit=False,
            )
        if self.audio_decoder is not None:
            audio_latents = torch.zeros(
                (2, 32, self.layout.packed.audio_frames),
                dtype=torch.float32,
                device=self.device,
            )
            yield ModuleWarmup("audio_decoder", (audio_latents,))


def validate_h3_entries(bindings: EntryBindings) -> None:
    """Validate only the numerical entries assigned to this static Worker."""

    expected = {"denoiser", "text_encoder", "video_decoder", "audio_decoder", "output"}
    if not bindings.entries or not set(bindings.entries) <= expected:
        raise ValueError(f"H3 entries must belong to {sorted(expected)}")
    for name, component in bindings.entries.items():
        config = component.parallel_config
        if name == "video_decoder":
            if component.distribution != "temporal_units" or component.units_per_rank != 1:
                raise ValueError("H3 video decoder requires temporal_units with native batch one")
            continue
        if component.distribution is not None:
            raise ValueError(f"H3 {name} requires model-parallel membership")
        if name in {"audio_decoder", "output"}:
            if len(component.ranks) != 1 or config.world_size != 1:
                raise ValueError(f"H3 {name} requires one local owner")
        elif name == "text_encoder":
            if config.pipeline_parallel_size != 1 or config.sequence_parallel_size != 1:
                raise ValueError("H3 text encoder supports direct tensor parallelism")
            if any(width % config.tensor_parallel_size for width in (64, 8, 25600)):
                raise ValueError("H3 encoder TP must divide query heads, KV heads, and MLP width")
        elif name == "denoiser":
            if config.pipeline_parallel_size > 50:
                raise ValueError("H3 pipeline stages cannot exceed its 50 transformer layers")
            if config.sequence_parallel.kind not in {
                "local",
                "ulysses",
                "allgather",
                "ring",
                "hybrid",
                "attention2d",
            }:
                raise ValueError("H3 sequence attention requires global sparse selection")
            tensor = config.tensor_parallel_size
            ulysses = dict(config.dimensions)["ulysses"]
            if 56 % (tensor * ulysses) or 5376 % tensor or 14336 % tensor:
                raise ValueError(
                    "H3 TP × Ulysses must divide heads; TP must divide hidden and MLP widths"
                )
            if tensor not in (1, 2, 4) or config.sequence_parallel_size not in (1, 2, 4):
                raise ValueError("H3 requires TP and sequence degrees in 1, 2, or 4")
