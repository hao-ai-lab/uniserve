"""Concrete fixed-profile MiniMax H3 model and resident request state."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from ...execution.batch import (
    NewRequest,
    RunKind,
)
from ...execution.device_transfer import ComponentTensorTransfer
from ...profiling import profile_range, synchronize_profile_range
from ..runtime import DedicatedStateGeometry, ResourceGeometry
from ..video import DecodeKind, DecodeOutput, VideoOutputGeometry, VideoRunner
from .packing import audio_latent_frames, patchify_video, unpatchify_video_into
from .placement import H3Placement
from .precision import H3LinearPrecisionPolicy
from .schedule import solver_step
from .state import (
    MIN_H3_FRAMES,
    PROFILE_AUDIO_RATE,
    PROFILE_FPS,
    PROFILE_HEIGHT,
    PROFILE_WIDTH,
    H3Layout,
    H3MediaScratch,
    H3Scratch,
    H3StatePool,
    H3StateSlot,
)
from .video_vae import H3VideoAssembler
from .weights import H3Components, load_h3_components

if TYPE_CHECKING:
    from ...worker.warmup import WarmupContext

__all__ = ["MiniMaxH3Runner"]


@dataclass(frozen=True, slots=True)
class _H3PageExecution:
    """Binds an H3 layout to scratch storage, transformer indices, and sparse-attention page metadata."""

    layout: H3Layout
    scratch: H3Scratch | None
    media: H3MediaScratch
    transformer_execution: Any
    base_tile_valid_sizes: torch.Tensor
    prompt_prefix_indices: torch.Tensor
    prompt_dense_indices: torch.Tensor
    prompt_prefix_counts: torch.Tensor


class MiniMaxH3Runner(VideoRunner):
    """Execute assigned H3 components and transfer products within one instance."""

    architecture = "MiniMaxH3Transformer3DModel"
    serving_dtype = "bfloat16"
    resource_geometry = ResourceGeometry(kv=False)
    supported_work = frozenset(
        {
            RunKind.DIFFUSION_PREPARE,
            RunKind.DIFFUSION_STEP,
            RunKind.DIFFUSION_DECODE,
            RunKind.DIFFUSION_FINALIZE,
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
        placement: H3Placement,
        components: H3Components,
        layout: H3Layout,
        *,
        max_state_slots: int,
    ) -> None:
        """Bind H3 model components to runtime-owned state, scratch, and device products."""

        super().__init__()

        self.placement = placement
        self.device = placement.process_group.device
        self.layout = layout
        self.transformer = components.transformer
        self.encoder = components.encoder
        self.video_vae = components.video_vae
        self.audio_vae = components.audio_vae
        for name, module in (
            ("denoiser", self.transformer),
            ("text_encoder", self.encoder),
            ("video_decoder", self.video_vae),
            ("audio_decoder", self.audio_vae),
        ):
            if placement.owns(name) != (module is not None):
                raise ValueError(f"H3 {name} materialization disagrees with assigned membership")
        self.video_assembler = H3VideoAssembler(self.device) if placement.owns("output") else None
        self.preparation_stream = torch.cuda.Stream(device=self.device)
        group = placement.process_group
        self.video_latent_transfer = ComponentTensorTransfer(
            group, placement.latent_producers, placement.decoder_ranks
        )
        self.video_output_transfer = ComponentTensorTransfer(
            group, placement.decoder_ranks, (placement.output_rank,)
        )
        self.audio_latent_transfer = ComponentTensorTransfer(
            group, placement.latent_producers, (placement.audio_rank,)
        )
        self.audio_output_transfer = ComponentTensorTransfer(
            group, (placement.audio_rank,), (placement.output_rank,)
        )
        encoder_members = placement.components["text_encoder"].ranks
        encoder_consumers = tuple(
            rank for rank in placement.input_ranks if rank not in encoder_members
        )
        self.encoder_transfer = (
            ComponentTensorTransfer(group, encoder_members[:1], encoder_consumers)
            if encoder_consumers
            else None
        )
        self.scratch_storage = None
        if self.transformer is not None:
            self.scratch_storage = H3Scratch.allocate(
                layout,
                placement.denoiser_mesh,
                block_params_shape=tuple(self.transformer.modulation_plan.blocks.shape[1:]),
                final_params_shape=(
                    tuple(self.transformer.modulation_plan.final.shape[1:])
                    if self.transformer.modulation_plan.final is not None
                    else (0,)
                ),
                attention_workspace_dtype=torch.bfloat16
                if self.transformer.attention_linear_precision == "bf16"
                else torch.uint8,
            )
        self.media_storage = H3MediaScratch.allocate(layout, placement)
        self.page_executions: dict[tuple[int, int, int], _H3PageExecution] = {}
        self.prompt_device = torch.empty(
            (1 + int(layout.packed.text_indices.numel()),),
            dtype=torch.long,
            device=self.device,
        )
        self.prompt_host = (
            torch.empty_like(self.prompt_device, device="cpu", pin_memory=True)
            if placement.owns("output")
            else None
        )

        # Size the state pool from live device memory after all model weights
        # and rank-local scratch allocations have reached their final shapes.
        if int(max_state_slots) < 2:
            raise ValueError("the FastH3 serving topology requires at least two state slots")
        torch.cuda.empty_cache()
        free_bytes, _total_bytes = torch.cuda.mem_get_info(self.device)
        state_slots = min(
            int(max_state_slots),
            int(free_bytes) // H3StatePool.bytes_per_slot(layout),
        )
        shared_slots = torch.tensor(state_slots, device=self.device, dtype=torch.int64)
        placement.process_group.all_reduce_min(shared_slots)
        state_slots = int(shared_slots.item())
        if state_slots < 2:
            raise RuntimeError("MiniMax H3 has insufficient CUDA memory for two state slots")
        self._state_slot_count = state_slots
        self.dedicated_state_geometry = DedicatedStateGeometry(
            slot_count=int(state_slots),
            persistent_units=int(layout.persistent_units),
            max_vae_grid_tokens=int(layout.packed.video_indices.numel()),
            rank=placement.process_group.rank,
            size=placement.process_group.world_size,
        )

    def _build_page_execution(self, layout: H3Layout) -> _H3PageExecution:
        """Build shape-specific sparse-attention metadata and a bounded scratch view."""

        text_tiles = int(layout.packed.text_indices.numel()) // 64
        if text_tiles < 1:
            raise ValueError("H3 page execution requires at least one text page")
        prefix = torch.arange(layout.packed.prefix_tiles, dtype=torch.int32)
        dense = torch.arange(
            layout.packed.prefix_tiles + layout.packed.video_tiles,
            dtype=torch.int32,
        )
        return _H3PageExecution(
            layout=layout,
            scratch=self.scratch_storage.view(layout) if self.scratch_storage is not None else None,
            media=self.media_storage.view(layout),
            transformer_execution=self.transformer.build_execution(layout)
            if self.transformer is not None
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

    def create_media_runtime(self, unresolved_window: int) -> tuple[object | None, object | None]:
        """Create muxing and bounded output storage on the declared output owner."""

        from ...execution.video import (
            VideoMuxCoordinator,
            VideoOutputRing,
            require_video_codecs,
        )

        if not self.placement.owns("output"):
            return None, None
        require_video_codecs()
        geometry = VideoOutputGeometry(
            frame_count=self.layout.frame_count,
            unit_frames=self.layout.reconstruction_unit_frames,
            width=PROFILE_WIDTH,
            height=PROFILE_HEIGHT,
            frame_rate=PROFILE_FPS,
            audio_rate=PROFILE_AUDIO_RATE,
        )
        return (
            VideoMuxCoordinator(rank=self.placement.output_rank),
            VideoOutputRing(
                state_slots=self._state_slot_count,
                unresolved_window=int(unresolved_window),
                max_video_frames_per_round=self.layout.max_video_round_frames,
                max_geometry=geometry,
            ),
        )

    def create_request_state(self) -> H3StatePool:
        """Allocate the resident request slots admitted by the deployment geometry."""

        return H3StatePool(
            self.layout,
            self._state_slot_count,
            self.device,
        )

    def synchronize_runtime(self) -> None:
        """Wait for all H3 work submitted to the local CUDA device."""

        torch.cuda.synchronize(self.device)

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: str,
        placement: H3Placement,
        *,
        max_state_slots: int,
        max_text_rows: int,
        max_video_seconds: float,
        cache_dir: str | None = None,
        revision: str | None = None,
        precision_policy: H3LinearPrecisionPolicy,
    ) -> "MiniMaxH3Runner":
        """Load a fixed-profile runner sized for the configured prompt and video limits."""

        text_capacity = ((int(max_text_rows) + 63) // 64) * 64
        raw_frames = math.floor(float(max_video_seconds) * 24.0 + 0.5)
        max_frames = int(raw_frames + (5 - raw_frames) % 17)
        if text_capacity < 64 or max_frames < MIN_H3_FRAMES:
            raise ValueError("H3 deployment capacity is smaller than a legal request")
        max_audio_frames = audio_latent_frames(max_frames)
        layout = H3Layout.build(
            placement,
            frames=max_frames,
            text_rows=text_capacity,
            audio_frames=max_audio_frames,
        )
        components = load_h3_components(
            checkpoint,
            placement,
            layout,
            cache_dir=cache_dir,
            revision=revision,
            precision_policy=precision_policy,
        )
        return cls(
            placement,
            components,
            layout,
            max_state_slots=max_state_slots,
        )

    def _page_execution_for_geometry(
        self,
        *,
        frame_count: int,
        audio_frames: int,
        token_count: int,
    ) -> _H3PageExecution:
        """Resolve or cache execution metadata for one video, audio, and prompt geometry."""

        page_rows = ((int(token_count) + 63) // 64) * 64
        shape_key = (int(frame_count), page_rows, int(audio_frames))
        if (
            page_rows > int(self.layout.packed.text_indices.numel())
            or frame_count > self.layout.frame_count
            or audio_frames > self.layout.packed.audio_frames
        ):
            raise ValueError("H3 media geometry exceeds the configured model capacity")
        execution = self.page_executions.get(shape_key)
        if execution is None:
            page_layout = H3Layout.build(
                self.placement,
                frames=frame_count,
                text_rows=page_rows,
                audio_frames=audio_frames,
                schedule=self.layout.schedule,
            )
            execution = self._build_page_execution(page_layout)
            self.page_executions[shape_key] = execution
        return execution

    def _page_execution_for_slot(self, slot: H3StateSlot) -> _H3PageExecution:
        """Resolve the shape-specific execution metadata bound to a live state slot."""

        if slot.shape_key is None:
            raise RuntimeError("H3 state slot has no active execution layout")
        try:
            return self.page_executions[slot.shape_key]
        except KeyError as error:
            raise RuntimeError("H3 state slot references an unavailable page layout") from error

    def _token_ids(
        self,
        prompt_token_ids: tuple[int, ...],
        expected_tokens: int,
    ) -> torch.Tensor:
        """Validate and distribute canonical token identifiers to component participants."""

        token_error: BaseException | None = None
        if self.placement.owns("output"):
            host = self.prompt_host
            if host is None:
                raise RuntimeError("H3 output owner has no prompt staging buffer")
            try:
                token_ids = tuple(int(value) for value in prompt_token_ids)
                if len(token_ids) != int(expected_tokens):
                    raise ValueError("H3 prompt tokens disagree with the admitted media geometry")
                if not 1 <= len(token_ids) <= self.layout.packed.text_indices.numel():
                    raise ValueError(
                        "H3 prompt must encode to between 1 and "
                        f"{self.layout.packed.text_indices.numel()} tokens"
                    )
                host[0] = len(token_ids)
                host[1 : 1 + len(token_ids)].copy_(torch.tensor(token_ids))
            except BaseException as error:
                host[0] = 0
                token_error = error
            self.prompt_device.copy_(host, non_blocking=True)
        self.placement.process_group.broadcast(
            self.prompt_device,
            src=self.placement.process_group.ranks.index(self.placement.output_rank),
        )
        count = int(self.prompt_device[0].item())
        if token_error is not None:
            raise token_error
        if not 1 <= count <= self.layout.packed.text_indices.numel():
            raise ValueError("H3 prompt tokenization failed on its input owner")
        return self.prompt_device[1 : 1 + count].view(1, -1)

    def _prepare_tile_metadata(
        self,
        execution: _H3PageExecution,
        slot: H3StateSlot,
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
        execution: _H3PageExecution,
        slot: H3StateSlot,
        text_rows: int,
    ) -> None:
        """Build packed multimodal positions and write rotary tables into one state slot."""

        transformer = self.transformer
        positions = execution.scratch.rotary_positions
        positions.copy_(execution.transformer_execution.positions)
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
    def prepare(self, slot: H3StateSlot, admission: NewRequest) -> None:
        """Bind request geometry, seed resident latents, and precompute conditioning state."""

        if slot.active:
            raise RuntimeError("H3 state slot is already active")
        media = admission.diffusion
        if media is None:
            raise ValueError("the H3 worker received an incompatible media profile")
        token_ids = self._token_ids(
            media.prompt_token_ids,
            media.geometry.prompt_tokens,
        )
        execution = self._page_execution_for_geometry(
            frame_count=media.geometry.frame_count,
            audio_frames=audio_latent_frames(media.geometry.frame_count),
            token_count=media.geometry.prompt_tokens,
        )
        expected_decode_units = (
            execution.layout.video_reconstruction_units + execution.layout.decoder_width - 1
        ) // execution.layout.decoder_width + 2
        if int(media.geometry.decode_units) != expected_decode_units:
            raise ValueError("the H3 worker received an invalid decode bound")

        slot.bind(execution.layout)
        layout = execution.layout
        if self.transformer is not None:
            self._initialize_latents(slot, layout, media.seed)
        # Text conditioning and all shape-dependent metadata are stable across
        # the four denoiser steps, so they are materialized once at admission.
        encoded = self.encoder(token_ids) if self.encoder is not None else None
        if self.encoder_transfer is not None:
            received = (
                execution.media.encoder_hidden[:, : token_ids.shape[1]]
                if self.placement.process_group.rank in self.encoder_transfer.consumers
                else None
            )
            self.encoder_transfer.broadcast(encoded, received)
            if received is not None:
                encoded = received
        if self.transformer is not None:
            if self.transformer.pipeline.first:
                if encoded is None:
                    raise RuntimeError("denoiser input owner did not receive text conditioning")
                refined = self.transformer.refine_text(encoded)
                slot.text_condition.zero_()
                slot.text_condition[:, : refined.shape[1]].copy_(refined)
            self._prepare_tile_metadata(execution, slot, int(token_ids.shape[1]))
            self._prepare_rotary(execution, slot, int(token_ids.shape[1]))
            torch.cuda.current_stream(self.device).wait_stream(self.preparation_stream)
        if slot.video_overlap is not None:
            slot.video_overlap.zero_()
        slot.request_key = admission.request_key
        slot.denoise_step = 0
        slot.next_video_unit = 0
        slot.audio_reconstructed = False

    def _initialize_latents(self, slot: H3StateSlot, layout: H3Layout, seed: int) -> None:
        """Initialize owned rows with the checkpoint's physical-layout-independent noise."""

        # Diffusers' CPU-generator path draws the full video tensor first and
        # then the audio rows. Reproducing that order preserves the public seed.
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        video_noise = torch.empty(
            (1, 24, layout.packed.video_frames, 48, 84),
            dtype=torch.float32,
            pin_memory=True,
        )
        video_noise.normal_(generator=generator)
        raster_rows = patchify_video(video_noise)[0]
        tiled_rows = raster_rows.index_select(0, layout.packed.video_raster_indices)
        video_owned = (layout.packed.video_indices >= layout.local_start) & (
            layout.packed.video_indices < layout.local_end
        )
        video_source = torch.empty(
            slot.video_rows.shape,
            dtype=torch.float32,
            pin_memory=True,
        )
        torch.index_select(
            tiled_rows,
            0,
            video_owned.nonzero(as_tuple=False).flatten(),
            out=video_source,
        )
        audio_rows = int(layout.packed.audio_indices.numel())
        audio_noise = torch.empty((audio_rows, 32), dtype=torch.float32, pin_memory=True)
        audio_noise.normal_(generator=generator)
        audio_owned = (layout.packed.audio_indices >= layout.local_start) & (
            layout.packed.audio_indices < layout.local_end
        )
        audio_source = torch.empty(
            slot.audio_rows.shape,
            dtype=torch.float32,
            pin_memory=True,
        )
        torch.index_select(
            audio_noise,
            0,
            audio_owned.nonzero(as_tuple=False).flatten(),
            out=audio_source,
        )
        with torch.cuda.stream(self.preparation_stream):
            slot.video_rows.copy_(video_source, non_blocking=True)
            slot.audio_rows.copy_(audio_source, non_blocking=True)

    @torch.inference_mode()
    def denoise(self, slot: H3StateSlot, start_step: int, step_count: int) -> None:
        """Advance both resident media latents by exactly one scheduled solver step."""

        if not slot.active:
            raise RuntimeError("H3 denoise references an inactive state slot")
        if int(step_count) != 1 or int(start_step) != slot.denoise_step:
            raise ValueError("H3 denoise placements must advance exactly one current step")
        if not 0 <= start_step < 4:
            raise ValueError("H3 denoise step is outside the four-evaluation ladder")
        if self.transformer is None:
            slot.denoise_step += 1
            return
        execution = self._page_execution_for_slot(slot)
        schedule = execution.layout.schedule
        scratch = execution.scratch

        # Modulation parameters are selected from the admission-time plan; the
        # transformer writes rank-local velocity tensors into shared scratch.
        self.transformer.select_adaln_step(scratch, start_step)
        self.transformer.bind_execution(execution.transformer_execution)
        with profile_range("uniserve.h3.denoise"):
            self.transformer.forward_local_prepared(slot, scratch)
            if self.transformer.pipeline.last:
                video_velocity = scratch.video_velocity
                audio_velocity = scratch.audio_velocity
                solver_step(
                    slot.video_rows,
                    video_velocity,
                    schedule.video_timesteps[start_step],
                    schedule.video_sigmas[start_step],
                    schedule.video_sigmas[start_step + 1],
                )
                solver_step(
                    slot.audio_rows,
                    audio_velocity,
                    schedule.audio_timesteps[start_step],
                    schedule.audio_sigmas[start_step],
                    schedule.audio_sigmas[start_step + 1],
                )
            self.transformer.pipeline.feedback((slot.video_rows, slot.audio_rows))
            synchronize_profile_range(self.device)
        slot.denoise_step += 1

    def _exchange_video_round(
        self, slot: H3StateSlot, start_unit: int, unit_count: int
    ) -> torch.Tensor | None:
        """Reconstruct decoder units from unique denoiser row owners."""

        execution = self._page_execution_for_slot(slot)
        scratch = execution.media
        send = scratch.video_send
        send.zero_()
        if self.placement.process_group.rank in self.placement.latent_producers:
            raster = scratch.local_video_raster
            rows = scratch.reconstruction_rows
            rows_per_frame = 24 * 42
            for offset in range(unit_count):
                start_frame = (start_unit + offset) * 5
                start_row = start_frame * rows_per_frame
                stop_row = (start_frame + 7) * rows_per_frame
                rows.zero_()
                selected = (raster >= start_row) & (raster < stop_row)
                rows.index_copy_(0, raster[selected] - start_row, slot.video_rows[selected])
                unpatchify_video_into(
                    rows,
                    send[offset].unsqueeze(0),
                    frames=7,
                    height=48,
                    width=84,
                )
        with profile_range("uniserve.collective kind=video_latent_exchange"):
            self.video_latent_transfer.exchange(send, scratch.video_receive)
        if self.video_vae is None:
            return None
        torch.sum(scratch.video_receive, dim=0, out=scratch.video_input[0])
        return scratch.video_input

    @torch.inference_mode()
    def reconstruct_video(
        self, slot: H3StateSlot, start_unit: int, unit_count: int
    ) -> torch.Tensor | None:
        """Decode assigned temporal units and assemble them on the output owner."""

        execution = self._page_execution_for_slot(slot)
        layout = execution.layout
        start_unit, unit_count = int(start_unit), int(unit_count)
        expected_count = min(layout.decoder_width, layout.video_reconstruction_units - start_unit)
        if (
            start_unit != slot.next_video_unit
            or start_unit % layout.decoder_width != 0
            or unit_count != expected_count
            or not 0 <= start_unit < layout.video_reconstruction_units
        ):
            raise ValueError("invalid H3 video reconstruction placement")
        scratch = execution.media
        latents = self._exchange_video_round(slot, start_unit, unit_count)
        send = scratch.empty_segments
        decoder_index = self.placement.decoder_index
        if self.video_vae is not None:
            if latents is None or decoder_index is None:
                raise RuntimeError("video decoder did not receive its latent product")
            segment = (
                self.video_vae.decode_segment(latents)
                if decoder_index < unit_count
                else scratch.segment_placeholder
            )
            if segment.dtype != torch.float16:
                raise RuntimeError("H3 video decoder returned an unexpected segment dtype")
            send = segment.unsqueeze(0)
        with profile_range("uniserve.collective kind=video_segment_gather"):
            self.video_output_transfer.exchange(send, scratch.segment_receive)
        slot.next_video_unit += unit_count
        if self.video_assembler is None:
            return None
        if slot.video_overlap is None:
            raise RuntimeError("H3 output owner lost its temporal overlap storage")
        overlap = None if start_unit == 0 else slot.video_overlap
        frame_start = 0
        for offset in range(unit_count):
            unit = start_unit + offset
            rgb, overlap = self.video_assembler(
                scratch.segment_receive[offset],
                overlap,
                final_unit=unit + 1 == layout.video_reconstruction_units,
            )
            valid_frames = layout.reconstruction_unit_frames[unit]
            if int(rgb.shape[0]) != valid_frames:
                raise RuntimeError("H3 video decoder returned an unexpected frame count")
            scratch.rgb_round[frame_start : frame_start + valid_frames].copy_(rgb)
            frame_start += valid_frames
        slot.video_overlap.copy_(overlap)
        return scratch.rgb_round[:frame_start]

    @torch.inference_mode()
    def reconstruct_audio(self, slot: H3StateSlot) -> torch.Tensor | None:
        """Reconstruct unique audio rows, decode once, and transfer ordered PCM."""

        if slot.audio_reconstructed:
            raise ValueError("invalid H3 audio reconstruction placement")
        execution = self._page_execution_for_slot(slot)
        layout, scratch = execution.layout, execution.media
        scratch.audio_send.zero_()
        if self.placement.process_group.rank in self.placement.latent_producers:
            scratch.audio_send[0].index_copy_(0, scratch.local_audio_raster, slot.audio_rows)
        with profile_range("uniserve.collective kind=audio_latent_gather"):
            self.audio_latent_transfer.exchange(scratch.audio_send, scratch.audio_receive)
        pcm = None
        if self.audio_vae is not None:
            torch.sum(scratch.audio_receive, dim=0, out=scratch.audio_input)
            scratch.audio_latents.copy_(
                scratch.audio_input.view(2, layout.packed.audio_frames, 32).permute(0, 2, 1)
            )
            decoded = self.audio_vae.decode(scratch.audio_latents)
            target_samples = round(layout.frame_count * PROFILE_AUDIO_RATE / PROFILE_FPS)
            if decoded.shape[0] < target_samples:
                raise RuntimeError("H3 audio decoder returned less than the fixed video duration")
            pcm = decoded[:target_samples]
        output = scratch.pcm if self.placement.owns("output") else None
        self.audio_output_transfer.broadcast(pcm, output)
        slot.audio_reconstructed = True
        return output

    def decode_kind(self, slot: H3StateSlot, cursor: int) -> DecodeKind:
        """Map the bounded output cursor to video, audio, or finalization work."""

        layout = self._page_execution_for_slot(slot).layout
        video_rounds = (
            layout.video_reconstruction_units + layout.decoder_width - 1
        ) // layout.decoder_width
        if 0 <= int(cursor) < video_rounds:
            return DecodeKind.VIDEO
        if int(cursor) == video_rounds:
            return DecodeKind.AUDIO
        if int(cursor) == video_rounds + 1:
            return DecodeKind.FINALIZE
        raise ValueError("H3 decode cursor is outside the request output bound")

    @torch.inference_mode()
    def decode(self, slot: H3StateSlot, cursor: int, max_units: int) -> DecodeOutput:
        """Execute one bounded output unit and describe its position in the media stream."""

        if int(max_units) != 1:
            raise ValueError("H3 decode calls advance exactly one bounded unit")
        kind = self.decode_kind(slot, cursor)
        layout = self._page_execution_for_slot(slot).layout
        if kind is DecodeKind.VIDEO:
            start_unit = int(cursor) * layout.decoder_width
            unit_count = min(
                layout.decoder_width,
                layout.video_reconstruction_units - start_unit,
            )
            return DecodeOutput(
                kind=kind,
                value=self.reconstruct_video(slot, start_unit, unit_count),
                unit_offset=start_unit,
                unit_count=unit_count,
            )
        if kind is DecodeKind.AUDIO:
            return DecodeOutput(kind, self.reconstruct_audio(slot), 0, 1)
        return DecodeOutput(kind, None, 0, 1)

    def finalize(self, request: object) -> None:
        """Complete the protocol finalization stage after all media units are published."""

        return None

    def output_geometry(self, request: object) -> VideoOutputGeometry:
        """Expose exact video/audio mux geometry for an active H3 request slot."""

        if not isinstance(request, H3StateSlot):
            raise TypeError("H3 output geometry requires a state slot")
        layout = self._page_execution_for_slot(request).layout
        return VideoOutputGeometry(
            frame_count=layout.frame_count,
            unit_frames=layout.reconstruction_unit_frames,
            width=PROFILE_WIDTH,
            height=PROFILE_HEIGHT,
            frame_rate=PROFILE_FPS,
            audio_rate=PROFILE_AUDIO_RATE,
        )

    @torch.inference_mode()
    def warmup(self, context: WarmupContext) -> None:
        """Materialize page shapes, kernels, collectives, and decoder graphs before admission."""

        state_pool = context.requests.model_state
        if not isinstance(state_pool, H3StatePool):
            raise RuntimeError("H3 warmup requires request-pool state")
        if self.transformer is not None:
            max_rows = int(self.layout.packed.text_indices.numel())

            def capacity_execution(token_count: int) -> _H3PageExecution:
                """Resolve the maximum media layout at one padded prompt capacity."""

                return self._page_execution_for_geometry(
                    frame_count=self.layout.frame_count,
                    audio_frames=self.layout.packed.audio_frames,
                    token_count=token_count,
                )

            # Cover the minimum request, every text-page residue class used by the
            # sparse kernels, and the configured maximum-capacity page shape.
            min_execution = self._page_execution_for_geometry(
                frame_count=MIN_H3_FRAMES,
                audio_frames=audio_latent_frames(MIN_H3_FRAMES),
                token_count=1,
            )
            generic_execution = capacity_execution(65)
            residue_executions = tuple(
                capacity_execution(token_count) for token_count in (129, 193, 257)
            )
            max_execution = capacity_execution(max_rows)
            warmup_executions = (
                generic_execution,
                min_execution,
                *residue_executions,
                max_execution,
            )
            for execution in warmup_executions:
                page_rows = int(execution.layout.packed.text_indices.numel())
                self.transformer.bind_execution(execution.transformer_execution)
                for slot in state_pool.slots:
                    slot.bind(execution.layout)
                    slot.text_condition.zero_()
                    slot.video_rows.zero_()
                    slot.audio_rows.zero_()
                    self._prepare_tile_metadata(execution, slot, page_rows)
                    self._prepare_rotary(execution, slot, page_rows)
                slot = state_pool.slots[0]
                self.transformer.select_adaln_step(execution.scratch, 0)
                self.transformer.forward_local_prepared(slot, execution.scratch)
            self.transformer.bind_execution(max_execution.transformer_execution)
        if self.video_vae is not None:
            video_latents = torch.zeros((1, 24, 7, 48, 84), dtype=torch.float32, device=self.device)
            self.video_vae.capture_decoder(video_latents)
        if self.video_assembler is not None:
            segment = self.media_storage.segment_receive[0]
            segment.zero_()
            self.video_assembler(segment, None, final_unit=False)
        if self.audio_vae is not None:
            audio_latents = torch.zeros(
                (2, 32, self.layout.packed.audio_frames),
                dtype=torch.float32,
                device=self.device,
            )
            self.audio_vae.warmup_decoder(audio_latents)
        torch.cuda.synchronize(self.device)
        for slot in state_pool.slots:
            slot.clear()
