"""Concrete fixed-profile MiniMax H3 model and resident request state."""

from __future__ import annotations

from typing import Any, Literal

import torch
from torch import nn

from ...execution.batch import DecodeKind, DecodePlacement, ForwardMode, MediaProfileId, NewRequest
from ...nn.mesh import DeviceMesh
from ...server.profiler import profile_range
from ..runtime import ResourceGeometry
from .packing import patchify_video, unpatchify_video_into
from .schedule import solver_step
from .state import H3Layout, H3Scratch, H3StatePool, H3StateSlot
from .weights import H3Components, load_h3_components

__all__ = ["MiniMaxH3Model"]


class MiniMaxH3Model(nn.Module):
    """One SP4 replica with TP4 conditioning and a serial shared scratch lane."""

    architecture = "MiniMaxH3Transformer3DModel"
    serving_dtype = "bfloat16"
    resource_geometry = ResourceGeometry(kv=False)
    supported_work = frozenset(
        {
            ForwardMode.GEN_TRANSITION,
            ForwardMode.GEN_FLOW,
            ForwardMode.GEN_DECODE,
            ForwardMode.MATERIALIZE,
        }
    )
    generation = None
    image_processor = None
    tensorized_mixed = False

    def __init__(
        self,
        mesh: DeviceMesh,
        components: H3Components,
        layout: H3Layout,
        *,
        max_state_slots: int,
    ) -> None:
        super().__init__()
        if mesh.size("sp") != 4 or mesh.size("tp") != 4:
            raise ValueError("the production FastH3 topology requires TP4/SP4")
        self.mesh = mesh
        self.layout = layout
        self.transformer = components.transformer
        self.encoder = components.encoder
        self.video_vae = components.video_vae
        self.audio_vae = components.audio_vae
        self.tokenizer = components.tokenizer
        self.checkpoint_digest = components.checkpoint.digest
        self.scratch = H3Scratch.allocate(layout, mesh.local_device)
        self.preparation_stream = torch.cuda.Stream(device=mesh.local_device)
        self.prompt_device = torch.empty(
            (1 + int(layout.packed.text_indices.numel()),),
            dtype=torch.long,
            device=mesh.local_device,
        )
        self.prompt_host = (
            torch.empty_like(self.prompt_device, device="cpu", pin_memory=True)
            if mesh.coord("tp") == 0
            else None
        )
        text_tiles = int(layout.packed.text_indices.numel()) // 64
        prefix_rows: list[torch.Tensor] = []
        dense_rows: list[torch.Tensor] = []
        prefix_counts: list[int] = []
        for valid_text_tiles in range(1, text_tiles + 1):
            prefix = (
                *range(valid_text_tiles),
                *range(text_tiles, layout.packed.prefix_tiles),
            )
            dense = (
                *prefix,
                *range(
                    layout.packed.prefix_tiles,
                    layout.packed.prefix_tiles + layout.packed.video_tiles,
                ),
            )
            prefix_row = torch.zeros(
                (layout.packed.prefix_tiles,), dtype=torch.int32
            )
            prefix_row[: len(prefix)] = torch.tensor(prefix, dtype=torch.int32)
            dense_row = torch.zeros(
                (layout.packed.prefix_tiles + layout.packed.video_tiles,),
                dtype=torch.int32,
            )
            dense_row[: len(dense)] = torch.tensor(dense, dtype=torch.int32)
            prefix_rows.append(prefix_row)
            dense_rows.append(dense_row)
            prefix_counts.append(len(prefix))
        self.register_buffer(
            "base_tile_valid_sizes",
            layout.packed.tile_valid_sizes.to(mesh.local_device),
            persistent=False,
        )
        self.register_buffer(
            "tile_row_offsets",
            torch.arange(64, dtype=torch.int32, device=mesh.local_device).view(1, 64),
            persistent=False,
        )
        self.register_buffer(
            "prompt_prefix_indices",
            torch.stack(prefix_rows).to(mesh.local_device),
            persistent=False,
        )
        self.register_buffer(
            "prompt_dense_indices",
            torch.stack(dense_rows).to(mesh.local_device),
            persistent=False,
        )
        self.register_buffer(
            "prompt_prefix_counts",
            torch.tensor(
                prefix_counts, dtype=torch.int32, device=mesh.local_device
            ),
            persistent=False,
        )
        if int(max_state_slots) < 2:
            raise ValueError("the FastH3 serving topology requires at least two state slots")
        torch.cuda.empty_cache()
        free_bytes, _total_bytes = torch.cuda.mem_get_info(mesh.local_device)
        state_slots = min(
            int(max_state_slots),
            int(free_bytes) // H3StatePool.bytes_per_slot(layout),
        )
        if state_slots < 2:
            raise RuntimeError("MiniMax H3 has insufficient CUDA memory for two state slots")
        self.states = H3StatePool(layout, state_slots, mesh.local_device)

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: str,
        mesh: DeviceMesh,
        *,
        max_state_slots: int,
        cache_dir: str | None = None,
        revision: str | None = None,
        attention_mode: Literal[
            "sparse_kernel", "sparse_oracle", "dense_oracle"
        ] = "sparse_kernel",
    ) -> "MiniMaxH3Model":
        layout = H3Layout.build(mesh)
        components = load_h3_components(
            checkpoint,
            mesh,
            layout,
            cache_dir=cache_dir,
            revision=revision,
            attention_mode=attention_mode,
        )
        return cls(mesh, components, layout, max_state_slots=max_state_slots)

    def _token_ids(self, prompt: str) -> torch.Tensor:
        token_error: BaseException | None = None
        if self.mesh.coord("tp") == 0:
            host = self.prompt_host
            if host is None:
                raise RuntimeError("rank zero has no H3 prompt staging buffer")
            try:
                tokenizer = self.tokenizer
                if tokenizer is None:
                    raise RuntimeError("rank zero has no H3 tokenizer")
                encoded: Any = tokenizer(prompt, add_special_tokens=False)
                values = (
                    encoded["input_ids"]
                    if isinstance(encoded, dict)
                    else encoded.input_ids
                )
                if values and isinstance(values[0], list):
                    if len(values) != 1:
                        raise ValueError(
                            "H3 presentation produced more than one token sequence"
                        )
                    values = values[0]
                token_ids = tuple(int(value) for value in values)
                if not 1 <= len(token_ids) <= self.layout.packed.text_indices.numel():
                    raise ValueError(
                        "H3 prompt must encode to between 1 and 1024 tokens"
                    )
                host[0] = len(token_ids)
                host[1 : 1 + len(token_ids)].copy_(torch.tensor(token_ids))
            except BaseException as error:
                host[0] = 0
                token_error = error
            self.prompt_device.copy_(host, non_blocking=True)
        self.mesh.broadcast(self.prompt_device, src=0, group="tp")
        count = int(self.prompt_device[0].item())
        if token_error is not None:
            raise token_error
        if not 1 <= count <= self.layout.packed.text_indices.numel():
            raise ValueError("H3 prompt tokenization failed on rank zero")
        return self.prompt_device[1 : 1 + count].view(1, -1)

    def _prepare_tile_metadata(self, slot: H3StateSlot, text_rows: int) -> None:
        packed = self.layout.packed
        valid = slot.tile_valid_sizes
        text_tiles = int(packed.text_indices.numel()) // 64
        full_tiles, remaining = divmod(int(text_rows), 64)
        valid.copy_(self.base_tile_valid_sizes)
        valid[:text_tiles].zero_()
        if full_tiles:
            valid[:full_tiles].fill_(64)
        if remaining:
            valid[full_tiles].fill_(remaining)
        torch.lt(
            self.tile_row_offsets,
            valid.view(-1, 1),
            out=slot.row_valid_mask.view(-1, 64),
        )
        table_row = (int(text_rows) - 1) // 64
        slot.prefix_key_indices.copy_(self.prompt_prefix_indices[table_row])
        slot.dense_key_indices.copy_(self.prompt_dense_indices[table_row])
        slot.prefix_count.copy_(self.prompt_prefix_counts[table_row])

    def _prepare_rotary(self, slot: H3StateSlot, text_rows: int) -> None:
        transformer = self.transformer
        positions = self.scratch.rotary_positions
        positions.copy_(transformer.positions)
        non_text_start = int(self.layout.packed.text_indices.numel())
        positions[non_text_start:, 0].add_(
            int(text_rows) - int(self.layout.packed.text_indices.numel())
        )
        transformer.rope.forward_into(
            positions,
            slot.rotary_cosine,
            slot.rotary_sine,
            self.scratch.rotary_frequencies,
        )

    @torch.inference_mode()
    def prepare(self, slot: H3StateSlot, admission: NewRequest) -> None:
        if slot.active:
            raise RuntimeError("H3 state slot is already active")
        media = admission.media
        if media is None or media.profile is not MediaProfileId.MINIMAX_H3_T2VA:
            raise ValueError("the H3 worker received an incompatible media profile")
        # Diffusers' CPU-generator path draws the full video tensor first and
        # then the audio rows. Reproducing that order preserves the public seed.
        generator = torch.Generator(device="cpu").manual_seed(int(media.seed))
        video_noise = torch.empty(
            (1, 24, 37, 48, 84),
            dtype=torch.float32,
            pin_memory=True,
        )
        video_noise.normal_(generator=generator)
        raster_rows = patchify_video(video_noise)[0]
        tiled_rows = raster_rows.index_select(0, self.layout.packed.video_raster_indices)
        video_owned = (self.layout.packed.video_indices >= self.layout.local_start) & (
            self.layout.packed.video_indices < self.layout.local_end
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
        audio_noise = torch.empty(
            (414, 32), dtype=torch.float32, pin_memory=True
        )
        audio_noise.normal_(generator=generator)
        audio_owned = (self.layout.packed.audio_indices >= self.layout.local_start) & (
            self.layout.packed.audio_indices < self.layout.local_end
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

        token_ids = self._token_ids(media.prompt)
        encoded = self.encoder(token_ids)
        refined = self.transformer.refine_text(encoded)
        slot.text_condition.zero_()
        slot.text_condition[:, : refined.shape[1]].copy_(refined)
        self._prepare_tile_metadata(slot, int(refined.shape[1]))
        self._prepare_rotary(slot, int(refined.shape[1]))
        torch.cuda.current_stream(self.mesh.local_device).wait_stream(
            self.preparation_stream
        )
        if slot.video_overlap is not None:
            slot.video_overlap.zero_()
        slot.request_key = admission.request_key
        slot.denoise_step = 0
        slot.next_video_unit = 0
        slot.audio_decoded = False

    @torch.inference_mode()
    def denoise(self, slot: H3StateSlot, start_step: int, step_count: int) -> None:
        if not slot.active:
            raise RuntimeError("H3 denoise references an inactive state slot")
        if int(step_count) != 1 or int(start_step) != slot.denoise_step:
            raise ValueError("H3 denoise placements must advance exactly one current step")
        if not 0 <= start_step < 4:
            raise ValueError("H3 denoise step is outside the four-evaluation ladder")
        schedule = self.layout.schedule
        video_velocity, audio_velocity = self.transformer.forward_local(
            slot,
            self.scratch,
            video_timestep=schedule.video_timesteps[start_step],
            audio_timestep=schedule.audio_timesteps[start_step],
        )
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
        slot.denoise_step += 1

    def _gather_video_span(self, slot: H3StateSlot, unit: int) -> torch.Tensor:
        start_frame = unit * 5
        frame_count = 7
        rows_per_frame = 24 * 42
        start_row = start_frame * rows_per_frame
        stop_row = (start_frame + frame_count) * rows_per_frame
        scratch = self.scratch
        rows = scratch.decode_rows[: frame_count * rows_per_frame]
        rows.zero_()
        raster = scratch.local_video_raster
        selected = (raster >= start_row) & (raster < stop_row)
        rows.index_copy_(
            0,
            raster[selected] - start_row,
            slot.video_rows[selected],
        )
        local = scratch.gather_input
        unpatchify_video_into(
            rows,
            local.unsqueeze(0),
            frames=frame_count,
            height=48,
            width=84,
        )
        gathered = scratch.gather
        with profile_range(
            f"uniserve.h3.collective kind=video_latent_gather rank={self.layout.sp_rank}"
        ):
            self.mesh.all_gather_into_tensor(
                gathered.reshape(self.layout.sp_size * 24, frame_count, 48, 84),
                local,
                "sp",
            )
        torch.sum(gathered, dim=0, out=local)
        return local.unsqueeze(0)

    @torch.inference_mode()
    def decode_video(self, slot: H3StateSlot, placement: DecodePlacement) -> torch.Tensor | None:
        unit = int(placement.start_unit)
        if (
            placement.kind is not DecodeKind.VIDEO
            or placement.unit_count != 1
            or unit != slot.next_video_unit
            or not 0 <= unit < self.layout.video_decode_units
        ):
            raise ValueError("invalid H3 video decode placement")
        latents = self._gather_video_span(slot, unit)
        owner = unit % self.layout.sp_size
        scratch = self.scratch
        scratch.rgb_send.zero_()
        scratch.overlap_send.zero_()
        valid_frames = self.layout.decode_unit_frames[unit]
        if self.layout.sp_rank == owner:
            rgb, overlap = self.video_vae.decode_unit(
                latents,
                unit,
                None if unit == 0 else slot.video_overlap,
                final_unit=unit + 1 == self.layout.video_decode_units,
            )
            scratch.rgb_send[:valid_frames].copy_(rgb)
            scratch.overlap_send.copy_(overlap)
        with profile_range(
            f"uniserve.h3.collective kind=video_overlap_gather rank={self.layout.sp_rank}"
        ):
            self.mesh.all_gather_into_tensor(
                scratch.overlap_gather.reshape(
                    self.layout.sp_size, 3, 5, 768, 1344
                ),
                scratch.overlap_send.reshape(1, 3, 5, 768, 1344),
                "sp",
            )
        if slot.video_overlap is None:
            raise RuntimeError("H3 state slot lost its persistent VAE overlap")
        slot.video_overlap.copy_(scratch.overlap_gather[owner])
        with profile_range(
            f"uniserve.h3.collective kind=video_rgb_gather rank={self.layout.sp_rank}"
        ):
            self.mesh.all_gather_into_tensor(
                scratch.rgb_gather.reshape(
                    self.layout.sp_size * 22, 768, 1344, 3
                ),
                scratch.rgb_send,
                "sp",
            )
        slot.next_video_unit += 1
        return scratch.rgb_gather[owner, :valid_frames] if self.layout.sp_rank == 0 else None

    @torch.inference_mode()
    def decode_audio(self, slot: H3StateSlot, placement: DecodePlacement) -> torch.Tensor | None:
        if (
            placement.kind is not DecodeKind.AUDIO
            or placement.start_unit != 0
            or placement.unit_count != 1
            or slot.audio_decoded
        ):
            raise ValueError("invalid H3 audio decode placement")
        scratch = self.scratch
        scratch.audio_input.zero_()
        raster = scratch.local_audio_raster
        scratch.audio_input.index_copy_(0, raster, slot.audio_rows)
        with profile_range(
            f"uniserve.h3.collective kind=audio_latent_gather rank={self.layout.sp_rank}"
        ):
            self.mesh.all_gather_into_tensor(
                scratch.audio_gather.reshape(self.layout.sp_size * 414, 32),
                scratch.audio_input,
                "sp",
            )
        torch.sum(scratch.audio_gather, dim=0, out=scratch.audio_input)
        slot.audio_decoded = True
        if self.layout.sp_rank != 0:
            return None
        if self.audio_vae is None:
            raise RuntimeError("rank zero has no resident H3 audio VAE")
        scratch.audio_latents.copy_(
            scratch.audio_input.view(2, 207, 32).permute(0, 2, 1)
        )
        pcm = self.audio_vae.decode(scratch.audio_latents)
        target_samples = round(124 * 32_000 / 24)
        if pcm.shape[0] < target_samples:
            raise RuntimeError("H3 audio decoder returned less than the fixed video duration")
        return pcm[:target_samples]

    @torch.inference_mode()
    def warmup(self) -> None:
        self.transformer.compile_blocks()
        self.video_vae.compile_decoder()
        slot = self.states.slots[0]
        slot.text_condition.zero_()
        slot.video_rows.zero_()
        slot.audio_rows.zero_()
        self._prepare_tile_metadata(
            slot,
            int(self.layout.packed.text_indices.numel()),
        )
        self._prepare_rotary(
            slot,
            int(self.layout.packed.text_indices.numel()),
        )
        schedule = self.layout.schedule
        self.transformer.forward_local(
            slot,
            self.scratch,
            video_timestep=schedule.video_timesteps[0],
            audio_timestep=schedule.audio_timesteps[0],
        )
        video_latents = torch.zeros(
            (1, 24, 7, 48, 84),
            dtype=torch.float32,
            device=self.mesh.local_device,
        )
        self.video_vae.decode_unit(
            video_latents,
            0,
            None,
            final_unit=False,
        )
        if self.audio_vae is not None:
            audio_latents = torch.zeros(
                (2, 32, 207),
                dtype=torch.float32,
                device=self.mesh.local_device,
            )
            self.audio_vae.decode(audio_latents)
        torch.cuda.synchronize(self.mesh.local_device)
        slot.clear()
