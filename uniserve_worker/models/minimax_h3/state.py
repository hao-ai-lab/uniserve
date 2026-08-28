"""Persistent request state and the single shared H3 execution scratch lane."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from ...execution.batch import RequestKey
from ...nn.mesh import DeviceMesh
from .packing import H3PackedLayout, build_packed_layout
from .schedule import H3Schedule

__all__ = ["H3Layout", "H3Scratch", "H3StatePool", "H3StateSlot"]

PROFILE_HEIGHT = 768
PROFILE_WIDTH = 1344
PROFILE_FRAMES = 124
PROFILE_FPS = 24
PROFILE_AUDIO_RATE = 32_000
VIDEO_DECODE_UNITS = 7
VIDEO_UNIT_FRAMES = (17, 17, 17, 17, 17, 17, 22)


@dataclass(frozen=True, slots=True)
class H3Layout:
    packed: H3PackedLayout
    schedule: H3Schedule
    sp_rank: int
    sp_size: int
    local_start: int
    local_end: int
    decode_unit_frames: tuple[int, ...] = VIDEO_UNIT_FRAMES

    @classmethod
    def build(
        cls,
        mesh: DeviceMesh,
        *,
        text_rows: int = 1024,
    ) -> "H3Layout":
        size = mesh.size("sp")
        rank = mesh.coord("sp")
        packed = build_packed_layout(text_rows=text_rows, row_multiple=64 * size)
        shard = packed.padded_rows // size
        return cls(
            packed=packed,
            schedule=H3Schedule.build(mesh.local_device),
            sp_rank=rank,
            sp_size=size,
            local_start=rank * shard,
            local_end=(rank + 1) * shard,
        )

    @property
    def local_rows(self) -> int:
        return self.local_end - self.local_start

    @property
    def video_decode_units(self) -> int:
        return len(self.decode_unit_frames)

    @property
    def persistent_units(self) -> int:
        # Scheduler capacity token: fixed bytes rounded to one MiB.
        video = self.packed.video_indices.numel() * 96 * 4
        audio = self.packed.audio_indices.numel() * 32 * 4
        text = self.packed.text_indices.numel() * 5120 * 2
        return (video + audio + text + (1 << 20) - 1) // (1 << 20)

    def local_indices(self, indices: torch.Tensor) -> torch.Tensor:
        selected = indices[(indices >= self.local_start) & (indices < self.local_end)]
        return selected - self.local_start

    @property
    def local_video_rows(self) -> int:
        return int(self.local_indices(self.packed.video_indices).numel())

    @property
    def local_audio_rows(self) -> int:
        return int(self.local_indices(self.packed.audio_indices).numel())

    @property
    def local_video_raster_indices(self) -> torch.Tensor:
        selected = (self.packed.video_indices >= self.local_start) & (
            self.packed.video_indices < self.local_end
        )
        return self.packed.video_raster_indices[selected]

    @property
    def local_audio_raster_indices(self) -> torch.Tensor:
        selected = (self.packed.audio_indices >= self.local_start) & (
            self.packed.audio_indices < self.local_end
        )
        return torch.arange(self.packed.audio_indices.numel(), dtype=torch.long)[selected]


@dataclass(slots=True)
class H3StateSlot:
    index: int
    text_condition: torch.Tensor
    video_rows: torch.Tensor
    audio_rows: torch.Tensor
    tile_valid_sizes: torch.Tensor
    prefix_key_indices: torch.Tensor
    dense_key_indices: torch.Tensor
    prefix_count: torch.Tensor
    row_valid_mask: torch.Tensor
    rotary_cosine: torch.Tensor
    rotary_sine: torch.Tensor
    request_key: RequestKey | None = None
    denoise_step: int = 0
    next_video_unit: int = 0
    audio_decoded: bool = False
    video_overlap: torch.Tensor | None = None

    @property
    def active(self) -> bool:
        return self.request_key is not None

    def clear(self) -> None:
        self.request_key = None
        self.denoise_step = 0
        self.next_video_unit = 0
        self.audio_decoded = False
        if self.video_overlap is not None:
            self.video_overlap.zero_()


class H3StatePool:
    def __init__(self, layout: H3Layout, slot_count: int, device: torch.device) -> None:
        if slot_count < 2:
            raise ValueError("the FastH3 serving topology requires at least two state slots")
        self.layout = layout
        text_rows = int(layout.packed.text_indices.numel())
        video_rows = layout.local_video_rows
        audio_rows = layout.local_audio_rows
        self.slots = tuple(
            H3StateSlot(
                index=index,
                text_condition=torch.empty(
                    (1, text_rows, 5376), dtype=torch.bfloat16, device=device
                ),
                video_rows=torch.empty((video_rows, 96), dtype=torch.float32, device=device),
                audio_rows=torch.empty((audio_rows, 32), dtype=torch.float32, device=device),
                tile_valid_sizes=layout.packed.tile_valid_sizes.to(device).clone(),
                prefix_key_indices=torch.zeros(
                    (layout.packed.prefix_tiles,), dtype=torch.int32, device=device
                ),
                dense_key_indices=torch.zeros(
                    (layout.packed.prefix_tiles + layout.packed.video_tiles,),
                    dtype=torch.int32,
                    device=device,
                ),
                prefix_count=torch.zeros((), dtype=torch.int32, device=device),
                row_valid_mask=torch.empty(
                    (layout.packed.padded_rows,), dtype=torch.bool, device=device
                ),
                rotary_cosine=torch.empty(
                    (layout.packed.padded_rows, 96), dtype=torch.float32, device=device
                ),
                rotary_sine=torch.empty(
                    (layout.packed.padded_rows, 96), dtype=torch.float32, device=device
                ),
                video_overlap=torch.empty((1, 3, 5, 768, 1344), dtype=torch.float16, device=device),
            )
            for index in range(1, slot_count + 1)
        )

    @staticmethod
    def bytes_per_slot(layout: H3Layout) -> int:
        """Return the exact persistent CUDA tensor bytes owned by one state slot."""

        packed = layout.packed
        text_rows = int(packed.text_indices.numel())
        video_rows = int(layout.local_video_rows)
        audio_rows = int(layout.local_audio_rows)
        padded_rows = int(packed.padded_rows)
        return sum(
            (
                text_rows * 5376 * 2,
                video_rows * 96 * 4,
                audio_rows * 32 * 4,
                int(packed.tile_valid_sizes.numel()) * 4,
                int(packed.prefix_tiles) * 4,
                (int(packed.prefix_tiles) + int(packed.video_tiles)) * 4,
                4,
                padded_rows,
                padded_rows * 96 * 4 * 2,
                3 * 5 * PROFILE_HEIGHT * PROFILE_WIDTH * 2,
            )
        )

    @property
    def slot_count(self) -> int:
        return len(self.slots)

    def get(self, index: int) -> H3StateSlot:
        if index < 1 or index > len(self.slots):
            raise ValueError(f"H3 request-pool index {index} is outside resident capacity")
        return self.slots[index - 1]

    def drop_session(self, session_id: int) -> None:
        for slot in self.slots:
            if slot.request_key is not None and slot.request_key.session_id == int(session_id):
                slot.clear()


@dataclass(slots=True)
class H3Scratch:
    packed_hidden: torch.Tensor
    local_text_hidden: torch.Tensor
    projected_input: torch.Tensor
    projected_input_bf16: torch.Tensor
    local_video_hidden: torch.Tensor
    local_audio_hidden: torch.Tensor
    video_velocity: torch.Tensor
    audio_velocity: torch.Tensor
    projection_buffer: torch.Tensor
    qkvg_send: torch.Tensor
    qkvg_exchange: torch.Tensor
    attention_output: torch.Tensor
    tile_scores: torch.Tensor
    block_indices: torch.Tensor
    block_counts: torch.Tensor
    pooled_query: torch.Tensor
    pooled_key: torch.Tensor
    pooled_value: torch.Tensor
    compressed_tiles: torch.Tensor
    topk_values: torch.Tensor
    topk_indices: torch.Tensor
    topk_indices_i32: torch.Tensor
    gather: torch.Tensor
    gather_input: torch.Tensor
    decode_rows: torch.Tensor
    audio_gather: torch.Tensor
    audio_input: torch.Tensor
    rgb_send: torch.Tensor
    rgb_gather: torch.Tensor
    overlap_send: torch.Tensor
    overlap_gather: torch.Tensor
    local_video_raster: torch.Tensor
    local_audio_raster: torch.Tensor
    audio_latents: torch.Tensor
    time_values: torch.Tensor
    rotary_positions: torch.Tensor
    rotary_frequencies: torch.Tensor

    @classmethod
    def allocate(cls, layout: H3Layout, device: torch.device) -> "H3Scratch":
        local_rows = layout.local_rows
        local_heads = 56 // layout.sp_size
        global_rows = layout.packed.padded_rows
        tile_count = (global_rows + 63) // 64
        keep_video_tiles = max(1, math.ceil(0.1 * layout.packed.video_tiles))
        local_text = int(layout.local_indices(layout.packed.text_indices).numel())
        local_video = layout.local_video_rows
        local_audio = layout.local_audio_rows
        max_projected_rows = max(local_video, local_audio)
        return cls(
            packed_hidden=torch.empty((1, local_rows, 5376), dtype=torch.bfloat16, device=device),
            local_text_hidden=torch.empty(
                (1, local_text, 5376), dtype=torch.bfloat16, device=device
            ),
            projected_input=torch.empty(
                (max_projected_rows, 5376), dtype=torch.float32, device=device
            ),
            projected_input_bf16=torch.empty(
                (max_projected_rows, 5376), dtype=torch.bfloat16, device=device
            ),
            local_video_hidden=torch.empty(
                (local_video, 5376), dtype=torch.bfloat16, device=device
            ),
            local_audio_hidden=torch.empty(
                (local_audio, 5376), dtype=torch.bfloat16, device=device
            ),
            video_velocity=torch.empty((local_video, 96), dtype=torch.float32, device=device),
            audio_velocity=torch.empty((local_audio, 32), dtype=torch.float32, device=device),
            projection_buffer=torch.empty(
                (local_rows, 56, 128), dtype=torch.bfloat16, device=device
            ),
            qkvg_send=torch.empty(
                (layout.sp_size, local_rows, local_heads, 4, 128),
                dtype=torch.bfloat16,
                device=device,
            ),
            qkvg_exchange=torch.empty(
                (global_rows, local_heads, 4, 128), dtype=torch.bfloat16, device=device
            ),
            attention_output=torch.empty(
                (global_rows, local_heads, 128), dtype=torch.bfloat16, device=device
            ),
            tile_scores=torch.empty(
                (local_heads, tile_count, tile_count), dtype=torch.float32, device=device
            ),
            block_indices=torch.empty(
                (
                    local_heads,
                    tile_count,
                    layout.packed.prefix_tiles + layout.packed.video_tiles,
                ),
                dtype=torch.int32,
                device=device,
            ),
            block_counts=torch.empty((local_heads, tile_count), dtype=torch.int32, device=device),
            pooled_query=torch.empty(
                (tile_count, local_heads, 128), dtype=torch.float32, device=device
            ),
            pooled_key=torch.empty(
                (tile_count, local_heads, 128), dtype=torch.float32, device=device
            ),
            pooled_value=torch.empty(
                (tile_count, local_heads, 128), dtype=torch.float32, device=device
            ),
            compressed_tiles=torch.empty(
                (local_heads, tile_count, 128), dtype=torch.float32, device=device
            ),
            topk_values=torch.empty(
                (local_heads, layout.packed.video_tiles, keep_video_tiles),
                dtype=torch.float32,
                device=device,
            ),
            topk_indices=torch.empty(
                (local_heads, layout.packed.video_tiles, keep_video_tiles),
                dtype=torch.long,
                device=device,
            ),
            topk_indices_i32=torch.empty(
                (local_heads, layout.packed.video_tiles, keep_video_tiles),
                dtype=torch.int32,
                device=device,
            ),
            gather=torch.empty((layout.sp_size, 24, 7, 48, 84), dtype=torch.float32, device=device),
            gather_input=torch.empty((24, 7, 48, 84), dtype=torch.float32, device=device),
            decode_rows=torch.empty((7 * 24 * 42, 96), dtype=torch.float32, device=device),
            audio_gather=torch.empty((layout.sp_size, 414, 32), dtype=torch.float32, device=device),
            audio_input=torch.empty((414, 32), dtype=torch.float32, device=device),
            rgb_send=torch.empty((22, 768, 1344, 3), dtype=torch.uint8, device=device),
            rgb_gather=torch.empty(
                (layout.sp_size, 22, 768, 1344, 3),
                dtype=torch.uint8,
                device=device,
            ),
            overlap_send=torch.empty((1, 3, 5, 768, 1344), dtype=torch.float16, device=device),
            overlap_gather=torch.empty(
                (layout.sp_size, 1, 3, 5, 768, 1344),
                dtype=torch.float16,
                device=device,
            ),
            local_video_raster=layout.local_video_raster_indices.to(device),
            local_audio_raster=layout.local_audio_raster_indices.to(device),
            audio_latents=torch.empty((2, 32, 207), dtype=torch.float32, device=device),
            time_values=torch.empty((2,), dtype=torch.float32, device=device),
            rotary_positions=torch.empty((global_rows, 3), dtype=torch.float32, device=device),
            rotary_frequencies=torch.empty(
                (global_rows, 3, 16), dtype=torch.float32, device=device
            ),
        )
