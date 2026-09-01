"""Persistent request state and the single shared H3 execution scratch lane."""

from __future__ import annotations

import math
from dataclasses import dataclass, fields

import torch

from ...backends.attention.video_sparse import video_sparse_selected_tiles
from ...execution.batch import RequestKey
from ...execution.bounded_storage import BoundedTensorStorage
from ...nn.mesh import DeviceMesh, SymmetricMemoryWorkspace
from .packing import H3PackedLayout, build_packed_layout
from .schedule import H3Schedule

__all__ = [
    "H3Layout",
    "H3Scratch",
    "H3StatePool",
    "H3StateSlot",
    "MIN_H3_FRAMES",
]

PROFILE_HEIGHT = 768
PROFILE_WIDTH = 1344
PROFILE_FPS = 24
PROFILE_AUDIO_RATE = 32_000
FASTH3_STEPS = 4
VIDEO_ROUND_UNITS = 4
MIN_H3_FRAMES = 22


def _reconstruction_unit_frames(frames: int) -> tuple[int, ...]:
    if frames < MIN_H3_FRAMES or frames % 17 != 5:
        raise ValueError("H3 frame count must have the form 17 * n + 5")
    units = (frames - 5) // 17
    return (17,) * (units - 1) + (MIN_H3_FRAMES,)


@dataclass(frozen=True, slots=True)
class H3Layout:
    packed: H3PackedLayout
    schedule: H3Schedule
    sp_rank: int
    sp_size: int
    local_start: int
    local_end: int
    frame_count: int
    reconstruction_unit_frames: tuple[int, ...]

    @classmethod
    def build(
        cls,
        mesh: DeviceMesh,
        *,
        frames: int,
        text_rows: int,
        audio_frames: int,
        schedule: H3Schedule | None = None,
    ) -> "H3Layout":
        size = mesh.size("sp")
        rank = mesh.coord("sp")
        packed = build_packed_layout(
            text_rows=text_rows,
            num_frames=frames,
            audio_frames=audio_frames,
            row_multiple=64 * size,
        )
        shard = packed.padded_rows // size
        return cls(
            packed=packed,
            schedule=H3Schedule.build(mesh.local_device) if schedule is None else schedule,
            sp_rank=rank,
            sp_size=size,
            local_start=rank * shard,
            local_end=(rank + 1) * shard,
            frame_count=int(frames),
            reconstruction_unit_frames=_reconstruction_unit_frames(int(frames)),
        )

    @property
    def shape_key(self) -> tuple[int, int, int]:
        return (
            self.frame_count,
            int(self.packed.text_indices.numel()),
            int(self.packed.audio_frames),
        )

    @property
    def video_round_frames(self) -> int:
        return max(
            sum(self.reconstruction_unit_frames[start : start + VIDEO_ROUND_UNITS])
            for start in range(0, self.video_reconstruction_units, VIDEO_ROUND_UNITS)
        )

    @property
    def local_rows(self) -> int:
        return self.local_end - self.local_start

    @property
    def video_reconstruction_units(self) -> int:
        return len(self.reconstruction_unit_frames)

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


@dataclass(frozen=True, slots=True)
class _H3StateBuffers:
    text_condition: torch.Tensor
    video_rows: torch.Tensor
    audio_rows: torch.Tensor
    tile_valid_sizes: torch.Tensor
    prefix_key_indices: torch.Tensor
    dense_key_indices: torch.Tensor
    prefix_count: torch.Tensor
    rotary_cosine: torch.Tensor
    rotary_sine: torch.Tensor
    block_adaln_plan: torch.Tensor
    final_adaln_plan: torch.Tensor
    video_overlap: torch.Tensor | None


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
    rotary_cosine: torch.Tensor
    rotary_sine: torch.Tensor
    block_adaln_plan: torch.Tensor
    final_adaln_plan: torch.Tensor
    video_overlap: torch.Tensor | None
    bounded: BoundedTensorStorage
    request_key: RequestKey | None = None
    shape_key: tuple[int, int, int] | None = None
    denoise_step: int = 0
    next_video_unit: int = 0
    audio_reconstructed: bool = False

    @property
    def active(self) -> bool:
        return self.request_key is not None

    def bind(self, layout: H3Layout) -> None:
        text_rows = int(layout.packed.text_indices.numel())
        if self.active and self.shape_key != layout.shape_key:
            raise RuntimeError("an active H3 state slot cannot change its execution layout")
        capacity = self.bounded.capacity
        views = self.bounded.bind(
            layout.shape_key,
            {
                "text_condition": (1, text_rows, 5376),
                "video_rows": (layout.local_video_rows, 96),
                "audio_rows": (layout.local_audio_rows, 32),
                "tile_valid_sizes": (int(layout.packed.tile_valid_sizes.numel()),),
                "prefix_key_indices": (int(layout.packed.prefix_tiles),),
                "dense_key_indices": (int(layout.packed.prefix_tiles + layout.packed.video_tiles),),
                "prefix_count": (),
                "rotary_cosine": (layout.packed.padded_rows, 96),
                "rotary_sine": (layout.packed.padded_rows, 96),
                "block_adaln_plan": tuple(capacity["block_adaln_plan"].shape),
                "final_adaln_plan": tuple(capacity["final_adaln_plan"].shape),
                "video_overlap": tuple(capacity["video_overlap"].shape),
            },
        )
        buffers = _H3StateBuffers(**views)
        self.text_condition = buffers.text_condition
        self.video_rows = buffers.video_rows
        self.audio_rows = buffers.audio_rows
        self.tile_valid_sizes = buffers.tile_valid_sizes
        self.prefix_key_indices = buffers.prefix_key_indices
        self.dense_key_indices = buffers.dense_key_indices
        self.prefix_count = buffers.prefix_count
        self.rotary_cosine = buffers.rotary_cosine
        self.rotary_sine = buffers.rotary_sine
        self.block_adaln_plan = buffers.block_adaln_plan
        self.final_adaln_plan = buffers.final_adaln_plan
        self.video_overlap = buffers.video_overlap
        self.shape_key = layout.shape_key

    def clear(self) -> None:
        self.request_key = None
        self.shape_key = None
        self.denoise_step = 0
        self.next_video_unit = 0
        self.audio_reconstructed = False
        if self.video_overlap is not None:
            self.video_overlap.zero_()


class H3StatePool:
    def __init__(
        self,
        layout: H3Layout,
        slot_count: int,
        device: torch.device,
        *,
        block_plan_shape: tuple[int, ...],
        final_plan_shape: tuple[int, ...],
    ) -> None:
        if slot_count < 2:
            raise ValueError("the FastH3 serving topology requires at least two state slots")
        self.layout = layout
        self.slots = tuple(
            self._allocate_slot(
                index,
                layout,
                device,
                block_plan_shape=block_plan_shape,
                final_plan_shape=final_plan_shape,
            )
            for index in range(1, slot_count + 1)
        )

    @staticmethod
    def _allocate_slot(
        index: int,
        layout: H3Layout,
        device: torch.device,
        *,
        block_plan_shape: tuple[int, ...],
        final_plan_shape: tuple[int, ...],
    ) -> H3StateSlot:
        text_capacity = int(layout.packed.text_indices.numel())
        video_capacity = min(int(layout.packed.video_indices.numel()), layout.local_rows)
        audio_capacity = min(int(layout.packed.audio_indices.numel()), layout.local_rows)
        tile_capacity = int(layout.packed.tile_valid_sizes.numel())
        prefix_capacity = int(layout.packed.prefix_tiles)
        dense_capacity = int(layout.packed.prefix_tiles + layout.packed.video_tiles)
        row_capacity = int(layout.packed.padded_rows)
        text_condition = torch.empty((1, text_capacity, 5376), dtype=torch.bfloat16, device=device)
        video_rows = torch.empty((video_capacity, 96), dtype=torch.float32, device=device)
        audio_rows = torch.empty((audio_capacity, 32), dtype=torch.float32, device=device)
        tile_valid_sizes = torch.empty((tile_capacity,), dtype=torch.int32, device=device)
        prefix_key_indices = torch.empty((prefix_capacity,), dtype=torch.int32, device=device)
        dense_key_indices = torch.empty((dense_capacity,), dtype=torch.int32, device=device)
        prefix_count = torch.zeros((), dtype=torch.int32, device=device)
        rotary_cosine = torch.empty((row_capacity, 96), dtype=torch.float32, device=device)
        rotary_sine = torch.empty((row_capacity, 96), dtype=torch.float32, device=device)
        block_adaln_plan = torch.empty(
            block_plan_shape,
            dtype=torch.bfloat16,
            device=device,
        )
        final_adaln_plan = torch.empty(
            final_plan_shape,
            dtype=torch.bfloat16,
            device=device,
        )
        video_overlap = torch.empty(
            (1, 3, 5, PROFILE_HEIGHT, PROFILE_WIDTH),
            dtype=torch.float16,
            device=device,
        )
        tensors = locals()
        bounded = BoundedTensorStorage(
            {
                name: tensors[name]
                for name in _H3StateBuffers.__dataclass_fields__
                if name != "video_overlap"
            }
            | {"video_overlap": video_overlap}
        )
        slot = H3StateSlot(
            index=index,
            text_condition=text_condition,
            video_rows=video_rows,
            audio_rows=audio_rows,
            tile_valid_sizes=tile_valid_sizes,
            prefix_key_indices=prefix_key_indices,
            dense_key_indices=dense_key_indices,
            prefix_count=prefix_count,
            rotary_cosine=rotary_cosine,
            rotary_sine=rotary_sine,
            block_adaln_plan=block_adaln_plan,
            final_adaln_plan=final_adaln_plan,
            video_overlap=video_overlap,
            bounded=bounded,
        )
        slot.bind(layout)
        slot.shape_key = None
        return slot

    @staticmethod
    def bytes_per_slot(
        layout: H3Layout,
        *,
        block_plan_shape: tuple[int, ...],
        final_plan_shape: tuple[int, ...],
    ) -> int:
        """Return the exact persistent CUDA tensor bytes owned by one state slot."""

        text_rows = int(layout.packed.text_indices.numel())
        video_rows = min(int(layout.packed.video_indices.numel()), layout.local_rows)
        audio_rows = min(int(layout.packed.audio_indices.numel()), layout.local_rows)
        tile_count = int(layout.packed.tile_valid_sizes.numel())
        prefix_tiles = int(layout.packed.prefix_tiles)
        dense_tiles = int(layout.packed.prefix_tiles + layout.packed.video_tiles)
        padded_rows = int(layout.packed.padded_rows)
        return sum(
            (
                text_rows * 5376 * 2,
                video_rows * 96 * 4,
                audio_rows * 32 * 4,
                tile_count * 4,
                prefix_tiles * 4,
                dense_tiles * 4,
                4,
                padded_rows * 96 * 4 * 2,
                math.prod(block_plan_shape) * 2,
                math.prod(final_plan_shape) * 2,
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
    projection_peers: tuple[torch.Tensor, ...]
    projection_exchange: SymmetricMemoryWorkspace
    projection_sync_input: torch.Tensor
    projection_sync_output: torch.Tensor
    attention_workspace: torch.Tensor
    attention_output: torch.Tensor
    tile_scores: torch.Tensor
    block_indices: torch.Tensor
    block_counts: torch.Tensor
    pooled_query: torch.Tensor
    pooled_key: torch.Tensor
    pooled_value: torch.Tensor
    compressed_tiles: torch.Tensor
    topk_indices_i32: torch.Tensor
    latent_send: torch.Tensor
    latent_exchange: torch.Tensor
    latent_input: torch.Tensor
    reconstruction_rows: torch.Tensor
    audio_gather: torch.Tensor
    audio_input: torch.Tensor
    segment_placeholder: torch.Tensor
    segment_gather: torch.Tensor | None
    rgb_round: torch.Tensor | None
    local_video_raster: torch.Tensor
    local_audio_raster: torch.Tensor
    audio_latents: torch.Tensor
    block_adaln_params: torch.Tensor
    final_adaln_params: torch.Tensor
    rotary_positions: torch.Tensor
    rotary_frequencies: torch.Tensor
    bounded: BoundedTensorStorage | None

    def view(self, layout: H3Layout) -> "H3Scratch":
        local_rows = int(layout.local_rows)
        global_rows = int(layout.packed.padded_rows)
        local_heads = 56 // layout.sp_size
        tiles = global_rows // 64
        prefix_width = int(layout.packed.prefix_tiles + layout.packed.video_tiles)
        local_text = int(layout.local_indices(layout.packed.text_indices).numel())
        local_video = int(layout.local_video_rows)
        local_audio = int(layout.local_audio_rows)
        projected_rows = max(local_video, local_audio)
        workspace_elements = global_rows * 5376
        if self.bounded is None:
            raise RuntimeError("H3 scratch storage has no bounded-view owner")
        shapes = {name: tuple(tensor.shape) for name, tensor in self.bounded.capacity.items()}
        shapes.update(
            {
                "packed_hidden": (1, local_rows, 5376),
                "local_text_hidden": (1, local_text, 5376),
                "projected_input": (projected_rows, 5376),
                "projected_input_bf16": (projected_rows, 5376),
                "local_video_hidden": (local_video, 5376),
                "local_audio_hidden": (local_audio, 5376),
                "video_velocity": (local_video, 96),
                "audio_velocity": (local_audio, 32),
                "attention_workspace": (workspace_elements,),
                "attention_output": (global_rows, local_heads, 128),
                "tile_scores": (local_heads, tiles, tiles),
                "block_indices": (local_heads, tiles, prefix_width),
                "block_counts": (local_heads, tiles),
                "pooled_query": (tiles, local_heads, 128),
                "pooled_key": (tiles, local_heads, 128),
                "pooled_value": (tiles, local_heads, 128),
                "compressed_tiles": (local_heads, tiles, 128),
                "topk_indices_i32": (
                    local_heads,
                    layout.packed.video_tiles,
                    video_sparse_selected_tiles(layout.packed.video_tiles),
                ),
                "audio_gather": (
                    layout.sp_size,
                    layout.packed.audio_indices.numel(),
                    32,
                ),
                "audio_input": (layout.packed.audio_indices.numel(), 32),
                "audio_latents": (2, 32, layout.packed.audio_frames),
                "rotary_positions": (global_rows, 3),
                "rotary_frequencies": (global_rows, 3, 16),
            }
        )
        if "rgb_round" in shapes:
            shapes["rgb_round"] = (layout.video_round_frames, 768, 1344, 3)
        for rank in range(len(self.projection_peers)):
            shapes[f"projection_peer_{rank}"] = (local_rows, 56, 128)
        views = self.bounded.bind(layout.shape_key, shapes)
        values = {field.name: getattr(self, field.name) for field in fields(self)}
        values.update({name: value for name, value in views.items() if name in values})
        values["projection_peers"] = tuple(
            views[f"projection_peer_{rank}"] for rank in range(len(self.projection_peers))
        )
        values["local_video_raster"] = layout.local_video_raster_indices.to(
            self.packed_hidden.device
        )
        values["local_audio_raster"] = layout.local_audio_raster_indices.to(
            self.packed_hidden.device
        )
        return H3Scratch(**values)

    @classmethod
    def allocate(
        cls,
        layout: H3Layout,
        mesh: DeviceMesh,
        *,
        block_params_shape: tuple[int, ...],
        final_params_shape: tuple[int, ...],
        attention_workspace_dtype: torch.dtype,
    ) -> "H3Scratch":
        device = mesh.local_device
        local_rows = layout.local_rows
        local_heads = 56 // layout.sp_size
        global_rows = layout.packed.padded_rows
        tile_count = (global_rows + 63) // 64
        keep_video_tiles = video_sparse_selected_tiles(layout.packed.video_tiles)
        local_text = min(int(layout.packed.text_indices.numel()), local_rows)
        local_video = min(int(layout.packed.video_indices.numel()), local_rows)
        local_audio = min(int(layout.packed.audio_indices.numel()), local_rows)
        max_projected_rows = max(local_video, local_audio)
        projection_exchange = mesh.symmetric_memory(
            (local_rows, 56, 128),
            dtype=torch.bfloat16,
            group="sp",
            name="video_attention_heads",
        )
        projection_peers = projection_exchange.peers
        scratch = cls(
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
            projection_peers=projection_peers,
            projection_exchange=projection_exchange,
            projection_sync_input=torch.full(
                (1,), layout.sp_rank, dtype=torch.int32, device=device
            ),
            projection_sync_output=torch.empty((layout.sp_size,), dtype=torch.int32, device=device),
            attention_workspace=torch.empty(
                global_rows * 5376,
                dtype=attention_workspace_dtype,
                device=device,
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
            topk_indices_i32=torch.empty(
                (local_heads, layout.packed.video_tiles, keep_video_tiles),
                dtype=torch.int32,
                device=device,
            ),
            latent_send=torch.empty(
                (layout.sp_size, 24, 7, 48, 84), dtype=torch.float32, device=device
            ),
            latent_exchange=torch.empty(
                (layout.sp_size, 24, 7, 48, 84), dtype=torch.float32, device=device
            ),
            latent_input=torch.empty((24, 7, 48, 84), dtype=torch.float32, device=device),
            reconstruction_rows=torch.empty((7 * 24 * 42, 96), dtype=torch.float32, device=device),
            audio_gather=torch.empty(
                (layout.sp_size, layout.packed.audio_indices.numel(), 32),
                dtype=torch.float32,
                device=device,
            ),
            audio_input=torch.empty(
                (layout.packed.audio_indices.numel(), 32), dtype=torch.float32, device=device
            ),
            segment_placeholder=torch.zeros(
                (1, 3, 25, 768, 1344), dtype=torch.float16, device=device
            ),
            segment_gather=(
                torch.empty(
                    (layout.sp_size, 1, 3, 25, 768, 1344),
                    dtype=torch.float16,
                    device=device,
                )
                if layout.sp_rank == 0
                else None
            ),
            rgb_round=(
                torch.empty(
                    (layout.video_round_frames, 768, 1344, 3),
                    dtype=torch.uint8,
                    device=device,
                )
                if layout.sp_rank == 0
                else None
            ),
            local_video_raster=layout.local_video_raster_indices.to(device),
            local_audio_raster=layout.local_audio_raster_indices.to(device),
            audio_latents=torch.empty(
                (2, 32, layout.packed.audio_frames), dtype=torch.float32, device=device
            ),
            block_adaln_params=torch.empty(
                block_params_shape,
                dtype=torch.bfloat16,
                device=device,
            ),
            final_adaln_params=torch.empty(
                final_params_shape,
                dtype=torch.bfloat16,
                device=device,
            ),
            rotary_positions=torch.empty((global_rows, 3), dtype=torch.float32, device=device),
            rotary_frequencies=torch.empty(
                (global_rows, 3, 16), dtype=torch.float32, device=device
            ),
            bounded=None,
        )
        variable_fields = (
            "packed_hidden",
            "local_text_hidden",
            "projected_input",
            "projected_input_bf16",
            "local_video_hidden",
            "local_audio_hidden",
            "video_velocity",
            "audio_velocity",
            "attention_workspace",
            "attention_output",
            "tile_scores",
            "block_indices",
            "block_counts",
            "pooled_query",
            "pooled_key",
            "pooled_value",
            "compressed_tiles",
            "topk_indices_i32",
            "audio_gather",
            "audio_input",
            "audio_latents",
            "rotary_positions",
            "rotary_frequencies",
        )
        tensors = {name: getattr(scratch, name) for name in variable_fields}
        if scratch.rgb_round is not None:
            tensors["rgb_round"] = scratch.rgb_round
        tensors.update(
            {f"projection_peer_{rank}": peer for rank, peer in enumerate(scratch.projection_peers)}
        )
        scratch.bounded = BoundedTensorStorage(tensors)
        return scratch
