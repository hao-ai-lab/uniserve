"""Tensor geometry and mathematical views for H3 computation."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from uniserve_worker.modeling.geometry import TensorOutputLayout

from ...modeling import resources
from ...nn.mesh import DeviceMesh
from ...nn.parallel import ParallelConfig
from ...nn.sparse_attention import video_sparse_selected_tiles
from ...transfer.layout import TensorRegion
from .encoder import H3TextEncoderConfig
from .packing import H3PackedLayout, build_packed_layout

if TYPE_CHECKING:
    from ...nn.video_attention import VideoAttention


__all__ = [
    "H3Layout",
    "tensor_output_layout",
    "MIN_H3_FRAMES",
]

PROFILE_HEIGHT = 768
PROFILE_WIDTH = 1344
PROFILE_FPS = 24
PROFILE_AUDIO_RATE = 32_000
MIN_H3_FRAMES = 22


def reconstruction_unit_frames(frames: int) -> tuple[int, ...]:
    """Convert generated frame count into overlapping decoder reconstruction units."""

    if frames < MIN_H3_FRAMES or frames % 17 != 5:
        raise ValueError("H3 frame count must have the form 17 * n + 5")
    units = (frames - 5) // 17
    return (17,) * (units - 1) + (MIN_H3_FRAMES,)


@dataclass(frozen=True, slots=True)
class H3Layout:
    """Combines packed multimodal geometry with sequence-parallel rank ownership."""

    packed: H3PackedLayout
    sp_rank: int
    sp_size: int
    tp_size: int
    ulysses_size: int
    sequence_kind: str
    context_col_size: int
    denoiser_participant: bool
    postprocess: bool
    local_start: int
    local_end: int
    frame_count: int
    reconstruction_unit_frames: tuple[int, ...]
    local_video_rows: int = field(init=False)
    local_audio_rows: int = field(init=False)

    def __post_init__(self) -> None:
        """Resolve modality extents once for this immutable packed layout.

        Request tensor binding uses these extents on every bounded operation.
        Filtering the packed CPU indices there would allocate masks and invoke
        the CPU tensor thread pool between consecutive device submissions.
        """

        object.__setattr__(
            self, "local_video_rows", int(self.local_indices(self.packed.video_indices).numel())
        )
        object.__setattr__(
            self, "local_audio_rows", int(self.local_indices(self.packed.audio_indices).numel())
        )

    @classmethod
    def build(
        cls,
        parallel: ParallelConfig,
        mesh: DeviceMesh | None,
        *,
        frames: int,
        text_rows: int,
        audio_frames: int,
        postprocess: bool = False,
    ) -> "H3Layout":
        """Partition one packed request evenly across the mesh sequence ranks."""

        config = parallel
        size = config.sequence_parallel_size
        rank = mesh.coord("sp") if mesh is not None else 0
        packed = build_packed_layout(
            text_rows=text_rows,
            num_frames=frames,
            audio_frames=audio_frames,
            # Four logical row partitions define tensorwise activation scales;
            # physical sequence ownership does not redefine those domains.
            row_multiple=64 * math.lcm(4, size),
        )
        shard = packed.padded_rows // size
        return cls(
            packed=packed,
            sp_rank=rank,
            sp_size=size,
            tp_size=config.tensor_parallel_size,
            ulysses_size=dict(config.dimensions)["ulysses"],
            sequence_kind=config.sequence_parallel.kind,
            context_col_size=dict(config.dimensions).get("cp_col", 1),
            denoiser_participant=mesh is not None,
            postprocess=postprocess,
            local_start=rank * shard if mesh is not None else 0,
            local_end=(rank + 1) * shard if mesh is not None else 0,
            frame_count=int(frames),
            reconstruction_unit_frames=reconstruction_unit_frames(int(frames)),
        )

    @property
    def attention_rows(self) -> int:
        """Rows owned by one context coordinate after Ulysses head exchange."""

        return self.packed.padded_rows // self.sp_size * self.ulysses_size

    @property
    def attention_start(self) -> int:
        return (self.sp_rank // self.ulysses_size) * self.attention_rows

    @property
    def attention_video_tiles(self) -> int:
        start = self.attention_start // 64
        end = start + self.attention_rows // 64
        prefix = self.packed.prefix_tiles
        return max(0, min(end, prefix + self.packed.video_tiles) - max(start, prefix))

    @property
    def local_rows(self) -> int:
        """Count packed transport rows owned by this sequence rank."""

        return self.local_end - self.local_start

    @property
    def video_reconstruction_units(self) -> int:
        """Count overlapping temporal segments needed to reconstruct the video."""

        return len(self.reconstruction_unit_frames)

    def local_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """Filter global packed indices to this rank and convert them to local offsets."""

        selected = indices[(indices >= self.local_start) & (indices < self.local_end)]
        return selected - self.local_start

    @property
    def local_video_raster_indices(self) -> torch.Tensor:
        """Map this rank's tile-major video rows back to global raster order."""

        selected = (self.packed.video_indices >= self.local_start) & (
            self.packed.video_indices < self.local_end
        )
        return self.packed.video_raster_indices[selected]

    @property
    def local_audio_raster_indices(self) -> torch.Tensor:
        """Map this rank's packed audio rows to the unsharded channel-major timeline."""

        selected = (self.packed.audio_indices >= self.local_start) & (
            self.packed.audio_indices < self.local_end
        )
        return torch.arange(self.packed.audio_indices.numel(), dtype=torch.long)[selected]


def state_specs(layout: H3Layout) -> dict[str, resources.TensorSchema]:
    """Describe the exact borrowed state for one logical media geometry.

    Native draws and gathered shard sources have host representation. Their
    physical pinning and capacity belong to the caller's request storage.
    """

    denoiser, packed = layout.denoiser_participant, layout.packed
    shapes = {
        "video_noise": ((int(denoiser), 24, packed.video_frames, 48, 84), torch.float32),
        "audio_noise": ((packed.audio_indices.numel() if denoiser else 0, 32), torch.float32),
        "video_source": ((layout.local_video_rows if denoiser else 0, 96), torch.float32),
        "audio_source": ((layout.local_audio_rows if denoiser else 0, 32), torch.float32),
        "text_condition": (
            (1, int(packed.text_indices.numel()) if denoiser else 0, 5376),
            torch.bfloat16,
        ),
        "video": ((layout.local_video_rows, 96), torch.float32),
        "audio": ((layout.local_audio_rows, 32), torch.float32),
        "tile_valid_sizes": ((packed.padded_rows // 64 if denoiser else 0,), torch.int32),
        "rotary_cosine": ((packed.padded_rows if denoiser else 0, 96), torch.float32),
        "rotary_sine": ((packed.padded_rows if denoiser else 0, 96), torch.float32),
    }
    return {
        name: resources.TensorSchema(
            shape,
            dtype,
            domain="host" if name.endswith(("_noise", "_source")) else "device",
            partition=(
                resources.PackedAxis(
                    0,
                    int(packed.video_indices.numel())
                    if name.startswith("video")
                    else int(packed.audio_indices.numel()),
                    packed.padded_rows,
                    layout.sp_size,
                )
                if denoiser and name in {"video", "audio", "video_source", "audio_source"}
                else None
            ),
        )
        for name, (shape, dtype) in shapes.items()
    }


def compute_specs(
    layout: H3Layout,
    mesh: DeviceMesh | None,
    *,
    block_params_shape: tuple[int, ...],
    final_params_shape: tuple[int, ...],
    attention: VideoAttention | None,
) -> dict[str, resources.TensorSchema]:
    """Describe exact numerical scratch for the local computation.

    Local modality extents describe this call, not the maximum capacity a
    runtime must retain across different prompt and temporal geometries.
    Context-attention views are supplied by its separately bound public layer.
    """

    shapes: dict[str, tuple[int, ...]] = {}
    if mesh is not None:
        local_rows = int(layout.local_rows)
        global_rows = int(layout.packed.padded_rows)
        local_heads = 56 // (layout.tp_size * layout.ulysses_size)
        tiles = global_rows // 64
        query_rows = layout.attention_rows
        query_tiles = query_rows // 64
        prefix_width = int(layout.packed.prefix_tiles + layout.packed.video_tiles)
        local_text = int(layout.local_indices(layout.packed.text_indices).numel())
        local_video = int(layout.local_video_rows)
        local_audio = int(layout.local_audio_rows)
        projected_rows = max(local_video, local_audio)
        shapes.update(
            {
                "packed_hidden": (1, local_rows, 5376),
                "rotary_positions": (global_rows, 3),
                "rotary_frequencies": (global_rows, 3, 16),
                "local_text_hidden": (1, local_text, 5376),
                "projected_input": (projected_rows, 5376),
                "projected_input_bf16": (projected_rows, 5376),
                "video_velocity": (local_video, 96),
                "audio_velocity": (local_audio, 32),
                "tile_scores": (local_heads, query_tiles, tiles),
                "block_indices": (local_heads, query_tiles, prefix_width),
                "block_counts": (local_heads, query_tiles),
                "pooled_query": (query_tiles, local_heads, 128),
                "pooled_key": (tiles, local_heads, 128),
                "pooled_value": (tiles, local_heads, 128),
                "compressed_tiles": (local_heads, query_tiles, 128),
                "topk_indices_i32": (
                    local_heads,
                    layout.attention_video_tiles,
                    video_sparse_selected_tiles(layout.packed.video_tiles),
                ),
            }
        )
        shapes.update(
            {
                "block_adaln_params": block_params_shape,
                "final_adaln_params": final_params_shape,
            }
        )
    integer = {
        "block_indices",
        "block_counts",
        "topk_indices_i32",
    }
    bf16 = {
        "packed_hidden",
        "local_text_hidden",
        "projected_input_bf16",
        "block_adaln_params",
        "final_adaln_params",
    }
    partitions = {}
    if mesh is not None:
        video_rows = int(layout.packed.video_indices.numel())
        audio_rows = int(layout.packed.audio_indices.numel())
        selections = {
            "local_text_hidden": (1, int(layout.packed.text_indices.numel())),
            "projected_input": (0, max(video_rows, audio_rows)),
            "projected_input_bf16": (0, max(video_rows, audio_rows)),
            "video_velocity": (0, video_rows),
            "audio_velocity": (0, audio_rows),
        }
        partitions = {
            name: resources.PackedAxis(axis, elements, global_rows, layout.sp_size)
            for name, (axis, elements) in selections.items()
        }
        # Query tiles are selected from the whole context interval. A different
        # prompt can move video tiles into this rank.
        partitions["topk_indices_i32"] = resources.PackedAxis(
            1, tiles, tiles, global_rows // query_rows
        )
    specs = {}
    for name, shape in shapes.items():
        dtype = torch.float32
        if name in integer:
            dtype = torch.int32
        elif name in bf16:
            dtype = torch.bfloat16
        specs[name] = resources.TensorSchema(shape, dtype, partition=partitions.get(name))
    if mesh is not None:
        if attention is None:
            raise ValueError("resident diffusion scratch requires its numerical attention layer")
        specs.update(attention.tensor_specs(global_rows, query_rows, dtype=torch.bfloat16).scratch)
    return specs


def tensor_output_layout(
    layout: H3Layout,
    entry: str,
    output_index: int,
    *,
    frames: int,
    unit_count: int | None,
    prompt_tokens: int,
) -> TensorOutputLayout | None:
    """Describe logical H3 tensor extents and mathematical sequence shards."""

    if entry == "text_encoder":
        return TensorOutputLayout((1, prompt_tokens, H3TextEncoderConfig().hidden_size))
    if entry == "denoiser":
        indices = layout.packed.video_indices if output_index == 0 else layout.packed.audio_indices
        count = layout.local_video_rows if output_index == 0 else layout.local_audio_rows
        if count == 0:
            return None
        width = 96 if output_index == 0 else 32
        start = int(torch.searchsorted(indices, layout.local_start))
        return TensorOutputLayout(
            (int(indices.numel()), width),
            TensorRegion((start, 0), (count, width)),
        )
    if entry == "video_decoder":
        if unit_count is None:
            raise ValueError("video reconstruction requires a temporal range")
        return TensorOutputLayout((unit_count, 1, 3, 25, PROFILE_HEIGHT, PROFILE_WIDTH))
    if entry == "audio_decoder":
        return TensorOutputLayout((round(frames * PROFILE_AUDIO_RATE / PROFILE_FPS), 2))
    raise ValueError(f"H3 entry {entry!r} has no Tensor result")
