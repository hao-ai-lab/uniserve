"""Fixed-profile H3 audio/video packing and RoPE coordinates."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

__all__ = [
    "AUDIO_CHANNELS",
    "AUDIO_TAG",
    "H3PackedLayout",
    "TEXT_TAG",
    "VIDEO_TAG",
    "audio_latent_frames",
    "build_packed_layout",
    "patchify_video",
    "unpatchify_video",
    "unpatchify_video_into",
    "video_latent_frames",
]

VIDEO_TAG = 0
TEXT_TAG = 1
AUDIO_TAG = 2
AUDIO_CHANNELS = 2
FPS = 24
AUDIO_LATENTS_PER_SECOND = 40
ROPE_FRAME_RESCALE = 5.0 / 3.0
ROPE_FRAMES_PER_LATENT = (1, 4, 4, 4, 4)
_ROPE_SPATIAL_SCALE = 32.0


def video_latent_frames(num_frames: int) -> int:
    """Convert a valid H3 output-frame count into temporal video-VAE latents."""

    if num_frames % 17 != 5:
        raise ValueError("H3 frame count must have the form 17 * n + 5")
    return (num_frames - 5) // 17 * 5 + 2


def audio_latent_frames(num_frames: int) -> int:
    """Size the 40 Hz audio latent timeline for a 24 Hz video frame count."""

    return math.ceil(num_frames / FPS * AUDIO_LATENTS_PER_SECOND)


@dataclass(frozen=True, slots=True)
class H3PackedLayout:
    """Defines the padded multimodal row order, indices, positions, and media geometry for one H3 request."""

    semantic_rows: int
    padded_rows: int
    position_ids: torch.Tensor
    token_tags: torch.Tensor
    text_indices: torch.Tensor
    audio_indices: torch.Tensor
    video_indices: torch.Tensor
    video_raster_indices: torch.Tensor
    video_untile_indices: torch.Tensor
    tile_valid_sizes: torch.Tensor
    prefix_tiles: int
    video_tiles: int
    video_frames: int
    latent_height: int
    latent_width: int
    audio_frames: int


def _spatial_grid(dim: int, patch: int, sqrt_area: float) -> torch.Tensor:
    """Choose a patch-aligned spatial extent near the target square-root area."""

    ratio = dim / sqrt_area
    left = (1.0 - ratio) / 2.0
    values = np.linspace(left, left + ratio, dim // patch, endpoint=False)
    return torch.from_numpy(values * _ROPE_SPATIAL_SCALE).to(torch.float64)


def _temporal_grid(count: int, origin: float) -> torch.Tensor:
    """Generate evenly spaced temporal coordinates from an origin."""

    spans = torch.tensor(
        [
            ROPE_FRAME_RESCALE * ROPE_FRAMES_PER_LATENT[index % len(ROPE_FRAMES_PER_LATENT)]
            for index in range(count)
        ],
        dtype=torch.float64,
    )
    return origin + torch.cat((torch.zeros(1, dtype=torch.float64), spans[:-1].cumsum(0)))


def build_packed_layout(
    *,
    text_rows: int,
    num_frames: int = 124,
    height: int = 768,
    width: int = 1344,
    patch_size: tuple[int, int, int] = (1, 2, 2),
    row_multiple: int = 256,
    audio_frames: int | None = None,
) -> H3PackedLayout:
    """Build the fixed-profile `[text | audio | tiled video | padding]` row layout."""

    if text_rows < 1 or height != 768 or width != 1344:
        raise ValueError("the FastH3 profile requires 1344x768 output and nonempty text")
    patch_t, patch_h, patch_w = patch_size
    latent_height, latent_width = height // 16, width // 16
    video_frames = video_latent_frames(num_frames)
    audio_frames = audio_latent_frames(num_frames) if audio_frames is None else int(audio_frames)
    if audio_frames < 1:
        raise ValueError("H3 audio latent frame count must be positive")
    if video_frames % patch_t or latent_height % patch_h or latent_width % patch_w:
        raise ValueError("fixed latent geometry is not divisible by the transformer patch")

    # Text and audio occupy dense 64-row tiles before the sparse video region.
    rows_per_frame = latent_height // patch_h * (latent_width // patch_w)
    audio_rows = AUDIO_CHANNELS * audio_frames
    video_rows = video_frames // patch_t * rows_per_frame
    if text_rows % 64:
        raise ValueError("the fixed text span must be tile-64 aligned")
    audio_block_rows = math.ceil(audio_rows / 64) * 64
    video_start = text_rows + audio_block_rows

    # Sparse attention consumes 4x4x4 spatiotemporal tiles. Boundary tiles reserve
    # 64 transport rows and pack their valid raster rows at the front.
    tile_t, tile_h, tile_w = (4, 4, 4)
    grid_t = video_frames // patch_t
    grid_h = latent_height // patch_h
    grid_w = latent_width // patch_w
    raster = torch.arange(video_rows, dtype=torch.long).reshape(grid_t, grid_h, grid_w)
    tiled_raster: list[torch.Tensor] = []
    video_valid_sizes: list[int] = []
    for t in range(math.ceil(grid_t / tile_t)):
        for h in range(math.ceil(grid_h / tile_h)):
            for w in range(math.ceil(grid_w / tile_w)):
                block = raster[
                    t * tile_t : min((t + 1) * tile_t, grid_t),
                    h * tile_h : min((h + 1) * tile_h, grid_h),
                    w * tile_w : min((w + 1) * tile_w, grid_w),
                ].reshape(-1)
                tiled_raster.append(block)
                video_valid_sizes.append(int(block.numel()))

    video_tiles = len(tiled_raster)
    video_transport_rows = video_tiles * 64
    transport_rows = video_start + video_transport_rows
    # The block-sparse kernel consumes tile pairs, so an odd logical tile count
    # receives one all-zero transport partner that is not a semantic model row.
    padded_rows = math.ceil(transport_rows / max(row_multiple, 128)) * max(row_multiple, 128)
    if padded_rows // 64 % 2:
        padded_rows += 64
    semantic_rows = text_rows + audio_rows + video_rows

    # Map semantic video raster rows to their tile-major transport positions.
    text_indices = torch.arange(text_rows, dtype=torch.long)
    audio_indices = torch.arange(text_rows, text_rows + audio_rows, dtype=torch.long)
    video_indices_parts: list[torch.Tensor] = []
    video_raster_parts: list[torch.Tensor] = []
    for tile_index, block in enumerate(tiled_raster):
        start = video_start + tile_index * 64
        video_indices_parts.append(torch.arange(start, start + block.numel(), dtype=torch.long))
        video_raster_parts.append(block)
    video_indices = torch.cat(video_indices_parts)
    video_raster_indices = torch.cat(video_raster_parts)
    raster_to_transport = torch.empty(video_rows, dtype=torch.long)
    raster_to_transport[video_raster_indices] = video_indices
    tags = torch.full((padded_rows,), VIDEO_TAG, dtype=torch.long)
    tags[text_indices] = TEXT_TAG
    tags[text_rows:video_start] = AUDIO_TAG

    # Rotary coordinates share a temporal origin at the end of the text prefix;
    # video rows additionally carry normalized height and width coordinates.
    positions = torch.zeros((padded_rows, 3), dtype=torch.float64)
    positions[text_indices, 0] = torch.arange(text_rows, dtype=torch.float64)
    sqrt_area = math.sqrt(latent_height * latent_width)
    height_grid = _spatial_grid(latent_height, patch_h, sqrt_area)
    width_grid = _spatial_grid(latent_width, patch_w, sqrt_area)
    spatial = torch.stack(
        [grid.reshape(-1) for grid in torch.meshgrid(height_grid, width_grid, indexing="ij")],
        dim=-1,
    )
    audio_time = float(text_rows) + torch.arange(audio_frames, dtype=torch.float64)
    positions[audio_indices, 0] = audio_time.repeat(AUDIO_CHANNELS)
    positions[audio_indices, 2] = torch.cat(
        (
            torch.full((audio_frames,), float(width_grid[0]), dtype=torch.float64),
            torch.full((audio_frames,), float(width_grid[-1]), dtype=torch.float64),
        )
    )
    temporal = _temporal_grid(video_frames, float(text_rows))
    video_positions = torch.empty((video_frames, rows_per_frame, 3), dtype=torch.float64)
    video_positions[:, :, 0] = temporal[:, None]
    video_positions[:, :, 1:] = spatial[None]
    positions[video_indices] = video_positions.reshape(-1, 3).index_select(0, video_raster_indices)

    # Per-tile valid counts let sparse attention ignore audio, video-boundary,
    # and pair-alignment padding without changing the fixed row allocation.
    tile_valid_sizes = torch.zeros((padded_rows // 64,), dtype=torch.int32)
    tile_valid_sizes[: text_rows // 64] = 64
    audio_tile_start = text_rows // 64
    for offset in range(audio_block_rows // 64):
        tile_valid_sizes[audio_tile_start + offset] = max(0, min(64, audio_rows - offset * 64))
    video_tile_start = video_start // 64
    tile_valid_sizes[video_tile_start : video_tile_start + video_tiles] = torch.tensor(
        video_valid_sizes, dtype=torch.int32
    )
    return H3PackedLayout(
        semantic_rows=semantic_rows,
        padded_rows=padded_rows,
        position_ids=positions,
        token_tags=tags,
        text_indices=text_indices,
        audio_indices=audio_indices,
        video_indices=video_indices,
        video_raster_indices=video_raster_indices,
        video_untile_indices=raster_to_transport,
        tile_valid_sizes=tile_valid_sizes,
        prefix_tiles=video_start // 64,
        video_tiles=video_tiles,
        video_frames=video_frames,
        latent_height=latent_height,
        latent_width=latent_width,
        audio_frames=audio_frames,
    )


def patchify_video(
    latents: torch.Tensor, patch_size: tuple[int, int, int] = (1, 2, 2)
) -> torch.Tensor:
    """Flatten `[B, C, T, H, W]` latents into raster-ordered spatiotemporal patch rows."""

    patch_t, patch_h, patch_w = patch_size
    batch, channels, frames, height, width = latents.shape
    rows = latents.reshape(
        batch,
        channels,
        frames // patch_t,
        patch_t,
        height // patch_h,
        patch_h,
        width // patch_w,
        patch_w,
    )
    return (
        rows.permute(0, 2, 4, 6, 1, 3, 5, 7)
        .reshape(batch, -1, channels * patch_t * patch_h * patch_w)
        .contiguous()
    )


def unpatchify_video(
    rows: torch.Tensor,
    *,
    frames: int,
    height: int,
    width: int,
    channels: int = 24,
    patch_size: tuple[int, int, int] = (1, 2, 2),
) -> torch.Tensor:
    """Restore raster patch rows to contiguous `[B, C, T, H, W]` video latents."""

    patch_t, patch_h, patch_w = patch_size
    value = rows.reshape(
        -1,
        frames // patch_t,
        height // patch_h,
        width // patch_w,
        channels,
        patch_t,
        patch_h,
        patch_w,
    )
    return (
        value.permute(0, 4, 1, 5, 2, 6, 3, 7)
        .reshape(-1, channels, frames, height, width)
        .contiguous()
    )


def unpatchify_video_into(
    rows: torch.Tensor,
    output: torch.Tensor,
    *,
    frames: int,
    height: int,
    width: int,
    channels: int = 24,
    patch_size: tuple[int, int, int] = (1, 2, 2),
) -> None:
    """Write raster patch rows into one caller-owned decoder input tensor."""

    patch_t, patch_h, patch_w = patch_size
    expected = (1, channels, frames, height, width)
    if tuple(output.shape) != expected:
        raise ValueError(f"video decode output must have shape {expected}")
    value = rows.reshape(
        1,
        frames // patch_t,
        height // patch_h,
        width // patch_w,
        channels,
        patch_t,
        patch_h,
        patch_w,
    ).permute(0, 4, 1, 5, 2, 6, 3, 7)
    output.view(
        1,
        channels,
        frames // patch_t,
        patch_t,
        height // patch_h,
        patch_h,
        width // patch_w,
        patch_w,
    ).copy_(value)
