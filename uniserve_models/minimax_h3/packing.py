"""Fixed-profile H3 audio/video packing and RoPE coordinates."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

__all__ = [
    "AUDIO_CHANNELS",
    "AUDIO_TAG",
    "Packing",
    "TEXT_TAG",
    "VIDEO_TAG",
    "audio_latent_frames",
    "build_packing",
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

    if type(num_frames) is not int or num_frames < 22 or num_frames % 17 != 5:
        raise ValueError("H3 frame count must have the form 17 * n + 5")
    return (num_frames - 5) // 17 * 5 + 2


def audio_latent_frames(num_frames: int) -> int:
    """Size the 40 Hz audio latent timeline for a 24 Hz video frame count."""

    return math.ceil(num_frames / FPS * AUDIO_LATENTS_PER_SECOND)


@dataclass(frozen=True, slots=True)
class Packing:
    """CPU indices for the mathematical text, stereo audio and tiled video order.

    num_tokens counts valid tokens. padded_tokens also includes tile-boundary
    and partition alignment, whose validity is represented separately.
    """

    num_tokens: int
    padded_tokens: int
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
        device="cpu",
    )
    return origin + torch.cat(
        (torch.zeros(1, dtype=torch.float64, device="cpu"), spans[:-1].cumsum(0))
    )


def build_packing(
    *,
    num_text_tokens: int,
    num_frames: int = 124,
    height: int = 768,
    width: int = 1344,
    patch_size: tuple[int, int, int] = (1, 2, 2),
    token_multiple: int = 256,
    audio_frames: int | None = None,
) -> Packing:
    """Build CPU coordinates for `[text | audio | tiled video | padding]` rows.

    These host metadata values remain concrete during deferred parameter
    construction. Execution supplies device views for numerical kernels."""

    if num_text_tokens < 1 or height != 768 or width != 1344:
        raise ValueError("the FastH3 profile requires 1344x768 output and nonempty text")
    if token_multiple < 1 or token_multiple % 64:
        raise ValueError("packing alignment must contain complete 64-token tiles")
    text_rows = math.ceil(num_text_tokens / 64) * 64
    patch_t, patch_h, patch_w = patch_size
    latent_height, latent_width = height // 16, width // 16
    video_frames = video_latent_frames(num_frames)
    audio_frames = audio_latent_frames(num_frames) if audio_frames is None else int(audio_frames)
    if audio_frames < 1:
        raise ValueError("H3 audio latent frame count must be positive")
    if video_frames % patch_t or latent_height % patch_h or latent_width % patch_w:
        raise ValueError("fixed latent dimensions are not divisible by the transformer patch")

    # Text and audio occupy dense 64-row tiles before the sparse video region.
    rows_per_frame = latent_height // patch_h * (latent_width // patch_w)
    audio_rows = AUDIO_CHANNELS * audio_frames
    video_rows = video_frames // patch_t * rows_per_frame
    audio_block_rows = math.ceil(audio_rows / 64) * 64
    video_start = text_rows + audio_block_rows

    # Sparse attention consumes 4x4x4 spatiotemporal tiles. Boundary tiles reserve
    # 64 transport rows and pack their valid raster rows at the front.
    tile_t, tile_h, tile_w = (4, 4, 4)
    grid_t = video_frames // patch_t
    grid_h = latent_height // patch_h
    grid_w = latent_width // patch_w
    tiles_t, tiles_h, tiles_w = (
        math.ceil(extent / tile) for extent, tile in zip((grid_t, grid_h, grid_w), (4, 4, 4))
    )
    video_tiles = tiles_t * tiles_h * tiles_w
    tile_ids = np.arange(video_tiles, dtype=np.int64)[:, None]
    offsets = np.arange(64, dtype=np.int64)[None, :]
    temporal_rows = tile_ids // (tiles_h * tiles_w) * tile_t + offsets // 16
    height_rows = tile_ids // tiles_w % tiles_h * tile_h + offsets // 4 % 4
    width_rows = tile_ids % tiles_w * tile_w + offsets % 4
    valid = (temporal_rows < grid_t) & (height_rows < grid_h) & (width_rows < grid_w)
    valid_sizes = valid.sum(axis=1, dtype=np.int32)
    # These are CPU index tables. NumPy's integer ufuncs avoid launching an
    # intra-op worker team for each small coordinate expression. Tensor views
    # retain the completed arrays without copying their backing.
    video_valid_sizes = torch.from_numpy(valid_sizes)
    video_raster_indices = torch.from_numpy(
        ((temporal_rows * grid_h + height_rows) * grid_w + width_rows)[valid]
    )
    # Boundary tiles compact raster rows at the front of each transport tile;
    # missing spatial columns must not leave holes between valid rows.
    video_indices = torch.from_numpy(
        (video_start + tile_ids * 64 + offsets)[offsets < valid_sizes[:, None]]
    )
    video_transport_rows = video_tiles * 64
    transport_rows = video_start + video_transport_rows
    # The block-sparse kernel consumes tile pairs, so an odd logical tile count
    # receives one all-zero transport partner that is not a semantic model row.
    padded_tokens = math.ceil(transport_rows / max(token_multiple, 128)) * max(token_multiple, 128)
    if padded_tokens // 64 % 2:
        padded_tokens += 64
    num_tokens = num_text_tokens + audio_rows + video_rows

    # Map semantic video raster rows to their tile-major transport positions.
    text_indices = torch.arange(num_text_tokens, dtype=torch.long, device="cpu")
    audio_indices = torch.arange(text_rows, text_rows + audio_rows, dtype=torch.long, device="cpu")
    raster_to_transport = torch.empty(video_rows, dtype=torch.long, device="cpu")
    raster_to_transport[video_raster_indices] = video_indices
    tags = torch.full((padded_tokens,), VIDEO_TAG, dtype=torch.long, device="cpu")
    tags[:text_rows] = TEXT_TAG
    tags[text_rows:video_start] = AUDIO_TAG

    # Rotary coordinates share a temporal origin at the end of the text prefix;
    # video rows additionally carry normalized height and width coordinates.
    positions = torch.zeros((padded_tokens, 3), dtype=torch.float64, device="cpu")
    positions[:text_rows, 0] = torch.arange(text_rows, dtype=torch.float64, device="cpu")
    sqrt_area = math.sqrt(latent_height * latent_width)
    height_grid = _spatial_grid(latent_height, patch_h, sqrt_area)
    width_grid = _spatial_grid(latent_width, patch_w, sqrt_area)
    spatial = torch.stack(
        [grid.reshape(-1) for grid in torch.meshgrid(height_grid, width_grid, indexing="ij")],
        dim=-1,
    )
    audio_time = float(text_rows) + torch.arange(audio_frames, dtype=torch.float64, device="cpu")
    positions[audio_indices, 0] = audio_time.repeat(AUDIO_CHANNELS)
    positions[audio_indices, 2] = torch.cat(
        (
            torch.full((audio_frames,), float(width_grid[0]), dtype=torch.float64, device="cpu"),
            torch.full((audio_frames,), float(width_grid[-1]), dtype=torch.float64, device="cpu"),
        )
    )
    temporal = _temporal_grid(video_frames, float(text_rows))
    video_positions = torch.empty(
        (video_frames, rows_per_frame, 3), dtype=torch.float64, device="cpu"
    )
    video_positions[:, :, 0] = temporal[:, None]
    video_positions[:, :, 1:] = spatial[None]
    positions[video_indices] = video_positions.reshape(-1, 3).index_select(0, video_raster_indices)

    positions[text_rows:, 0].add_(num_text_tokens - text_rows)

    # Per-tile valid counts let sparse attention ignore audio, video-boundary,
    # and pair-alignment padding without changing the fixed row allocation.
    tile_valid_sizes = torch.zeros((padded_tokens // 64,), dtype=torch.int32, device="cpu")
    for offset in range(text_rows // 64):
        tile_valid_sizes[offset] = min(64, num_text_tokens - offset * 64)
    audio_tile_start = text_rows // 64
    for offset in range(audio_block_rows // 64):
        tile_valid_sizes[audio_tile_start + offset] = max(0, min(64, audio_rows - offset * 64))
    video_tile_start = video_start // 64
    tile_valid_sizes[video_tile_start : video_tile_start + video_tiles] = video_valid_sizes
    return Packing(
        num_tokens=num_tokens,
        padded_tokens=padded_tokens,
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
