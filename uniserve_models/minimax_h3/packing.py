"""H3 packed sequences, their rotary coordinates and latent row orders.

Every H3 denoiser evaluates one packed sequence per sample whose rows carry
a token tag and a (time, height, width) rotary coordinate. Two packings
exist, one per attention kind:

* ``TilePacking`` (sparse attention): 64-row tiles ``[text | stereo audio |
  tiled video | padding]``. Text and audio form the dense prefix that every
  query attends; video rows are grouped into 4x4x4 spatiotemporal tiles.
* ``DensePacking`` (dense attention): ``[target video | target audio | text
  and conditions | padding]``. Attention is permutation-equivariant given
  the rotary coordinates, so the generated rows lead at fixed offsets and the
  request-dependent text and condition rows follow them, contiguous, with
  padding only at the end.

All tables are CPU tensors built on the host.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from uniserve.media import image

__all__ = [
    "AUDIO_CHANNELS",
    "AUDIO_TAG",
    "CANVAS_MULTIPLE",
    "DensePacking",
    "TEXT_TAG",
    "TilePacking",
    "VIDEO_TAG",
    "audio_latent_frames",
    "dense_packing",
    "dense_tables",
    "latent_raster",
    "patchify_video",
    "tile_packing",
    "unpatchify_video",
    "unpatchify_video_into",
    "video_latent_frames",
    "video_order",
]

# Token tags. A tag is also the token's group among each timestep's three
# modulation rows (see ``Denoiser._metadata``).
VIDEO_TAG = 0
TEXT_TAG = 1
AUDIO_TAG = 2
AUDIO_CHANNELS = 2
FPS = 24
AUDIO_LATENTS_PER_SECOND = 40
ROPE_FRAME_RESCALE = 5.0 / 3.0
# Output frames covered by each of the five latent frames of one 17-frame
# clip: 1 + 4 * 4 = 17.
ROPE_FRAMES_PER_LATENT = (1, 4, 4, 4, 4)
_ROPE_SPATIAL_SCALE = 32.0
# A canvas side survives the VAE's 16x spatial compression and remains a
# whole number of 2x2 transformer patches.
CANVAS_MULTIPLE = 32
# Timestep groups a packed row reads its modulation under: the generated
# video timestep (text and padding rows too), the generated audio timestep,
# the visual-condition timestep and the audio-reference timestep.
VIDEO_GROUP, AUDIO_GROUP, VISUAL_CONDITION_GROUP, AUDIO_CONDITION_GROUP = (
    0,
    1,
    2,
    3,
)


def latent_raster(canvas: image.Config) -> tuple[int, int]:
    """Return the (height, width) latent raster of ``canvas``.

    Raises:
        ValueError: A canvas side is not a positive multiple of 32.
    """
    if (
        not isinstance(canvas, image.Config)
        or canvas.height % CANVAS_MULTIPLE
        or canvas.width % CANVAS_MULTIPLE
    ):
        raise ValueError(
            f"H3 canvas sides must be multiples of {CANVAS_MULTIPLE} pixels"
        )
    return canvas.height // 16, canvas.width // 16


def video_latent_frames(num_frames: int) -> int:
    """Convert a valid H3 output-frame count into temporal video-VAE latents.

    Each 17-frame unit contributes five latent frames and the trailing
    5-frame overlap two more.

    Raises:
        ValueError: ``num_frames`` is not an int of the form ``17 * n + 5``
            with ``n`` positive.
    """
    if type(num_frames) is not int or num_frames < 22 or num_frames % 17 != 5:
        raise ValueError("H3 frame count must have the form 17 * n + 5")
    return (num_frames - 5) // 17 * 5 + 2


def audio_latent_frames(num_frames: int) -> int:
    """Size the 40 Hz audio latent timeline for a 24 Hz video frame count.

    The count is the video duration in audio latents rounded to the nearest
    integer, as the checkpoint's training pipeline sizes it. At 24 fps and
    40 Hz the exact value ``num_frames * 5 / 3`` is never halfway between
    two integers, so the rounding direction is unambiguous.
    """
    return round(num_frames * AUDIO_LATENTS_PER_SECOND / FPS)


@dataclass(frozen=True, slots=True)
class TilePacking:
    """CPU indices for the mathematical text, stereo audio and tiled video
    order of sparse attention.

    num_tokens counts valid tokens. padded_tokens also includes tile-boundary
    and partition alignment, whose validity is represented separately.

    Attributes:
        position_ids: [padded_tokens, 3] FP64 (time, height, width) rotary
            coordinates of every packed row.
        token_tags: [padded_tokens] int64 ``VIDEO_TAG``/``TEXT_TAG``/
            ``AUDIO_TAG`` per row; rows past the video region carry
            ``VIDEO_TAG``.
        text_indices: Packed rows of the text tokens.
        audio_indices: Packed rows of the audio tokens, channel-major.
        video_indices: Packed rows of the video tokens, ascending.
        video_raster_indices: Raster row of each entry of ``video_indices``.
        video_untile_indices: Packed row of each video raster row, the
            inverse of ``video_raster_indices``.
        tile_valid_sizes: [padded_tokens // 64] int32 valid rows per tile.
        prefix_tiles: Tiles of the dense text/audio prefix.
        video_tiles: Tiles of the video region, including partial tiles.
        video_frames: Temporal latent frames.
        latent_height: Latent raster height before patching.
        latent_width: Latent raster width before patching.
        audio_frames: Audio latent frames per stereo channel.
    """  # noqa: D205

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
    """Return rotary coordinates of the patches along one spatial axis.

    The ``dim // patch`` coordinates are evenly spaced over an interval of
    length ``dim / sqrt_area`` centered on 0.5, so both axes share the scale
    of the raster's geometric-mean side, and are then multiplied by
    ``_ROPE_SPATIAL_SCALE``.
    """  # noqa: E501
    ratio = dim / sqrt_area
    left = (1.0 - ratio) / 2.0
    values = np.linspace(left, left + ratio, dim // patch, endpoint=False)
    return torch.from_numpy(values * _ROPE_SPATIAL_SCALE).to(torch.float64)


def _temporal_grid(count: int, origin: float) -> torch.Tensor:
    """Place ``count`` latent frames on the rotary time axis from ``origin``.

    The gap after a latent frame is ``ROPE_FRAME_RESCALE`` times the output
    frames it covers, cycling through ``ROPE_FRAMES_PER_LATENT``.
    """
    spans = torch.tensor(
        [
            ROPE_FRAME_RESCALE
            * ROPE_FRAMES_PER_LATENT[index % len(ROPE_FRAMES_PER_LATENT)]
            for index in range(count)
        ],
        dtype=torch.float64,
        device="cpu",
    )
    return origin + torch.cat(
        (
            torch.zeros(1, dtype=torch.float64, device="cpu"),
            spans[:-1].cumsum(0),
        )
    )


def tile_packing(
    *,
    num_text_tokens: int,
    num_frames: int,
    canvas: image.Config,
    patch_size: tuple[int, int, int] = (1, 2, 2),
    token_multiple: int = 256,
    audio_frames: int | None = None,
    text_rows: int | None = None,
) -> TilePacking:
    """Build CPU coordinates for `[text | audio | tiled video | padding]` rows.

    These host metadata values remain concrete during deferred parameter
    construction. Execution supplies device views for numerical kernels.

    Args:
        num_text_tokens: Prompt tokens, the valid leading rows of the text
            region.
        num_frames: Output video frames, of the form ``17 * n + 5``.
        canvas: Output raster, sides multiples of 32.
        patch_size: Transformer (time, height, width) patch on the latents.
        token_multiple: Row alignment, a multiple of 64. ``padded_tokens``
            rounds up to a multiple of ``max(token_multiple, 128)``, plus one
            tile when that leaves an odd tile count.
        audio_frames: Audio latent frames per channel; defaults to the
            40 Hz timeline of ``num_frames``.
        text_rows: Rows of the text region, whole 64-row tiles holding at
            least the prompt; defaults to the prompt's own tiles. Rows past
            the prompt are invalid in every tile they occupy, and the media
            rotary timeline still starts at the prompt's exact length, so a
            larger region changes the packed shapes but not the positions,
            tags or validity of any prompt, audio or video token.

    Raises:
        ValueError: A canvas off the 32-pixel grid, empty text, a frame
            count H3 does not generate, a nonpositive audio length, an
            alignment that is not a positive multiple of the 64-row tile, a
            text region that is not whole tiles holding the prompt, or a
            patch that does not divide the latents.
    """
    if num_text_tokens < 1:
        raise ValueError("H3 packing requires nonempty text")
    if token_multiple < 1 or token_multiple % 64:
        raise ValueError(
            "packing alignment must contain complete 64-token tiles"
        )
    if text_rows is None:
        text_rows = math.ceil(num_text_tokens / 64) * 64
    elif text_rows % 64 or text_rows < num_text_tokens:
        raise ValueError(
            "the H3 text region must be whole 64-row tiles holding the prompt"
        )
    patch_t, patch_h, patch_w = patch_size
    # Latents are 16x spatially compressed relative to the output raster.
    latent_height, latent_width = latent_raster(canvas)
    video_frames = video_latent_frames(num_frames)
    audio_frames = (
        audio_latent_frames(num_frames)
        if audio_frames is None
        else int(audio_frames)
    )
    if audio_frames < 1:
        raise ValueError("H3 audio latent frame count must be positive")
    if (
        video_frames % patch_t
        or latent_height % patch_h
        or latent_width % patch_w
    ):
        raise ValueError(
            "fixed latent dimensions are not divisible by the transformer patch"
        )

    # Text and audio occupy dense 64-row tiles before the sparse video region.
    rows_per_frame = latent_height // patch_h * (latent_width // patch_w)
    audio_rows = AUDIO_CHANNELS * audio_frames
    video_rows = video_frames // patch_t * rows_per_frame
    audio_block_rows = math.ceil(audio_rows / 64) * 64
    video_start = text_rows + audio_block_rows

    # Sparse attention consumes 4x4x4 spatiotemporal tiles. Boundary tiles
    # reserve 64 transport rows and pack their valid raster rows at the
    # front.
    tile_t, tile_h, tile_w = (4, 4, 4)
    grid_t = video_frames // patch_t
    grid_h = latent_height // patch_h
    grid_w = latent_width // patch_w
    tiles_t, tiles_h, tiles_w = (
        math.ceil(extent / tile)
        for extent, tile in zip((grid_t, grid_h, grid_w), (4, 4, 4))
    )
    video_tiles = tiles_t * tiles_h * tiles_w
    # Tiles are ordered time-major over the tile grid. Within a tile, row
    # offset o addresses the (time, height, width) cell
    # (o // 16, o // 4 % 4, o % 4) before boundary compaction.
    tile_ids = np.arange(video_tiles, dtype=np.int64)[:, None]
    offsets = np.arange(64, dtype=np.int64)[None, :]
    temporal_rows = tile_ids // (tiles_h * tiles_w) * tile_t + offsets // 16
    height_rows = tile_ids // tiles_w % tiles_h * tile_h + offsets // 4 % 4
    width_rows = tile_ids % tiles_w * tile_w + offsets % 4
    valid = (
        (temporal_rows < grid_t)
        & (height_rows < grid_h)
        & (width_rows < grid_w)
    )
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
    padded_tokens = math.ceil(transport_rows / max(token_multiple, 128)) * max(
        token_multiple, 128
    )
    if padded_tokens // 64 % 2:
        padded_tokens += 64
    num_tokens = num_text_tokens + audio_rows + video_rows

    # Map semantic video raster rows to their tile-major transport positions.
    text_indices = torch.arange(num_text_tokens, dtype=torch.long, device="cpu")
    audio_indices = torch.arange(
        text_rows, text_rows + audio_rows, dtype=torch.long, device="cpu"
    )
    raster_to_transport = torch.empty(
        video_rows, dtype=torch.long, device="cpu"
    )
    raster_to_transport[video_raster_indices] = video_indices
    tags = torch.full(
        (padded_tokens,), VIDEO_TAG, dtype=torch.long, device="cpu"
    )
    tags[:text_rows] = TEXT_TAG
    tags[text_rows:video_start] = AUDIO_TAG

    # Rotary coordinates share a temporal origin at the end of the text prefix;
    # video rows additionally carry normalized height and width coordinates.
    positions = torch.zeros(
        (padded_tokens, 3), dtype=torch.float64, device="cpu"
    )
    positions[:text_rows, 0] = torch.arange(
        text_rows, dtype=torch.float64, device="cpu"
    )
    sqrt_area = math.sqrt(latent_height * latent_width)
    height_grid = _spatial_grid(latent_height, patch_h, sqrt_area)
    width_grid = _spatial_grid(latent_width, patch_w, sqrt_area)
    spatial = torch.stack(
        [
            grid.reshape(-1)
            for grid in torch.meshgrid(height_grid, width_grid, indexing="ij")
        ],
        dim=-1,
    )
    # Both stereo channels share the audio timeline; the first channel sits
    # at the leftmost video width coordinate and the second at the
    # rightmost.
    audio_time = float(text_rows) + torch.arange(
        audio_frames, dtype=torch.float64, device="cpu"
    )
    positions[audio_indices, 0] = audio_time.repeat(AUDIO_CHANNELS)
    positions[audio_indices, 2] = torch.cat(
        (
            torch.full(
                (audio_frames,),
                float(width_grid[0]),
                dtype=torch.float64,
                device="cpu",
            ),
            torch.full(
                (audio_frames,),
                float(width_grid[-1]),
                dtype=torch.float64,
                device="cpu",
            ),
        )
    )
    temporal = _temporal_grid(video_frames, float(text_rows))
    video_positions = torch.empty(
        (video_frames, rows_per_frame, 3), dtype=torch.float64, device="cpu"
    )
    video_positions[:, :, 0] = temporal[:, None]
    video_positions[:, :, 1:] = spatial[None]
    positions[video_indices] = video_positions.reshape(-1, 3).index_select(
        0, video_raster_indices
    )

    # Move the media time origin from the tile-rounded text prefix to the
    # exact prompt length, so the first audio frame and video latent frame
    # sit at time num_text_tokens, right after the last prompt token.
    positions[text_rows:, 0].add_(num_text_tokens - text_rows)

    # Per-tile valid counts let sparse attention ignore audio, video-boundary,
    # and pair-alignment padding without changing the fixed row allocation.
    tile_valid_sizes = torch.zeros(
        (padded_tokens // 64,), dtype=torch.int32, device="cpu"
    )
    for offset in range(text_rows // 64):
        tile_valid_sizes[offset] = max(
            0, min(64, num_text_tokens - offset * 64)
        )
    audio_tile_start = text_rows // 64
    for offset in range(audio_block_rows // 64):
        tile_valid_sizes[audio_tile_start + offset] = max(
            0, min(64, audio_rows - offset * 64)
        )
    video_tile_start = video_start // 64
    tile_valid_sizes[video_tile_start : video_tile_start + video_tiles] = (
        video_valid_sizes
    )
    return TilePacking(
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


@dataclass(frozen=True, slots=True)
class DensePacking:
    """Row extents of one dense layout: ``[video | audio | prefix | padding]``.

    The generated video rows (raster order: latent frame, then patch row,
    then patch column) start at row 0 and the generated stereo audio rows
    (channel-major) follow them, so a sequence shard's generated rows depend
    on the layout alone. The prefix region holds a request's text rows and
    then its condition rows, contiguous; the rest of the region and the
    alignment rows are padding, which attention excludes.

    Attributes:
        video_rows: Generated video rows.
        audio_rows: Generated audio rows, both stereo channels.
        text_rows: Text capacity of the prefix region.
        condition_rows: Condition capacity of the prefix region.
        padded_tokens: All rows, a multiple of the layout's alignment.
    """

    video_rows: int
    audio_rows: int
    text_rows: int
    condition_rows: int
    padded_tokens: int

    @property
    def prefix_start(self) -> int:
        """First row of the prefix region."""
        return self.video_rows + self.audio_rows

    @property
    def zero_row(self) -> int:
        """Row of the prefix source that holds zeros (see ``dense_tables``)."""
        return self.text_rows + self.condition_rows


def dense_packing(
    *,
    num_frames: int,
    canvas: image.Config,
    text_rows: int,
    condition_rows: int,
    token_multiple: int,
    patch_size: tuple[int, int, int] = (1, 2, 2),
) -> DensePacking:
    """Size the dense layout of one frame count, canvas and prefix capacity.

    ``token_multiple`` aligns the padded row count so that every sequence
    rank holds an equal shard.

    Raises:
        ValueError: A frame count H3 does not generate, a canvas off the
            32-pixel grid, nonpositive text or negative condition capacity,
            or a nonpositive alignment.
    """
    if (
        type(text_rows) is not int
        or text_rows < 1
        or type(condition_rows) is not int
        or condition_rows < 0
        or type(token_multiple) is not int
        or token_multiple < 1
    ):
        raise ValueError(
            "a dense layout needs text rows, a condition capacity and a "
            "positive alignment"
        )
    patch_t, patch_h, patch_w = patch_size
    latent_height, latent_width = latent_raster(canvas)
    video_rows = (
        video_latent_frames(num_frames)
        // patch_t
        * (latent_height // patch_h)
        * (latent_width // patch_w)
    )
    audio_rows = AUDIO_CHANNELS * audio_latent_frames(num_frames)
    rows = video_rows + audio_rows + text_rows + condition_rows
    return DensePacking(
        video_rows=video_rows,
        audio_rows=audio_rows,
        text_rows=text_rows,
        condition_rows=condition_rows,
        padded_tokens=math.ceil(rows / token_multiple) * token_multiple,
    )


@dataclass(frozen=True, slots=True)
class DenseTables:
    """One request's row tables in a dense layout.

    Attributes:
        position_ids: [padded_tokens, 3] FP64 rotary coordinates; padding
            rows hold zeros.
        token_tags: [padded_tokens] int64 modulation tags; padding rows carry
            ``VIDEO_TAG``.
        groups: [padded_tokens] int64 timestep groups (``VIDEO_GROUP`` ...).
        prefix_index: [padded_tokens] int64 row of the prefix source each
            packed row takes: text rows first, then condition rows, and the
            source's zero row for generated and padding rows.
        used: Rows attention attends: generated rows, text and conditions.
    """

    position_ids: torch.Tensor
    token_tags: torch.Tensor
    groups: torch.Tensor
    prefix_index: torch.Tensor
    used: int


def dense_tables(
    packing: DensePacking,
    *,
    num_frames: int,
    canvas: image.Config,
    num_text_tokens: int,
    patch_size: tuple[int, int, int] = (1, 2, 2),
) -> DenseTables:
    """Build a text-only request's tables in its dense layout.

    The coordinates are the released contract: text row ``i`` sits at
    ``(i, 0, 0)``; the media clock starts at the prompt length, the audio
    rows advancing one unit per latent with the stereo channels at the two
    ends of the width grid, and the video frames at
    ``(5 / 3) * (1, 4, 4, 4, 4)[k mod 5]`` spacing over the aspect-normalized
    spatial grid.

    Raises:
        ValueError: The prompt does not fit the layout's text rows.
    """
    if not 1 <= num_text_tokens <= packing.text_rows:
        raise ValueError("the prompt must fit its layout's text rows")
    _, patch_h, patch_w = patch_size
    latent_height, latent_width = latent_raster(canvas)
    rows = packing.padded_tokens
    frames = video_latent_frames(num_frames)
    audio_frames = audio_latent_frames(num_frames)
    origin = float(num_text_tokens)

    positions = torch.zeros((rows, 3), dtype=torch.float64, device="cpu")
    sqrt_area = math.sqrt(latent_height * latent_width)
    height_grid = _spatial_grid(latent_height, patch_h, sqrt_area)
    width_grid = _spatial_grid(latent_width, patch_w, sqrt_area)
    spatial = torch.stack(
        [
            grid.reshape(-1)
            for grid in torch.meshgrid(height_grid, width_grid, indexing="ij")
        ],
        dim=-1,
    )
    video = torch.empty(
        (frames, spatial.shape[0], 3), dtype=torch.float64, device="cpu"
    )
    video[:, :, 0] = _temporal_grid(frames, origin)[:, None]
    video[:, :, 1:] = spatial[None]
    positions[: packing.video_rows] = video.reshape(-1, 3)

    audio = slice(packing.video_rows, packing.prefix_start)
    positions[audio, 0] = (
        origin + torch.arange(audio_frames, dtype=torch.float64, device="cpu")
    ).repeat(AUDIO_CHANNELS)
    positions[audio, 2] = torch.cat(
        (
            torch.full(
                (audio_frames,), float(width_grid[0]), dtype=torch.float64
            ),
            torch.full(
                (audio_frames,), float(width_grid[-1]), dtype=torch.float64
            ),
        )
    )
    text = slice(packing.prefix_start, packing.prefix_start + num_text_tokens)
    positions[text, 0] = torch.arange(
        num_text_tokens, dtype=torch.float64, device="cpu"
    )

    tags = torch.full((rows,), VIDEO_TAG, dtype=torch.int64, device="cpu")
    tags[audio] = AUDIO_TAG
    tags[text] = TEXT_TAG
    groups = torch.full((rows,), VIDEO_GROUP, dtype=torch.int64, device="cpu")
    groups[audio] = AUDIO_GROUP

    prefix_index = torch.full(
        (rows,), packing.zero_row, dtype=torch.int64, device="cpu"
    )
    prefix_index[text] = torch.arange(num_text_tokens, dtype=torch.int64)
    return DenseTables(
        position_ids=positions,
        token_tags=tags,
        groups=groups,
        prefix_index=prefix_index,
        used=packing.prefix_start + num_text_tokens,
    )


def video_order(
    attention, *, num_frames: int, canvas: image.Config
) -> torch.Tensor:
    """Return the raster row of each generated video row in packed order.

    ``attention`` is a denoiser's attention configuration
    (``config.DenseAttention`` or ``config.SparseAttention``): dense packing
    keeps the raster order; tile packing orders rows tile-major, which does
    not depend on the prompt.
    """
    from .config import DenseAttention

    if isinstance(attention, DenseAttention):
        height, width = latent_raster(canvas)
        return torch.arange(
            video_latent_frames(num_frames) * (height // 2) * (width // 2),
            dtype=torch.int64,
        )
    return tile_packing(
        num_text_tokens=64, num_frames=num_frames, canvas=canvas
    ).video_raster_indices


def patchify_video(
    latents: torch.Tensor, patch_size: tuple[int, int, int] = (1, 2, 2)
) -> torch.Tensor:
    """Flatten `[B, C, T, H, W]` latents into raster-ordered spatiotemporal patch rows."""  # noqa: E501
    patch_t, patch_h, patch_w = patch_size
    batch, channels, frames, height, width = latents.shape
    # Split each axis into (cell, patch) pairs, then walk cells in raster order.
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
    """Restore raster patch rows to contiguous `[B, C, T, H, W]` video latents."""  # noqa: E501
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
