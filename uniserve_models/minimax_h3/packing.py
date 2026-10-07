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
from enum import StrEnum

import numpy as np
import torch

from uniserve.media import image
from uniserve.model import Condition, ConditionRole

__all__ = [
    "AUDIO_CHANNELS",
    "AUDIO_TAG",
    "CANVAS_MULTIPLE",
    "DensePacking",
    "Segment",
    "SegmentKind",
    "TEXT_TAG",
    "TilePacking",
    "VIDEO_TAG",
    "audio_latent_frames",
    "condition_segments",
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
# The audio VAE's 32 kHz samples per latent.
AUDIO_HOP = 32_000 // AUDIO_LATENTS_PER_SECOND
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


class SegmentKind(StrEnum):
    """What one condition contributes to the packed sequence."""

    # One latent frame of the target canvas anchoring a generated frame.
    KEYFRAME = "keyframe"
    # One latent frame of a reference image at its own raster.
    IMAGE = "image"
    # A reference video's latent frames, after its soundtrack's audio rows.
    VIDEO = "video"
    # A reference track's stereo audio rows.
    AUDIO = "audio"


@dataclass(frozen=True, slots=True)
class Segment:
    """One condition's rows in the packed sequence.

    Attributes:
        kind: What the condition contributes.
        condition: The condition's position in request order.
        anchor: The generated frame a keyframe anchors; None otherwise.
        latent_frames: Visual latent frames; 0 for audio alone.
        latent_height: Visual latent raster height; 0 for audio alone.
        latent_width: Visual latent raster width; 0 for audio alone.
        audio_frames: Audio latent frames per stereo channel; 0 without
            audio.
    """

    kind: SegmentKind
    condition: int
    anchor: ConditionRole | None
    latent_frames: int
    latent_height: int
    latent_width: int
    audio_frames: int

    @property
    def video_rows(self) -> int:
        """Visual rows: one per 2x2 patch of every latent frame."""
        return (
            self.latent_frames
            * (self.latent_height // 2)
            * (self.latent_width // 2)
        )

    @property
    def audio_rows(self) -> int:
        """Channel-major stereo audio rows."""
        return AUDIO_CHANNELS * self.audio_frames


def condition_segments(
    conditions: tuple[Condition, ...], canvas: image.Config
) -> tuple[Segment, ...]:
    """Order a request's conditions as the packed sequence holds them.

    Keyframes come first, in request order, then the references in request
    order: the released ``fl2va`` layout holds only keyframes, and a
    ``ref2va`` request's keyframes lead its references as in the SGLang
    layout, the only implementation of that combination. A one-frame
    reference is an image; a longer one is a video, with its soundtrack
    when it carries audio; a reference without pixels is audio.

    Raises:
        ValueError: A keyframe off the target canvas, or a raster or frame
            count the VAE does not encode.
    """
    keyframes: list[Segment] = []
    references: list[Segment] = []
    for index, condition in enumerate(conditions):
        pixels = condition.video
        height = width = frames = 0
        if pixels is not None:
            height, width = latent_raster(pixels.frame)
            frames = (
                1
                if pixels.num_frames == 1
                else video_latent_frames(pixels.num_frames)
            )
        audio = math.ceil(condition.audio_samples / AUDIO_HOP)
        if condition.role != ConditionRole.REFERENCE:
            if pixels is None or pixels.frame != canvas:
                raise ValueError("H3 keyframes are fitted to the target canvas")
            kind = SegmentKind.KEYFRAME
        elif pixels is None:
            kind = SegmentKind.AUDIO
        else:
            kind = SegmentKind.IMAGE if frames == 1 else SegmentKind.VIDEO
        if kind == SegmentKind.IMAGE and audio:
            raise ValueError("an H3 image reference carries no audio")
        segment = Segment(
            kind,
            index,
            condition.role if kind == SegmentKind.KEYFRAME else None,
            frames,
            height,
            width,
            audio,
        )
        (keyframes if kind == SegmentKind.KEYFRAME else references).append(
            segment
        )
    return (*keyframes, *references)


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


def _frame_grid(
    latent_height: int, latent_width: int, patch: tuple[int, int, int]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return one latent frame's ``(h, w)`` coordinates and its width axis.

    Both axes share the scale of the raster's geometric-mean side
    (``_spatial_grid``); rows are row-major over the 2x2 patches.
    """
    _, patch_h, patch_w = patch
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
    return spatial, width_grid


def _place_audio(
    positions: torch.Tensor,
    rows: slice | torch.Tensor,
    frames: int,
    origin: float,
    width_grid: torch.Tensor,
) -> None:
    """Place one channel-major stereo block from ``origin``.

    ``rows`` selects the block's packed rows, as a slice or an int64 index
    tensor. Audio advances one unit per latent; it has no height coordinate,
    and the two channels sit at the two ends of ``width_grid``.
    """
    positions[rows, 0] = (
        origin + torch.arange(frames, dtype=torch.float64, device="cpu")
    ).repeat(AUDIO_CHANNELS)
    positions[rows, 2] = torch.cat(
        (
            torch.full((frames,), float(width_grid[0]), dtype=torch.float64),
            torch.full((frames,), float(width_grid[-1]), dtype=torch.float64),
        )
    )


def _place_frames(
    positions: torch.Tensor,
    rows: slice,
    frames: int,
    origin: float,
    spatial: torch.Tensor,
) -> None:
    """Place ``frames`` latent frames of one raster from ``origin``."""
    block = torch.empty(
        (frames, spatial.shape[0], 3), dtype=torch.float64, device="cpu"
    )
    block[:, :, 0] = _temporal_grid(frames, origin)[:, None]
    block[:, :, 1:] = spatial[None]
    positions[rows] = block.reshape(-1, 3)


def _video_span(frames: int) -> float:
    """Rotary time ``frames`` latent frames occupy, summed sequentially.

    The reference advances its ``ref2va`` clock past a reference video by
    this Python ``sum``; ``_keyframe_span`` sums the same series pairwise,
    and the two differ in the last place from 16 latent frames on.
    """
    return sum(
        ROPE_FRAME_RESCALE
        * ROPE_FRAMES_PER_LATENT[index % len(ROPE_FRAMES_PER_LATENT)]
        for index in range(frames)
    )


def _keyframe_span(frames: int) -> float:
    """Rotary time ``frames`` latent frames occupy, summed pairwise.

    The reference anchors a last keyframe with numpy's pairwise sum of the
    frame spans (see ``_video_span``).
    """
    spans = np.ones(frames, dtype=np.float64) * ROPE_FRAME_RESCALE
    cycle = len(ROPE_FRAMES_PER_LATENT)
    for offset in range(cycle):
        spans[offset::cycle] *= ROPE_FRAMES_PER_LATENT[offset]
    return float(spans.sum())


def dense_tables(
    packing: DensePacking,
    *,
    num_frames: int,
    canvas: image.Config,
    num_text_tokens: int,
    segments: tuple[Segment, ...] = (),
    vision_spans: tuple[tuple[int, int], ...] = (),
    patch_size: tuple[int, int, int] = (1, 2, 2),
) -> DenseTables:
    """Build one request's tables in its dense layout.

    The coordinates are the released contract (spec section 2.3). Text row
    ``i`` sits at ``(i, 0, 0)`` and the media clock starts at the prompt
    length. References advance it in packed order: an image by one unit at
    its own spatial grid; an audio track by its latent count, its channels
    at the ends of the target width grid; a video by the longer of its
    soundtrack and its frames, the soundtrack starting with the frames at
    the ends of the video's own width grid. The generated audio and video
    start where the references leave the clock, the audio one unit per
    latent and the video frames at ``(5 / 3) * (1, 4, 4, 4, 4)[k mod 5]``
    spacing. A keyframe sits on the generated timeline at the target grid:
    at its start, or at its last latent frame.

    Text rows in ``vision_spans`` are tagged video; visual condition rows
    read the visual-condition timestep group and audio reference rows the
    audio-reference group.

    Raises:
        ValueError: The prompt and conditions do not fit the layout's prefix
            region, or a vision span leaves the prompt.
    """
    condition_rows = sum(
        segment.video_rows + segment.audio_rows for segment in segments
    )
    if (
        not 1 <= num_text_tokens <= packing.text_rows
        or condition_rows > packing.condition_rows
    ):
        raise ValueError(
            "the prompt and conditions must fit their layout's prefix region"
        )
    if any(
        not 0 <= start < stop <= num_text_tokens for start, stop in vision_spans
    ):
        raise ValueError("vision spans must lie within the prompt")
    latent_height, latent_width = latent_raster(canvas)
    rows = packing.padded_tokens
    frames = video_latent_frames(num_frames)
    audio_frames = audio_latent_frames(num_frames)
    spatial, width_grid = _frame_grid(latent_height, latent_width, patch_size)

    positions = torch.zeros((rows, 3), dtype=torch.float64, device="cpu")
    tags = torch.full((rows,), VIDEO_TAG, dtype=torch.int64, device="cpu")
    groups = torch.full((rows,), VIDEO_GROUP, dtype=torch.int64, device="cpu")
    prefix_index = torch.full(
        (rows,), packing.zero_row, dtype=torch.int64, device="cpu"
    )

    text = slice(packing.prefix_start, packing.prefix_start + num_text_tokens)
    positions[text, 0] = torch.arange(
        num_text_tokens, dtype=torch.float64, device="cpu"
    )
    tags[text] = TEXT_TAG
    for start, stop in vision_spans:
        tags[text.start + start : text.start + stop] = VIDEO_TAG
    prefix_index[text] = torch.arange(num_text_tokens, dtype=torch.int64)

    # Condition rows follow the text in packed order and gather the prefix
    # source rows after its text capacity.
    cursor = text.stop
    prefix_index[cursor : cursor + condition_rows] = torch.arange(
        packing.text_rows,
        packing.text_rows + condition_rows,
        dtype=torch.int64,
    )
    clock = float(num_text_tokens)
    keyframes = []
    for segment in segments:
        audio = slice(cursor, cursor + segment.audio_rows)
        visual = slice(audio.stop, audio.stop + segment.video_rows)
        cursor = visual.stop
        tags[audio], groups[audio] = AUDIO_TAG, AUDIO_CONDITION_GROUP
        tags[visual], groups[visual] = VIDEO_TAG, VISUAL_CONDITION_GROUP
        if segment.kind == SegmentKind.KEYFRAME:
            # Placed once the generated timeline's origin is known.
            keyframes.append((segment, visual))
        elif segment.kind == SegmentKind.IMAGE:
            grid, _ = _frame_grid(
                segment.latent_height, segment.latent_width, patch_size
            )
            positions[visual, 0] = clock
            positions[visual, 1:] = grid
            # An image takes one integer slot, not a latent frame's span.
            clock += 1.0
        elif segment.kind == SegmentKind.AUDIO:
            _place_audio(
                positions, audio, segment.audio_frames, clock, width_grid
            )
            clock += float(segment.audio_frames)
        else:
            grid, own_width = _frame_grid(
                segment.latent_height, segment.latent_width, patch_size
            )
            _place_audio(
                positions, audio, segment.audio_frames, clock, own_width
            )
            _place_frames(positions, visual, segment.latent_frames, clock, grid)
            clock += max(
                float(segment.audio_frames), _video_span(segment.latent_frames)
            )
    for segment, visual in keyframes:
        positions[visual, 0] = (
            clock
            if segment.anchor == ConditionRole.FIRST_FRAME
            else clock + _keyframe_span(frames) - ROPE_FRAME_RESCALE
        )
        positions[visual, 1:] = spatial

    _place_frames(
        positions, slice(0, packing.video_rows), frames, clock, spatial
    )
    audio = slice(packing.video_rows, packing.prefix_start)
    _place_audio(positions, audio, audio_frames, clock, width_grid)
    tags[audio], groups[audio] = AUDIO_TAG, AUDIO_GROUP
    return DenseTables(
        position_ids=positions,
        token_tags=tags,
        groups=groups,
        prefix_index=prefix_index,
        used=cursor,
    )


# The (time, height, width) token grid of one video tile of each tile size of
# region packing; dense segments fill whole tiles of the same rows.
VIDEO_TILE_SHAPES = {64: (4, 4, 4), 128: (4, 4, 8)}


def tile_order(
    grid: tuple[int, int, int], shape: tuple[int, int, int]
) -> tuple[np.ndarray, np.ndarray]:
    """Order a ``(time, height, width)`` token grid tile-major.

    Tiles of ``shape`` cover the grid time-major, then by height, then by
    width; the grid's far edges cut boundary tiles short. Within a tile the
    rows inside the grid follow in row-major order, packed at the tile's
    front.

    Returns:
        The raster row (row-major over the grid) of every row in tile-major
        order, and the int32 row count of every tile.
    """
    counts = [math.ceil(extent / size) for extent, size in zip(grid, shape)]
    tile_t, tile_h, tile_w = shape
    tiles = np.arange(counts[0] * counts[1] * counts[2], dtype=np.int64)
    offsets = np.arange(tile_t * tile_h * tile_w, dtype=np.int64)[None, :]
    time = tiles[:, None] // (counts[1] * counts[2]) * tile_t + offsets // (
        tile_h * tile_w
    )
    height = tiles[:, None] // counts[2] % counts[1] * tile_h + (
        offsets // tile_w % tile_h
    )
    width = tiles[:, None] % counts[2] * tile_w + offsets % tile_w
    inside = (time < grid[0]) & (height < grid[1]) & (width < grid[2])
    raster = ((time * grid[1] + height) * grid[2] + width)[inside]
    return raster, inside.sum(axis=1, dtype=np.int32)


def _video_grid(
    frames: int, latent_height: int, latent_width: int
) -> tuple[int, int, int]:
    """Token grid of ``frames`` latent frames of a latent raster."""
    return frames, latent_height // 2, latent_width // 2


def region_tiles(segment: Segment, tile: int) -> int:
    """Tiles one condition occupies in a region packing of ``tile`` rows.

    Its audio rows fill whole tiles, then an image's rows fill whole tiles
    and a video's rows fill the tiles of its token grid.

    Raises:
        ValueError: A keyframe, which region packing does not hold.
    """
    if segment.kind == SegmentKind.KEYFRAME:
        raise ValueError("H3 region packing holds no keyframes")
    tiles = math.ceil(segment.audio_rows / tile)
    if segment.kind == SegmentKind.IMAGE:
        tiles += math.ceil(segment.video_rows / tile)
    elif segment.kind == SegmentKind.VIDEO:
        grid = _video_grid(
            segment.latent_frames, segment.latent_height, segment.latent_width
        )
        tiles += math.prod(
            math.ceil(extent / size)
            for extent, size in zip(grid, VIDEO_TILE_SHAPES[tile])
        )
    return tiles


@dataclass(frozen=True, slots=True)
class RegionPacking:
    """Row extents of one multi-region sparse layout.

    ``[text | conditions | target audio | target video | padding]``, each
    region whole tiles of ``tile`` rows. Text and conditions are capacities
    that a request fills from their start (``region_tables``); the generated
    audio and video rows sit at positions the layout fixes. The generated
    video's rows are tile-major over its token grid (``tile_order``) and the
    stereo audio rows are channel-major.

    Attributes:
        tile: Rows of one tile, a key of ``VIDEO_TILE_SHAPES``.
        text_rows: Text capacity, whole tiles.
        condition_rows: Condition capacity, whole tiles.
        audio_rows: Generated audio rows, both stereo channels.
        video_indices: [video rows] int64 packed row of each generated video
            row, ascending.
        video_raster_indices: [video rows] int64 raster row of each entry of
            ``video_indices``.
        video_valid_sizes: [video tiles] int32 rows of each video tile.
        padded_tokens: All rows, a multiple of the layout's alignment.
    """

    tile: int
    text_rows: int
    condition_rows: int
    audio_rows: int
    video_indices: torch.Tensor
    video_raster_indices: torch.Tensor
    video_valid_sizes: torch.Tensor
    padded_tokens: int

    @property
    def audio_start(self) -> int:
        """First row of the generated audio tiles."""
        return self.text_rows + self.condition_rows

    @property
    def video_start(self) -> int:
        """First row of the generated video tiles."""
        return (
            self.audio_start
            + math.ceil(self.audio_rows / self.tile) * self.tile
        )

    @property
    def audio_indices(self) -> torch.Tensor:
        """[audio rows] int64 packed rows of the generated audio, ascending."""
        return torch.arange(
            self.audio_start,
            self.audio_start + self.audio_rows,
            dtype=torch.int64,
        )

    @property
    def zero_row(self) -> int:
        """Row of the prefix source that holds zeros (see ``region_tables``)."""
        return self.text_rows + self.condition_rows


def region_packing(
    *,
    num_frames: int,
    canvas: image.Config,
    text_rows: int,
    condition_rows: int,
    tile: int,
    token_multiple: int,
) -> RegionPacking:
    """Size the multi-region layout of one frame count, canvas and capacity.

    ``token_multiple``, a multiple of ``tile``, aligns the padded row count so
    that every sequence rank holds an equal shard of whole tiles.

    Raises:
        ValueError: A tile size without a video tile shape, capacities that
            are not whole tiles, a frame count H3 does not generate, a canvas
            off the 32-pixel grid, or an alignment that is not whole tiles.
    """
    if (
        tile not in VIDEO_TILE_SHAPES
        or type(text_rows) is not int
        or text_rows < tile
        or text_rows % tile
        or type(condition_rows) is not int
        or condition_rows < 0
        or condition_rows % tile
        or type(token_multiple) is not int
        or token_multiple < tile
        or token_multiple % tile
    ):
        raise ValueError(
            "a region layout needs whole-tile text and condition capacities "
            "and a whole-tile alignment"
        )
    latent_height, latent_width = latent_raster(canvas)
    grid = _video_grid(
        video_latent_frames(num_frames), latent_height, latent_width
    )
    raster, sizes = tile_order(grid, VIDEO_TILE_SHAPES[tile])
    audio_rows = AUDIO_CHANNELS * audio_latent_frames(num_frames)
    video_start = (
        text_rows + condition_rows + math.ceil(audio_rows / tile) * tile
    )
    offsets = np.arange(tile, dtype=np.int64)[None, :]
    rows = video_start + np.arange(sizes.size, dtype=np.int64)[:, None] * tile
    video_indices = (rows + offsets)[offsets < sizes[:, None]]
    end = video_start + sizes.size * tile
    return RegionPacking(
        tile=tile,
        text_rows=text_rows,
        condition_rows=condition_rows,
        audio_rows=audio_rows,
        video_indices=torch.from_numpy(video_indices),
        video_raster_indices=torch.from_numpy(raster),
        video_valid_sizes=torch.from_numpy(sizes),
        padded_tokens=math.ceil(end / token_multiple) * token_multiple,
    )


@dataclass(frozen=True, slots=True)
class RegionTables:
    """One request's row and tile tables in its multi-region layout.

    Attributes:
        position_ids: [padded_tokens, 3] FP64 rotary coordinates; rows
            outside the request hold zeros.
        token_tags: [padded_tokens] int64 modulation tags; rows outside the
            request carry ``VIDEO_TAG``.
        groups: [padded_tokens] int64 timestep groups.
        prefix_index: [padded_tokens] int64 row of the prefix source each
            packed row takes: the text rows, then the conditions' rows in
            packed order, and the source's zero row for every generated row
            and every row outside the request.
        valid_sizes: [tiles] int32 rows of each tile; zero for a tile the
            request leaves empty.
        tile_regions: [tiles] int32 video region of each tile, ``-1`` for a
            dense or empty tile (``uniserve.nn.attention.vsa.Regions``).
        region_starts: [tiles] int32 tiles below each region index.
        region_keep: [tiles] int32 key tiles a video query keeps of each
            region.
    """

    position_ids: torch.Tensor
    token_tags: torch.Tensor
    groups: torch.Tensor
    prefix_index: torch.Tensor
    valid_sizes: torch.Tensor
    tile_regions: torch.Tensor
    region_starts: torch.Tensor
    region_keep: torch.Tensor


def kept_tiles(sparsity: float, tiles: int) -> int:
    """Tiles of a region of ``tiles`` that a video query keeps at ``sparsity``.

    ``ceil((1 - sparsity) * tiles)``, at least one and at most every tile,
    evaluated in the reference's double-precision arithmetic.
    """
    return max(1, min(math.ceil((1 - sparsity) * tiles), tiles))


def region_tables(
    packing: RegionPacking,
    *,
    num_frames: int,
    canvas: image.Config,
    num_text_tokens: int,
    segments: tuple[Segment, ...],
    vision_spans: tuple[tuple[int, int], ...],
    sparsity: float,
    reference_keep: float,
) -> RegionTables:
    """Build one request's tables in its multi-region layout.

    The rows and their coordinates are the released ``ref2va`` packed
    sequence ``[text | references | target audio | target video]`` of
    ``dense_tables``, held in tiles (FastVideo's ``p2_multi_region`` policy).
    The prompt fills the text tiles from the first. The references follow in
    packed order from the first condition tile, each segment in its own
    tiles: a reference's audio rows, then an image's rows fill whole dense
    tiles, and a reference video forms its own region of video tiles over its
    token grid. The generated audio fills dense tiles and the generated
    video forms the last region. A video query keeps
    ``kept_tiles(1 - reference_keep, n)`` tiles of a reference region of
    ``n`` tiles and ``kept_tiles(sparsity, n)`` of the generated video.

    Raises:
        ValueError: A keyframe, conditions or a prompt that do not fit the
            layout's capacities, or a vision span outside the prompt.
    """
    tile = packing.tile
    if (
        not 1 <= num_text_tokens <= packing.text_rows
        or sum(region_tiles(segment, tile) for segment in segments) * tile
        > packing.condition_rows
    ):
        raise ValueError(
            "the prompt and conditions must fit their layout's capacities"
        )
    if any(
        not 0 <= start < stop <= num_text_tokens for start, stop in vision_spans
    ):
        raise ValueError("vision spans must lie within the prompt")
    rows, tiles = packing.padded_tokens, packing.padded_tokens // tile
    latent_height, latent_width = latent_raster(canvas)
    frames = video_latent_frames(num_frames)
    spatial, width_grid = _frame_grid(latent_height, latent_width, (1, 2, 2))

    positions = torch.zeros((rows, 3), dtype=torch.float64)
    tags = torch.full((rows,), VIDEO_TAG, dtype=torch.int64)
    groups = torch.full((rows,), VIDEO_GROUP, dtype=torch.int64)
    prefix_index = torch.full((rows,), packing.zero_row, dtype=torch.int64)
    valid_sizes = torch.zeros(tiles, dtype=torch.int32)
    tile_regions = torch.full((tiles,), -1, dtype=torch.int32)
    region_tile_counts: list[int] = []

    def dense_rows(first_tile: int, count: int) -> torch.Tensor:
        # Packed rows of a dense segment of ``count`` rows filling tiles from
        # ``first_tile``: whole tiles, the last one partially.
        used = math.ceil(count / tile)
        sizes = torch.full((used,), tile, dtype=torch.int32)
        sizes[-1] = count - (used - 1) * tile
        valid_sizes[first_tile : first_tile + used] = sizes
        return torch.arange(first_tile * tile, first_tile * tile + count)

    def region(first_tile: int, grid: tuple[int, int, int]):
        # A video region's tiles from ``first_tile``: returns the packed row
        # of each grid row in tile-major order with that row's raster index.
        raster, sizes = tile_order(grid, VIDEO_TILE_SHAPES[tile])
        offsets = np.arange(tile, dtype=np.int64)[None, :]
        starts = (first_tile + np.arange(sizes.size, dtype=np.int64)) * tile
        packed = (starts[:, None] + offsets)[offsets < sizes[:, None]]
        valid_sizes[first_tile : first_tile + sizes.size] = torch.from_numpy(
            sizes
        )
        tile_regions[first_tile : first_tile + sizes.size] = len(
            region_tile_counts
        )
        region_tile_counts.append(int(sizes.size))
        return torch.from_numpy(packed), torch.from_numpy(raster)

    # The prompt: text row i at (i, 0, 0), vision tokens tagged video.
    text = dense_rows(0, num_text_tokens)
    positions[text, 0] = torch.arange(num_text_tokens, dtype=torch.float64)
    tags[text] = TEXT_TAG
    for start, stop in vision_spans:
        tags[start:stop] = VIDEO_TAG
    prefix_index[text] = torch.arange(num_text_tokens)

    # References advance the media clock in packed order, as in
    # ``dense_tables``; their prefix source rows follow the text capacity.
    clock = float(num_text_tokens)
    next_tile = packing.text_rows // tile
    source = packing.text_rows
    for segment in segments:
        if segment.audio_rows:
            audio = dense_rows(next_tile, segment.audio_rows)
            next_tile += math.ceil(segment.audio_rows / tile)
            prefix_index[audio] = torch.arange(
                source, source + segment.audio_rows
            )
            source += segment.audio_rows
            tags[audio], groups[audio] = AUDIO_TAG, AUDIO_CONDITION_GROUP
        if segment.kind == SegmentKind.IMAGE:
            visual = dense_rows(next_tile, segment.video_rows)
            next_tile += math.ceil(segment.video_rows / tile)
            prefix_index[visual] = torch.arange(
                source, source + segment.video_rows
            )
            source += segment.video_rows
            grid, _ = _frame_grid(
                segment.latent_height, segment.latent_width, (1, 2, 2)
            )
            positions[visual, 0] = clock
            positions[visual, 1:] = grid
            tags[visual], groups[visual] = VIDEO_TAG, VISUAL_CONDITION_GROUP
            # An image takes one integer slot, not a latent frame's span.
            clock += 1.0
        elif segment.kind == SegmentKind.AUDIO:
            _place_audio(
                positions, audio, segment.audio_frames, clock, width_grid
            )
            clock += float(segment.audio_frames)
        else:
            grid, own_width = _frame_grid(
                segment.latent_height, segment.latent_width, (1, 2, 2)
            )
            if segment.audio_rows:
                _place_audio(
                    positions, audio, segment.audio_frames, clock, own_width
                )
            visual, raster = region(
                next_tile,
                _video_grid(
                    segment.latent_frames,
                    segment.latent_height,
                    segment.latent_width,
                ),
            )
            next_tile += region_tile_counts[-1]
            prefix_index[visual] = source + raster
            block = torch.empty(
                (segment.latent_frames, grid.shape[0], 3), dtype=torch.float64
            )
            block[:, :, 0] = _temporal_grid(segment.latent_frames, clock)[
                :, None
            ]
            block[:, :, 1:] = grid[None]
            positions[visual] = block.reshape(-1, 3).index_select(0, raster)
            source += segment.video_rows
            tags[visual], groups[visual] = VIDEO_TAG, VISUAL_CONDITION_GROUP
            clock += max(
                float(segment.audio_frames), _video_span(segment.latent_frames)
            )

    # The generated audio and video start where the references leave the
    # clock; their rows are the layout's.
    audio = dense_rows(packing.audio_start // tile, packing.audio_rows)
    _place_audio(
        positions,
        audio,
        packing.audio_rows // AUDIO_CHANNELS,
        clock,
        width_grid,
    )
    tags[audio], groups[audio] = AUDIO_TAG, AUDIO_GROUP
    video_tile = packing.video_start // tile
    video_tiles = packing.video_valid_sizes.numel()
    valid_sizes[video_tile : video_tile + video_tiles] = (
        packing.video_valid_sizes
    )
    tile_regions[video_tile : video_tile + video_tiles] = len(
        region_tile_counts
    )
    region_tile_counts.append(video_tiles)
    block = torch.empty((frames, spatial.shape[0], 3), dtype=torch.float64)
    block[:, :, 0] = _temporal_grid(frames, clock)[:, None]
    block[:, :, 1:] = spatial[None]
    positions[packing.video_indices] = block.reshape(-1, 3).index_select(
        0, packing.video_raster_indices
    )

    # References keep their reference rate, the generated video its
    # sparsity; a region's start counts every tile of a lower index, the
    # dense and empty tiles (index -1) included.
    region_keep = torch.zeros(tiles, dtype=torch.int32)
    region_starts = torch.zeros(tiles, dtype=torch.int32)
    below = int((tile_regions < 0).sum())
    for index, count in enumerate(region_tile_counts):
        last = index == len(region_tile_counts) - 1
        region_keep[index] = kept_tiles(
            sparsity if last else 1.0 - reference_keep, count
        )
        region_starts[index] = below
        below += count
    return RegionTables(
        position_ids=positions,
        token_tags=tags,
        groups=groups,
        prefix_index=prefix_index,
        valid_sizes=valid_sizes,
        tile_regions=tile_regions,
        region_starts=region_starts,
        region_keep=region_keep,
    )


def video_order(
    attention, *, num_frames: int, canvas: image.Config
) -> torch.Tensor:
    """Return the raster row of each generated video row in packed order.

    ``attention`` is a denoiser's attention configuration
    (``config.DenseAttention`` or ``config.SparseAttention``): dense packing
    keeps the raster order; tile packing (64-row tiles of one region) and
    region packing (``reference_keep`` set) order the rows tile-major, which
    depends on neither the prompt nor the conditions.
    """
    from .config import DenseAttention

    if isinstance(attention, DenseAttention):
        height, width = latent_raster(canvas)
        return torch.arange(
            video_latent_frames(num_frames) * (height // 2) * (width // 2),
            dtype=torch.int64,
        )
    if attention.reference_keep is not None:
        height, width = latent_raster(canvas)
        raster, _ = tile_order(
            _video_grid(video_latent_frames(num_frames), height, width),
            VIDEO_TILE_SHAPES[attention.tile],
        )
        return torch.from_numpy(raster)
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
