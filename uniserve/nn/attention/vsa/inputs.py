"""Explicit Video Sparse Attention tile domains and borrowed numerical views."""

from __future__ import annotations

from dataclasses import dataclass

import torch

# Rows of one query tile and of one key block that block attention serves.
TILE_SIZES = (64, 128)


@dataclass(frozen=True, slots=True)
class Pattern:
    """Immutable selection cardinalities, independent of mutable selected IDs.

    Counts describe either one shared head or every head. Each bounds its
    query tile's key tiles: the block map's device counts, which backends
    read, may be lower where prefix key tiles hold no valid rows. A dense
    prefix has at most the keys ``range(dense_key_tiles)``. ``tile`` is the
    rows of one query tile and of one key block. Backends may specialize this
    declared mathematical domain without inspecting device data on the host.
    """

    row_counts: tuple[tuple[int, ...], ...]
    dense_prefix_tiles: int
    dense_key_tiles: int
    tile: int

    def __post_init__(self):
        if self.tile not in TILE_SIZES:
            raise ValueError(f"VSA tiles hold one of {TILE_SIZES} rows")
        if (
            not isinstance(self.row_counts, tuple)
            or not self.row_counts
            or not self.row_counts[0]
            or any(
                not isinstance(row, tuple)
                or len(row) != len(self.row_counts[0])
                for row in self.row_counts
            )
            or any(
                type(count) is not int or count < 1
                for row in self.row_counts
                for count in row
            )
            or not 0 <= self.dense_prefix_tiles <= len(self.row_counts[0])
            or self.dense_key_tiles < 0
            or any(
                count != self.dense_key_tiles
                for row in self.row_counts
                for count in row[: self.dense_prefix_tiles]
            )
        ):
            raise ValueError(
                "VSA pattern has invalid tile counts or dense visibility"
            )

    def counts(
        self,
        *,
        num_heads: int,
        query_tiles: int,
        index_width: int,
        out: torch.Tensor,
    ) -> torch.Tensor:
        if (
            len(self.row_counts) not in (1, num_heads)
            or len(self.row_counts[0]) != query_tiles
            or any(
                count > index_width for row in self.row_counts for count in row
            )
            or out.shape != (num_heads, query_tiles)
            or out.dtype != torch.int32
        ):
            raise ValueError(
                "VSA pattern does not match the requested block-map dimensions"
            )
        out.copy_(
            torch.tensor(
                self.row_counts, device="cpu", dtype=torch.int32
            ).expand(num_heads, -1)
        )
        return out


@dataclass(frozen=True, slots=True)
class BlockInput:
    """Per-head selected key-block indices over the declared query-tile
    domain.
    """  # noqa: D205

    pattern: Pattern
    block_indices: torch.Tensor
    block_counts: torch.Tensor
    valid_sizes: torch.Tensor
    query_tile_offset: int

    def __post_init__(self):
        if (
            self.block_indices.ndim != 3
            or self.block_counts.shape != self.block_indices.shape[:2]
            or self.valid_sizes.ndim != 1
            or self.query_tile_offset < 0
            or any(
                value.dtype != torch.int32
                for value in (
                    self.block_indices,
                    self.block_counts,
                    self.valid_sizes,
                )
            )
            or len(self.pattern.row_counts[0]) != self.block_counts.shape[1]
            or len(self.pattern.row_counts)
            not in (1, self.block_counts.shape[0])
        ):
            raise ValueError(
                "VSA block input requires matching tile counts and int32 "
                "index tensors"
            )


@dataclass(frozen=True, slots=True)
class Input:
    """Dense tile-64 prefix and video domains that selection and compression
    share.

    The dense prefix spans ``prefix_tiles`` tiles, but a prefix tile may hold
    no valid rows, as when a text segment has capacity for a longer prompt.
    ``prefix_count`` (int32 device scalar) counts the live prefix tiles, those
    with valid rows; ``prefix_key_indices`` lists them first, and
    ``dense_key_indices`` lists them followed by every video tile. Entries
    past the live ones are unread. Every static count here is an upper bound
    that fixes shapes; the device tables fix which tiles attend, so one
    captured call serves every validity pattern of its domain.
    """  # noqa: D205

    padded_tokens: int
    prefix_tiles: int
    video_tiles: int
    valid_tiles: int
    valid_sizes: torch.Tensor
    prefix_key_indices: torch.Tensor
    dense_key_indices: torch.Tensor
    prefix_count: torch.Tensor

    def __post_init__(self):
        if (
            self.padded_tokens < 128
            or self.padded_tokens % 128
            or min(self.prefix_tiles, self.video_tiles) < 0
            or self.valid_tiles != self.prefix_tiles + self.video_tiles
            or not 0 < self.valid_tiles <= self.padded_tokens // 64
            or self.valid_sizes.shape != (self.padded_tokens // 64,)
            or self.prefix_key_indices.shape != (self.prefix_tiles,)
            or self.dense_key_indices.shape != (self.valid_tiles,)
            or self.prefix_count.numel() != 1
            or any(
                value.dtype != torch.int32
                for value in (
                    self.valid_sizes,
                    self.prefix_key_indices,
                    self.dense_key_indices,
                    self.prefix_count,
                )
            )
        ):
            raise ValueError(
                "VSA input requires complete tile-64 prefix and video index "
                "domains"
            )

    def pattern(
        self,
        query_tiles: int,
        *,
        selected_tiles: int,
        query_tile_offset: int = 0,
    ) -> Pattern:
        """Key-tile count bounds per query tile for a window of the domain.

        Prefix query tiles see at most every dense key tile; video query
        tiles see at most the dense prefix plus their selected video tiles;
        trailing padding tiles attend to a single tile so their count stays
        positive. The block map's device counts never exceed these; they are
        lower where prefix tiles are empty.
        """
        if (
            not 1 <= selected_tiles <= self.video_tiles
            or query_tiles < 1
            or query_tile_offset < 0
        ):
            raise ValueError(
                "VSA selection requires positive bounded video and query "
                "tile counts"
            )
        if query_tile_offset + query_tiles > self.padded_tokens // 64:
            raise ValueError(
                "VSA query tiles exceed their complete token domain"
            )

        counts = tuple(
            self.valid_tiles
            if tile < self.prefix_tiles
            else self.prefix_tiles + selected_tiles
            if tile < self.valid_tiles
            else 1
            for tile in range(
                query_tile_offset, query_tile_offset + query_tiles
            )
        )
        return Pattern(
            (counts,),
            max(0, min(query_tiles, self.prefix_tiles - query_tile_offset)),
            self.valid_tiles,
            64,
        )


# ``Segments.tile_segments`` entry of a tile outside every video segment.
DENSE_TILE = -1


@dataclass(frozen=True, slots=True)
class Segments:
    """A tile-major packed sequence whose video tiles form several segments.

    Every packed row belongs to one ``tile``-row tile. A dense tile's queries
    attend every live key tile, and every video query attends every live
    dense tile. A tile of video segment ``r`` is a video tile: its queries
    attend, besides the dense tiles, the ``segment_keep[s]`` best-scoring key
    tiles of every segment ``s``, each segment selected independently. A tile
    whose valid size is zero is empty: it neither attends nor is attended,
    so a layout can reserve capacity that a request leaves unused. The
    physical tile order is free; selection reads the tables alone.

    Every table is a device int32 vector with one entry per tile, so the
    shapes depend on ``padded_tokens`` alone and one captured call serves
    every assignment of tiles to segments.

    Attributes:
        tile: Rows of one tile, one of ``TILE_SIZES``.
        padded_tokens: Packed rows, whole tiles.
        valid_sizes: Valid rows of each tile; they lead the tile's rows.
        tile_segments: ``DENSE_TILE`` for a dense or empty tile, else the
            tile's segment index.
        segment_starts: For each segment index ``r``, the number of tiles
            whose ``tile_segments`` entry is below ``r``: the first position
            of segment ``r`` when the key tiles are ordered by segment index.
            Entries past the last segment are unread.
        segment_keep: For each segment index, the key tiles a video query
            keeps of that segment, at least one and at most the segment's
            tiles. Entries past the last segment are unread.
    """

    tile: int
    padded_tokens: int
    valid_sizes: torch.Tensor
    tile_segments: torch.Tensor
    segment_starts: torch.Tensor
    segment_keep: torch.Tensor

    def __post_init__(self):
        if (
            self.tile not in TILE_SIZES
            or self.padded_tokens < self.tile
            or self.padded_tokens % self.tile
        ):
            raise ValueError("VSA segments require whole tiles of a VSA tile")
        tiles = self.padded_tokens // self.tile
        if any(
            value.shape != (tiles,) or value.dtype != torch.int32
            for value in (
                self.valid_sizes,
                self.tile_segments,
                self.segment_starts,
                self.segment_keep,
            )
        ):
            raise ValueError(
                "VSA segment tables require one int32 entry per tile"
            )

    @property
    def tiles(self) -> int:
        """Tiles of the packed sequence."""
        return self.padded_tokens // self.tile


@dataclass(frozen=True, slots=True)
class NormRope:
    """Learned Q/K normalization and partial rotation applied while packing.

    Each head is RMS-normalized in fp32 with its learned weight, then an even
    prefix twice the width of the compact factors is rotated split-half; the
    remaining columns are only normalized. ``cos`` and ``sin`` hold one row
    of compact factors per padded token, indexed by global token position.
    """

    query_weight: torch.Tensor
    key_weight: torch.Tensor
    eps: float
    cos: torch.Tensor
    sin: torch.Tensor

    def __post_init__(self):
        if (
            self.query_weight.shape != self.key_weight.shape
            or self.query_weight.ndim != 1
            or self.cos.ndim != 2
            or self.sin.shape != self.cos.shape
            or self.cos.shape[1] * 2 > self.query_weight.shape[0]
            or self.eps < 0
        ):
            raise ValueError(
                "VSA norm-rope needs head-wide weights and compact factors "
                "within the head"
            )


@dataclass(frozen=True, slots=True)
class Workspace:
    """Borrow scratch for tile pooling, selection, compression and fine
    attention.
    """  # noqa: D205

    attention_output: torch.Tensor
    tile_scores: torch.Tensor
    block_counts: torch.Tensor
    block_indices: torch.Tensor
    pooled_query: torch.Tensor
    pooled_key: torch.Tensor
    pooled_value: torch.Tensor
    compressed_tiles: torch.Tensor
