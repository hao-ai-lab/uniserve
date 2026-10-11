"""Explicit Video Sparse Attention tile domains and block selections."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class Pattern:
    """Immutable selection cardinalities, independent of mutable selected IDs.

    Counts describe either one shared head or every head. Each bounds its
    query tile's key tiles: the block map's device counts, which backends
    read, may be lower where prefix key tiles hold no valid rows. A dense
    prefix has at most the keys ``range(dense_key_tiles)``. ``tile`` is the
    rows of one query tile and of one key block; resolving a backend refuses
    a tile no installed kernel serves. Backends may specialize this declared
    mathematical domain without inspecting device data on the host.
    """

    row_counts: tuple[tuple[int, ...], ...]
    dense_prefix_tiles: int
    dense_key_tiles: int
    tile: int

    def __post_init__(self):
        if (
            type(self.tile) is not int
            or self.tile < 1
            or not isinstance(self.row_counts, tuple)
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
                "VSA pattern has an invalid tile, tile counts or dense "
                "visibility"
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
        tile: Rows of one tile.
        padded_tokens: Packed rows, whole tiles.
        valid_sizes: Valid rows of each tile; they lead the tile's
            rows.
        tile_segments: ``DENSE_TILE`` for a dense or empty tile, else the
            tile's segment index.
        segment_starts: For each segment index ``r``, the number of tiles
            whose ``tile_segments`` entry is below ``r``: the first position
            of segment ``r`` when the key tiles are ordered by segment index.
            Entries past the last segment are unread.
        segment_keep: For each segment index, the key tiles a video query
            keeps of that segment, at least one and at most the segment's
            tiles. Entries past the last segment are unread.
        dense_prefix_tiles: Leading tiles that are dense or empty in every
            assignment of the layout, so their queries attend every live key
            tile; a backend may evaluate them as one dense prefix. Zero
            states none.
        key_bounds: For each tile, an upper bound on the key tiles its
            queries attend, which a backend that plans from static counts
            sizes its plan by; ``None`` states every tile for every query.
            A dense tile's queries attend every live tile.
    """

    tile: int
    padded_tokens: int
    valid_sizes: torch.Tensor
    tile_segments: torch.Tensor
    segment_starts: torch.Tensor
    segment_keep: torch.Tensor
    dense_prefix_tiles: int = 0
    key_bounds: tuple[int, ...] | None = None

    def __post_init__(self):
        if (
            self.tile < 1
            or self.padded_tokens < self.tile
            or self.padded_tokens % self.tile
        ):
            raise ValueError("VSA segments require whole tiles")
        tiles = self.padded_tokens // self.tile
        if not 0 <= self.dense_prefix_tiles <= tiles or (
            self.key_bounds is not None
            and (
                len(self.key_bounds) != tiles
                or any(not 1 <= bound <= tiles for bound in self.key_bounds)
                or any(
                    bound != tiles
                    for bound in self.key_bounds[: self.dense_prefix_tiles]
                )
            )
        ):
            raise ValueError(
                "VSA dense prefix and key bounds must lie within the tiles"
            )
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
