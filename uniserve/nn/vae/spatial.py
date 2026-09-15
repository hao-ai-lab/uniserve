"""Shared spatial tiling for numerical video decoders."""

from __future__ import annotations

import math

import torch
from torch import nn

from uniserve.nn.video import blend_decoded_overlap


def split_tiles(
    length: int, tile_size: int, minimum_overlap: int, alignment: int
) -> tuple[list[int], list[int], list[int]]:
    """Cover an aligned output extent, distributing excess overlap in order."""

    if (
        min(length, tile_size, alignment) < 1
        or not 0 <= minimum_overlap < tile_size
        or any(value % alignment for value in (length, tile_size, minimum_overlap))
    ):
        raise ValueError("spatial tiles and overlap must align with the decoder scale")
    if tile_size >= length:
        return [0], [length], []
    tile_count = math.ceil(length / tile_size)
    while tile_size * tile_count - minimum_overlap * (tile_count - 1) < length:
        tile_count += 1
    overlaps = [minimum_overlap] * (tile_count - 1)
    remaining = tile_size * tile_count - sum(overlaps) - length
    for index in range(remaining // alignment):
        overlaps[index % (tile_count - 1)] += alignment
    starts = [0]
    for overlap in overlaps:
        starts.append(starts[-1] + tile_size - overlap)
    return starts, [tile_size] * tile_count, overlaps


def stitch_tiles(
    tiles: list[list[torch.Tensor]],
    height_overlaps: list[int],
    width_overlaps: list[int],
) -> torch.Tensor:
    """Blend the original neighboring tiles vertically, then horizontally.

    Neighbors remain unmodified so corner arithmetic follows the same order as
    edge arithmetic. Trimming happens only after both blends. The decoded dtype
    determines the cross-fade weights and intermediate rounding.
    """

    assembled_rows: list[torch.Tensor] = []
    for row_index, row in enumerate(tiles):
        assembled: list[torch.Tensor] = []
        for column_index, tile in enumerate(row):
            if row_index:
                tile = blend_decoded_overlap(
                    tiles[row_index - 1][column_index], tile, height_overlaps[row_index - 1], -2
                )
            if column_index:
                tile = blend_decoded_overlap(
                    row[column_index - 1], tile, width_overlaps[column_index - 1], -1
                )
            if row_index + 1 < len(tiles) and height_overlaps[row_index]:
                tile = tile[..., : -height_overlaps[row_index], :]
            if column_index + 1 < len(row) and width_overlaps[column_index]:
                tile = tile[..., :, : -width_overlaps[column_index]]
            assembled.append(tile)
        assembled_rows.append(torch.cat(assembled, dim=-1))
    return torch.cat(assembled_rows, dim=-2)


class SpatialDecoder(nn.Module):
    """Decode an aligned raster directly or as one batch of overlapping tiles.

    The concrete ``forward`` defines any latent-channel projection before its
    decoder. Tile extents and overlaps are output pixels, aligned to the
    spatial compression ratio. Blending retains decoded-dtype rounding.
    """

    def __init__(
        self,
        decoder: nn.Module,
        *,
        spatial_compression: int,
        tile_height: int,
        tile_width: int,
        overlap_height: int,
        overlap_width: int,
    ):
        super().__init__()
        for extent, overlap in ((tile_height, overlap_height), (tile_width, overlap_width)):
            split_tiles(extent, extent, overlap, spatial_compression)
        self.decoder = decoder
        self.spatial_compression = spatial_compression
        self.tile_height, self.tile_width = tile_height, tile_width
        self.overlap_height, self.overlap_width = overlap_height, overlap_width

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        return self.decoder(latents)

    def decode(self, latents: torch.Tensor, *, tiled: bool) -> torch.Tensor:
        """Restore NCTHW latents, preserving sample independence within each tile."""
        if latents.ndim != 5 or latents.shape[0] < 1:
            raise ValueError("spatial decoding requires a nonempty NCTHW latent")
        if not tiled:
            return self(latents)
        ratio = self.spatial_compression
        y_indices, y_lengths, y_overlaps = split_tiles(
            int(latents.shape[-2]) * ratio, self.tile_height, self.overlap_height, ratio
        )
        x_indices, x_lengths, x_overlaps = split_tiles(
            int(latents.shape[-1]) * ratio, self.tile_width, self.overlap_width, ratio
        )
        tiles = torch.cat(
            tuple(
                latents[
                    ...,
                    y_pos // ratio : (y_pos + y_length) // ratio,
                    x_pos // ratio : (x_pos + x_length) // ratio,
                ]
                for y_pos, y_length in zip(y_indices, y_lengths, strict=True)
                for x_pos, x_length in zip(x_indices, x_lengths, strict=True)
            ),
            dim=0,
        )
        decoded = self(tiles)
        flat_tiles = decoded.split(latents.shape[0], dim=0)
        columns = len(x_indices)
        rows = [
            list(flat_tiles[start : start + columns])
            for start in range(0, len(flat_tiles), columns)
        ]
        return stitch_tiles(rows, y_overlaps, x_overlaps)
