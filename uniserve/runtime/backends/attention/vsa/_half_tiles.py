"""128-row VSA tile selections rewritten in a 64-row kernel's tiles.

A block-sparse kernel whose selection unit is a 64-row tile attends a
selection of 128-row tiles once this rule rewrites it in 64-row tiles:

* Each 128-row query tile becomes its two 64-row halves, and both halves
  select the key tiles the full tile selects.
* Each selected 128-row key tile ``j`` becomes kernel tiles ``2j`` and
  ``2j + 1``, so the count of every selection row doubles.
* A key tile's valid rows lead the tile, so its ``valid`` rows split as
  ``min(valid, 64)`` in kernel tile ``2j`` and the rest,
  ``max(valid - 64, 0)``, in kernel tile ``2j + 1``.

Every query row therefore attends exactly the valid keys of its 128-row
selection, provided the kernel's online softmax gives a kernel tile with
no valid rows no weight. Selection entries past a row's count are
rewritten by the same rule; the count bounds them, so the kernel never
reads them.

The source selection is ``indices`` ``[heads, query tiles, width]``, its
entries contiguous along the last axis, and ``counts`` ``[heads, query
tiles]``, contiguous along the tiles; ``sizes`` holds the valid rows of
each 128-row key tile contiguously. Head and tile strides of the source
are kernel arguments, so a strided view of the selection needs no copy.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch

from uniserve.tensors import BufferConfig

try:  # pragma: no cover - CUDA provider dependency.
    import triton
    import triton.language as tl
except BaseException:  # pragma: no cover
    triton = None

# Entries per step: ``_split_selection_kernel`` rewrites a selection row
# this many entries at a time, and each ``_split_sizes_kernel`` program
# splits this many key tiles.
_BLOCK = 128


def available() -> bool:
    """Report whether Triton, which runs the rewrite, is installed."""
    return triton is not None


def work_areas(
    *, heads: int, query_tiles: int, width: int, key_tiles: int
) -> dict[str, BufferConfig]:
    """Describe the work areas ``split`` writes for one selection.

    Args:
        heads: Heads of the 128-row tile selection.
        query_tiles: Its 128-row query tiles.
        width: Its entries per query tile.
        key_tiles: The 128-row key tiles it selects from.

    Returns:
        Compact ``indices``, ``counts`` and ``sizes``: the selection in
        kernel tiles and the valid rows of every kernel key tile.
    """
    return {
        "indices": BufferConfig(
            (heads, 2 * query_tiles, 2 * width), torch.int32
        ),
        "counts": BufferConfig((heads, 2 * query_tiles), torch.int32),
        "sizes": BufferConfig((2 * key_tiles,), torch.int32),
    }


if triton is not None:

    @triton.jit
    def _split_selection_kernel(
        source_indices,
        source_counts,
        indices,
        counts,
        source_stride_head,
        source_stride_tile,
        count_stride_head,
        stride_head,
        stride_tile,
        query_tiles,
        width,
        block: tl.constexpr,
    ):
        """Write one 128-row query tile's selection as two 64-row rows.

        Selected key tile ``j`` becomes kernel tiles ``2j`` and ``2j + 1``,
        and both query halves select the same key tiles; entries past the
        live count are copied but never read, since the count bounds them.
        Shapes and strides are runtime values, so every layout shares one
        compiled kernel.
        """
        row = tl.program_id(0)
        head = row // query_tiles
        tile = row % query_tiles
        count = tl.load(source_counts + head * count_stride_head + tile)
        for half in tl.static_range(2):
            tl.store(
                counts + head * (2 * query_tiles) + 2 * tile + half,
                2 * count,
            )
        for start in range(0, width, block):
            offsets = start + tl.arange(0, block)
            active = offsets < width
            selected = tl.load(
                source_indices
                + head * source_stride_head
                + tile * source_stride_tile
                + offsets,
                mask=active,
                other=0,
            )
            for half in tl.static_range(2):
                destination = (
                    indices
                    + head * stride_head
                    + (2 * tile + half) * stride_tile
                    + 2 * offsets
                )
                tl.store(destination, 2 * selected, mask=active)
                tl.store(destination + 1, 2 * selected + 1, mask=active)

    @triton.jit
    def _split_sizes_kernel(
        source_sizes, sizes, key_tiles, block: tl.constexpr
    ):
        """Split each 128-row key tile's leading valid rows between halves."""
        offsets = tl.program_id(0) * block + tl.arange(0, block)
        active = offsets < key_tiles
        valid = tl.load(source_sizes + offsets, mask=active, other=0)
        tl.store(sizes + 2 * offsets, tl.minimum(valid, 64), mask=active)
        tl.store(
            sizes + 2 * offsets + 1, tl.maximum(valid - 64, 0), mask=active
        )


def split(
    indices: torch.Tensor,
    counts: torch.Tensor,
    sizes: torch.Tensor,
    work: Mapping[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Rewrite a 128-row tile selection in kernel tiles.

    Args:
        indices: ``[heads, query tiles, width]`` selected key tiles of
            every query tile.
        counts: ``[heads, query tiles]`` live entries of every row of
            ``indices``.
        sizes: Valid rows of every 128-row key tile.
        work: The work areas ``work_areas`` describes for this selection.

    Returns:
        ``work``'s ``indices``, ``counts`` and ``sizes``, holding the
        selection in kernel tiles.
    """
    heads, query_tiles, width = (int(size) for size in indices.shape)
    key_tiles = int(sizes.numel())
    _split_selection_kernel[(heads * query_tiles,)](
        indices,
        counts,
        work["indices"],
        work["counts"],
        int(indices.stride(0)),
        int(indices.stride(1)),
        int(counts.stride(0)),
        int(work["indices"].stride(0)),
        int(work["indices"].stride(1)),
        query_tiles,
        width,
        _BLOCK,
        num_warps=4,
        num_stages=1,
    )
    _split_sizes_kernel[(triton.cdiv(key_tiles, _BLOCK),)](
        sizes, work["sizes"], key_tiles, _BLOCK, num_warps=4
    )
    return work["indices"], work["counts"], work["sizes"]
