"""The 128-to-64-row tile split writes the selection its rule defines.

Each 128-row query tile becomes two 64-row halves selecting the same key
tiles, each selected key tile ``j`` becomes kernel tiles ``2j`` and
``2j + 1``, and a key tile's ``valid`` rows split as ``min(valid, 64)``
and the rest. Entries past a row's count follow the same rule.
"""

import pytest
import torch

from uniserve.runtime.backends.attention.vsa import _half_tiles

pytestmark = [pytest.mark.unit, pytest.mark.gpu]

# Valid rows of a key tile in each class the split distinguishes: empty,
# within the first half, exactly the first half, reaching into the second
# half and full.
_SIZE_CLASSES = (0, 1, 63, 64, 65, 127, 128)


def _reference(indices, counts, sizes):
    """The split rule in PyTorch."""
    heads, query_tiles, width = indices.shape
    halves = indices.repeat_interleave(2, dim=1)
    split_indices = torch.stack((2 * halves, 2 * halves + 1), dim=-1)
    return (
        split_indices.reshape(heads, 2 * query_tiles, 2 * width),
        (2 * counts).repeat_interleave(2, dim=1),
        torch.stack(
            (sizes.clamp(max=64), (sizes - 64).clamp(min=0)), dim=-1
        ).reshape(-1),
    )


def _allocation(heads, query_tiles, width, key_tiles, generator):
    """A random selection inside a wider allocation (see ``_window``).

    Every row keeps a leading count of distinct key tiles below the width;
    entries past the count are padding of arbitrary key tiles.
    """
    indices = torch.randint(
        0, key_tiles, (heads, query_tiles + 3, width + 5), generator=generator
    ).to(torch.int32)
    for head in range(heads):
        for tile in range(query_tiles):
            indices[head, tile + 2, :width] = torch.randperm(
                key_tiles, generator=generator
            )[:width]
    counts = torch.randint(
        0, width, (heads, query_tiles + 3), generator=generator
    ).to(torch.int32)
    return indices, counts


def _window(indices, counts, query_tiles, width):
    """The selection: a window whose head and tile strides are not compact."""
    rows = slice(2, 2 + query_tiles)
    return indices[:, rows, :width], counts[:, rows]


def _sizes(key_tiles, generator):
    """Valid rows of each key tile, covering every value 0..128 if it can.

    With fewer than 129 key tiles, the sizes hold every class of
    ``_SIZE_CLASSES``; random sizes fill the remaining tiles.
    """
    covered = (
        torch.arange(129)
        if key_tiles >= 129
        else torch.tensor(_SIZE_CLASSES, dtype=torch.int64)
    )
    filled = torch.randint(
        0, 129, (key_tiles - covered.numel(),), generator=generator
    )
    order = torch.randperm(key_tiles, generator=generator)
    return torch.cat((covered, filled))[order].to(torch.int32)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize(
    ("heads", "query_tiles", "width", "key_tiles"),
    [
        # One head, one query tile selecting at most one key tile.
        (1, 1, 2, 7),
        # Several heads and query tiles over a few key tiles.
        (3, 7, 5, 9),
        # Every valid size 0..128, and more key tiles than one program
        # of the size split covers.
        (24, 40, 33, 300),
        # A width beyond one step of the selection split.
        (5, 130, 200, 257),
    ],
)
def test_split_matches_the_rule(heads, query_tiles, width, key_tiles):
    if not _half_tiles.available():
        pytest.skip("Triton is unavailable")
    generator = torch.Generator().manual_seed(
        heads * 1_000_003 + query_tiles * 1009 + width * 31 + key_tiles
    )
    allocation = _allocation(heads, query_tiles, width, key_tiles, generator)
    indices, counts = _window(*allocation, query_tiles, width)
    sizes = _sizes(key_tiles, generator)
    assert set(_SIZE_CLASSES) <= set(sizes.tolist())
    assert bool((counts < width).all())

    device = torch.device("cuda")
    device_indices, device_counts = _window(
        *(value.to(device) for value in allocation), query_tiles, width
    )
    # The split reads the selection through the allocation's strides.
    assert device_indices.stride() == (
        (query_tiles + 3) * (width + 5),
        width + 5,
        1,
    )
    assert device_counts.stride() == (query_tiles + 3, 1)
    work = {
        # Every entry the split leaves unwritten keeps this value.
        name: torch.full(config.shape, -7, dtype=config.dtype, device=device)
        for name, config in _half_tiles.work_areas(
            heads=heads,
            query_tiles=query_tiles,
            width=width,
            key_tiles=key_tiles,
        ).items()
    }
    split = _half_tiles.split(
        device_indices, device_counts, sizes.to(device), work
    )

    assert all(
        actual is work[name]
        for actual, name in zip(split, ("indices", "counts", "sizes"))
    )
    for actual, expected in zip(
        split, _reference(indices, counts, sizes), strict=True
    ):
        assert torch.equal(actual.cpu(), expected.to(torch.int32))
