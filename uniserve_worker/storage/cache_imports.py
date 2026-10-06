"""Numerical KV readback and page-representation conversion.

Rust owns import admission, workspaces, cancellation and physical retirement.
These routines allocate conversion tensors, construct borrowed views and
convert page representations. Native execution selects and retains reads.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import torch

from uniserve.cache import mha
from uniserve.cache.state import decode_region
from uniserve.quantization import QuantizedTensor, Quantizer
from uniserve_worker._uniserve_ipc import KVImport, KVImporter

if TYPE_CHECKING:
    from uniserve_worker.storage.kv_cache import KVCacheManager

__all__ = [
    "KVImport",
    "KVImporter",
    "cache_transfer_workspace_bytes",
    "import_page_sizes",
]


def cache_transfer_workspace_bytes(
    *,
    page_elements: int,
    page_scales: int,
    capacity: int,
) -> int:
    """Return the startup bytes for every bounded import workspace.

    Startup memory accounting (`bootstrap.report`) reserves this amount, so it
    must match the `KVWorkspace` allocation in `_allocate`.
    ``page_elements`` is the largest ``page_tokens * layers * kv_heads *
    head_dim`` of any cache group and ``page_scales`` the largest
    ``page_tokens * layers * kv_heads``. Each import thread has two raw pages
    (one per K/V field) large enough for float64 elements, one FP32
    conversion page, and an FP32 scale buffer of ``[page_tokens, 2, layers,
    kv_heads]``: a destination span touches at most one source page per
    token and one source scale group per head. The tail of a raw page serves
    as rounding scratch while it holds an FP8 input. No prefix-sized
    conversion allocation is needed.
    """
    workers = min(4, int(capacity))
    # 20 bytes per element: two raw float64 pages (2 x 8) plus one FP32
    # conversion page (4). Scale entries are FP32 (4 bytes each), two per
    # token, layer and head.
    return workers * (20 * int(page_elements) + 8 * int(page_scales))


def import_page_sizes(pool: KVCacheManager) -> tuple[int, int]:
    """Return the largest group page's elements and scale entries.

    The pair sizes `cache_transfer_workspace_bytes` for ``pool``: elements
    are ``page_tokens * layers * kv_heads * head_dim`` and scale entries
    ``page_tokens * layers * kv_heads``, each the largest over the groups.
    """
    elements = scales = 0
    for group in pool.cache.groups:
        rows = group.page_tokens * len(group.layers) * group.num_kv_heads
        elements = max(elements, rows * group.head_dim)
        scales = max(scales, rows)
    return elements, scales


@dataclass(slots=True)
class KVWorkspace:
    """Fixed page buffers for one KV copy worker.

    With ``elements`` the element count of the largest group page (``page
    tokens * layers * kv_heads * head_dim``) and ``scales`` its scale
    entries (``page tokens * layers * kv_heads``), ``raw`` is uint8
    ``[2, elements * 8]``: one buffer page per K/V field, sized for float64
    source elements. ``values`` is a flat FP32 ``[elements]`` conversion
    page and ``scales`` a flat FP32 ``[2 * scales]`` buffer for the source
    scale rows of one span; each group views them in its own shape.
    """

    raw: torch.Tensor
    values: torch.Tensor
    scales: torch.Tensor


def _allocate(pool: KVCacheManager, workers: int) -> tuple[KVWorkspace, ...]:
    elements, scales = import_page_sizes(pool)
    device = pool.cache.device
    return tuple(
        KVWorkspace(
            raw=torch.empty(
                (2, elements * 8), dtype=torch.uint8, device=device
            ),
            values=torch.empty(elements, dtype=torch.float32, device=device),
            scales=torch.empty(2 * scales, dtype=torch.float32, device=device),
        )
        for _ in range(workers)
    )


def _direct_views(
    pool: KVCacheManager,
    index: int,
    column: int,
    spans: tuple[tuple[int, int, int], ...],
) -> tuple[
    tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...] | None], ...
]:
    """Borrow one layer's values and scales at native-selected unit spans."""
    name = pool.cache.groups[index].layers[column]
    state = pool.cache.state(name)
    fields = []
    for field in ("key", "value"):
        tensor = state.tensors[field]
        if isinstance(tensor, QuantizedTensor):
            buffers = tensor.buffers()
            values = buffers["values"]
            scales = tuple(
                buffers["scale"][unit : unit + 1] for unit, _, _ in spans
            )
        else:
            values, scales = tensor, None

        # The single-layer axis matches the exported token/layer/head layout.
        views = tuple(
            values[unit, offset : offset + count].unsqueeze(1)
            for unit, offset, count in spans
        )
        fields.append((views, scales))
    return tuple(fields)


def _conversion_views(
    workspace: KVWorkspace,
    dtype: str,
    shape: tuple[int, ...],
    scale_shape: tuple[int, ...],
) -> tuple[tuple[torch.Tensor, ...], torch.Tensor | None]:
    """Borrow one conversion page's raw K/V and optional FP8 scale views."""
    from math import prod

    storage_dtype = getattr(torch, dtype)
    nbytes = prod(shape) * workspace.raw.view(storage_dtype).element_size()
    raw = tuple(
        workspace.raw[field, :nbytes].view(storage_dtype).reshape(shape)
        for field in range(2)
    )
    scales = (
        workspace.scales[: prod(scale_shape)].view(scale_shape)
        if scale_shape
        else None
    )
    return raw, scales


def _copy_converted(
    pool: KVCacheManager,
    index: int,
    workspace: KVWorkspace,
    raw: tuple[torch.Tensor, ...],
    scales: torch.Tensor | None,
    units: tuple[int, ...],
    offset: int,
    source_offset: int,
    page_tokens: int,
    scale_head_size: int,
    compute_dtype: torch.dtype,
) -> None:
    """Decode one source span and encode it into the selected destination units.

    Source scales are grouped by source page and head range. Destination
    encoding and rounding remain owned by the numerical cache library.
    """
    advertised = pool.info.groups[index]
    cache_group = pool.cache.groups[index]
    columns = pool.cache.planes.columns
    count, num_layers, num_heads, head_dim = raw[0].shape
    trailing = (num_layers, num_heads, head_dim)
    elements = raw[0].numel()

    for field_index, source in enumerate(raw):
        values = source
        if scales is not None:
            values = workspace.values[:elements].view(count, *trailing)
            # Intersect source scale pages and head groups here; the
            # numerical cache library owns decoding and rounding.
            # Rounding scratch lies in the raw page past the FP8
            # input, which occupies at most its first eighth. FP32 and
            # FP64 compute dtypes need no rounding scratch.
            rounded = (
                workspace.raw[field_index, elements * 4 :].view(compute_dtype)
                if compute_dtype not in {torch.float32, torch.float64}
                else workspace.raw[field_index, :0].view(compute_dtype)
            )
            # Walk source pages (one scale row each) and, within each,
            # this worker's heads grouped by source scale group.
            cursor, scale_index, token_offset = 0, 0, source_offset
            while cursor < count:
                length = min(page_tokens - token_offset, count - cursor)
                head, head_group = 0, 0
                while head < num_heads:
                    heads = min(
                        scale_head_size
                        - (advertised.kv_head_offset + head) % scale_head_size,
                        num_heads - head,
                    )
                    for layer in range(num_layers):
                        page_region = (
                            slice(cursor, cursor + length),
                            layer,
                            slice(head, head + heads),
                        )
                        encoded = source[page_region]
                        numerical = Quantizer("fp8").from_tensors(
                            {
                                "values": encoded,
                                "scale": scales[
                                    scale_index,
                                    field_index,
                                    layer,
                                    head_group,
                                ].reshape(()),
                            },
                            shape=tuple(encoded.shape),
                            dtype=compute_dtype,
                        )
                        decode_region(
                            numerical,
                            tuple(slice(0, size) for size in encoded.shape),
                            device=values.device,
                            workspace={
                                "values": values[page_region],
                                "rounded": rounded,
                            },
                        )
                    head += heads
                    head_group += 1
                cursor += length
                scale_index += 1
                token_offset = 0

        # Store the converted tokens into each layer's unit of the
        # destination page.
        for column_index, name in enumerate(cache_group.layers):
            unit = units[column_index // columns]
            trailing_slice = (slice(0, num_heads), slice(0, head_dim))
            # The cache manager admits only MHA state layers.
            state = cast(mha.State, pool.cache.state(name))
            state.copy_region(
                values[:, column_index],
                field=("key", "value")[field_index],
                block=unit,
                source_slice=(slice(0, count), *trailing_slice),
                target_slice=(
                    slice(offset, offset + count),
                    *trailing_slice,
                ),
                workspace={},
            )
