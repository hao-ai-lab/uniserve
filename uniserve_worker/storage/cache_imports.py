"""Numerical KV readback and page-representation conversion.

Rust owns import admission, workspaces, cancellation and physical retirement.
These routines allocate conversion tensors and copy direct or converted page
spans through the public cache and transport interfaces.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, cast

import torch

from uniserve.cache import mha
from uniserve.cache.state import decode_region
from uniserve.quantization import QuantizedTensor, Quantizer
from uniserve_worker._uniserve_ipc import KVImport, KVImporter
from uniserve_worker.protocol.transfer import (
    KvGroupTransfer,
    KvTransfer,
    TensorTransfer,
)
from uniserve_worker.transport.fetch import fetch_tensor
from uniserve_worker.transport.interface import Transport
from uniserve_worker.transport.ticket import TransferTicket

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
    ``[2, elements * 8]``: one staging page per K/V field, sized for float64
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


def _copy(
    pool: KVCacheManager,
    importer: KVImporter,
    write: KVImport,
    transports: Mapping[str, Transport],
    workspace: KVWorkspace,
) -> None:
    publication = write.publication
    for index, group in enumerate(publication.groups):
        if not group.tensors:
            continue

        if _direct(pool, publication, group, index):
            _copy_direct(pool, importer, write, index, group, transports)
        else:
            _copy_converted(
                pool, importer, write, index, group, transports, workspace
            )


def _fetch(
    importer: KVImporter,
    write: KVImport,
    tensor: TensorTransfer,
    destination: torch.Tensor | tuple[torch.Tensor, ...],
    transports: Mapping[str, Transport],
    *,
    region: tuple[slice, ...] | None = None,
) -> tuple[TransferTicket, ...]:
    # Start transport reads of ``region`` of the source tensor into
    # ``destination``, using only locations whose backend this worker has
    # a transport for. Each started read is retained on ``write``.
    importer._require_active(write)
    return fetch_tensor(
        tensor,
        destination,
        bindings={
            (location.source, location.backend): transports[location.backend]
            for location in tensor.locations
            if location.backend in transports
        },
        region=region,
        retain=partial(importer._retain, write),
    )


def _direct(
    pool: KVCacheManager,
    publication: KvTransfer,
    group: KvGroupTransfer,
    index: int,
) -> bool:
    """Whether a group's source bytes can land in destination units as is.

    The direct path needs an identical page representation. For FP8 the
    page scales must also transfer unchanged: equal page sizes, a
    page-aligned carried interval, the same compute dtype, and all of
    this rank's heads in one source scale group, because each
    destination unit holds one scale per column and K/V field.
    """
    info = pool.info
    advertised = info.groups[index]
    if group.tensors[0].dtype != info.dtype:
        return False
    if info.dtype != "float8_e4m3fn":
        return True
    page_tokens = pool.shapes[index].page_tokens
    return (
        group.page_tokens == page_tokens
        # A partial installed page owns its destination scale. Its base
        # may have arrived through another TP layout.
        and group.start % page_tokens == 0
        and publication.compute_dtype
        == str(pool.compute_dtypes[index]).removeprefix("torch.")
        and advertised.kv_head_offset // group.scale_head_size
        == (advertised.kv_head_offset + advertised.num_kv_heads - 1)
        // group.scale_head_size
    )


def _copy_direct(
    pool: KVCacheManager,
    importer: KVImporter,
    write: KVImport,
    index: int,
    group: KvGroupTransfer,
    transports: Mapping[str, Transport],
) -> None:
    """Fetch one group's source values, and FP8 scales, into its units.

    The fetch region selects each of this worker's layers and KV heads
    by their offsets on the publication's group layer and head axes.
    """
    publication = write.publication
    table = write.tables[index]
    carried = publication.published_extent - group.start
    advertised = pool.info.groups[index]
    axis = pool.axes[index]
    cache_group = pool.cache.groups[index]
    columns = pool.cache.planes.columns
    page_tokens = table.shape.page_tokens
    # Pages the carried tokens touch, as (absolute page, offset, count).
    pages = []
    position = group.start
    while position < publication.published_extent:
        page, offset = divmod(position, page_tokens)
        count = min(
            publication.published_extent - position, page_tokens - offset
        )
        pages.append((page, offset, count))
        position += count

    # ``layer`` indexes the publication's group layer axis.
    for column_index, name in enumerate(cache_group.layers):
        row = column_index // columns
        units = table.row(row)
        layer = axis.offset + column_index
        # Limit physical reads to one layer, independent of model depth.
        tickets: list[TransferTicket] = []
        state = pool.cache.state(name)
        for field_index, tensor_name in enumerate(("key", "value")):
            tensor = state.tensors[tensor_name]
            values = (
                tensor.buffers()["values"]
                if isinstance(tensor, QuantizedTensor)
                else tensor
            )
            # Each span view is [tokens, 1, kv heads, head dim]: the
            # unsqueezed axis matches the single-layer fetch region.
            destination = tuple(
                values[
                    units[page - table.start_page], offset : offset + count
                ].unsqueeze(1)
                for page, offset, count in pages
            )
            tickets.extend(
                _fetch(
                    importer,
                    write,
                    group.tensors[field_index],
                    destination,
                    transports,
                    region=(
                        slice(0, carried),
                        slice(layer, layer + 1),
                        slice(
                            advertised.kv_head_offset,
                            advertised.kv_head_offset + advertised.num_kv_heads,
                        ),
                        slice(0, advertised.head_dim),
                    ),
                )
            )
            if isinstance(tensor, QuantizedTensor):
                # Page-aligned intervals and equal page sizes make source
                # page ``i`` of the carried tokens destination span ``i``.
                scales = tensor.buffers()["scale"]
                destination = tuple(
                    scales[
                        units[page - table.start_page] : units[
                            page - table.start_page
                        ]
                        + 1
                    ]
                    for page, _, _ in pages
                )
                head = advertised.kv_head_offset // group.scale_head_size
                tickets.extend(
                    _fetch(
                        importer,
                        write,
                        group.tensors[2],
                        destination,
                        transports,
                        region=(
                            slice(0, len(pages)),
                            slice(field_index, field_index + 1),
                            slice(layer, layer + 1),
                            slice(head, head + 1),
                        ),
                    )
                )
        importer._consume(write, tuple(tickets))
        for ticket in tickets:
            ticket.close()

    # Mark the units initialized only while the import is still active.
    importer._require_active(write)
    pool.cache.mark_initialized(
        tuple(
            unit
            for page, _, _ in pages
            for unit in table.units[
                (page - table.start_page) * table.shape.units_per_page : (
                    page - table.start_page + 1
                )
                * table.shape.units_per_page
            ]
        )
    )


def _copy_converted(
    pool: KVCacheManager,
    importer: KVImporter,
    write: KVImport,
    index: int,
    group: KvGroupTransfer,
    transports: Mapping[str, Transport],
    workspace: KVWorkspace,
) -> None:
    """Copy one group's tokens span by span through the conversion page.

    For each destination page span, fetches the source K/V tokens of this
    worker's shard into ``workspace.raw``. FP8 sources are decoded per
    source page and scale head group into ``workspace.values``; other
    sources are copied as fetched. Each layer is then written through
    `mha.State.copy_region`, which applies the destination encoding.
    """
    publication = write.publication
    table = write.tables[index]
    start = group.start
    carried = publication.published_extent - start
    advertised = pool.info.groups[index]
    axis = pool.axes[index]
    cache_group = pool.cache.groups[index]
    columns = pool.cache.planes.columns
    page_tokens = table.shape.page_tokens
    num_layers = len(cache_group.layers)
    num_heads = advertised.num_kv_heads
    head_dim = advertised.head_dim
    dtype = getattr(torch, group.tensors[0].dtype)
    quantized = dtype is torch.float8_e4m3fn
    trailing = (num_layers, num_heads, head_dim)
    itemsize = workspace.raw.view(dtype).element_size()
    # Tokens of the carried interval already copied.
    logical = 0

    while logical < carried:
        position = start + logical
        page, offset = divmod(position, page_tokens)
        count = min(carried - logical, page_tokens - offset)
        elements = count * num_layers * num_heads * head_dim
        # Raw staging views per K/V field: [tokens, layers, kv heads, dim].
        raw = tuple(
            workspace.raw[field, : elements * itemsize]
            .view(dtype)
            .reshape(count, *trailing)
            for field in range(2)
        )
        # Fetch this span's tokens and this worker's layer/head shard.
        region = tuple(
            slice(first, first + extent)
            for first, extent in zip(
                (logical, axis.offset, advertised.kv_head_offset, 0),
                (count, *trailing),
                strict=True,
            )
        )
        tickets = tuple(
            ticket
            for tensor, destination in zip(group.tensors[:2], raw, strict=True)
            for ticket in _fetch(
                importer, write, tensor, destination, transports, region=region
            )
        )

        # Token offset of this span within its first source page.
        source_offset = position % group.page_tokens
        if quantized:
            head_start = advertised.kv_head_offset // group.scale_head_size
            head_end = (
                advertised.kv_head_offset + num_heads - 1
            ) // group.scale_head_size + 1
            # Source scale rows cover the publication pages this span
            # touches, counted from the source page holding the first
            # carried token.
            scale_start = (
                position // group.page_tokens - start // group.page_tokens
            )
            scale_count = (
                source_offset + count + group.page_tokens - 1
            ) // group.page_tokens
            scales = workspace.scales[
                : scale_count * 2 * num_layers * (head_end - head_start)
            ].view(scale_count, 2, num_layers, head_end - head_start)
            tickets += _fetch(
                importer,
                write,
                group.tensors[2],
                scales,
                transports,
                region=(
                    slice(scale_start, scale_start + scale_count),
                    slice(0, 2),
                    slice(axis.offset, axis.offset + num_layers),
                    slice(head_start, head_end),
                ),
            )

        importer._consume(write, tickets)

        for field_index, source in enumerate(raw):
            values = source
            if quantized:
                values = workspace.values[:elements].view(count, *trailing)
                # Intersect source scale pages and head groups here; the
                # numerical cache library owns decoding and rounding.
                compute_dtype = getattr(torch, publication.compute_dtype)
                # Rounding scratch lies in the raw page past the FP8
                # input, which occupies at most its first eighth. FP32 and
                # FP64 compute dtypes need no rounding scratch.
                rounded = (
                    workspace.raw[field_index, elements * 4 :].view(
                        compute_dtype
                    )
                    if compute_dtype not in {torch.float32, torch.float64}
                    else workspace.raw[field_index, :0].view(compute_dtype)
                )
                # Walk source pages (one scale row each) and, within each,
                # this worker's heads grouped by source scale group.
                cursor, scale_index, token_offset = 0, 0, source_offset
                while cursor < count:
                    length = min(
                        group.page_tokens - token_offset, count - cursor
                    )
                    head, head_group = 0, 0
                    while head < num_heads:
                        heads = min(
                            group.scale_head_size
                            - (advertised.kv_head_offset + head)
                            % group.scale_head_size,
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
                unit = table.row(column_index // columns)[
                    page - table.start_page
                ]
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

        # The next span fetches into the same workspace buffers, so this
        # span's conversion and copies must finish first.
        importer._drain(write)
        for ticket in tickets:
            ticket.close()
        logical += count
