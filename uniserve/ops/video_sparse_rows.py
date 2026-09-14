"""Shared sparse input layouts and row-wise compression epilogues."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None

_TILE = 64


@dataclass(frozen=True, slots=True)
class SparseAttentionPattern:
    """Host-known sparsity contract paired with mutable device block indices.

    ``row_counts`` contains one count per query tile, either for one shared
    head or for every head. Device counts must agree with these immutable
    cardinalities. A different cardinality requires a different pattern and
    plan; changing selected block IDs or valid rows does not.

    The optional dense prefix declares query tiles whose selected keys are
    exactly ``range(dense_key_tiles)``. This enables dense execution without
    asking a numerical backend to infer the model's selection rule.
    """

    row_counts: tuple[tuple[int, ...], ...]
    dense_prefix_tiles: int = 0
    dense_key_tiles: int = 0

    def __post_init__(self) -> None:
        if (
            not self.row_counts
            or not self.row_counts[0]
            or any(len(row) != len(self.row_counts[0]) for row in self.row_counts)
            or any(count < 1 for row in self.row_counts for count in row)
            or not 0 <= self.dense_prefix_tiles <= len(self.row_counts[0])
            or self.dense_key_tiles < 0
            or (self.dense_prefix_tiles > 0 and self.dense_key_tiles == 0)
            or any(
                count != self.dense_key_tiles
                for row in self.row_counts
                for count in row[: self.dense_prefix_tiles]
            )
        ):
            raise ValueError("sparse pattern has invalid row counts or dense visibility")

    def counts(self, heads: int, query_tiles: int, index_width: int) -> torch.Tensor:
        """Materialize validated CPU counts for planning a concrete head layout."""

        if (
            len(self.row_counts) not in (1, heads)
            or len(self.row_counts[0]) != query_tiles
            or any(count > index_width for row in self.row_counts for count in row)
        ):
            raise ValueError("sparse pattern does not match the device block-map geometry")
        counts = torch.tensor(self.row_counts, dtype=torch.int32)
        return counts.expand(heads, -1)


@dataclass
class SparseRowExecution:
    """Own selected query maps for one serialized row-wise attention execution."""

    fine_attention: Callable[..., torch.Tensor]
    plans: dict[tuple[object, ...], tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = field(
        default_factory=dict
    )

    def prepare(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        mask_block_indices: torch.Tensor,
        mask_block_count: torch.Tensor,
        valid_sizes: torch.Tensor,
        pattern: SparseAttentionPattern,
        gate: torch.Tensor,
        compressed: torch.Tensor,
        attention_output: torch.Tensor,
        owners: int,
        chunk_rows: int,
        packed: torch.Tensor | None = None,
    ) -> Callable[[slice, tuple[torch.Tensor, ...]], None]:
        """Produce paired owner intervals against the complete selected key domain.

        Counts and indices are read from current device metadata on every call.
        Each numerical launch finishes using scratch before its epilogue writes
        transport destinations; downstream consumers may then reuse that scratch.
        """

        del pattern
        if (
            query.ndim != 3
            or query.shape != key.shape
            or query.shape != value.shape
            or owners < 1
            or query.shape[0] % (owners * _TILE)
            or chunk_rows < _TILE
            or chunk_rows % _TILE
        ):
            raise ValueError("sparse row production requires tile-aligned equal QKV intervals")
        rows, heads, width = query.shape
        owner_rows = rows // owners
        if packed is None:
            packed = pack_sparse_input_rows(
                query, key, value, valid_sizes, owners=owners, chunk_rows=chunk_rows, row_major=True
            )
        if (
            packed.shape != (3, rows, heads, width)
            or packed.dtype != query.dtype
            or packed.device != query.device
            or not packed.is_contiguous()
        ):
            raise ValueError("prepared sparse rows must match the complete row-major QKV layout")

        def produce(interval: slice, outputs: tuple[torch.Tensor, ...]) -> None:
            start, stop = interval.start, interval.stop
            count = stop - start
            if (
                start < 0
                or start % chunk_rows
                or count != min(chunk_rows, owner_rows - start)
                or len(outputs) != owners
                or any(
                    output.shape != (count, heads, width)
                    or not output.is_contiguous()
                    or output.dtype != query.dtype
                    or output.device != query.device
                    for output in outputs
                )
            ):
                raise ValueError("sparse row destinations must match the prepared owner interval")
            signature = (
                query.device,
                heads,
                rows,
                owners,
                start,
                count,
                mask_block_indices.shape[-1],
            )
            plan = self.plans.get(signature)
            if plan is None:
                tiles = owners * count // _TILE
                local = torch.arange(tiles)
                selected = local // (count // _TILE) * (owner_rows // _TILE)
                selected += start // _TILE + local % (count // _TILE)
                plan = (
                    selected.to(device=query.device),
                    torch.empty(
                        (heads, tiles, mask_block_indices.shape[-1]),
                        device=query.device,
                        dtype=torch.int32,
                    ),
                    torch.empty((heads, tiles), device=query.device, dtype=torch.int32),
                )
                self.plans[signature] = plan
            selected, indices, counts = plan
            torch.index_select(mask_block_indices, 1, selected, out=indices)
            torch.index_select(mask_block_count, 1, selected, out=counts)
            elements = owners * count * heads * width
            packed_query = packed[0].view(-1).narrow(0, start * owners * heads * width, elements)
            packed_query = packed_query.view(owners * count, heads, width)
            output = attention_output.view(-1)[:elements].view_as(packed_query)
            attended = self.fine_attention(
                packed_query, packed[1], packed[2], output, indices, counts, valid_sizes
            )
            compose_attention(
                attended,
                gate,
                compressed,
                list(outputs),
                0,
                owner_rows=owner_rows,
                start_row=start,
            )

        return produce


if triton is not None:

    @triton.jit
    def _pack_masked_qkv_kernel(
        query,
        key,
        value,
        valid_sizes,
        packed,
        query_stride_row: tl.constexpr,
        query_stride_head: tl.constexpr,
        key_stride_row: tl.constexpr,
        key_stride_head: tl.constexpr,
        value_stride_row: tl.constexpr,
        value_stride_head: tl.constexpr,
        rows: tl.constexpr,
        input_rows: tl.constexpr,
        row_start: tl.constexpr,
        heads: tl.constexpr,
        width: tl.constexpr,
        tile_rows: tl.constexpr,
        block_rows: tl.constexpr,
        owners: tl.constexpr,
        chunk_rows: tl.constexpr,
        row_major: tl.constexpr,
    ):
        """Pack interval queries and full masked K/V in the provider's physical layout."""

        input_offsets = (tl.program_id(0) * block_rows + tl.arange(0, block_rows)).to(tl.int64)
        row_offsets = row_start + input_offsets
        head = tl.program_id(1)
        columns = tl.arange(0, width)
        row_mask = input_offsets[:, None] < input_rows
        valid_rows = tl.load(
            valid_sizes + row_offsets // tile_rows, mask=input_offsets < input_rows, other=0
        )
        key_mask = row_mask & ((row_offsets % tile_rows)[:, None] < valid_rows[:, None])
        if row_major:
            destination = row_offsets[:, None] * heads * width + head * width + columns[None, :]
        else:
            destination = head * rows * width + row_offsets[:, None] * width + columns[None, :]

        query_values = tl.load(
            query
            + input_offsets[:, None] * query_stride_row
            + head * query_stride_head
            + columns[None, :],
            mask=row_mask,
            other=0.0,
        )
        key_values = tl.load(
            key
            + input_offsets[:, None] * key_stride_row
            + head * key_stride_head
            + columns[None, :],
            mask=key_mask,
            other=0.0,
        )
        value_values = tl.load(
            value
            + input_offsets[:, None] * value_stride_row
            + head * value_stride_head
            + columns[None, :],
            mask=key_mask,
            other=0.0,
        )
        component_size = heads * rows * width
        owner_rows = rows // owners
        owner = row_offsets // owner_rows
        local_row = row_offsets % owner_rows
        segment = local_row // chunk_rows
        count = tl.minimum(chunk_rows, owner_rows - segment * chunk_rows)
        interval_offset = segment * chunk_rows * owners * heads * width
        interval_row = owner * count + local_row % chunk_rows
        if row_major:
            query_destination = interval_offset + interval_row * heads * width + head * width
        else:
            query_destination = (
                interval_offset + head * owners * count * width + interval_row * width
            )
        tl.store(packed + query_destination[:, None] + columns, query_values, mask=row_mask)
        tl.store(packed + component_size + destination, key_values, mask=row_mask)
        tl.store(packed + 2 * component_size + destination, value_values, mask=row_mask)

    @triton.jit
    def _compose_rows_kernel(
        attended,
        gate,
        compressed,
        output,
        attended_stride_head: tl.constexpr,
        attended_stride_row: tl.constexpr,
        gate_stride_row: tl.constexpr,
        gate_stride_head: tl.constexpr,
        rows: tl.constexpr,
        width: tl.constexpr,
        tile_rows: tl.constexpr,
        block_rows: tl.constexpr,
    ):
        """Fuse the attention result with trained compression."""

        row_offsets = (tl.program_id(0) * block_rows + tl.arange(0, block_rows)).to(tl.int64)
        head = tl.program_id(1)
        columns = tl.arange(0, width)
        mask = row_offsets[:, None] < rows
        attended_values = tl.load(
            attended
            + head * attended_stride_head
            + row_offsets[:, None] * attended_stride_row
            + columns[None, :],
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        gate_values = tl.load(
            gate + row_offsets[:, None] * gate_stride_row + head * gate_stride_head + columns,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        compressed_values = tl.load(
            compressed
            + head * (rows // tile_rows) * width
            + (row_offsets[:, None] // tile_rows) * width
            + columns,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        values = attended_values + gate_values * compressed_values
        tl.store(
            output + row_offsets[:, None] * tl.num_programs(1) * width + head * width + columns,
            values,
            mask=mask,
        )

    @triton.jit
    def _compose_shards_kernel(
        attended,
        gate,
        compressed,
        outputs,
        attended_stride_head: tl.constexpr,
        attended_stride_row: tl.constexpr,
        gate_stride_row: tl.constexpr,
        gate_stride_head: tl.constexpr,
        rows: tl.constexpr,
        local_rows: tl.constexpr,
        local_heads: tl.constexpr,
        global_heads: tl.constexpr,
        source_rank: tl.constexpr,
        width: tl.constexpr,
        tile_rows: tl.constexpr,
        block_rows: tl.constexpr,
        owner_rows: tl.constexpr,
        start_row: tl.constexpr,
        global_rows: tl.constexpr,
    ):
        """Compose sparse outputs and route local heads into row-owner shards."""

        row_offsets = (tl.program_id(0) * block_rows + tl.arange(0, block_rows)).to(tl.int64)
        local_row_offsets = row_offsets % local_rows
        global_row_offsets = row_offsets // local_rows * owner_rows + start_row + local_row_offsets
        head = tl.program_id(1)
        columns = tl.arange(0, width)
        mask = row_offsets[:, None] < rows
        attended_values = tl.load(
            attended
            + head * attended_stride_head
            + row_offsets[:, None] * attended_stride_row
            + columns[None, :],
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        gate_values = tl.load(
            gate
            + global_row_offsets[:, None] * gate_stride_row
            + head * gate_stride_head
            + columns,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        compressed_values = tl.load(
            compressed
            + head * (global_rows // tile_rows) * width
            + (global_row_offsets[:, None] // tile_rows) * width
            + columns,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        values = attended_values + gate_values * compressed_values
        destination = (
            local_row_offsets[:, None] * global_heads * width
            + (source_rank * local_heads + head) * width
            + columns
        )
        destination_rank = row_offsets[:, None] // local_rows
        for destination_index in tl.static_range(len(outputs)):
            tl.store(
                outputs[destination_index] + destination,
                values,
                mask=mask & (destination_rank == destination_index),
            )


def pack_sparse_input_rows(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    valid_sizes: torch.Tensor,
    *,
    owners: int = 1,
    chunk_rows: int | None = None,
    packed: torch.Tensor | None = None,
    row_start: int = 0,
    row_major: bool = False,
) -> torch.Tensor:
    """Publish a Q/K/V row interval into the provider's complete input layout.

    Caller-owned storage supports out-of-order disjoint intervals produced by
    a distributed projection. Every row must be published before attention
    consumes the buffer. Validity and destination offsets use global rows.
    Storage is (3, rows, heads, width) for the native provider and
    (3, heads, rows, width) for flattened BSR. Queries use owner-interval
    order within their component; K/V retain global row order.
    """

    assert triton is not None
    input_rows, heads, width = (int(size) for size in query.shape)
    if packed is None:
        shape = (3, input_rows, heads, width) if row_major else (3, heads, input_rows, width)
        packed = torch.empty(shape, dtype=query.dtype, device=query.device)
    rows = packed.shape[1 if row_major else 2]
    expected_shape = (3, rows, heads, width) if row_major else (3, heads, rows, width)
    if (
        query.shape != key.shape
        or query.shape != value.shape
        or packed.shape != expected_shape
        or packed.dtype != query.dtype
        or packed.device != query.device
        or not packed.is_contiguous()
        or row_start < 0
        or row_start + input_rows > rows
        or owners < 1
        or rows % owners
        or (chunk_rows is not None and chunk_rows < 1)
    ):
        raise ValueError("sparse input rows must fit matching packed owner storage")
    block_rows = 8
    _pack_masked_qkv_kernel[(triton.cdiv(input_rows, block_rows), heads)](
        query,
        key,
        value,
        valid_sizes,
        packed,
        int(query.stride(0)),
        int(query.stride(1)),
        int(key.stride(0)),
        int(key.stride(1)),
        int(value.stride(0)),
        int(value.stride(1)),
        rows,
        input_rows,
        row_start,
        heads,
        width,
        _TILE,
        block_rows,
        owners,
        rows if chunk_rows is None else chunk_rows,
        row_major,
        num_warps=4,
        num_stages=1,
    )
    return packed


def compose_attention(
    attended: torch.Tensor,
    gate: torch.Tensor,
    compressed: torch.Tensor,
    outputs: list[torch.Tensor],
    source_rank: int,
    *,
    owner_rows: int | None = None,
    start_row: int = 0,
) -> None:
    """Fuse trained compression into masked attention and route complete head rows.

    Numerical providers own key validity and softmax normalization. Composition
    accumulates in FP32 before writing the destination dtype.
    """

    assert triton is not None
    heads, rows, width = (int(size) for size in attended.shape[1:])
    block_rows = 8
    grid = (triton.cdiv(rows, block_rows), heads)
    if len(outputs) == 1 and owner_rows is None:
        _compose_rows_kernel[grid](
            attended,
            gate,
            compressed,
            outputs[0],
            int(attended.stride(1)),
            int(attended.stride(2)),
            int(gate.stride(0)),
            int(gate.stride(1)),
            rows,
            width,
            _TILE,
            block_rows,
            num_warps=4,
            num_stages=1,
        )
        return
    _compose_shards_kernel[grid](
        attended,
        gate,
        compressed,
        tuple(outputs),
        int(attended.stride(1)),
        int(attended.stride(2)),
        int(gate.stride(0)),
        int(gate.stride(1)),
        rows,
        rows // len(outputs),
        heads,
        int(outputs[0].shape[1]),
        source_rank,
        width,
        _TILE,
        block_rows,
        rows // len(outputs) if owner_rows is None else owner_rows,
        start_row,
        int(gate.shape[0]),
        num_warps=4,
        num_stages=1,
    )
