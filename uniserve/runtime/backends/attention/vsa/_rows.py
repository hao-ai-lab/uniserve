"""Runtime-owned query maps for streamed VSA block attention."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import torch

from uniserve.nn.attention.vsa.inputs import Pattern
from uniserve.ops.video_sparse_rows import compose_attention, pack_sparse_input_rows

_TILE = 64


@dataclass
class _Rows:
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
        pattern: Pattern,
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
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError("prepare VSA query intervals before graph capture")
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
