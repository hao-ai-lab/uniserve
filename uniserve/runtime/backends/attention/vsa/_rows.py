"""Runtime-owned query maps for streamed VSA block attention."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import torch

from uniserve.nn.attention.vsa.inputs import Pattern
from uniserve.ops.video_sparse_rows import (
    compose_attention,
    pack_sparse_input_rows,
)

_TILE = 64


@dataclass
class _Rows:
    """Own selected query maps for one row-wise attention execution.

    Own selected query maps for one serialized row-wise attention execution.
    """

    fine_attention: Callable[..., torch.Tensor]
    plans: dict[
        tuple[object, ...], tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ] = field(default_factory=dict)

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
        """Produce paired owner intervals.

        Produce paired owner intervals against the complete selected key
        domain.

        Counts and indices are read from current device metadata on every
        call. Each numerical launch finishes using scratch before its
        epilogue writes transport destinations; downstream consumers may
        then reuse that scratch.
        """
        # The row path reads the live device block maps on every call, so the
        # static pattern carries no information here.
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
            raise ValueError(
                "sparse row production requires tile-aligned equal QKV "
                "intervals"
            )

        rows, heads, width = query.shape
        owner_rows = rows // owners
        if packed is None:
            packed = pack_sparse_input_rows(
                query,
                key,
                value,
                valid_sizes,
                owners=owners,
                chunk_rows=chunk_rows,
                row_major=True,
            )
        if (
            packed.shape != (3, rows, heads, width)
            or packed.dtype != query.dtype
            or packed.device != query.device
            or not packed.is_contiguous()
        ):
            raise ValueError(
                "prepared sparse rows must match the complete row-major "
                "QKV layout"
            )

        # One fine-attention launch covers the complete packed query domain:
        # its query tiles are laid out segment by segment, each segment
        # holding every owner's rows of one exchange interval. Each interval
        # then composes its own rows from that output, so the numerical launch
        # stays whole while transport pipelines the intervals.
        segments = tuple(
            (start, min(chunk_rows, owner_rows - start))
            for start in range(0, owner_rows, chunk_rows)
        )
        launched = False

        def launch() -> None:
            signature = (
                query.device,
                heads,
                rows,
                owners,
                chunk_rows,
                mask_block_indices.shape[-1],
            )
            plan = self.plans.get(signature)
            if plan is None:
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError(
                        "prepare VSA query intervals before graph capture"
                    )

                # Map each segment's packed local tiles onto its owners' query
                # tiles in the full row domain: owner stride owner_rows,
                # offset start.
                selected = []
                for start, count in segments:
                    local = torch.arange(owners * count // _TILE)
                    tiles = local // (count // _TILE) * (owner_rows // _TILE)
                    tiles += start // _TILE + local % (count // _TILE)
                    selected.append(tiles)
                selected = torch.cat(selected).to(device=query.device)
                plan = (
                    selected,
                    torch.empty(
                        (heads, rows // _TILE, mask_block_indices.shape[-1]),
                        device=query.device,
                        dtype=torch.int32,
                    ),
                    torch.empty(
                        (heads, rows // _TILE),
                        device=query.device,
                        dtype=torch.int32,
                    ),
                )
                self.plans[signature] = plan

            selected, indices, counts = plan
            torch.index_select(mask_block_indices, 1, selected, out=indices)
            torch.index_select(mask_block_count, 1, selected, out=counts)
            output = attention_output.view(-1)[: rows * heads * width]
            self.fine_attention(
                packed[0],
                packed[1],
                packed[2],
                output.view(rows, heads, width),
                indices,
                counts,
                valid_sizes,
            )

        def produce(interval: slice, outputs: tuple[torch.Tensor, ...]) -> None:
            nonlocal launched
            start, stop = interval.start, interval.stop
            count = stop - start
            if (
                (start, count) not in segments
                or len(outputs) != owners
                or any(
                    output.shape != (count, heads, width)
                    or not output.is_contiguous()
                    or output.dtype != query.dtype
                    or output.device != query.device
                    for output in outputs
                )
            ):
                raise ValueError(
                    "sparse row destinations must match a prepared owner "
                    "interval"
                )
            if not launched:
                launch()
                launched = True

            # The interval's rows sit contiguously in the packed output, all
            # owners together; the kernel consumed [1, heads, rows, width].
            attended = attention_output.view(-1)[: rows * heads * width]
            attended = attended.view(rows, heads, width)[
                start * owners : start * owners + owners * count
            ]
            compose_attention(
                attended.unsqueeze(0).transpose(1, 2),
                gate,
                compressed,
                list(outputs),
                0,
                owner_rows=owner_rows,
                start_row=start,
            )

        return produce
