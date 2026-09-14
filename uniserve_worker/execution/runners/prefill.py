"""Numerical token inputs for bounded packed startup preparation."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from ..model_entry import ModelEntry

if TYPE_CHECKING:
    from ..model_runner import ModelRunner

from collections.abc import Callable, Sequence

import torch

from uniserve.attention.inputs import physical_columns
from uniserve.math import bucketed_length, ceil_div
from uniserve.model.tensors import TokenSelection
from uniserve_worker.protocol.batch import ForwardMode

from ...runtime.cache_manager import CacheManager
from ..batch import ExecutionOutput, InputBatch
from ..graph_inputs import PrefillShape
from ..input_buffers import InputBuffers
from ..rows import ForwardRow


def stage_text(
    buffers: InputBuffers,
    cache: CacheManager,
    tokens: tuple[tuple[int, ...], ...],
    pages: Sequence[Sequence[int]],
    *,
    packed: bool,
    prefixes: tuple[int, ...] | None = None,
    decode: bool = False,
    selection: TokenSelection = TokenSelection.LAST_LOGITS,
    slots: tuple[int, ...] | None = None,
) -> InputBatch:
    """Use serving's staging and attention preparation with numerical inputs."""

    rows = len(tokens)
    lengths = tuple(len(value) for value in tokens)
    prefixes = (0,) * rows if prefixes is None else prefixes
    positions = tuple(
        torch.arange(prefix, prefix + length, dtype=torch.int64)
        for prefix, length in zip(prefixes, lengths, strict=True)
    )
    indexes = tuple(
        torch.stack((value, torch.zeros_like(value), torch.zeros_like(value)))
        for value in positions
    )
    attention = physical_columns(
        pages=pages,
        prefix_lens=prefixes,
        query_lens=lengths,
        causal_rows=(True,) * rows,
        write_rows=(True,) * rows,
        positions=indexes,
        token_rows=(True,) * rows,
        text_local_indices=((),) * rows,
        width=min(buffers.max_blocks_per_row, bucketed_length(max(1, max(map(len, pages))))),
        block_size=cache.cache.page_size,
        packed=packed,
        decode=decode,
    )
    slots = slots or tuple(range(1, rows + 1))
    mode = ForwardMode.DECODE if decode else ForwardMode.PREFILL
    batch = buffers.stage(
        tuple(
            ForwardRow(
                forward_mode=mode,
                token_ids=torch.tensor(value, dtype=torch.int64),
                positions=position,
                selection=selection,
                request_pool_idx=slot,
                seq_len=prefix,
                write_kv=True,
            )
            for value, position, slot, prefix in zip(
                tokens, positions, slots, prefixes, strict=True
            )
        ),
        forward_mode=mode,
        attention=attention,
    )
    if decode:
        # Startup captures the same force-finish address used by live decode.
        force_finish = buffers.decode_force_finish[:rows]
        force_finish.zero_()
        batch = replace(batch, decode_force_finish=force_finish)
    return batch


def prepare_prefill(
    runner: ModelRunner,
    entry: ModelEntry,
    buffers: InputBuffers,
    forward: Callable[[InputBatch], ExecutionOutput],
    shapes: tuple[PrefillShape, ...],
    *,
    packed: bool,
) -> None:
    """Capture each selected physical token/row bucket in footprint order."""

    for shape in sorted(
        shapes,
        key=lambda item: (item.token_bucket * item.row_bucket, item.token_bucket),
        reverse=True,
    ):
        lengths = (shape.token_bucket - shape.live_rows + 1, *(1,) * (shape.live_rows - 1))
        counts = tuple(ceil_div(length, runner.worker_config.block_size) for length in lengths)
        with runner.kv_cache.startup_pages(sum(counts)) as scratch:
            pages = tuple(
                scratch[sum(counts[:index]) : sum(counts[: index + 1])]
                for index in range(len(counts))
            )
            batch = stage_text(
                buffers, runner.kv_cache, tuple((0,) * n for n in lengths), pages, packed=packed
            )
            runner.capture_batch(entry, batch, forward)
