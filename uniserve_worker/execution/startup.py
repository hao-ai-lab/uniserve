"""Prepare bounded text inputs and CUDA graphs during worker startup."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import replace
from typing import TYPE_CHECKING

import torch

from uniserve.math import ceil_div
from uniserve_worker.execution.sampling import TokenSelection
from uniserve_worker.protocol.call import ForwardMode

from ..runtime.cache_manager import CacheManager
from .attention import from_blocks
from .batch import ExecutionOutput, InputBatch
from .component_binding import ComponentBinding
from .input_buffers import InputBuffers
from .rows import ForwardRow

if TYPE_CHECKING:
    from .graph_inputs import PrefillShape
    from .model_runner import ModelRunner


def stage_text(
    buffers: InputBuffers,
    cache: CacheManager,
    tokens: tuple[tuple[int, ...], ...],
    pages: Sequence[Sequence[int]],
    *,
    prefixes: tuple[int, ...] | None = None,
    decode: bool = False,
    selection: TokenSelection = TokenSelection.LAST_LOGITS,
    causal: bool = True,
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

    attention = from_blocks(
        pages=pages,
        query_lengths=lengths,
        prefix_lengths=prefixes,
        causal=(causal,) * rows,
        write=(True,) * rows,
        block_size=cache.info.block_size,
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
                causal=causal,
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
    entry: ComponentBinding,
    buffers: InputBuffers,
    forward: Callable[[InputBatch], ExecutionOutput],
    shapes: tuple[PrefillShape, ...],
) -> None:
    """Capture each selected physical token/row bucket in footprint order."""
    for shape in sorted(
        shapes,
        key=lambda item: (
            item.token_bucket * item.row_bucket,
            item.token_bucket,
        ),
        reverse=True,
    ):
        # One row holds the long prompt; the remaining live rows hold one token.
        lengths = (
            shape.token_bucket - shape.live_rows + 1,
            *(1,) * (shape.live_rows - 1),
        )
        counts = tuple(
            ceil_div(length, runner.worker_config.block_size)
            for length in lengths
        )

        with runner.kv_cache.startup_pages(sum(counts)) as scratch:
            pages = tuple(
                scratch[sum(counts[:index]) : sum(counts[: index + 1])]
                for index in range(len(counts))
            )
            batch = stage_text(
                buffers,
                runner.kv_cache,
                tuple((0,) * count for count in lengths),
                pages,
                causal=shape.causal,
                selection=shape.selection,
            )
            runner.capture_batch(entry, batch, forward)


def prepare_decode(
    runner: ModelRunner,
    entry: ComponentBinding,
    buffers: InputBuffers,
    forward: Callable[[InputBatch], ExecutionOutput],
) -> None:
    """Prepare valid one-token prefixes.

    then capture each decode batch bucket.
    """
    row_counts = (
        tuple(reversed(runner.decode_shapes[entry]))
        if runner.worker_config.graph_policy != "off"
        else (1,)
    )
    for rows in row_counts:
        with runner.kv_cache.startup_pages(rows) as scratch:
            pages = tuple((page,) for page in scratch)

            # Warm the one-token prompt eagerly so the decode capture below
            # reads valid K/V prefixes instead of uninitialized pages.
            prompt = stage_text(buffers, runner.kv_cache, ((0,),) * rows, pages)
            runner.eager_batch(entry, prompt, forward)

            batch = stage_text(
                buffers,
                runner.kv_cache,
                ((0,),) * rows,
                pages,
                prefixes=(1,) * rows,
                decode=True,
            )

            # Capture with every row live, then restore the caller's predicates.
            predicates = runner.decode_predicates
            saved = None if predicates is None else predicates.clone()
            try:
                if predicates is not None:
                    predicates[1 : rows + 1] = True
                runner.capture_batch(entry, batch, forward)
            finally:
                if saved is not None and predicates is not None:
                    predicates.copy_(saved)
