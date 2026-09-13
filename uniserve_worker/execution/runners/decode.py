"""Decode startup inputs with valid KV context and greedy state."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..model_entry import ModelEntry

if TYPE_CHECKING:
    from ..model_runner import ModelRunner

from collections.abc import Callable

from ..batch import ExecutionOutput, InputBatch
from ..input_buffers import InputBuffers
from .prefill import stage_text


def prepare_decode(
    runner: ModelRunner,
    entry: ModelEntry,
    buffers: InputBuffers,
    forward: Callable[[InputBatch], ExecutionOutput],
    *,
    packed: bool,
) -> None:
    """Prepare valid one-token prefixes, then capture each decode batch bucket."""

    for rows in (
        tuple(reversed(runner.decode_shapes[entry]))
        if (runner.worker_config.graph_policy != "off")
        else (1,)
    ):
        with runner.kv_cache.startup_pages(rows) as scratch:
            pages = tuple((page,) for page in scratch)
            prompt = stage_text(buffers, runner.kv_cache, ((0,),) * rows, pages, packed=packed)
            forward(prompt)
            batch = stage_text(
                buffers,
                runner.kv_cache,
                ((0,),) * rows,
                pages,
                packed=packed,
                prefixes=(1,) * rows,
                decode=True,
            )
            predicates = runner.decode_predicates
            saved = None if predicates is None else predicates.clone()
            try:
                if predicates is not None:
                    predicates[1 : rows + 1] = True
                runner.capture_batch(entry, batch, forward)
            finally:
                if saved is not None and predicates is not None:
                    predicates.copy_(saved)
