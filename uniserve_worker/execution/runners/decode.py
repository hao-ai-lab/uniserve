"""Decode startup inputs with valid KV context and greedy state."""

from collections.abc import Callable

from ..forward_batch import ForwardBatch, ForwardOutput
from ..input_buffers import InputBuffers
from .packed import PackedRunner
from .prefill import stage_text


def prepare_decode(
    runner: PackedRunner,
    buffers: InputBuffers,
    forward: Callable[[ForwardBatch], ForwardOutput],
    *,
    packed: bool,
) -> None:
    """Prepare valid one-token prefixes, then capture each decode batch bucket."""

    for rows in tuple(reversed(runner.decode_batch_sizes)) if runner.enabled else (1,):
        with runner.cache_pool.startup_pages(rows) as scratch:
            pages = tuple((page,) for page in scratch)
            prompt = stage_text(buffers, runner.cache_pool, ((0,),) * rows, pages, packed=packed)
            forward(prompt)
            batch = stage_text(
                buffers,
                runner.cache_pool,
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
                runner.capture(batch, forward)
            finally:
                if saved is not None and predicates is not None:
                    predicates.copy_(saved)
