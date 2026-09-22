"""Call identities and scalar bounds derived from logical execution.

state.
"""

from __future__ import annotations

from dataclasses import replace

from uniserve_worker.execution.output import PendingOutput
from uniserve_worker.foundation.errors import (
    invalid_descriptor,
    unsupported_setup,
)
from uniserve_worker.protocol.call import Call, CallStatus
from uniserve_worker.protocol.output import FinishFlags
from uniserve_worker.runtime.block_tables import BlockTables
from uniserve_worker.runtime.request import RequestProgress

from .batch_state import BatchState
from .rows import CallIdentity


def output_generations(call: Call) -> tuple[int, ...]:
    """List logical generations in descriptor output order."""
    return tuple(
        int(reference.generation) for reference in call.tensor_outputs()
    )


def require_progress(output: PendingOutput) -> RequestProgress:
    """Require real request progress for a state-consuming numerical.

    call.
    """
    progress = output.progress
    if progress is None:
        raise invalid_descriptor("call does not consume request progress")
    return progress


def execution_runtime(
    request: PendingOutput,
    cache: tuple[int, int, int, int] | None,
    *,
    flow_step: int | None = None,
    computed_len: int | None = None,
) -> RequestProgress:
    """Project the coordinates used by a call's numerical consumers.

    The call states them; the cache tuple overrides the KV extents when the
    block tables resolved a different accepted prefix.
    """
    progress = request.progress
    if cache is None:
        visible = progress.kv_visible_len
        computed = progress.kv_computed_len
    else:
        _slot, _group, visible, _capacity = cache
        computed = visible if computed_len is None else int(computed_len)
    return replace(
        progress,
        kv_visible_len=visible,
        kv_computed_len=computed,
        flow_step=progress.flow_step if flow_step is None else int(flow_step),
    )


def cache_coordinates(
    request: PendingOutput,
    *,
    tables: BlockTables | None,
    group_id: int = 0,
) -> tuple[int, int, int, int]:
    """Resolve the request slot, cache group.

    accepted prefix and physical token capacity.
    """
    slot = int(request.request.request_pool_idx)
    # Scheduler columns may reserve the full unobserved verifier prefix. The
    # call states the accepted extent its numerical consumers run at.
    visible = int(request.progress.kv_visible_len)

    pool = tables
    if pool is None:
        raise unsupported_setup("call requires request-to-token storage")
    pool.pages(slot, group_id)
    capacity = pool.allocated_length(slot)

    if visible > capacity:
        raise invalid_descriptor(
            "call visibility exceeds scheduler block table"
        )
    return slot, int(group_id), visible, capacity


def call_identity(call: Call) -> CallIdentity:
    """Form the batch-local identity from generation and call ID."""
    return call.request_key, call.call_id


def _predicated_outcome(
    call: Call,
    *,
    state: BatchState,
) -> PendingOutput:
    """Construct an inactive outcome while preserving declared product.

    generations.
    """
    request = state.pending_output(call.request_key.request_id)
    request.status = CallStatus.PREDICATED
    request.progress = execution_runtime(request, None)
    request.finish_flags = FinishFlags()
    request.product_generations = ()
    return request
