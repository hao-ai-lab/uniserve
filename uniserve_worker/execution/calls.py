"""Per-call identity and coordinate helpers shared by the execution modules.

Call-kind handlers (such as `token`, `diffusion` and `image`), batch
preparation and scheduling use these to key calls within a batch, to read
the coordinates a call states, and to resolve the KV-cache coordinates a
numerical call runs at.
"""

from __future__ import annotations

from dataclasses import replace

from uniserve_worker.errors import invalid_descriptor, unsupported_setup
from uniserve_worker.execution.batch import BatchState
from uniserve_worker.execution.output import PendingOutput
from uniserve_worker.execution.request import RequestProgress
from uniserve_worker.protocol.call import Call, CallStatus
from uniserve_worker.protocol.identity import CallIdentity
from uniserve_worker.protocol.output import FinishFlags
from uniserve_worker.storage.block_tables import BlockTables


def output_generations(call: Call) -> tuple[int, ...]:
    """Return the generation of each tensor output the call declares.

    Values follow `Call.tensor_outputs` order. Handlers store them as the
    completion's ``product_generations``.
    """
    return tuple(
        int(reference.generation) for reference in call.tensor_outputs()
    )


def require_progress(output: PendingOutput) -> RequestProgress:
    """Return a pending output's progress for a state-consuming call.

    Raises:
        WorkerError: ``invalid_descriptor`` when the output has no progress.
    """
    progress = output.progress
    if progress is None:
        raise invalid_descriptor("call does not consume request progress")
    return progress


def execution_runtime(
    request: PendingOutput,
    cache: tuple[int, int, int] | None,
    *,
    flow_step: int | None = None,
    computed_len: int | None = None,
) -> RequestProgress:
    """Project the coordinates used by a call's numerical consumers.

    Returns a copy of ``request.progress``; the output is not modified. With
    ``cache`` None, the KV extents are kept from ``request.progress`` and
    ``computed_len`` is ignored. With a `cache_coordinates` tuple, the visible
    length comes from the tuple and the computed length equals it unless
    ``computed_len`` is given. ``flow_step`` replaces the solver step when
    given.
    """
    progress = request.progress
    if cache is None:
        visible = progress.kv_visible_len
        computed = progress.kv_computed_len
    else:
        _slot, visible, _capacity = cache
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
) -> tuple[int, int, int]:
    """Resolve a request's KV-cache coordinates across its cache groups.

    Returns:
        ``(slot, visible, capacity)``: the request-pool slot, the accepted
        visible prefix the call states (tokens), and the token capacity
        every installed group table of the slot covers.

    Raises:
        WorkerError: ``unsupported_setup`` when ``tables`` is None;
            ``invalid_descriptor`` when the slot lacks a block table of some
            cache group or the visible prefix exceeds the capacity.
    """
    slot = int(request.request.request_pool_idx)
    # Scheduler columns may reserve the full unobserved verifier prefix. The
    # call states the accepted extent its numerical consumers run at.
    visible = int(request.progress.kv_visible_len)

    pool = tables
    if pool is None:
        raise unsupported_setup("call requires request-to-token storage")
    # Called only for its check that every group's table is installed.
    for group in range(len(pool.groups)):
        pool.table(slot, group)
    capacity = pool.allocated_length(slot)

    if visible > capacity:
        raise invalid_descriptor(
            "call visibility exceeds scheduler block table"
        )
    return slot, visible, capacity


def call_identity(call: Call) -> CallIdentity:
    """Return the call's ``(request_key, call_id)`` identity.

    The request key carries the request epoch, so calls of different epochs
    of one request id never share an identity.
    """
    return call.request_key, call.call_id


def _predicated_outcome(
    call: Call,
    *,
    state: BatchState,
) -> PendingOutput:
    """Mark a call whose completion predicate is false as predicated.

    Reuses the call's reserved `PendingOutput`: sets `CallStatus.PREDICATED`,
    keeps its progress coordinates, and clears finish flags and product
    generations, which a predicated completion must not carry
    (`RequestOutput.validate` rejects them).
    """
    request = state.pending_output(call.request_key.request_id)
    request.status = CallStatus.PREDICATED
    request.progress = execution_runtime(request, None)
    request.finish_flags = FinishFlags()
    request.product_generations = ()
    return request
