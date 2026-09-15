"""ScheduledRequest identities and scalar bounds derived from logical execution state."""

from __future__ import annotations

from dataclasses import replace

from uniserve_worker.execution.output import PendingOutput
from uniserve_worker.foundation.errors import invalid_descriptor, unsupported_setup
from uniserve_worker.protocol.operation import OpStatus, ScheduledRequest
from uniserve_worker.protocol.output import FinishFlags
from uniserve_worker.runtime.block_tables import BlockTables
from uniserve_worker.runtime.request import RequestProgress

from .batch_state import BatchState
from .rows import OperationIdentity


def output_generations(operation: ScheduledRequest) -> tuple[int, ...]:
    """List logical generations in descriptor output order."""

    return tuple(int(reference.generation) for reference in operation.tensor_outputs())


def require_progress(output: PendingOutput) -> RequestProgress:
    """Require real request progress for a state-consuming numerical operation."""

    progress = output.projected_progress
    if progress is None:
        raise invalid_descriptor("operation does not consume request progress")
    return progress


def input_progress(output: PendingOutput) -> RequestProgress | None:
    """Read the stable predecessor, preferring actual acceptance when it is known."""

    predecessor = output.predecessor
    if isinstance(predecessor, PendingOutput):
        return predecessor.accepted_progress or predecessor.projected_progress
    return predecessor


def execution_runtime(
    request: PendingOutput,
    cache: tuple[int, int, int, int] | None,
    *,
    flow_step: int | None = None,
    computed_len: int | None = None,
) -> RequestProgress | None:
    """Project actual state consumers without constructing progress for stateless work."""

    progress = request.projected_progress
    if progress is None:
        return None
    parent = input_progress(request)
    assert parent is not None

    if cache is None:
        visible = parent.kv_visible_len
        computed = parent.kv_computed_len
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
    """Resolve the request slot, cache group, accepted prefix and physical token capacity."""

    slot = int(request.request.request_pool_idx)
    # Scheduler columns may reserve the full unobserved verifier prefix.
    # The request state owns the accepted extent used by numerical consumers.
    parent = input_progress(request)
    visible = 0 if parent is None else int(parent.kv_visible_len)

    pool = tables
    if pool is None:
        raise unsupported_setup("operation requires request-to-token storage")
    pool.pages(slot, group_id)
    capacity = pool.allocated_length(slot)

    if visible > capacity:
        raise invalid_descriptor("operation visibility exceeds scheduler block table")
    return slot, int(group_id), visible, capacity


def operation_identity(operation: ScheduledRequest) -> OperationIdentity:
    """Form the completion group-local identity from request generation and operation id."""

    return operation.request_key, operation.op_id


def _predicated_outcome(
    operation: ScheduledRequest,
    completion_group: int,
    *,
    state: BatchState,
) -> PendingOutput:
    """Construct an inactive outcome while preserving declared product generations."""

    request = state.pending_output(completion_group, operation.request_key.request_id)
    request.status = OpStatus.PREDICATED
    request.projected_progress = execution_runtime(request, None)
    request.finish_flags = FinishFlags()
    request.product_generations = ()
    return request
