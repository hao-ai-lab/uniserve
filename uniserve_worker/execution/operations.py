"""Operation identities and scalar geometry derived from logical execution state."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from uniserve_worker.execution.batch import (
    FinishFlags,
    FixedCheckpoint,
    LogicalLengths,
    OpCode,
    Operation,
    OpStatus,
    ProductRef,
    TokenSpan,
)
from uniserve_worker.execution.rows import LaneState, LatentExecution, OperationIdentity, Outcome
from uniserve_worker.foundation.errors import invalid_descriptor, unsupported_setup
from uniserve_worker.runtime.req_to_token_pool import ReqToTokenPool
from uniserve_worker.runtime.request import RequestDraft

if TYPE_CHECKING:
    from uniserve_worker.config import WorkerConfig


def output_generations(operation: Operation) -> tuple[int, ...]:
    """List logical generations in descriptor output order."""

    return tuple(int(reference.generation) for reference in operation.outputs)


def latent_row(operation: Operation, scope: LaneState) -> LatentExecution:
    """Resolve the staged physical latent params for an operation in this lane."""

    row = scope.latent_rows.get(operation_identity(operation))
    if row is None:
        raise invalid_descriptor("trajectory operation has no staged latent params")
    return row


def fixed_parent(operation: Operation) -> FixedCheckpoint:
    """Require a host-resolved parent checkpoint for a depth-one state transition."""

    point = operation.state_parent.point
    if not isinstance(point, FixedCheckpoint):
        raise invalid_descriptor("operation names a device parent; depth one commits fixed")
    return point


def logical_lengths(
    operation: Operation,
    request: RequestDraft,
    cache: tuple[int, int, int, int] | None,
    *,
    latent_len: int | None = None,
    computed_len: int | None = None,
) -> LogicalLengths:
    """Construct post-operation semantic lengths from request state and optional cache coordinates."""

    parent = request.request.parent_runtime(operation.parent)
    if cache is None:
        visible = parent.kv_visible_len
        computed = parent.kv_computed_len
    else:
        _slot, _group, visible, _capacity = cache
        computed = visible if computed_len is None else int(computed_len)
    return LogicalLengths(
        token_len=request.logical_position,
        kv_visible_len=visible,
        kv_computed_len=computed,
        latent_len=request.flow_step if latent_len is None else int(latent_len),
    )


def request_row(scope: LaneState, request_id: int) -> RequestDraft:
    """Return the unique staged request draft for an identifier within the current lane."""

    try:
        return scope.request_rows[int(request_id)]
    except KeyError:
        raise invalid_descriptor(f"lane has no request row for request {request_id}") from None


def cache_coordinates(
    operation: Operation,
    scope: LaneState,
    *,
    tables: ReqToTokenPool | None,
    group_id: int = 0,
) -> tuple[int, int, int, int]:
    """Resolve visible, computed, and physical KV coordinates for an operation and cache group."""

    request = request_row(scope, operation.request_key.request_id)
    slot = int(request.request.request_pool_idx)
    rows = scope.forward_rows.get(operation_identity(operation), ())
    descriptor = next(
        (row for row in rows if int(row.request_pool_index) == slot),
        None,
    )
    parent = request.request.parent_runtime(operation.parent)
    visible = int(parent.kv_visible_len) if descriptor is None else int(descriptor.seq_len)
    pool = tables
    if pool is None:
        raise unsupported_setup("operation requires request-to-token storage")
    pool.pages(slot, group_id)
    capacity = pool.allocated_length(slot)
    if visible > capacity:
        raise invalid_descriptor("operation visibility exceeds scheduler block table")
    return slot, int(group_id), visible, capacity


def operation_identity(operation: Operation) -> OperationIdentity:
    """Form the lane-local identity from request generation and operation id."""

    return operation.request_key, int(operation.op_id)


def _completion_devices(
    operations: tuple[Operation, ...], *, config: WorkerConfig
) -> tuple[str, ...]:
    """List distinct devices that may contribute asynchronous completion fields."""

    worker_config = config
    generation_device = worker_config.generation_device
    device = worker_config.device
    selected: list[str] = []
    for operation in operations:
        target = (
            generation_device
            if generation_device is not None
            and operation.kind
            in {
                OpCode.DIFFUSION_PREPARE,
                OpCode.DIFFUSION_STEP,
                OpCode.DIFFUSION_DECODE,
                OpCode.MEDIA_APPEND,
                OpCode.DIFFUSION_FINALIZE,
            }
            else device
        )
        if target not in selected:
            selected.append(target)
    return tuple(selected)


def _operation_device(operation: Operation, *, config: WorkerConfig) -> torch.device:
    """Resolve the execution device for an operation's model phase."""

    return (
        torch.device((config.generation_device or config.device))
        if operation.kind
        in {
            OpCode.DIFFUSION_PREPARE,
            OpCode.DIFFUSION_STEP,
            OpCode.DIFFUSION_FINALIZE,
        }
        else torch.device(config.device)
    )


def _predicated_outcome(
    operation: Operation,
    scope: LaneState,
) -> Outcome:
    """Construct an inactive outcome while preserving declared product generations."""

    request = request_row(scope, operation.request_key.request_id)
    lengths = logical_lengths(operation, request, None)
    selected = request.request.resolve_version(operation.parent)
    if operation.parent is not None and (
        selected is None or not isinstance(selected.point, FixedCheckpoint)
    ):
        raise invalid_descriptor("predicated operation parent has no selected fixed checkpoint")
    selected_point = (
        0 if selected is None else int(cast(FixedCheckpoint, selected.point).point_index)
    )
    return Outcome(
        status=OpStatus.PREDICATED,
        selected_point=selected_point,
        logical_lengths=lengths,
        token_span=TokenSpan(base=int(lengths.token_len), len=0),
        finish_flags=FinishFlags(),
        product_generations=(),
    )


def product_identity(reference: ProductRef) -> OperationIdentity:
    """Identify the logical operation that produced a reference."""

    return reference.request_key, int(reference.producer_op_id)
