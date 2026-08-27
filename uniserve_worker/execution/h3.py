"""Resident MiniMax H3 trajectory, decode, and mux actions."""

from __future__ import annotations

from pathlib import Path

import torch

from uniserve_worker.batch import (
    Admission,
    Batch,
    BatchPartition,
    DecodeKind,
    DecodePlacement,
    FinishFlags,
    FixedPoint,
    ForwardMode,
    Operation,
    OpStatus,
    TokenSpan,
)
from uniserve_worker.foundation.errors import capability_mismatch, invalid_descriptor
from uniserve_worker.models.minimax_h3 import MiniMaxH3Model
from uniserve_worker.models.minimax_h3.execution import (
    DeferredH3Task,
    H3MuxCoordinator,
    H3OutputRingLease,
)
from uniserve_worker.models.minimax_h3.state import H3StateSlot
from uniserve_worker.nn.mesh import DeviceMesh
from uniserve_worker.server.completion import PinnedOutputBuffer
from uniserve_worker.server.cpu_tasks import CpuTaskReservation
from uniserve_worker.server.profiler import profile_range

from .resources import ExecutionResources
from .rows import OperationState, Outcome, PartitionState


def trajectory_placement(partition: BatchPartition, operation: Operation):
    selected = tuple(
        placement
        for placement in partition.latent_placements
        if placement.request_key == operation.request_key
        and int(placement.op_id) == int(operation.op_id)
    )
    if len(selected) != 1:
        raise invalid_descriptor("H3 trajectory operation has no exact latent placement")
    return selected[0]


def decode_placement(partition: BatchPartition, operation: Operation) -> DecodePlacement:
    selected = tuple(
        placement
        for placement in partition.decode_placements
        if placement.request_key == operation.request_key
        and int(placement.op_id) == int(operation.op_id)
    )
    if len(selected) != 1:
        raise invalid_descriptor("H3 decode operation has no exact decode placement")
    return selected[0]


def validate_batch(runtime: ExecutionResources, batch: Batch) -> None:
    """Validate the fixed H3 admission and spool contract before state is staged."""

    if not isinstance(runtime.model, MiniMaxH3Model):
        return
    admissions = {admission.request_key: admission for admission in batch.admissions}
    transitions = {
        operation.request_key: operation
        for operation in batch.operations
        if operation.work is ForwardMode.GEN_TRANSITION
    }
    if set(admissions) != set(transitions):
        raise invalid_descriptor("H3 admissions must exactly match transition operations")
    for request_key, admission in admissions.items():
        output_path = _output_path(runtime, admission)
        slot = runtime.model.states.get(int(admission.request_pool_idx))
        if slot.active:
            raise invalid_descriptor("H3 admission targets an occupied request slot")
        operation = transitions[request_key]
        point = operation.parent.point
        if (
            int(operation.parent.producer_op_id) != 0
            or not isinstance(point, FixedPoint)
            or point.semantic_digest != admission.digest
        ):
            raise invalid_descriptor("H3 transition does not name its admission root")
        if output_path.name in {".", ".."}:
            raise invalid_descriptor("H3 output has an invalid filename")


def execute_action(
    model: MiniMaxH3Model,
    mesh: DeviceMesh,
    mux: H3MuxCoordinator | None,
    operation: Operation,
    partition: BatchPartition,
    slot: H3StateSlot,
    buffer: PinnedOutputBuffer,
    reservation: CpuTaskReservation | None,
    ring_lease: H3OutputRingLease | None,
) -> tuple[DeferredH3Task, ...]:
    """Run one H3 quantum after its resident request state has been bound."""

    variant = operation.work
    if variant is ForwardMode.GEN_TRANSITION:
        placement = trajectory_placement(partition, operation)
        if int(placement.start_step) != 0 or int(placement.step_count) != 0:
            raise invalid_descriptor("H3 transition placement must carry zero denoise steps")
        return ()
    if variant is ForwardMode.GEN_FLOW:
        placement = trajectory_placement(partition, operation)
        model.denoise(slot, int(placement.start_step), int(placement.step_count))
        return ()
    if variant is ForwardMode.GEN_DECODE:
        placement = decode_placement(partition, operation)
        if placement.kind is DecodeKind.VIDEO:
            rgb = model.decode_video(slot, placement)
            if mesh.coord("sp") != 0:
                return ()
            if rgb is None or reservation is None or ring_lease is None or mux is None:
                raise RuntimeError("rank zero lost its H3 video capture resources")
            with profile_range(
                f"uniserve.h3.decode_copy request={_request_label(operation)} "
                f"op={operation.op_id} kind=video unit={placement.start_unit} "
                f"rank={mesh.coord('sp')}"
            ):
                capture = buffer.capture_bytes_into(rgb, ring_lease.storage)
            try:
                return (
                    mux.video(
                        operation.request_key,
                        placement.start_unit,
                        capture,
                        reservation,
                        ring_lease,
                        operation.op_id,
                    ),
                )
            except BaseException:
                ring_lease.defer_until_capture_ready(capture)
                raise
        pcm = model.decode_audio(slot, placement)
        if mesh.coord("sp") != 0:
            return ()
        if pcm is None or reservation is None or ring_lease is None or mux is None:
            raise RuntimeError("rank zero lost its H3 audio capture resources")
        with profile_range(
            f"uniserve.h3.decode_copy request={_request_label(operation)} "
            f"op={operation.op_id} kind=audio rank={mesh.coord('sp')}"
        ):
            capture = buffer.capture_bytes_into(pcm.view(torch.uint8), ring_lease.storage)
        try:
            return (
                mux.audio(
                    operation.request_key,
                    capture,
                    reservation,
                    ring_lease,
                    operation.op_id,
                ),
            )
        except BaseException:
            ring_lease.defer_until_capture_ready(capture)
            raise
    if variant is ForwardMode.MATERIALIZE:
        if mesh.coord("sp") != 0:
            return ()
        if reservation is None or mux is None:
            raise RuntimeError("rank zero lost its H3 materialize reservation")
        return (mux.materialize(operation.request_key, reservation, operation.op_id),)
    raise invalid_descriptor(f"unsupported H3 work variant {variant.value!r}")


def run_action(runtime: ExecutionResources, state: OperationState) -> bool:
    """Land one ready H3 action without constructing a packed forward row."""

    if state.phase != "initial" or not isinstance(runtime.model, MiniMaxH3Model):
        return False
    operation = state.operation
    if operation.work not in {
        ForwardMode.GEN_TRANSITION,
        ForwardMode.GEN_FLOW,
        ForwardMode.GEN_DECODE,
        ForwardMode.MATERIALIZE,
    }:
        return False
    scope = state.partition
    if not isinstance(scope, PartitionState):
        raise RuntimeError("H3 action lost its partition state")
    model = runtime.model
    session = runtime.request_row(scope, operation.request_key.session_id)
    slot = model.states.get(int(session.request_pool_idx))
    mux = runtime._h3_mux
    if runtime.mesh.coord("sp") == 0 and mux is None:
        raise capability_mismatch("rank zero has no H3 mux resources")
    identity = runtime.operation_identity(operation)
    if operation.work is ForwardMode.GEN_TRANSITION:
        admission = scope.admissions.get(operation.request_key)
        if admission is None:
            raise invalid_descriptor("H3 transition has no matching admission")
        model.prepare(slot, admission)
        output_path = _output_path(runtime, admission)
        if runtime.mesh.coord("sp") == 0:
            assert mux is not None
            mux.open(operation.request_key, output_path)
    tasks = execute_action(
        model,
        runtime.mesh,
        mux,
        operation,
        scope.partition,
        slot,
        scope.completion,
        scope.cpu_tasks.get(identity),
        scope.h3_output_leases.get(identity),
    )
    session.flow_step = int(slot.denoise_step)
    if operation.work is ForwardMode.MATERIALIZE:
        session.flow_step = 0
    state.outcome = Outcome(
        status=OpStatus.OK,
        selected_point=1 if operation.advances_state else 0,
        logical_lengths=runtime.logical_lengths(
            operation,
            session,
            None,
            latent_len=int(session.flow_step),
        ),
        token_span=TokenSpan(base=int(session.logical_position), len=0),
        finish_flags=FinishFlags(),
        product_generations=runtime.output_generations(operation),
        completion_tasks=tasks,
    )
    state.phase = "done"
    return True


def _output_path(runtime: ExecutionResources, admission: Admission) -> Path:
    media = admission.media
    spool = runtime._media_spool
    if media is None or spool is None:
        raise invalid_descriptor("H3 admission is missing its media output")
    output_path = Path(media.output_path).expanduser()
    try:
        output_parent = output_path.parent.resolve(strict=True)
    except OSError as error:
        raise invalid_descriptor("H3 output directory is unavailable") from error
    if output_parent != spool or output_path.suffix != ".mp4":
        raise invalid_descriptor("H3 output must be an MP4 in the configured media spool")
    return output_parent / output_path.name


def _request_label(operation: Operation) -> str:
    key = operation.request_key
    return f"{key.authority_id}:{key.session_id}:{key.epoch}"


__all__ = [
    "decode_placement",
    "execute_action",
    "run_action",
    "trajectory_placement",
    "validate_batch",
]
