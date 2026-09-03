"""Common bounded video decode, capture, and artifact actions."""

from __future__ import annotations

import concurrent.futures
from threading import RLock
from typing import Callable, cast

import numpy as np
import torch

from uniserve_worker.execution.batch import (
    DecodePlacement,
    FinishFlags,
    FixedCheckpoint,
    MediaOutput,
    Operation,
    OpStatus,
    PosixShmArtifact,
    RequestKey,
    Run,
    RunKind,
    RunLane,
    TokenSpan,
)
from uniserve_worker.execution.output import ByteCapture, CpuJob, OutputBuffer
from uniserve_worker.foundation.errors import (
    invalid_descriptor,
    resource_error,
    unsupported_setup,
)
from uniserve_worker.media.mux import AvMuxConfig, AvMuxSession, require_media_codecs
from uniserve_worker.models.video import DecodeKind, VideoOutputGeometry, VideoRunner
from uniserve_worker.nn.mesh import DeviceMesh
from uniserve_worker.profiling import profile_range
from uniserve_worker.runtime.cpu import CpuTaskReservation
from uniserve_worker.transfer.tickets import ShmTransport

from .resources import ExecutionResources
from .rows import LaneState, OperationState, Outcome


def require_video_codecs() -> None:
    require_media_codecs("libx264", "aac")


class VideoOutputRing:
    """Bounded pinned captures shared by bounded video decode implementations."""

    def __init__(
        self,
        *,
        state_slots: int,
        unresolved_window: int,
        max_video_frames_per_round: int,
        max_geometry: VideoOutputGeometry,
    ) -> None:
        self.video_capacity = int(state_slots) * int(unresolved_window)
        self.audio_capacity = int(state_slots)
        if min(self.video_capacity, self.audio_capacity) < 1:
            raise ValueError("video output-ring capacities must be positive")
        video_bytes = (
            int(max_video_frames_per_round)
            * int(max_geometry.height)
            * int(max_geometry.width)
            * 3
        )
        audio_bytes = (
            round(
                int(max_geometry.frame_count)
                * int(max_geometry.audio_rate)
                / int(max_geometry.frame_rate)
            )
            * 2
            * 2
        )
        if min(video_bytes, audio_bytes) < 1:
            raise ValueError("video output-ring media capacities must be positive")
        self._video_storage = tuple(
            torch.empty(video_bytes, dtype=torch.uint8, pin_memory=True)
            for _ in range(self.video_capacity)
        )
        self._audio_storage = tuple(
            torch.empty(audio_bytes, dtype=torch.uint8, pin_memory=True)
            for _ in range(self.audio_capacity)
        )
        self._video_free = list(range(self.video_capacity - 1, -1, -1))
        self._audio_free = list(range(self.audio_capacity - 1, -1, -1))
        self._lock = RLock()

    def reserve(self, kind: str) -> "VideoOutputRingLease":
        if kind not in {"video", "audio"}:
            raise ValueError(f"unknown video output-ring kind {kind!r}")
        with self._lock:
            free = self._video_free if kind == "video" else self._audio_free
            if not free:
                raise resource_error(f"video {kind} output ring is exhausted")
            index = free.pop()
        return VideoOutputRingLease(self, kind, index)

    def _storage(self, kind: str, index: int) -> torch.Tensor:
        values = self._video_storage if kind == "video" else self._audio_storage
        return values[int(index)]

    def _release(self, kind: str, index: int) -> None:
        with self._lock:
            free = self._video_free if kind == "video" else self._audio_free
            capacity = self.video_capacity if kind == "video" else self.audio_capacity
            if int(index) in free or not 0 <= int(index) < capacity:
                raise RuntimeError("video output-ring ownership is invalid")
            free.append(int(index))

    @property
    def used(self) -> tuple[int, int]:
        with self._lock:
            return (
                self.video_capacity - len(self._video_free),
                self.audio_capacity - len(self._audio_free),
            )


class VideoOutputRingLease:
    __slots__ = ("_ring", "kind", "index", "_released")

    def __init__(self, ring: VideoOutputRing, kind: str, index: int) -> None:
        self._ring = ring
        self.kind = kind
        self.index = int(index)
        self._released = False

    @property
    def storage(self) -> torch.Tensor:
        if self._released:
            raise RuntimeError("video output-ring storage was accessed after release")
        return self._ring._storage(self.kind, self.index)

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        self._ring._release(self.kind, self.index)

    def defer_until_capture_ready(self, capture: ByteCapture) -> None:
        if self._released:
            return
        if capture.ready():
            self.release()
            return
        self._released = True
        capture.buffer.retain_until_ready(
            _DeferredRingRelease(self._ring, self.kind, self.index)
        )

    def __del__(self) -> None:
        self.release()


class _DeferredRingRelease:
    __slots__ = ("_ring", "_kind", "_index")

    def __init__(self, ring: VideoOutputRing, kind: str, index: int) -> None:
        self._ring = ring
        self._kind = kind
        self._index = int(index)

    def __del__(self) -> None:
        self._ring._release(self._kind, self._index)


class VideoMuxCoordinator:
    """Request-indexed mux sessions with independent video and audio tails."""

    def __init__(self) -> None:
        self._sessions: dict[RequestKey, AvMuxSession] = {}
        self._video_tails: dict[
            RequestKey, concurrent.futures.Future[object] | None
        ] = {}
        self._audio_tails: dict[
            RequestKey, concurrent.futures.Future[object] | None
        ] = {}

    def open(self, request_key: RequestKey, *, geometry) -> None:
        if request_key in self._sessions:
            raise RuntimeError("video mux session is already active")
        self._sessions[request_key] = AvMuxSession(
            AvMuxConfig(
                width=int(geometry.width),
                height=int(geometry.height),
                frame_count=int(geometry.frame_count),
                frame_rate=int(geometry.frame_rate),
                audio_rate=int(geometry.audio_rate),
                video_unit_frames=tuple(int(value) for value in geometry.unit_frames),
            )
        )
        self._video_tails[request_key] = None
        self._audio_tails[request_key] = None

    def _task(
        self,
        request_key: RequestKey,
        reservation: CpuTaskReservation,
        action: Callable[[AvMuxSession], object],
        capture: ByteCapture | None,
        dependencies: tuple[concurrent.futures.Future[object], ...],
        ring_lease: VideoOutputRingLease | None = None,
        *,
        profile_name: str,
    ) -> CpuJob:
        session = self._sessions.get(request_key)
        if session is None:
            raise RuntimeError("video mux session is not active")
        return CpuJob(
            reservation,
            lambda: action(session),
            capture=capture,
            dependencies=dependencies,
            release=None if ring_lease is None else ring_lease.release,
            defer_release=(
                None if ring_lease is None else ring_lease.defer_until_capture_ready
            ),
            profile_name=profile_name,
        )

    def video(
        self,
        request_key: RequestKey,
        start_unit: int,
        unit_count: int,
        capture: ByteCapture,
        reservation: CpuTaskReservation,
        ring_lease: VideoOutputRingLease,
        operation_id: int,
    ) -> CpuJob:
        dependency = self._video_tails[request_key]
        task = self._task(
            request_key,
            reservation,
            lambda session: session.write_video(
                start_unit, unit_count, capture.numpy()
            ),
            capture,
            () if dependency is None else (dependency,),
            ring_lease,
            profile_name=(
                f"uniserve.video.mux request={_key_label(request_key)} "
                f"op={operation_id} kind=video start_unit={start_unit} "
                f"unit_count={unit_count} rank=0"
            ),
        )
        self._video_tails[request_key] = task.promise
        return task

    def audio(
        self,
        request_key: RequestKey,
        capture: ByteCapture,
        reservation: CpuTaskReservation,
        ring_lease: VideoOutputRingLease,
        operation_id: int,
    ) -> CpuJob:
        dependency = self._audio_tails[request_key]
        task = self._task(
            request_key,
            reservation,
            lambda session: session.write_audio(
                capture.numpy().reshape(-1).view(np.int16).reshape(-1, 2)
            ),
            capture,
            () if dependency is None else (dependency,),
            ring_lease,
            profile_name=(
                f"uniserve.video.mux request={_key_label(request_key)} "
                f"op={operation_id} kind=audio rank=0"
            ),
        )
        self._audio_tails[request_key] = task.promise
        return task

    def finalize_artifact(
        self,
        request_key: RequestKey,
        reservation: CpuTaskReservation,
        operation_id: int,
    ) -> CpuJob:
        dependencies = tuple(
            tail
            for tail in (self._video_tails[request_key], self._audio_tails[request_key])
            if tail is not None
        )

        def publish(session: AvMuxSession) -> MediaOutput:
            payload = session.close()
            locator = ShmTransport.publish_bytes(payload)
            self._sessions.pop(request_key, None)
            self._video_tails.pop(request_key, None)
            self._audio_tails.pop(request_key, None)
            return MediaOutput(
                handle=PosixShmArtifact(name=locator.handle.decode()),
                bytes=locator.nbytes,
            )

        return self._task(
            request_key,
            reservation,
            publish,
            None,
            dependencies,
            profile_name=(
                f"uniserve.video.mux request={_key_label(request_key)} "
                f"op={operation_id} kind=artifact rank=0"
            ),
        )

    def drop(self, request_id: int) -> None:
        selected = [key for key in self._sessions if key.request_id == int(request_id)]
        for key in selected:
            session = self._sessions.pop(key)
            self._video_tails.pop(key, None)
            self._audio_tails.pop(key, None)
            session.abort()

    def close(self) -> None:
        for session in self._sessions.values():
            session.abort()
        self._sessions.clear()
        self._video_tails.clear()
        self._audio_tails.clear()


def _key_label(request_key: RequestKey) -> str:
    return f"{request_key.authority_id}:{request_key.request_id}:{request_key.epoch}"


def trajectory_placement(lane: RunLane, operation: Operation):
    selected = tuple(
        placement
        for placement in lane.latent_placements
        if placement.request_key == operation.request_key
        and int(placement.op_id) == int(operation.op_id)
    )
    if len(selected) != 1:
        raise invalid_descriptor("video trajectory operation has no exact latent placement")
    return selected[0]


def decode_placement(
    lane: RunLane, operation: Operation
) -> DecodePlacement:
    selected = tuple(
        placement
        for placement in lane.decode_placements
        if placement.request_key == operation.request_key
        and int(placement.op_id) == int(operation.op_id)
    )
    if len(selected) != 1:
        raise invalid_descriptor(
            "video decode operation has no exact decode placement"
        )
    return selected[0]


def validate_batch(runtime: ExecutionResources, batch: Run) -> None:
    """Validate video admission requirements before staging state."""

    if not isinstance(runtime.model, VideoRunner):
        return
    admissions = {admission.request_key: admission for admission in batch.admissions}
    preparations = {
        operation.request_key: operation
        for operation in batch.operations
        if operation.kind is RunKind.DIFFUSION_PREPARE
    }
    if set(admissions) != set(preparations):
        raise invalid_descriptor("video starts must exactly match preparation operations")
    for request_key, admission in admissions.items():
        slot = runtime.requests.model_state_slot(admission.request_pool_idx)
        if slot.active:
            raise invalid_descriptor("video start targets an occupied request slot")
        operation = preparations[request_key]
        point = operation.parent.point
        if (
            int(operation.parent.op_id) != 0
            or not isinstance(point, FixedCheckpoint)
            or int(point.point_index) != 0
        ):
            raise invalid_descriptor("video preparation does not name its request root")


def execute_action(
    model: VideoRunner,
    mesh: DeviceMesh,
    mux: object | None,
    operation: Operation,
    lane: RunLane,
    slot: object,
    buffer: OutputBuffer,
    reservation: CpuTaskReservation | None,
    ring_lease: object | None,
) -> tuple[CpuJob, ...]:
    """Run one video quantum after its resident request state has been bound."""

    variant = operation.kind
    if variant is RunKind.DIFFUSION_PREPARE:
        placement = trajectory_placement(lane, operation)
        if int(placement.start_step) != 0 or int(placement.step_count) != 0:
            raise invalid_descriptor("video preparation placement must carry zero denoise steps")
        return ()
    if variant is RunKind.DIFFUSION_STEP:
        placement = trajectory_placement(lane, operation)
        model.denoise(slot, int(placement.start_step), int(placement.step_count))
        return ()
    if variant is RunKind.DIFFUSION_DECODE:
        placement = decode_placement(lane, operation)
        decoded = model.decode(
            slot, int(placement.cursor), int(placement.max_units)
        )
        if decoded.kind is DecodeKind.VIDEO:
            rgb = decoded.value
            if mesh.coord("sp") != 0:
                return ()
            if rgb is None or reservation is None or ring_lease is None or mux is None:
                raise RuntimeError("rank zero lost its video capture resources")
            with profile_range(
                f"uniserve.video.decode_copy request={_request_label(operation)} "
                f"op={operation.op_id} kind=video unit={decoded.unit_offset} "
                f"rank={mesh.coord('sp')}"
            ):
                capture = buffer.capture_bytes_into(rgb, ring_lease.storage)
            try:
                return (
                    mux.video(
                        operation.request_key,
                        decoded.unit_offset,
                        decoded.unit_count,
                        capture,
                        reservation,
                        ring_lease,
                        operation.op_id,
                    ),
                )
            except BaseException:
                ring_lease.defer_until_capture_ready(capture)
                raise
        if decoded.kind is not DecodeKind.AUDIO:
            raise invalid_descriptor("video bounded decode returned an unexpected action")
        pcm = decoded.value
        if mesh.coord("sp") != 0:
            return ()
        if pcm is None or reservation is None or ring_lease is None or mux is None:
            raise RuntimeError("rank zero lost its audio capture resources")
        with profile_range(
            f"uniserve.video.decode_copy request={_request_label(operation)} "
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
    if variant is RunKind.DIFFUSION_FINALIZE:
        placement = decode_placement(lane, operation)
        decoded = model.decode(
            slot, int(placement.cursor), int(placement.max_units)
        )
        if decoded.kind is not DecodeKind.FINALIZE:
            raise invalid_descriptor("video final decode did not select artifact finalization")
        model.finalize(slot)
        if mesh.coord("sp") != 0:
            return ()
        if reservation is None or mux is None:
            raise RuntimeError("rank zero lost its artifact reservation")
        return (mux.finalize_artifact(operation.request_key, reservation, operation.op_id),)
    raise invalid_descriptor(f"unsupported video work variant {variant.value!r}")


def run_action(runtime: ExecutionResources, state: OperationState) -> bool:
    """Land one ready video action without constructing a packed forward row."""

    if state.phase != "initial" or not isinstance(runtime.model, VideoRunner):
        return False
    operation = state.operation
    if operation.kind not in {
        RunKind.DIFFUSION_PREPARE,
        RunKind.DIFFUSION_STEP,
        RunKind.DIFFUSION_DECODE,
        RunKind.DIFFUSION_FINALIZE,
    }:
        return False
    scope = state.lane
    if not isinstance(scope, LaneState):
        raise RuntimeError("video action lost its physical run state")
    model = cast(VideoRunner, runtime.model)
    request = runtime.request_row(scope, operation.request_key.request_id)
    slot = runtime.requests.model_state_slot(request.request_pool_idx)
    mux = runtime._media_mux
    if runtime.mesh.coord("sp") == 0 and mux is None:
        raise unsupported_setup("rank zero has no video mux resources")
    identity = runtime.operation_identity(operation)
    if operation.kind is RunKind.DIFFUSION_PREPARE:
        admission = scope.admissions.get(operation.request_key)
        if admission is None:
            raise invalid_descriptor("video preparation has no matching start command")
        model.prepare(slot, admission)
        if runtime.mesh.coord("sp") == 0:
            assert mux is not None
            media = admission.diffusion
            if media is None:
                raise invalid_descriptor("video preparation has no media parameters")
            geometry = model.output_geometry(slot)
            mux.open(operation.request_key, geometry=geometry)
    tasks = execute_action(
        model,
        runtime.mesh,
        mux,
        operation,
        scope.lane,
        slot,
        scope.completion,
        scope.cpu_tasks.get(identity),
        scope.media_output_leases.get(identity),
    )
    request.flow_step = int(slot.denoise_step)
    if operation.kind is RunKind.DIFFUSION_FINALIZE:
        request.flow_step = 0
    state.outcome = Outcome(
        status=OpStatus.OK,
        selected_point=1 if operation.advances_state else 0,
        logical_lengths=runtime.logical_lengths(
            operation,
            request,
            None,
            latent_len=int(request.flow_step),
        ),
        token_span=TokenSpan(base=int(request.logical_position), len=0),
        finish_flags=FinishFlags(),
        product_generations=runtime.output_generations(operation),
        completion_tasks=tasks,
        next_cursor=(
            int(decode_placement(scope.lane, operation).cursor)
            + int(decode_placement(scope.lane, operation).max_units)
            if operation.kind in {RunKind.DIFFUSION_DECODE, RunKind.DIFFUSION_FINALIZE}
            else 0
        ),
        done=operation.kind is RunKind.DIFFUSION_FINALIZE,
    )
    state.phase = "done"
    return True


def _request_label(operation: Operation) -> str:
    key = operation.request_key
    return f"{key.authority_id}:{key.request_id}:{key.epoch}"


__all__ = [
    "VideoMuxCoordinator",
    "VideoOutputRing",
    "VideoOutputRingLease",
    "decode_placement",
    "execute_action",
    "run_action",
    "require_video_codecs",
    "trajectory_placement",
    "validate_batch",
]
