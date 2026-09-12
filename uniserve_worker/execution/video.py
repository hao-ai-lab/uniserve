"""Common bounded video decode, capture, and artifact actions."""

from __future__ import annotations

import concurrent.futures
from threading import RLock
from typing import TYPE_CHECKING, Callable

import numpy as np
import torch

from uniserve_worker.execution.batch import (
    ComputationId,
    DecodeRange,
    FinishFlags,
    MediaOutput,
    MediaTrack,
    OpStatus,
    PipelineStage,
    PosixShmArtifact,
    RequestKey,
    Run,
    RunLane,
    ScheduledRequest,
    TensorPublication,
)
from uniserve_worker.execution.output import ByteCapture, CpuJob
from uniserve_worker.foundation.errors import (
    invalid_descriptor,
    resource_error,
    unsupported_setup,
)
from uniserve_worker.media.mux import AvMuxConfig, AvMuxSession, require_media_codecs
from uniserve_worker.media.storage import publish_media_bytes
from uniserve_worker.models.video import VideoModel, VideoOutputGeometry
from uniserve_worker.profiling import profile_range
from uniserve_worker.runtime.cpu import CpuTaskReservation

from . import operations as operation_geometry
from .rows import OperationState, Outcome

if TYPE_CHECKING:
    from collections.abc import Mapping

    from ..models.runtime import ExecutionModel
    from ..runtime.device_products import DeviceProducts
    from ..runtime.encoder_cache import EncoderCache
    from ..runtime.request import RequestPool
    from ..transfer.tickets import Transport
    from .model_runner import ModelRunner
    from .video import VideoMuxCoordinator


def require_video_codecs() -> None:
    """Verify that the model’s configured video and audio codec dependencies are installed."""

    require_media_codecs("libx264", "aac")


def create_media_resources(
    model: VideoModel,
    *,
    rank: int,
    owns_output: bool,
    state_slots: int,
    unresolved_window: int,
) -> tuple[VideoMuxCoordinator | None, VideoOutputRing | None]:
    """Provision public output resources from the declared media geometry."""

    if not model.owns_media_output or not owns_output:
        return None, None
    require_video_codecs()
    return (
        VideoMuxCoordinator(rank=rank),
        VideoOutputRing(
            state_slots=state_slots,
            unresolved_window=unresolved_window,
            max_video_frames_per_round=model.decode_frame_capacity,
            max_geometry=model.output_capacity,
        ),
    )


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
        """Allocate bounded video and audio tensors with independent free-slot queues."""

        self.video_capacity = int(state_slots) * int(unresolved_window)
        self.audio_capacity = int(state_slots)
        if min(self.video_capacity, self.audio_capacity) < 1:
            raise ValueError("video output-ring capacities must be positive")
        video_bytes = (
            int(max_video_frames_per_round) * int(max_geometry.height) * int(max_geometry.width) * 3
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
        """Lease an unused video or audio output slot from the fixed ring."""

        if kind not in {"video", "audio"}:
            raise ValueError(f"unknown video output-ring kind {kind!r}")
        with self._lock:
            free = self._video_free if kind == "video" else self._audio_free
            if not free:
                raise resource_error(f"video {kind} output ring is exhausted")
            index = free.pop()
        return VideoOutputRingLease(self, kind, index)

    def _storage(self, kind: str, index: int) -> torch.Tensor:
        """Return backing storage for one typed output-ring slot."""

        values = self._video_storage if kind == "video" else self._audio_storage
        return values[int(index)]

    def _release(self, kind: str, index: int) -> None:
        """Return a typed output-ring slot to its free queue."""

        with self._lock:
            free = self._video_free if kind == "video" else self._audio_free
            capacity = self.video_capacity if kind == "video" else self.audio_capacity
            if int(index) in free or not 0 <= int(index) < capacity:
                raise RuntimeError("video output-ring ownership is invalid")
            free.append(int(index))

    @property
    def used(self) -> tuple[int, int]:
        """Return the number of output-ring slots currently leased."""

        with self._lock:
            return (
                self.video_capacity - len(self._video_free),
                self.audio_capacity - len(self._audio_free),
            )


class VideoOutputRingLease:
    """Grants exclusive access to one video output slot until immediate or deferred release."""

    __slots__ = ("_ring", "kind", "index", "_released")

    def __init__(self, ring: VideoOutputRing, kind: str, index: int) -> None:
        """Take exclusive ownership of one typed output-ring slot."""

        self._ring = ring
        self.kind = kind
        self.index = int(index)
        self._released = False

    @property
    def storage(self) -> torch.Tensor:
        """Expose the leased output tensor while this ring slot remains owned."""

        if self._released:
            raise RuntimeError("video output-ring storage was accessed after release")
        return self._ring._storage(self.kind, self.index)

    def release(self) -> None:
        """Return this output slot to the ring exactly once."""

        if self._released:
            return
        self._released = True
        self._ring._release(self.kind, self.index)

    def defer_until_ready(self, completion: concurrent.futures.Future[None]) -> None:
        """Keep this output slot leased until its existing device fence completes."""

        if self._released:
            return
        if completion.done():
            self.release()
            return
        self._released = True
        ring, kind, index = self._ring, self.kind, self.index
        completion.add_done_callback(lambda _future: ring._release(kind, index))


class VideoMuxCoordinator:
    """Request-indexed mux sessions with independent video and audio tails."""

    def __init__(self, *, rank: int) -> None:
        """Initialize per-request mux sessions and temporal overlap tails."""

        self.rank = rank
        self._sessions: dict[RequestKey, AvMuxSession] = {}
        self._video_tails: dict[RequestKey, concurrent.futures.Future[object] | None] = {}
        self._audio_tails: dict[RequestKey, concurrent.futures.Future[object] | None] = {}

    def open(self, request_key: RequestKey, *, geometry) -> None:
        """Create the request-owned mux session for a validated output geometry."""

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
        """Submit one ordered mux action and release its reservation and ring lease on completion."""

        session = self._sessions.get(request_key)
        if session is None:
            raise RuntimeError("video mux session is not active")
        return CpuJob(
            reservation,
            lambda: action(session),
            capture=capture,
            dependencies=dependencies,
            release=None if ring_lease is None else ring_lease.release,
            defer_release=(None if ring_lease is None else ring_lease.defer_until_ready),
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
        """Schedule ordered RGB frame encoding from a captured output-ring slot."""

        dependency = self._video_tails[request_key]
        task = self._task(
            request_key,
            reservation,
            lambda session: session.write_video(start_unit, unit_count, capture.numpy()),
            capture,
            () if dependency is None else (dependency,),
            ring_lease,
            profile_name=(
                f"uniserve.video.mux request={_key_label(request_key)} "
                f"op={operation_id} kind=video start_unit={start_unit} "
                f"unit_count={unit_count} rank={self.rank}"
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
        """Schedule PCM encoding from a captured output-ring slot."""

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
                f"op={operation_id} kind=audio rank={self.rank}"
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
        """Schedule mux finalization and shared-memory publication after all segment jobs."""

        dependencies = tuple(
            tail
            for tail in (self._video_tails[request_key], self._audio_tails[request_key])
            if tail is not None
        )

        def publish(session: AvMuxSession) -> MediaOutput:
            """Close the mux session, publish its bytes, and release request-local tails."""

            payload = session.close()
            name = publish_media_bytes(payload)
            self._sessions.pop(request_key, None)
            self._video_tails.pop(request_key, None)
            self._audio_tails.pop(request_key, None)
            return MediaOutput(
                handle=PosixShmArtifact(name=name),
                bytes=len(payload),
            )

        return self._task(
            request_key,
            reservation,
            publish,
            None,
            dependencies,
            profile_name=(
                f"uniserve.video.mux request={_key_label(request_key)} "
                f"op={operation_id} kind=artifact rank={self.rank}"
            ),
        )

    def drop(self, request_id: int) -> None:
        """Abort and remove every mux session owned by a request identifier."""

        selected = [key for key in self._sessions if key.request_id == int(request_id)]
        for key in selected:
            session = self._sessions.pop(key)
            self._video_tails.pop(key, None)
            self._audio_tails.pop(key, None)
            session.abort()

    def close(self) -> None:
        """Abort all active mux sessions and reject new media work."""

        for session in self._sessions.values():
            session.abort()
        self._sessions.clear()
        self._video_tails.clear()
        self._audio_tails.clear()


def _key_label(request_key: RequestKey) -> str:
    """Format a stable request key for media task profiling."""

    return f"{request_key.engine_id}:{request_key.request_id}:{request_key.request_epoch}"


def trajectory_params(lane: RunLane, operation: ScheduledRequest):
    """Return the unique latent trajectory params assigned to an operation."""

    selected = tuple(
        params
        for params in lane.latent_params
        if params.request_key == operation.request_key and params.op_id == operation.op_id
    )
    if len(selected) != 1:
        raise invalid_descriptor("video trajectory operation has no exact latent params")
    return selected[0]


def decode_range(lane: RunLane, operation: ScheduledRequest) -> DecodeRange:
    """Return the unique reconstruction params assigned to an operation."""

    selected = tuple(
        params
        for params in lane.decode_ranges
        if params.request_key == operation.request_key and params.op_id == operation.op_id
    )
    if len(selected) != 1:
        raise invalid_descriptor("video decode operation has no exact decode params")
    return selected[0]


def validate_batch(batch: Run, *, execution_model: ExecutionModel) -> None:
    """Validate video admission requirements before staging state."""

    if not isinstance(execution_model, VideoModel):
        return
    for operation in batch.operations:
        if operation.kind is not PipelineStage.LATENT_PREPARATION:
            continue
        if operation.predecessor != ComputationId(0, 0):
            raise invalid_descriptor("video preparation does not name its request root")


def run_action(
    state: OperationState,
    *,
    device_products: DeviceProducts,
    encoder_cache: EncoderCache,
    media_mux: VideoMuxCoordinator | None,
    execution_model: ExecutionModel,
    publication_transports: Mapping[str, Transport],
    request_pool: RequestPool,
    model_runner: ModelRunner,
) -> bool:
    """Land one ready video action without constructing a packed forward row."""

    if state.phase != "initial" or not isinstance(execution_model, VideoModel):
        return False
    operation = state.operation
    if operation.kind not in {
        PipelineStage.LATENT_PREPARATION,
        PipelineStage.DENOISING,
        PipelineStage.VIDEO_DECODING,
        PipelineStage.AUDIO_DECODING,
        PipelineStage.VIDEO_ENCODING,
        PipelineStage.AUDIO_ENCODING,
        PipelineStage.MUXING,
    }:
        return False
    scope = state.lane
    model = execution_model
    request = operation_geometry.request_row(scope, operation.request_key.request_id)
    media = request.request.admission.diffusion
    if media is None:
        raise invalid_descriptor("video operation has no admitted media geometry")
    scratch = model_runner.scratch
    if scratch is None:
        raise RuntimeError("video execution has no allocated scratch storage")
    media_geometry = media
    prompt_token_ids = request.request.admission.prompt_token_ids
    num_prompt_tokens = len(prompt_token_ids)
    metadata = model_runner.prepare_geometry(
        model.execution_key(media_geometry, num_prompt_tokens),
        lambda: model.build_execution(
            media_geometry, num_prompt_tokens, scratch, model_runner.context_workspace
        ),
    )
    slot = model.request_tensors(request_pool.tensors(request.request.request_pool_idx), metadata)
    mux = media_mux
    if (
        operation.kind
        in {PipelineStage.VIDEO_ENCODING, PipelineStage.AUDIO_ENCODING, PipelineStage.MUXING}
        and mux is None
    ):
        raise unsupported_setup("output owner has no video mux resources")
    identity = operation_geometry.operation_identity(operation)
    from . import transfer

    tasks: tuple[CpuJob, ...] = ()
    products: tuple[TensorPublication, ...] = ()
    if operation.kind is PipelineStage.LATENT_PREPARATION:
        params = trajectory_params(scope.lane, operation)
        if int(params.start_step) != 0 or int(params.step_count) != 0:
            raise invalid_descriptor("video preparation params must carry zero denoise steps")
        inputs = operation.inputs
        if len(inputs) != 1:
            raise invalid_descriptor("video preparation requires one conditioning Tensor")
        conditioning = device_products.consume(
            inputs[0],
            consumer_op_id=operation.op_id,
            device=model_runner.operation_device(operation),
        )
        scope.device_reads.append(conditioning)
        if conditioning.region is not None:
            raise invalid_descriptor("video preparation requires complete conditioning coverage")
        encoded = conditioning.tensor
        with model_runner.preparing_inputs(model.initialize_tensors(slot, media.seed)):
            if model.conditioner is not None:
                if encoded is None:
                    raise invalid_descriptor("conditioning module has no input Tensor")
                result = model_runner.run_module(
                    "conditioner",
                    encoded,
                )
                scope.observations.append(result.observation)
                if len(result.values) != 1:
                    raise invalid_descriptor("conditioning computation must return one Tensor")
                encoded = result.values[0]
            model.prepare_tensors(slot, metadata, encoded, num_prompt_tokens)
    elif operation.kind is PipelineStage.DENOISING:
        params = trajectory_params(scope.lane, operation)
        start_step, step_count = int(params.start_step), int(params.step_count)
        if start_step != request.flow_step:
            raise invalid_descriptor("denoising does not begin at the selected request step")
        schedule = model_runner.schedule
        if schedule is None:
            raise RuntimeError("video execution lost its bound diffusion schedule")
        result = model_runner.run_denoising(
            slot,
            metadata,
            start_step,
            step_count,
            schedule,
            slot=request.request.request_pool_idx,
            geometry=model.execution_key(media_geometry, num_prompt_tokens),
        )
        scope.observations.append(result.observation)
        request.flow_step = start_step + step_count
        if operation.outputs:
            if request.flow_step != media.num_inference_steps:
                raise invalid_descriptor("final latent products require completed denoising")
            products = transfer.publish_tensors(
                operation,
                result.values,
                scope,
                device_products=device_products,
                encoder_cache=encoder_cache,
                publication_transports=publication_transports,
            )
    elif operation.kind in {
        PipelineStage.VIDEO_DECODING,
        PipelineStage.AUDIO_DECODING,
        PipelineStage.VIDEO_ENCODING,
        PipelineStage.AUDIO_ENCODING,
    }:
        params = decode_range(scope.lane, operation)
        inputs = operation.inputs
        if len(inputs) != 1:
            raise invalid_descriptor("media reconstruction requires one Tensor input")
        read = device_products.consume(
            inputs[0],
            consumer_op_id=operation.op_id,
            device=model_runner.operation_device(operation),
        )
        scope.device_reads.append(read)
        if read.region is not None:
            raise invalid_descriptor("media reconstruction requires complete input coverage")
        cursor, count = params.cursor, params.max_units
        track = (
            MediaTrack.AUDIO
            if operation.kind in {PipelineStage.AUDIO_DECODING, PipelineStage.AUDIO_ENCODING}
            else MediaTrack.VIDEO
        )
        if operation.kind in {PipelineStage.VIDEO_DECODING, PipelineStage.AUDIO_DECODING}:
            value = model.decoder_input(metadata, read.tensor, track, cursor, count)
            result = model_runner.run_module(
                operation.entry,
                value,
            )
            scope.observations.append(result.observation)
            if len(result.values) != 1:
                raise invalid_descriptor("media decoder must return one numerical tensor")
            decoded = model.decoder_output(metadata, result.values[0], track)
            products = transfer.publish_tensors(
                operation,
                (decoded,),
                scope,
                device_products=device_products,
                encoder_cache=encoder_cache,
                publication_transports=publication_transports,
            )
        else:
            owner = request.request
            if owner.media_finalized:
                raise invalid_descriptor("media output is already finalized")
            if track is MediaTrack.VIDEO and (
                cursor != owner.media_video_units or cursor + count > media.num_decode_chunks
            ):
                raise invalid_descriptor("video assembly requires the next temporal range")
            if track is MediaTrack.AUDIO and owner.media_audio_written:
                raise invalid_descriptor("audio output is already written")
            assert mux is not None
            if owner.media_video_units == 0 and not owner.media_audio_written:
                mux.open(operation.request_key, geometry=model.output_geometry(media))
            reservation = scope.cpu_tasks.get(identity)
            ring_lease = scope.media_output_leases.get(identity)
            if reservation is None or ring_lease is None:
                raise RuntimeError("media output has no reserved capture storage")
            if track is MediaTrack.VIDEO:
                value = model.assemble_video(slot, metadata, read.tensor, cursor, count)
            else:
                value = read.tensor.view(torch.uint8)
            with profile_range(
                f"uniserve.video.decode_copy request={_request_label(operation)} op={operation.op_id} kind={track.value}"
            ):
                capture = scope.completion.capture_bytes_into(value, ring_lease.storage)
            try:
                if track is MediaTrack.VIDEO:
                    tasks = (
                        mux.video(
                            operation.request_key,
                            cursor,
                            count,
                            capture,
                            reservation,
                            ring_lease,
                            operation.op_id,
                        ),
                    )
                    owner.media_video_units += count
                else:
                    tasks = (
                        mux.audio(
                            operation.request_key, capture, reservation, ring_lease, operation.op_id
                        ),
                    )
                    owner.media_audio_written = True
            except BaseException:
                ring_lease.defer_until_ready(capture.buffer.completion_future())
                raise
    else:
        owner = request.request
        if (
            owner.media_video_units != media.num_decode_chunks
            or not owner.media_audio_written
            or owner.media_finalized
        ):
            raise invalid_descriptor("media finalization requires both completed output tracks")
        reservation = scope.cpu_tasks.get(identity)
        if reservation is None or mux is None:
            raise RuntimeError("media finalization has no reserved CPU task")
        tasks = (mux.finalize_artifact(operation.request_key, reservation, operation.op_id),)
        owner.media_finalized = True
    scope.completion_jobs.extend(tasks)
    state.outcome = Outcome(
        status=OpStatus.OK,
        runtime=operation_geometry.execution_runtime(
            request,
            None,
            flow_step=int(request.flow_step),
        ),
        finish_flags=FinishFlags(),
        product_generations=operation_geometry.output_generations(operation),
        completion_tasks=tasks,
        products=products,
    )
    state.phase = "done"
    return True


def _request_label(operation: ScheduledRequest) -> str:
    """Format a stable request and operation label for media work."""

    key = operation.request_key
    return f"{key.engine_id}:{key.request_id}:{key.request_epoch}"


__all__ = [
    "VideoMuxCoordinator",
    "VideoOutputRing",
    "VideoOutputRingLease",
    "decode_range",
    "run_action",
    "require_video_codecs",
    "trajectory_params",
    "validate_batch",
]


def require_media_output_ring(ring: VideoOutputRing | None) -> VideoOutputRing:
    """Require bounded media storage on the configured output rank."""

    if ring is None:
        raise unsupported_setup("operation requires the media output owner's ring")
    return ring
