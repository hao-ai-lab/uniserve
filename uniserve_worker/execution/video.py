"""Common bounded video decode, capture, and artifact actions."""

from __future__ import annotations

import concurrent.futures
from threading import RLock
from typing import TYPE_CHECKING, Callable

import numpy as np
import torch

from uniserve_worker.execution.batch import (
    DecodeRange,
    FinishFlags,
    FixedCheckpoint,
    MediaOutput,
    MediaTrack,
    OpCode,
    Operation,
    OpStatus,
    PosixShmArtifact,
    ProductKind,
    ProductPayload,
    RequestKey,
    Run,
    RunLane,
    TokenSpan,
)
from uniserve_worker.execution.output import ByteCapture, CpuJob
from uniserve_worker.foundation.errors import (
    invalid_descriptor,
    resource_error,
    unsupported_setup,
)
from uniserve_worker.media.mux import AvMuxConfig, AvMuxSession, require_media_codecs
from uniserve_worker.models.video import VideoModel, VideoOutputGeometry
from uniserve_worker.profiling import profile_range
from uniserve_worker.runtime.cpu import CpuTaskReservation
from uniserve_worker.runtime.device import allocate_shared_memory

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


def _publish_media_bytes(payload: bytes) -> str:
    """Transfer ownership of final media storage to the host artifact consumer."""

    from multiprocessing import resource_tracker

    if not payload:
        raise ValueError("shared-memory media publication must not be empty")
    shm = allocate_shared_memory(len(payload))
    try:
        buffer = shm.buf
        if buffer is None:
            raise RuntimeError("shared-memory artifact has no writable buffer")
        buffer[: len(payload)] = payload
    except BaseException:
        shm.unlink()
        raise
    finally:
        shm.close()
    resource_tracker.unregister("/" + shm.name.lstrip("/"), "shared_memory")
    return shm.name


def require_video_codecs() -> None:
    """Verify that the model’s configured video and audio codec dependencies are installed."""

    require_media_codecs("libx264", "aac")


def create_media_resources(
    model: VideoModel,
    *,
    device: torch.device | str,
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
            device=device,
            state_slots=state_slots,
            unresolved_window=unresolved_window,
            max_video_frames_per_round=model.decode_frame_capacity,
            max_geometry=model.output_capacity,
        ),
    )


class VideoOutputRing:
    """Bounded host captures, page-locked when their producer executes on CUDA."""

    def __init__(
        self,
        *,
        device: torch.device | str,
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
        pin_memory = torch.device(device).type == "cuda"
        self._video_storage = tuple(
            torch.empty(video_bytes, dtype=torch.uint8, pin_memory=pin_memory)
            for _ in range(self.video_capacity)
        )
        self._audio_storage = tuple(
            torch.empty(audio_bytes, dtype=torch.uint8, pin_memory=pin_memory)
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
            name = _publish_media_bytes(payload)
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

    return f"{request_key.authority_id}:{request_key.request_id}:{request_key.epoch}"


def trajectory_params(lane: RunLane, operation: Operation):
    """Return the unique latent trajectory params assigned to an operation."""

    selected = tuple(
        params
        for params in lane.latent_params
        if params.request_key == operation.request_key and int(params.op_id) == int(operation.op_id)
    )
    if len(selected) != 1:
        raise invalid_descriptor("video trajectory operation has no exact latent params")
    return selected[0]


def decode_range(lane: RunLane, operation: Operation) -> DecodeRange:
    """Return the unique reconstruction params assigned to an operation."""

    selected = tuple(
        params
        for params in lane.decode_ranges
        if params.request_key == operation.request_key and int(params.op_id) == int(operation.op_id)
    )
    if len(selected) != 1:
        raise invalid_descriptor("video decode operation has no exact decode params")
    return selected[0]


def validate_batch(batch: Run, *, execution_model: ExecutionModel) -> None:
    """Validate video admission requirements before staging state."""

    if not isinstance(execution_model, VideoModel):
        return
    for operation in batch.operations:
        if operation.kind is not OpCode.DIFFUSION_PREPARE:
            continue
        parent = operation.state_parent
        point = parent.point
        if parent.op_id != 0 or not isinstance(point, FixedCheckpoint) or point.point_index != 0:
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
        OpCode.DIFFUSION_PREPARE,
        OpCode.DIFFUSION_STEP,
        OpCode.DIFFUSION_DECODE,
        OpCode.MEDIA_APPEND,
        OpCode.DIFFUSION_FINALIZE,
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
    media_geometry = model.media_geometry(media)
    metadata = model_runner.prepare_geometry(
        model.execution_key(media_geometry),
        lambda: model.build_execution(media_geometry, scratch, model_runner.context_workspace),
    )
    slot = model.request_tensors(
        request_pool.tensors(request.request.request_pool_idx), media_geometry, metadata
    )
    mux = media_mux
    if operation.kind in {OpCode.MEDIA_APPEND, OpCode.DIFFUSION_FINALIZE} and mux is None:
        raise unsupported_setup("output owner has no video mux resources")
    identity = operation_geometry.operation_identity(operation)
    from . import transfer

    tasks: tuple[CpuJob, ...] = ()
    products: tuple[ProductPayload, ...] = ()
    next_cursor = 0
    if operation.kind is OpCode.DIFFUSION_PREPARE:
        params = trajectory_params(scope.lane, operation)
        if int(params.start_step) != 0 or int(params.step_count) != 0:
            raise invalid_descriptor("video preparation params must carry zero denoise steps")
        if len(media.prompt_token_ids) != media.geometry.prompt_tokens:
            raise invalid_descriptor("video prompt tokens disagree with admitted geometry")
        inputs = tuple(
            product for product in operation.inputs if product.kind is ProductKind.TENSOR
        )
        expected_inputs = 3 if media.references else 1
        if len(inputs) != expected_inputs:
            raise invalid_descriptor("video preparation conditioning products are incomplete")
        presentation_tags = reference_image = None
        if media.references:
            if len(media.references) != 1 or inputs[2] != media.references[0].pixels:
                raise invalid_descriptor("video preparation requires its declared image pixels")
            tags_read = device_products.consume(
                inputs[1],
                consumer_op_id=operation.op_id,
                device=model_runner.operation_device(operation),
            )
            scope.device_reads.append(tags_read)
            if tags_read.region is not None:
                raise invalid_descriptor("presentation tags require complete product coverage")
            presentation_tags = tags_read.tensor
            # Inline decoded pixels are request-owned host tensors, whereas a
            # transported pixel product is leased from the device-product owner.
            pixels = scope.input_tensors.get(inputs[2])
            if pixels is None:
                pixel_read = device_products.consume(
                    inputs[2],
                    consumer_op_id=operation.op_id,
                    device=model_runner.operation_device(operation),
                )
                scope.device_reads.append(pixel_read)
                if pixel_read.region is not None:
                    raise invalid_descriptor("reference pixels require complete product coverage")
                pixels = pixel_read.tensor
            if pixels is not None:
                if pixels.ndim != 4 or pixels.shape[0] != 1:
                    raise invalid_descriptor("reference pixels must contain one THWC image")
                reference_image = pixels[0]
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
            if media.references:
                model.prepare_tensors(
                    slot,
                    metadata,
                    encoded,
                    model.conditioning_rows(media_geometry),
                    presentation_tags=presentation_tags,
                    reference_image=reference_image,
                )
            else:
                model.prepare_tensors(slot, metadata, encoded, len(media.prompt_token_ids))
    elif operation.kind is OpCode.DIFFUSION_STEP:
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
            geometry=model.execution_key(media_geometry),
        )
        scope.observations.append(result.observation)
        request.flow_step = start_step + step_count
        if any(output.kind is ProductKind.TENSOR for output in operation.outputs):
            if request.flow_step != media.geometry.denoise_steps:
                raise invalid_descriptor("final latent products require completed denoising")
            products = transfer.publish_tensors(
                operation,
                result.values,
                scope,
                device_products=device_products,
                encoder_cache=encoder_cache,
                publication_transports=publication_transports,
            )
    elif operation.kind in {OpCode.DIFFUSION_DECODE, OpCode.MEDIA_APPEND}:
        params = decode_range(scope.lane, operation)
        inputs = tuple(
            product for product in operation.inputs if product.kind is ProductKind.TENSOR
        )
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
        cursor, count, track = params.cursor, params.max_units, params.track
        next_cursor = cursor + count
        if operation.kind is OpCode.DIFFUSION_DECODE:
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
                cursor != owner.media_video_units or cursor + count > media.geometry.video_units
            ):
                raise invalid_descriptor("video assembly requires the next temporal range")
            if track is MediaTrack.AUDIO and owner.media_audio_written:
                raise invalid_descriptor("audio output is already written")
            assert mux is not None
            if owner.media_video_units == 0 and not owner.media_audio_written:
                mux.open(operation.request_key, geometry=model.output_geometry(media.geometry))
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
            owner.media_video_units != media.geometry.video_units
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
        selected_point=1 if operation.advances_state else 0,
        logical_lengths=operation_geometry.logical_lengths(
            operation,
            request,
            None,
            latent_len=int(request.flow_step),
        ),
        token_span=TokenSpan(base=int(request.logical_position), len=0),
        finish_flags=FinishFlags(),
        product_generations=operation_geometry.output_generations(operation),
        completion_tasks=tasks,
        products=products,
        next_cursor=next_cursor,
        done=operation.kind is OpCode.DIFFUSION_FINALIZE,
    )
    state.phase = "done"
    return True


def _request_label(operation: Operation) -> str:
    """Format a stable request and operation label for media work."""

    key = operation.request_key
    return f"{key.authority_id}:{key.request_id}:{key.epoch}"


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
