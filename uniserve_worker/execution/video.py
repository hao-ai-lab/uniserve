"""Common bounded video decode, capture, and artifact actions."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import torch

from uniserve_worker.foundation.errors import (
    invalid_descriptor,
    unsupported_setup,
)
from uniserve_worker.media.buffers import MediaBuffers
from uniserve_worker.media.mux import MediaMux, require_media_codecs
from uniserve_worker.models.video import VideoModel
from uniserve_worker.profiling import profile_range
from uniserve_worker.protocol.batch import (
    ComputationId,
    DecodeRange,
    FinishFlags,
    MediaTrack,
    OpStatus,
    PipelineStage,
    ScheduleBatch,
    ScheduledRequest,
    TensorPublication,
)
from uniserve_worker.runtime.cpu import CpuTask

from . import operations as operation_geometry
from .batch_state import BatchState
from .diffusion_state import DiffusionState
from .output import PendingOutput

if TYPE_CHECKING:
    from collections.abc import Mapping

    from ..models.runtime import ExecutionModel
    from ..runtime.request import RequestPool
    from ..runtime.tensor_store import TensorStore
    from ..transfer.tickets import Transport
    from .model_runner import ModelRunner


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
) -> tuple[MediaMux | None, MediaBuffers | None]:
    """Provision public output resources from the declared media geometry."""

    if not model.owns_media_output or not owns_output:
        return None, None
    require_video_codecs()
    return (
        MediaMux(rank=rank),
        MediaBuffers(
            state_slots=state_slots,
            unresolved_window=unresolved_window,
            max_video_frames_per_round=model.decode_frame_capacity,
            max_geometry=model.output_capacity,
        ),
    )


def trajectory_params(operation: ScheduledRequest, *, state: BatchState):
    """Return the unique latent trajectory params assigned to an operation."""

    selected = tuple(
        params
        for params in state.batch.latent_params
        if params.request_key == operation.request_key and params.op_id == operation.op_id
    )
    if len(selected) != 1:
        raise invalid_descriptor("video trajectory operation has no exact latent params")
    return selected[0]


def decode_range(operation: ScheduledRequest, *, state: BatchState) -> DecodeRange:
    """Return the unique reconstruction params assigned to an operation."""

    selected = tuple(
        params
        for params in state.batch.decode_ranges
        if params.request_key == operation.request_key and params.op_id == operation.op_id
    )
    if len(selected) != 1:
        raise invalid_descriptor("video decode operation has no exact decode params")
    return selected[0]


def validate_batch(batch: ScheduleBatch, *, execution_model: ExecutionModel) -> None:
    """Validate video admission requirements before staging state."""

    if not isinstance(execution_model, VideoModel):
        return
    for operation in batch.operations:
        if operation.kind is not PipelineStage.LATENT_PREPARATION:
            continue
        if operation.predecessor != ComputationId(0, 0):
            raise invalid_descriptor("video preparation does not name its request root")


def execute(
    operation: ScheduledRequest,
    completion_group: int,
    *,
    state: BatchState,
    tensor_store: TensorStore,
    media_mux: MediaMux | None,
    execution_model: ExecutionModel,
    publication_transports: Mapping[str, Transport],
    request_pool: RequestPool,
    model_runner: ModelRunner,
) -> PendingOutput:
    """Land one ready video action without constructing a packed forward row."""

    if not isinstance(execution_model, VideoModel):
        raise invalid_descriptor("video execution requires a video model")
    model = execution_model
    request = state.pending_output(completion_group, operation.request_key.request_id)
    media = request.request.admission.diffusion
    if media is None:
        raise invalid_descriptor("video operation has no admitted media geometry")
    scratch = model_runner.scratch
    if scratch is None:
        raise RuntimeError("video execution has no allocated scratch storage")
    media_geometry = media
    prompt_token_ids = request.request.admission.prompt_token_ids
    num_prompt_tokens = len(prompt_token_ids)
    diffusion = model_runner.diffusion
    if diffusion is None:
        raise RuntimeError("video execution has no diffusion owner")
    trajectory = request.request.diffusion
    if trajectory is None:
        geometry = model.execution_key(media_geometry, num_prompt_tokens)
        metadata = diffusion.prepare_geometry(
            geometry,
            lambda: model.build_execution(
                media_geometry, num_prompt_tokens, scratch, model_runner.context_workspace
            ),
        )
        trajectory = DiffusionState(
            geometry=geometry,
            schedule=model_runner.schedule,
            tensors=model.request_tensors(
                request_pool.tensors(request.request.request_pool_idx), metadata
            ),
            metadata=metadata,
        )
        request.request.diffusion = trajectory
    slot, metadata = trajectory.tensors, trajectory.metadata
    mux = media_mux
    if (
        operation.kind
        in {PipelineStage.VIDEO_ENCODING, PipelineStage.AUDIO_ENCODING, PipelineStage.MUXING}
        and mux is None
    ):
        raise unsupported_setup("output owner has no video mux resources")
    from . import transfer

    tasks: tuple[CpuTask, ...] = ()
    products: tuple[TensorPublication, ...] = ()
    if operation.kind is PipelineStage.LATENT_PREPARATION:
        params = trajectory_params(operation, state=state)
        if int(params.start_step) != 0 or int(params.step_count) != 0:
            raise invalid_descriptor("video preparation params must carry zero denoise steps")
        inputs = operation.inputs
        if len(inputs) != 1:
            raise invalid_descriptor("video preparation requires one conditioning Tensor")
        conditioning = tensor_store.consume(
            inputs[0],
            consumer_op_id=operation.op_id,
            device=model_runner.operation_devices(operation)[0],
        )
        request.device_reads.append(conditioning)
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
                if result.stats is None:
                    raise RuntimeError("module output has no execution statistics")
                state.group_forward_stats[completion_group].append(result.stats)
                if len(result.values) != 1:
                    raise invalid_descriptor("conditioning computation must return one Tensor")
                encoded = result.values[0]
            model.prepare_tensors(slot, metadata, encoded, num_prompt_tokens)
    elif operation.kind is PipelineStage.DENOISING:
        params = trajectory_params(operation, state=state)
        start_step, step_count = int(params.start_step), int(params.step_count)
        if start_step != operation_geometry.require_progress(request).flow_step:
            raise invalid_descriptor("denoising does not begin at the selected request step")
        schedule = trajectory.schedule
        if schedule is None:
            raise RuntimeError("video execution lost its bound diffusion schedule")
        result = model_runner.run_denoising(
            slot,
            metadata,
            start_step,
            step_count,
            schedule,
            slot=request.request.request_pool_idx,
            geometry=trajectory.geometry,
        )
        if result.stats is None:
            raise RuntimeError("module output has no execution statistics")
        state.group_forward_stats[completion_group].append(result.stats)
        request.projected_progress = replace(
            operation_geometry.require_progress(request), flow_step=start_step + step_count
        )
        if operation.outputs:
            if operation_geometry.require_progress(request).flow_step != media.num_inference_steps:
                raise invalid_descriptor("final latent products require completed denoising")
            products = transfer.publish_tensors(
                operation,
                result.values,
                completion_group,
                tensor_store=tensor_store,
                publication_transports=publication_transports,
                state=state,
            )
    elif operation.kind in {
        PipelineStage.VIDEO_DECODING,
        PipelineStage.AUDIO_DECODING,
        PipelineStage.VIDEO_ENCODING,
        PipelineStage.AUDIO_ENCODING,
    }:
        params = decode_range(operation, state=state)
        inputs = operation.inputs
        if len(inputs) != 1:
            raise invalid_descriptor("media reconstruction requires one Tensor input")
        read = tensor_store.consume(
            inputs[0],
            consumer_op_id=operation.op_id,
            device=model_runner.operation_devices(operation)[0],
        )
        request.device_reads.append(read)
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
            if result.stats is None:
                raise RuntimeError("module output has no execution statistics")
            state.group_forward_stats[completion_group].append(result.stats)
            if len(result.values) != 1:
                raise invalid_descriptor("media decoder must return one numerical tensor")
            decoded = model.decoder_output(metadata, result.values[0], track)
            products = transfer.publish_tensors(
                operation,
                (decoded,),
                completion_group,
                tensor_store=tensor_store,
                publication_transports=publication_transports,
                state=state,
            )
        else:
            assert mux is not None
            mux.open(operation.request_key, geometry=model.output_geometry(media))
            mux.validate_track(operation.request_key, track, cursor, count)
            reservation = request.completion_tasks[0] if request.completion_tasks else None
            ring_lease = request.media_lease
            if reservation is None or ring_lease is None:
                raise RuntimeError("media output has no reserved capture storage")
            if track is MediaTrack.VIDEO:
                value = model.assemble_video(slot, metadata, read.tensor, cursor, count)
            else:
                value = read.tensor.view(torch.uint8)
            with profile_range(
                f"uniserve.video.decode_copy request={_request_label(operation)} op={operation.op_id} kind={track.value}"
            ):
                capture = state.group_buffers[completion_group].capture_bytes_into(
                    value, ring_lease.storage
                )
            try:
                if track is MediaTrack.VIDEO:
                    tasks = (
                        mux.video(
                            operation.request_key,
                            cursor,
                            count,
                            capture,
                            state.group_buffers[completion_group],
                            reservation,
                            ring_lease,
                            operation.op_id,
                        ),
                    )
                else:
                    tasks = (
                        mux.audio(
                            operation.request_key,
                            capture,
                            state.group_buffers[completion_group],
                            reservation,
                            ring_lease,
                            operation.op_id,
                        ),
                    )
            except BaseException:
                ring_lease.defer_until_ready(
                    state.group_buffers[completion_group].completion_future()
                )
                raise
    else:
        reservation = request.completion_tasks[0] if request.completion_tasks else None
        if reservation is None or mux is None:
            raise RuntimeError("media finalization has no reserved CPU task")
        tasks = (mux.finalize_artifact(operation.request_key, reservation, operation.op_id),)
    # Configured CpuTask now owns the media lease through its final CPU read.
    request.media_lease = None
    request.status = OpStatus.OK
    # Reconstruction and mux operations consume products without advancing the
    # diffusion trajectory. Keep their progress absent; denoising retains the
    # step already projected above through the same completion boundary.
    request.projected_progress = operation_geometry.execution_runtime(request, None)
    request.finish_flags = FinishFlags()
    request.product_generations = operation_geometry.output_generations(operation)
    request.completion_tasks = tasks
    request.products = products
    return request


def _request_label(operation: ScheduledRequest) -> str:
    """Format a stable request and operation label for media work."""

    key = operation.request_key
    return f"{key.engine_id}:{key.request_id}:{key.request_epoch}"


__all__ = [
    "decode_range",
    "execute",
    "require_video_codecs",
    "trajectory_params",
    "validate_batch",
]


def require_media_output_ring(ring: MediaBuffers | None) -> MediaBuffers:
    """Require bounded media storage on the configured output rank."""

    if ring is None:
        raise unsupported_setup("operation requires the media output owner's ring")
    return ring
