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
from uniserve_worker.modeling.batch import DecodeBatch, DiffusionBatch
from uniserve_worker.modeling.components import Call
from uniserve_worker.modeling.decoder import DecodeKind
from uniserve_worker.modeling.diffusion import DiffusionMixin
from uniserve_worker.modeling.geometry import MediaShape
from uniserve_worker.modeling.tensors import TensorViews
from uniserve_worker.modeling.video import VideoMixin
from uniserve_worker.nn.rng import diffusion_noise
from uniserve_worker.profiling import profile_range
from uniserve_worker.protocol.batch import (
    ComputationId,
    DecodeRange,
    DiffusionSamplingParams,
    FinishFlags,
    MediaTrack,
    OpStatus,
    PipelineStage,
    ScheduleBatch,
    ScheduledRequest,
    TensorPublication,
)
from uniserve_worker.runtime.cpu import CpuTask
from uniserve_worker.runtime.tensors import (
    bind_scratch,
    bind_state,
    prepare_constants,
    stage_tensor,
)

from . import operations as operation_geometry
from .batch_state import BatchState
from .denoising import denoising_batch
from .diffusion_state import DiffusionState
from .output import PendingOutput

if TYPE_CHECKING:
    from collections.abc import Mapping

    from ..modeling.model import Model
    from ..runtime.request import RequestPool
    from ..runtime.tensor_buffers import TensorBuffers
    from ..runtime.tensor_store import TensorStore
    from ..transfer.tickets import Transport
    from .model_runner import ModelRunner


def require_video_codecs() -> None:
    """Verify that the model’s configured video and audio codec dependencies are installed."""

    require_media_codecs("libx264", "aac")


def warmup_denoising(model: Model, runner: ModelRunner, storage: tuple[TensorBuffers, ...]) -> None:
    """Prepare representative legal media geometries using the runtime cache."""

    if runner.diffusion is None or runner.diffusion.model is not model:
        return
    if not isinstance(model, VideoMixin) or not isinstance(model, DiffusionMixin):
        raise TypeError("video denoising requires its numerical capabilities")
    scratch, diffusion, schedule = runner.scratch, runner.diffusion, runner.schedule
    if scratch is None or diffusion is None or schedule is None or not storage:
        raise RuntimeError("denoiser warmup requires its bound state, scratch, and schedule")
    maximum = model.output_capacity.frame_count
    alignment = model.text_alignment
    shapes = (
        (maximum, alignment + 1),
        (model.min_frames, 1),
        *((maximum, pages * alignment + 1) for pages in (2, 3, 4)),
        (maximum, model.text_max_tokens),
    )
    prepared = set()
    for frames, tokens in shapes:
        if tokens > model.text_max_tokens:
            continue
        shape = model.output_geometry(frames)
        geometry = DiffusionSamplingParams(
            num_frames=frames,
            num_decode_chunks=len(shape.unit_frames),
            num_inference_steps=model.num_inference_steps,
            seed=0,
        )
        numerical_shape = video_shape(model, geometry, tokens)
        if numerical_shape in prepared:
            continue
        prepared.add(numerical_shape)
        page_shape = metadata_shape(model, numerical_shape)
        constants, views_scratch = diffusion.prepare_geometry(
            page_shape,
            lambda: (
                prepare_constants(
                    model, Call.DIFFUSION, page_shape, device=runner.worker_config.device
                ),
                bind_scratch(
                    model,
                    Call.DIFFUSION,
                    numerical_shape,
                    scratch,
                ),
            ),
        )
        # Warmup executes one numerical request. Other slots are initialized
        # with their own prompt state when a request is actually admitted.
        views = bind_state(model, Call.DIFFUSION, numerical_shape, storage[0])
        spec = model.diffusion_spec(numerical_shape, model.num_inference_steps)
        initialize_latents(model, numerical_shape, views, constants, views_scratch, 0)
        views["text_condition"].zero_()
        for modality in spec.modalities:
            views[modality.name].zero_()
        diffusion.warmup(
            denoising_batch(model, numerical_shape, views, schedule, 0),
            schedule,
            state=views,
            constants=constants,
            scratch=views_scratch,
            geometry=page_shape,
        )


def initialize_latents(
    model: Model,
    shape: MediaShape,
    tensors: TensorViews,
    constants: TensorViews,
    scratch: TensorViews,
    seed: int,
) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    """Generate complete native modalities, then stage their numerical shards.

    Request storage owns normal-draw and shard-staging buffers. Their leading
    batch views represent this one logical request; preparation itself receives
    only numerical tensors and cached mathematical indices.
    """

    if not isinstance(model, DiffusionMixin):
        raise TypeError("latent initialization requires diffusion capability")
    spec = model.diffusion_spec(shape, model.num_inference_steps)
    noise = {item.name: tensors[f"{item.name}_noise"].unsqueeze(0) for item in spec.modalities}
    sources = {item.name: tensors[f"{item.name}_source"].unsqueeze(0) for item in spec.modalities}
    state = {**{name: value.unsqueeze(0) for name, value in tensors.items()}, **sources}
    diffusion_noise(
        spec,
        seeds=(seed,),
        device=next(iter(noise.values())).device,
        dtype=torch.float32,
        out=noise,
    )
    model.prepare_latents(
        DiffusionBatch({name: (value[0],) for name, value in sources.items()}, (shape,)),
        noise=noise,
        state=state,
        constants=constants,
        scratch=scratch,
    )
    return tuple(
        (tensors[f"{item.name}"], tensors[f"{item.name}_source"]) for item in spec.modalities
    )


def video_shape(model: Model, media: DiffusionSamplingParams, tokens: int) -> MediaShape:
    """Resolve admitted protocol bounds into one exact numerical media shape."""

    if not isinstance(model, VideoMixin):
        raise TypeError("video geometry requires video capability")
    output = model.output_geometry(media.num_frames)
    if (
        media.num_decode_chunks != len(output.unit_frames)
        or media.num_inference_steps != model.num_inference_steps
        or not 0 < tokens <= model.text_max_tokens
    ):
        raise invalid_descriptor("video request has invalid computation bounds")
    return MediaShape(output.height, output.width, frames=media.num_frames, prompt_tokens=tokens)


def metadata_shape(model: VideoMixin, shape: MediaShape) -> MediaShape:
    """Cache page geometry independently of request-specific prompt validity."""

    alignment = model.text_alignment
    tokens = ((shape.prompt_tokens + alignment - 1) // alignment) * alignment
    return replace(shape, prompt_tokens=tokens)


def prepare_call(
    model: Model,
    runner: ModelRunner,
    trajectory: DiffusionState,
    call: Call,
    shape: MediaShape,
    storage: TensorBuffers | None,
) -> tuple[TensorViews, TensorViews, TensorViews]:
    """Retain only the current call's declared numerical resources for a request.

    Denoising shares immutable geometry through the runner's bounded cache.
    Reconstruction constants belong to this request through its final window;
    backing allocations and graph retirement stay outside model preparation.
    """

    if call not in trajectory.constants:
        backing = runner.scratch
        if backing is None:
            raise RuntimeError("media computation requires allocated scratch")
        trajectory.tensors[call] = bind_state(model, call, shape, storage)
        if call is Call.DIFFUSION:
            diffusion = runner.diffusion
            if diffusion is None:
                raise RuntimeError("denoising requires its execution owner")
            if not isinstance(model, VideoMixin):
                raise TypeError("media diffusion requires video geometry")
            page_shape = metadata_shape(model, shape)
            trajectory.geometry = page_shape
            constants, scratch = diffusion.prepare_geometry(
                page_shape,
                lambda: (
                    prepare_constants(model, call, page_shape, device=runner.worker_config.device),
                    bind_scratch(model, call, shape, backing),
                ),
            )
        else:
            constants = prepare_constants(model, call, shape, device=runner.worker_config.device)
            scratch = bind_scratch(model, call, shape, backing)
        trajectory.constants[call] = constants
        trajectory.scratch[call] = scratch
    return trajectory.tensors[call], trajectory.constants[call], trajectory.scratch[call]


def warmup_postprocess(
    model: Model, runner: ModelRunner, storage: tuple[TensorBuffers, ...]
) -> None:
    """Exercise one real output window using runtime-owned numerical storage."""

    if not isinstance(model, VideoMixin):
        raise TypeError("video postprocessing requires video capability")
    output = runner.bindings.get("output")
    if output is None or not output.owns:
        return
    if not storage or runner.scratch is None:
        raise RuntimeError("video warmup requires bound state and scratch")
    shape = model.output_geometry(model.output_capacity.frame_count)
    numerical_shape = MediaShape(shape.height, shape.width, frames=shape.frame_count)
    constants = prepare_constants(
        model, Call.POSTPROCESS_VIDEO, numerical_shape, device=runner.worker_config.device
    )
    state = bind_state(model, Call.POSTPROCESS_VIDEO, numerical_shape, storage[0])
    views = bind_scratch(model, Call.POSTPROCESS_VIDEO, numerical_shape, runner.scratch)
    window = model.decode_windows(shape)[0]
    segment = torch.zeros(
        (1, 3, window.segment_frames, shape.height, shape.width),
        dtype=state["video_overlap"].dtype,
        device=state["video_overlap"].device,
    )
    result = model.postprocess_video(
        (segment,), (window,), state=state, constants=constants, scratch=views
    )
    result.validate(
        model.tensor_specs(Call.POSTPROCESS_VIDEO, numerical_shape), state=state, scratch=views
    )


def create_media_resources(
    model: VideoMixin,
    *,
    rank: int,
    owns_output: bool,
    state_slots: int,
    unresolved_window: int,
) -> tuple[MediaMux | None, MediaBuffers | None]:
    """Provision public output resources from the declared media geometry."""

    if not owns_output:
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


def validate_batch(batch: ScheduleBatch, *, execution_model: Model) -> None:
    """Validate video admission requirements before staging state."""

    if not isinstance(execution_model, VideoMixin):
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
    execution_model: Model,
    publication_transports: Mapping[str, Transport],
    request_pool: RequestPool,
    model_runner: ModelRunner,
) -> PendingOutput:
    """Land one ready video action without constructing a packed forward row."""

    if not isinstance(execution_model, VideoMixin):
        raise invalid_descriptor("video execution requires a video model")
    model = execution_model
    request = state.pending_output(completion_group, operation.request_key.request_id)
    media = request.request.admission.diffusion
    if media is None:
        raise invalid_descriptor("video operation has no admitted media geometry")
    scratch = model_runner.scratch
    if scratch is None:
        raise RuntimeError("video execution has no allocated scratch storage")
    numerical_shape = video_shape(model, media, len(request.request.admission.prompt_token_ids))
    trajectory = request.request.diffusion
    if trajectory is None:
        trajectory = DiffusionState(geometry=numerical_shape, schedule=model_runner.schedule)
        request.request.diffusion = trajectory
    calls = {
        PipelineStage.LATENT_PREPARATION: Call.DIFFUSION,
        PipelineStage.DENOISING: Call.DIFFUSION,
        PipelineStage.VIDEO_DECODING: Call.DECODE_VIDEO,
        PipelineStage.AUDIO_DECODING: Call.DECODE_AUDIO,
        PipelineStage.VIDEO_ENCODING: Call.POSTPROCESS_VIDEO,
    }
    call = calls.get(operation.kind) if isinstance(operation.kind, PipelineStage) else None
    slot, constants, call_scratch = (
        ({}, {}, {})
        if call is None
        else prepare_call(
            model,
            model_runner,
            trajectory,
            call,
            numerical_shape,
            (
                request_pool.tensors(request.request.request_pool_idx)
                if model_runner.tensor_resources.state
                else None
            ),
        )
    )
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
        if not isinstance(model, DiffusionMixin):
            raise invalid_descriptor("latent preparation requires diffusion capability")
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
        initial = (
            initialize_latents(
                model,
                numerical_shape,
                slot,
                constants,
                call_scratch,
                media.seed,
            )
            if model_runner.bindings[operation.entry].owns
            else ()
        )
        with model_runner.preparing_inputs(initial):
            if "conditioning" in model_runner.encoder_kinds:
                if encoded is None:
                    raise invalid_descriptor("conditioning module has no input Tensor")
                result = model_runner.run_encoder(
                    "conditioning",
                    encoded,
                )
                if result.stats is None:
                    raise RuntimeError("module output has no execution statistics")
                state.group_forward_stats[completion_group].append(result.stats)
                if len(result.values) != 1:
                    raise invalid_descriptor("conditioning computation must return one Tensor")
                encoded = result.values[0]
                stage_tensor(encoded, slot["text_condition"])
    elif operation.kind is PipelineStage.DENOISING:
        if not isinstance(model, DiffusionMixin):
            raise invalid_descriptor("denoising requires diffusion capability")
        params = trajectory_params(operation, state=state)
        start_step, step_count = int(params.start_step), int(params.step_count)
        if start_step != operation_geometry.require_progress(request).flow_step:
            raise invalid_descriptor("denoising does not begin at the selected request step")
        schedule = trajectory.schedule
        if schedule is None:
            raise RuntimeError("video execution lost its bound diffusion schedule")
        if not call_scratch:
            raise RuntimeError("denoising requires bound numerical scratch")
        result = model_runner.run_denoising(
            denoising_batch(
                model,
                numerical_shape,
                slot,
                schedule,
                start_step,
            ),
            step_count,
            schedule,
            state=slot,
            constants=constants,
            scratch=call_scratch,
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
            window = None
            if track is MediaTrack.VIDEO:
                windows = model.decode_windows(model.output_geometry(media.num_frames))
                binding = model_runner.bindings[operation.entry]
                position = binding.config.ranks.index(binding.process_group.rank)
                if position >= count or cursor < 0 or cursor + count > len(windows):
                    raise invalid_descriptor("video decoder assignment exceeds its window range")
                window = windows[cursor + position]
            elif cursor != 0 or count != 1:
                raise invalid_descriptor("audio decoding requires one complete stereo latent")
            kind: DecodeKind = "video" if track is MediaTrack.VIDEO else "audio"
            decoder_shape = MediaShape(
                model.output_capacity.height,
                model.output_capacity.width,
                frames=media.num_frames,
            )
            result = model_runner.run_decoder(
                kind,
                DecodeBatch((read.tensor,), (decoder_shape,), () if window is None else (window,)),
                constants=constants,
                scratch=call_scratch,
            )
            if result.stats is None:
                raise RuntimeError("module output has no execution statistics")
            state.group_forward_stats[completion_group].append(result.stats)
            if len(result.values) != 1:
                raise invalid_descriptor("media decoder must return one numerical tensor")
            products = transfer.publish_tensors(
                operation,
                result.values,
                completion_group,
                tensor_store=tensor_store,
                publication_transports=publication_transports,
                state=state,
            )
        else:
            assert mux is not None
            shape = model.output_geometry(media.num_frames)
            mux.open(operation.request_key, geometry=shape)
            mux.validate_track(operation.request_key, track, cursor, count)
            reservation = request.completion_tasks[0] if request.completion_tasks else None
            ring_lease = request.media_lease
            if reservation is None or ring_lease is None:
                raise RuntimeError("media output has no reserved capture storage")
            if track is MediaTrack.VIDEO:
                windows = model.decode_windows(shape)[cursor : cursor + count]
                if len(windows) != count or read.tensor.shape[0] != count:
                    raise invalid_descriptor("video assembly exceeds its logical window range")
                processed = model.postprocess_video(
                    tuple(read.tensor.unbind(0)),
                    windows,
                    state=slot,
                    constants=constants,
                    scratch=call_scratch,
                )
                processed.validate(
                    model.tensor_specs(
                        Call.POSTPROCESS_VIDEO,
                        MediaShape(shape.height, shape.width, frames=shape.frame_count),
                    ),
                    state=slot,
                    scratch=call_scratch,
                )
                frames = processed.values["video"][0]
                if frames is None:
                    raise RuntimeError("video postprocessing returned no frames")
                value = frames
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
