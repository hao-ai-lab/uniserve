"""Bounded execution of numerical video capabilities and media artifact actions."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import torch

from uniserve.media import video
from uniserve.model import AudioDecoder, VideoDecoder, VideoPostprocessor
from uniserve.profiling import profile_range
from uniserve.tensors import TensorOutput
from uniserve_worker.foundation.errors import invalid_descriptor, unsupported_setup
from uniserve_worker.media.buffers import MediaBuffers
from uniserve_worker.media.mux import MediaMux, require_media_codecs
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

from . import operations
from .batch_state import BatchState
from .diffusion_state import VideoState
from .output import PendingOutput
from .tensors import concatenate_views

if TYPE_CHECKING:
    from collections.abc import Mapping

    from uniserve.runtime.tensor_buffers import TensorBuffers

    from ..runtime.request import RequestPool
    from ..runtime.tensor_store import TensorStore
    from ..transfer.tickets import Transport
    from .model_runner import ModelRunner


def require_video_codecs() -> None:
    """Verify the output owner's configured video and audio encoders."""
    require_media_codecs("libx264", "aac")


def video_shape(runner: ModelRunner, media: DiffusionSamplingParams, tokens: int):
    """Resolve admission into exact numerical dimensions, without prompt padding."""
    factory = runner.media
    decoder = runner.video_decoder
    if factory is None or decoder is None:
        raise invalid_descriptor("video input requires denoising and reconstruction capabilities")
    try:
        size = factory.size(media.num_frames, tokens)
        windows = decoder.frame_slices(size.num_frames)
    except ValueError as error:
        raise invalid_descriptor(str(error)) from error
    if media.num_decode_chunks != len(windows) or media.num_inference_steps != factory.num_steps:
        raise invalid_descriptor("video request has invalid computation bounds")
    return size


def audio_samples(runner: ModelRunner, num_frames: int) -> int:
    decoder = runner.audio_decoder
    output = runner.video_postprocessor
    return round(num_frames * decoder.sample_rate / output.frame_rate)


def prepare_call(runner: ModelRunner, trajectory: VideoState, operation, storage):
    """Bind request state and prepare the exact call's runtime-owned resources.

    Only request state survives in the trajectory. Contexts own their constants
    and workspace and may be retired after their dependent graphs drain.
    """
    kind, size = operation.kind, trajectory.size
    if kind in {PipelineStage.LATENT_PREPARATION, PipelineStage.DENOISING}:
        if runner.diffusion is None:
            raise RuntimeError("rank does not own denoising execution")
        if "denoising" not in trajectory.tensors:
            if storage is None:
                raise RuntimeError("denoising requires reserved request storage")
            trajectory.tensors["denoising"] = storage.view(runner.media.buffers(size))
        context = runner.diffusion.prepare_inputs(size, size)
        return trajectory.tensors["denoising"], context
    if kind is PipelineStage.VIDEO_DECODING:
        return {}, runner.prepare_module(operation.entry, size.num_frames, method="decode")
    if kind is PipelineStage.AUDIO_DECODING:
        decoder = runner.component(kind)
        frames = decoder.latent_frames(audio_samples(runner, size.num_frames))
        return {}, runner.prepare_module(operation.entry, frames, method="decode")
    if kind is PipelineStage.VIDEO_ENCODING:
        component = runner.component(kind)
        if "video_overlap" not in trajectory.tensors:
            if storage is None:
                raise RuntimeError("video assembly requires reserved overlap storage")
            trajectory.tensors["video_overlap"] = storage.view(
                component.state_buffers(size.num_frames)
            )
        return trajectory.tensors["video_overlap"], runner.prepare_module(
            operation.entry, size.num_frames, method="forward"
        )
    return {}, None


@torch.inference_mode()
def warmup_denoising(runner: ModelRunner, storage: tuple[TensorBuffers, ...]) -> None:
    """Compile representative native tile boundaries without advancing samples."""
    factory, diffusion = runner.media, runner.diffusion
    if diffusion is None:
        return
    if factory is None or not storage:
        raise RuntimeError("denoising warmup requires its input factory and request storage")
    maximum = factory.maximum
    decoder = runner.video_decoder
    final_window = decoder.frame_slices(maximum.num_frames)[-1]
    minimum = final_window.stop - final_window.start
    shapes = (
        (maximum.num_frames, 65),
        (minimum, 1),
        *((maximum.num_frames, tokens) for tokens in (129, 193, 257)),
        (maximum.num_frames, maximum.num_text_tokens),
    )
    schedules, prepared = factory.schedules(device=runner.worker_config.device), set()
    for frames, tokens in shapes:
        if tokens > maximum.num_text_tokens:
            continue
        size = factory.size(frames, tokens)
        if size in prepared:
            continue
        prepared.add(size)
        context = diffusion.prepare_inputs(size, size)
        views = storage[0].view(factory.buffers(size))
        # Warmup has no prompt and uses zero numerical samples. The first real
        # request draws and stages its own seeded native noise before execution.
        with context.activate():
            views["text_condition"].zero_()
            for name in factory.denoiser.modalities:
                views[name].zero_()
        diffusion.warmup(
            factory.bind(size, views, schedules, 0), schedules, state=views, input_key=size
        )


@torch.inference_mode()
def warmup_decoders(runner: ModelRunner) -> None:
    """Prepare reconstruction kernels with the admitted maximum latent extent."""
    size = runner.media.maximum
    for (name, _, method), (binding, call) in runner._module_entries.items():
        if method != "decode":
            continue
        module = call.module
        if isinstance(module, VideoDecoder):
            shape = runner.media.denoiser.latent_shape("video", size)
            latent = torch.zeros(shape, dtype=torch.float32, device=binding.device)
            runner.run_module(
                name,
                (latent,),
                method="decode",
                size=size.num_frames,
                frames=(module.frame_slices(size.num_frames)[0],),
                num_frames=(size.num_frames,),
            )
        elif isinstance(module, AudioDecoder):
            shape = runner.media.denoiser.latent_shape("audio", size)
            latent = torch.zeros(shape, dtype=torch.float32, device=binding.device)
            samples = audio_samples(runner, size.num_frames)
            runner.run_module(
                name,
                (latent,),
                method="decode",
                size=module.latent_frames(samples),
                num_samples=(samples,),
            )


@torch.inference_mode()
def warmup_postprocess(runner: ModelRunner, storage: tuple[TensorBuffers, ...]) -> None:
    """Exercise one real output window using its public numerical interface."""
    entries = [
        (name, call)
        for (name, _, _), (_, call) in runner._module_entries.items()
        if isinstance(call.module, VideoPostprocessor)
    ]
    if not entries:
        return
    if not storage:
        raise RuntimeError("video output warmup requires request storage")
    name, call = entries[0]
    frames = runner.media.maximum.num_frames
    decoder = runner.video_decoder
    window = decoder.frame_slices(frames)[0]
    layout = decoder.output_layout(frames)["video"]
    segment = torch.zeros(
        (1, *layout.shape[1:]), dtype=layout.dtype, device=runner.bindings[name].device
    )
    state = storage[0].view(call.module.state_buffers(frames))
    runner.run_module(
        name,
        (
            TensorOutput(
                segment, replace(layout, local_slice=(slice(0, 1), *layout.local_slice[1:]))
            ),
        ),
        method="forward",
        size=frames,
        frames=(window,),
        num_frames=(frames,),
        state=state,
    )


def create_media_resources(
    runner: ModelRunner, *, rank: int, owns_output: bool, state_slots: int, unresolved_window: int
):
    """Reserve output captures from duration, raster and explicit sampling clocks."""
    if not owns_output:
        return None, None
    require_video_codecs()
    decoder = runner.video_decoder
    output = runner.video_postprocessor
    audio = runner.audio_decoder
    frames = runner.media.maximum.num_frames
    return MediaMux(rank=rank), MediaBuffers(
        state_slots=state_slots,
        unresolved_window=unresolved_window,
        max_video_frames_per_round=frames,
        video=video.Config(frames, decoder.frame_size),
        frame_rate=output.frame_rate,
        audio_rate=audio.sample_rate,
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


def validate_batch(
    batch: ScheduleBatch, *, postprocessor: VideoPostprocessor | None
) -> None:
    """Validate video admission requirements before staging state."""

    if postprocessor is None:
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
    publication_transports: Mapping[str, Transport],
    request_pool: RequestPool,
    model_runner: ModelRunner,
) -> PendingOutput:
    """Land one ready video action without constructing a packed forward row."""

    if model_runner.video_postprocessor is None:
        raise invalid_descriptor("video execution requires a video model")
    request = state.pending_output(completion_group, operation.request_key.request_id)
    media = request.request.admission.diffusion
    if media is None:
        raise invalid_descriptor("video operation has no admitted media geometry")
    numerical_shape = video_shape(
        model_runner, media, len(request.request.admission.prompt_token_ids)
    )
    trajectory = request.request.diffusion
    if trajectory is None:
        trajectory = VideoState(
            size=numerical_shape,
            schedules=dict(model_runner.media.schedules(device=model_runner.worker_config.device)),
        )
        request.request.diffusion = trajectory
    if not isinstance(trajectory, VideoState) or trajectory.size != numerical_shape:
        raise invalid_descriptor("video request changed its admitted numerical dimensions")
    slot, context = prepare_call(
        model_runner,
        trajectory,
        operation,
        request_pool.tensors(request.request.request_pool_idx)
        if model_runner.state_buffers
        else None,
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
        initial = model_runner.media.initialize(
            numerical_shape,
            slot,
            seed=media.seed,
            constants=context.constants,
            workspace=context.workspace,
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
                slot["text_condition"].copy_(
                    encoded.reshape_as(slot["text_condition"]), non_blocking=True
                )
    elif operation.kind is PipelineStage.DENOISING:
        params = trajectory_params(operation, state=state)
        start_step, step_count = int(params.start_step), int(params.step_count)
        if start_step != operations.require_progress(request).flow_step:
            raise invalid_descriptor("denoising does not begin at the selected request step")
        if step_count != 1:
            raise invalid_descriptor("video denoising requires one selected numerical step")
        result = model_runner.run_denoising(
            model_runner.media.bind(numerical_shape, slot, trajectory.schedules, start_step),
            trajectory.schedules,
            state=slot,
            slot=request.request.request_pool_idx,
            input_key=numerical_shape,
        )
        if result.stats is None:
            raise RuntimeError("module output has no execution statistics")
        state.group_forward_stats[completion_group].append(result.stats)
        request.projected_progress = replace(
            operations.require_progress(request), flow_step=start_step + step_count
        )
        if operation.outputs:
            if operations.require_progress(request).flow_step != media.num_inference_steps:
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
                windows = model_runner.video_decoder.frame_slices(media.num_frames)
                binding = model_runner.bindings[operation.entry]
                position = binding.config.ranks.index(binding.process_group.global_rank)
                if position >= count or cursor < 0 or cursor + count > len(windows):
                    raise invalid_descriptor("video decoder assignment exceeds its window range")
                window = windows[cursor + position]
            elif cursor != 0 or count != 1:
                raise invalid_descriptor("audio decoding requires one complete stereo latent")
            if read.tensor is None:
                raise invalid_descriptor("media decoder has no complete numerical input")
            if track is MediaTrack.VIDEO:
                result = model_runner.run_module(
                    operation.entry,
                    (read.tensor,),
                    method="decode",
                    size=media.num_frames,
                    frames=(window,),
                    num_frames=(media.num_frames,),
                )
            else:
                samples = audio_samples(model_runner, media.num_frames)
                decoder = model_runner.component(operation.kind)
                result = model_runner.run_module(
                    operation.entry,
                    (read.tensor,),
                    method="decode",
                    size=decoder.latent_frames(samples),
                    num_samples=(samples,),
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
            decoder = model_runner.video_decoder
            windows = decoder.frame_slices(media.num_frames)
            mux.open(
                operation.request_key,
                video=video.Config(media.num_frames, decoder.frame_size),
                frame_rate=model_runner.video_postprocessor.frame_rate,
                audio_rate=model_runner.audio_decoder.sample_rate,
                video_unit_frames=tuple(window.stop - window.start for window in windows),
            )
            mux.validate_track(operation.request_key, track, cursor, count)
            reservation = request.completion_tasks[0] if request.completion_tasks else None
            ring_lease = request.media_lease
            if reservation is None or ring_lease is None:
                raise RuntimeError("media output has no reserved capture storage")
            if track is MediaTrack.VIDEO:
                windows = windows[cursor : cursor + count]
                if len(windows) != count or read.tensor.shape[0] != count:
                    raise invalid_descriptor("video assembly exceeds its logical window range")
                layout = decoder.output_layout(media.num_frames)["video"]
                segments = tuple(
                    TensorOutput(
                        value.unsqueeze(0),
                        replace(
                            layout,
                            local_slice=(
                                slice(cursor + index, cursor + index + 1),
                                *layout.local_slice[1:],
                            ),
                        ),
                    )
                    for index, value in enumerate(read.tensor.unbind(0))
                )
                processed = model_runner.run_module(
                    operation.entry,
                    segments,
                    method="forward",
                    size=media.num_frames,
                    frames=windows,
                    num_frames=(media.num_frames,) * count,
                    state=slot,
                )
                state.group_forward_stats[completion_group].append(processed.stats)
                # Concatenation borrows one flat byte span. Restore the RGB
                # raster axes before handing its pinned capture to the encoder.
                value = concatenate_views(processed.values).view(
                    -1, decoder.frame_size.height, decoder.frame_size.width, 3
                )
            else:
                value = read.tensor.view(torch.uint8)
            with profile_range(
                f"uniserve.video.decode_copy request={_request_label(operation)} "
                f"step={operation.op_id.batch_id} op={operation.op_id.request_index} kind={track.value}"
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
    request.projected_progress = operations.execution_runtime(request, None)
    request.finish_flags = FinishFlags()
    request.product_generations = operations.output_generations(operation)
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
