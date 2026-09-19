"""Bounded execution of numerical video capabilities and media artifact.

actions.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import TYPE_CHECKING

import torch

from uniserve.model import AudioDecoder, VideoDecoder, VideoPostprocessor
from uniserve.tensors import TensorOutput, concatenate_views
from uniserve_worker.foundation.errors import (
    invalid_descriptor,
    unsupported_setup,
)
from uniserve_worker.media.mux import AvMuxConfig
from uniserve_worker.protocol.batch import (
    Batch,
    DecodeRange,
    DiffusionParams,
    MediaTrack,
    TensorPublication,
)
from uniserve_worker.protocol.call import (
    Call,
    CallStatus,
    MediaCall,
)
from uniserve_worker.protocol.identity import CallId
from uniserve_worker.protocol.output import FinishFlags

from . import calls
from .batch_state import BatchState
from .diffusion_state import VideoState
from .output import PendingOutput

if TYPE_CHECKING:
    from collections.abc import Mapping

    from uniserve.runtime.tensor_buffers import TensorBuffers

    from ..runtime.request import RequestPool
    from ..runtime.tensor_store import TensorStore
    from ..transfer.tickets import Transport
    from .model_runner import ModelRunner


def video_shape(runner: ModelRunner, media: DiffusionParams, tokens: int):
    """Resolve admission into exact numerical dimensions.

    without prompt padding.
    """
    builder = runner.media_builder
    decoder = runner.video_decoder
    if builder is None or decoder is None:
        raise invalid_descriptor(
            "video input requires denoising and reconstruction capabilities"
        )

    try:
        size = builder.size(media.num_frames, tokens)
        windows = decoder.frame_slices(size.num_frames)
    except ValueError as error:
        raise invalid_descriptor(str(error)) from error

    if (
        media.video_units != len(windows)
        or media.num_inference_steps != builder.num_steps
    ):
        raise invalid_descriptor("video request has invalid computation bounds")
    return size


def audio_samples(runner: ModelRunner, num_frames: int) -> int:
    """Return the audio sample count spanning the given video frame count."""
    decoder = runner.audio_decoder
    output = runner.video_postprocessor
    return round(num_frames * decoder.sample_rate / output.frame_rate)


def audio_unit_count(runner: ModelRunner, entry: str) -> int:
    """Return the media units the audio decoder's ranks reconstruct together.

    The division follows the component's placement, which the engine and every
    rank read from the same document, so both sides name the same units without
    carrying the count on the wire.
    """
    binding = runner.bindings[entry]
    if binding.config.distribution is None:
        return 1
    return len(binding.config.ranks) * max(1, binding.config.units_per_rank)


def audio_unit_windows(
    runner: ModelRunner, entry: str, num_samples: int
) -> tuple[slice, ...]:
    """Return every audio media unit's latent window for one request."""
    decoder = runner.audio_decoder
    units = audio_unit_count(runner, entry)
    if decoder.latent_frames(num_samples) < units:
        raise unsupported_setup(
            "audio decoder placement declares more media units than the "
            "request has latent frames"
        )
    return decoder.unit_frames(num_samples, units)


def prepare_call(runner: ModelRunner, trajectory: VideoState, call, storage):
    """Bind request state and prepare the exact call's runtime-owned resources.

    Only request state survives in the trajectory. Contexts own their constants
    and workspace and may be retired after their dependent graphs drain.
    """
    kind, size = call.kind, trajectory.size

    if kind in {MediaCall.LATENT_PREPARATION, MediaCall.DENOISING}:
        if runner.denoising is None:
            raise RuntimeError("rank does not own denoising execution")
        if "denoising" not in trajectory.tensors:
            if storage is None:
                raise RuntimeError(
                    "denoising requires reserved request storage"
                )
            trajectory.tensors["denoising"] = storage.view(
                runner.media_builder.buffers(size)
            )
        context = runner.denoising.prepare_inputs(size, size)
        return trajectory.tensors["denoising"], context

    if kind is MediaCall.VIDEO_DECODING:
        # A decode round also converts its media unit to RGB, cross-faded with
        # the neighbouring unit's tail held in the request's overlap state.
        postprocessor = runner.video_postprocessor
        if "video_overlap" not in trajectory.tensors:
            if storage is None:
                raise RuntimeError(
                    "video reconstruction requires reserved overlap storage"
                )
            trajectory.tensors["video_overlap"] = storage.view(
                postprocessor.state_buffers(size.num_frames)
            )
        runner.prepare_module(call.component, size.num_frames, method="forward")
        return trajectory.tensors["video_overlap"], runner.prepare_module(
            call.component, size.num_frames, method="decode"
        )

    if kind is MediaCall.AUDIO_DECODING:
        decoder = runner.component(kind)
        frames = decoder.latent_frames(audio_samples(runner, size.num_frames))
        return {}, runner.prepare_module(
            call.component, frames, method="decode"
        )

    return {}, None


@torch.inference_mode()
def warmup_denoising(
    runner: ModelRunner, storage: tuple[TensorBuffers, ...]
) -> None:
    """Compile representative native tile boundaries without advancing.

    samples.
    """
    builder, denoising = runner.media_builder, runner.denoising
    if denoising is None:
        return
    if builder is None or not storage:
        raise RuntimeError(
            "denoising warmup requires its input builder and request storage"
        )

    maximum = builder.maximum
    decoder = runner.video_decoder
    final_window = decoder.frame_slices(maximum.num_frames)[-1]
    minimum = final_window.stop - final_window.start
    # Representative (frames, text tokens) tile boundaries to compile.
    shapes = (
        (maximum.num_frames, 65),
        (minimum, 1),
        *((maximum.num_frames, tokens) for tokens in (129, 193, 257)),
        (maximum.num_frames, maximum.num_text_tokens),
    )

    schedules, prepared = (
        builder.schedules(device=runner.worker_config.device),
        set(),
    )
    for frames, tokens in shapes:
        if tokens > maximum.num_text_tokens:
            continue
        size = builder.size(frames, tokens)
        if size in prepared:
            continue
        prepared.add(size)
        context = denoising.prepare_inputs(size, size)
        views = storage[0].view(builder.buffers(size))
        # Warmup has no prompt and uses zero numerical samples. The first real
        # request draws and stages its own seeded native noise before execution.
        with context.activate():
            views["text_condition"].zero_()
            for name in builder.denoiser.modalities:
                views[name].zero_()
        denoising.warmup(
            builder.bind(size, views, schedules, 0),
            schedules,
            state=views,
            input_key=size,
        )


def declared_sizes(runner: ModelRunner) -> tuple:
    """Resolve the worker's declared video shapes into numerical sizes.

    A declared shape names the duration and prompt length of the requests a
    deployment serves. Admission converts a duration at the output sampling
    clock and rounds it up to the next complete native window, so a declared
    duration resolves to the frame count its requests carry.
    """
    builder = runner.media_builder
    if builder is None:
        return ()

    frame_rate = runner.video_postprocessor.frame_rate
    sizes = []
    for seconds, num_text_tokens in runner.worker_config.video_graph_shapes:
        frames = builder.denoiser.legal_frame_count(
            int(seconds * frame_rate + 0.5)
        )
        try:
            size = builder.size(frames, num_text_tokens)
        except ValueError as error:
            raise invalid_descriptor(
                f"declared video graph shape {seconds} s x "
                f"{num_text_tokens} tokens exceeds worker capacity: {error}"
            ) from error
        if size not in sizes:
            sizes.append(size)
    return tuple(sizes)


def decoded_units(runner: ModelRunner, name: str, count: int) -> tuple:
    """List the media unit indices this rank decodes for one unit count.

    The engine hands each decode round as many units as the component has
    ranks, in rank order, so a rank's share follows from its position.
    """
    binding = runner.bindings[name]
    ranks = binding.config.ranks
    position = ranks.index(binding.process_group.global_rank)
    units, cursor = [], 0
    while cursor < count:
        width = min(len(ranks), count - cursor)
        if position < width:
            units.append(cursor + position)
        cursor += width
    return tuple(units)


@torch.inference_mode()
def capture_denoising(
    runner: ModelRunner, storage: tuple[TensorBuffers, ...]
) -> None:
    """Make every declared video shape's denoising ladder resident.

    A denoising graph is captured per ladder step at each declared size and
    serves every request slot, since it reaches a slot's storage through the
    device slot index rather than through the slot's addresses. A request
    that first meets its shape would otherwise pay one capture per step on
    its own path. Capture is collective across the component's ranks and
    belongs here, where warmup holds them in lockstep. Samples are unchanged
    when this returns; a shape the deployment does not declare still serves
    and captures on first use.
    """
    builder, denoising = runner.media_builder, runner.denoising
    sizes = declared_sizes(runner)
    if denoising is None or not sizes or not denoising.captures:
        return
    if builder is None or not storage:
        raise RuntimeError(
            "denoising capture requires its input builder and request storage"
        )

    schedules = builder.schedules(device=runner.worker_config.device)
    for size in sizes:
        context = denoising.prepare_inputs(size, size)
        # Any slot's rows serve the capture; the first slot's do. The values
        # it reads are irrelevant, and the first real request stages its own
        # seeded noise and conditioning before replay.
        views = storage[0].view(builder.buffers(size))
        with context.activate():
            views["text_condition"].zero_()
            for name in builder.denoiser.modalities:
                views[name].zero_()

        for index in range(builder.num_steps):
            denoising.capture(
                builder.bind(size, views, schedules, index),
                schedules,
                state=views,
                slot=1,
                input_key=size,
            )


@torch.inference_mode()
def warmup_decoders(runner: ModelRunner) -> None:
    """Prepare reconstruction kernels for the admitted and declared extents.

    A decoder's prepared context and captured graph follow its frame count and,
    for video, the media unit it reconstructs, so a declared duration warms
    every unit this rank decodes at that duration. The admitted maximum stays
    covered so an undeclared duration still meets compiled kernels.
    """
    builder = runner.media_builder
    frames = tuple(
        dict.fromkeys(
            size.num_frames
            for size in (builder.maximum, *declared_sizes(runner))
        )
    )
    for (name, _, method), (binding, call) in runner._module_calls.items():
        if method != "decode":
            continue
        module = call.module
        for num_frames in frames:
            size = builder.size(num_frames, builder.maximum.num_text_tokens)
            if isinstance(module, VideoDecoder):
                shape = builder.denoiser.latent_shape("video", size)
                latent = torch.zeros(
                    shape, dtype=torch.float32, device=binding.device
                )
                windows = module.frame_slices(num_frames)
                for unit in decoded_units(runner, name, len(windows)):
                    runner.run_module(
                        name,
                        (latent,),
                        method="decode",
                        size=num_frames,
                        frames=(windows[unit],),
                        num_frames=(num_frames,),
                    )
            elif isinstance(module, AudioDecoder):
                shape = builder.denoiser.latent_shape("audio", size)
                latent = torch.zeros(
                    shape, dtype=torch.float32, device=binding.device
                )
                samples = audio_samples(runner, num_frames)
                windows = audio_unit_windows(runner, name, samples)
                for unit in decoded_units(runner, name, len(windows)):
                    runner.run_module(
                        name,
                        (latent,),
                        method="decode",
                        size=module.latent_frames(samples),
                        frames=(windows[unit],),
                        num_samples=(samples,),
                    )


@torch.inference_mode()
def warmup_conditioning(runner: ModelRunner) -> None:
    """Compile and capture the conditioning path at every declared length.

    The text encoder and the denoiser's conditioning encoder each prepare one
    context and capture one graph per exact token count. Running them in their
    serving order gives the conditioning encoder the same input the request
    path hands it, so a declared prompt length is warmed here rather than on
    the first request that carries it.
    """
    kinds = runner.encoder_kinds
    if "text" not in kinds:
        return
    for length in dict.fromkeys(
        size.num_text_tokens for size in declared_sizes(runner)
    ):
        encoded = runner.run_encoder(
            "text", runner.stage_text_tokens((0,) * length)
        )
        if "conditioning" in kinds:
            runner.run_encoder("conditioning", *encoded.values)


@torch.inference_mode()
def warmup_postprocess(
    runner: ModelRunner, storage: tuple[TensorBuffers, ...]
) -> None:
    """Exercise real output windows through the public numerical interface.

    One post-processing call converts the media unit this rank reconstructed, so
    its prepared context follows the frame count. Every rank of the ring
    prepares every declared duration even where it holds no unit in a round,
    because preparing a context binds the ring's communication resources and
    that binding spans the whole ring.
    """
    entries = [
        (name, call)
        for (name, _, _), (_, call) in runner._module_calls.items()
        if isinstance(call.module, VideoPostprocessor)
    ]
    if not entries:
        return
    if not storage:
        raise RuntimeError("video output warmup requires request storage")

    name, call = entries[0]
    builder, decoder = runner.media_builder, runner.video_decoder
    binding = runner.bindings[name]
    device = binding.device
    position = binding.config.ranks.index(binding.process_group.global_rank)
    units_per_round = len(binding.config.ranks)
    for frames in dict.fromkeys(
        size.num_frames for size in (builder.maximum, *declared_sizes(runner))
    ):
        runner.prepare_module(name, frames, method="forward")
        windows = decoder.frame_slices(frames)
        layout = decoder.output_layout(frames)["video"]
        state = storage[0].view(call.module.state_buffers(frames))
        cursor = 0
        while cursor < len(windows):
            count = min(units_per_round, len(windows) - cursor)
            if position < count:
                unit = cursor + position
                unit_outputs = (
                    TensorOutput(
                        torch.zeros(
                            (1, *layout.shape[1:]),
                            dtype=layout.dtype,
                            device=device,
                        ),
                        replace(
                            layout,
                            local_slice=(
                                slice(unit, unit + 1),
                                *layout.local_slice[1:],
                            ),
                        ),
                    ),
                )
                runner.run_module(
                    name,
                    unit_outputs,
                    method="forward",
                    size=frames,
                    frames=(windows[unit],),
                    num_frames=(frames,),
                    state=state,
                    unit_count=count,
                )
            cursor += count


def mux_config(runner: ModelRunner, media) -> AvMuxConfig:
    """Return the container settings one request's media units encode under."""
    decoder = runner.video_decoder
    windows = decoder.frame_slices(media.num_frames)
    return AvMuxConfig(
        width=decoder.frame_size.width,
        height=decoder.frame_size.height,
        frame_count=media.num_frames,
        frame_rate=runner.video_postprocessor.frame_rate,
        audio_rate=runner.audio_decoder.sample_rate,
        video_unit_frames=tuple(
            window.stop - window.start for window in windows
        ),
    )


def trajectory_params(call: Call, *, state: BatchState):
    """Return the unique latent trajectory params assigned to an call."""
    selected = tuple(
        params
        for params in state.batch.latent_params
        if params.request_key == call.request_key
        and params.call_id == call.call_id
    )
    if len(selected) != 1:
        raise invalid_descriptor(
            "video trajectory call has no exact latent params"
        )
    return selected[0]


def decode_range(call: Call, *, state: BatchState) -> DecodeRange:
    """Return the unique reconstruction params assigned to an call."""
    selected = tuple(
        params
        for params in state.batch.decode_ranges
        if params.request_key == call.request_key
        and params.call_id == call.call_id
    )
    if len(selected) != 1:
        raise invalid_descriptor("video decode call has no exact decode params")
    return selected[0]


def validate_batch(
    batch: Batch,
    *,
    postprocessor: VideoPostprocessor | None,
    predecessors: Mapping[CallId, CallId | None],
) -> None:
    """Validate video admission requirements before staging state.

    Latent preparation opens a request's trajectory, so it must be the first
    state-advancing call of its request: the call it follows is the admission
    root, not an earlier step.
    """
    if postprocessor is None:
        return
    for call in batch.calls:
        if call.kind is not MediaCall.LATENT_PREPARATION:
            continue
        if predecessors.get(call.call_id) != CallId(0, 0):
            raise invalid_descriptor(
                "video preparation does not follow its request root"
            )


def execute(
    call: Call,
    completion_group: int,
    *,
    state: BatchState,
    tensor_store: TensorStore,
    publication_transports: Mapping[str, Transport],
    request_pool: RequestPool,
    model_runner: ModelRunner,
) -> PendingOutput:
    """Land one ready video call without constructing a model input row.

    Latent preparation, denoising steps and decode rounds run here on the
    device; a video decode round also converts its media unit to RGB and
    publishes it as a host product for the host ranks that encode it.
    """
    if model_runner.video_postprocessor is None:
        raise invalid_descriptor("video execution requires a video model")
    request = state.pending_output(
        completion_group, call.request_key.request_id
    )
    media = request.request.admission.diffusion
    if media is None:
        raise invalid_descriptor("video call has no admitted media dimensions")
    numerical_shape = video_shape(
        model_runner, media, len(request.request.admission.prompt_token_ids)
    )

    trajectory = request.request.diffusion
    if trajectory is None:
        trajectory = VideoState(
            size=numerical_shape,
            schedules=dict(
                model_runner.media_builder.schedules(
                    device=model_runner.worker_config.device
                )
            ),
        )
        request.request.diffusion = trajectory
    if (
        not isinstance(trajectory, VideoState)
        or trajectory.size != numerical_shape
    ):
        raise invalid_descriptor(
            "video request changed its admitted numerical dimensions"
        )

    slot, context = prepare_call(
        model_runner,
        trajectory,
        call,
        request_pool.tensors(request.request.request_pool_idx)
        if model_runner.state_buffers
        else None,
    )

    from . import transfer

    products: tuple[TensorPublication, ...] = ()
    if call.kind is MediaCall.LATENT_PREPARATION:
        params = trajectory_params(call, state=state)
        if int(params.start_step) != 0 or int(params.step_count) != 0:
            raise invalid_descriptor(
                "video preparation params must carry zero denoise steps"
            )
        inputs = call.inputs
        if len(inputs) != 1:
            raise invalid_descriptor(
                "video preparation requires one conditioning Tensor"
            )
        conditioning = tensor_store.consume(
            inputs[0],
            consumer_call_id=call.call_id,
            device=model_runner.call_devices(call)[0],
        )
        request.device_reads.append(conditioning)
        if conditioning.region is not None:
            raise invalid_descriptor(
                "video preparation requires complete conditioning coverage"
            )

        encoded = conditioning.tensor
        initial = model_runner.media_builder.initialize(
            numerical_shape,
            slot,
            seed=media.seed,
            constants=context.constants,
            workspace=context.workspace,
        )
        with model_runner.preparing_inputs(initial):
            if "conditioning" in model_runner.encoder_kinds:
                if encoded is None:
                    raise invalid_descriptor(
                        "conditioning module has no input Tensor"
                    )
                result = model_runner.run_encoder(
                    "conditioning",
                    encoded,
                )
                if result.stats is None:
                    raise RuntimeError(
                        "module output has no execution statistics"
                    )
                state.group_forward_stats[completion_group].append(result.stats)
                if len(result.values) != 1:
                    raise invalid_descriptor(
                        "conditioning computation must return one Tensor"
                    )
                encoded = result.values[0]
                slot["text_condition"].copy_(
                    encoded.reshape_as(slot["text_condition"]),
                    non_blocking=True,
                )

    elif call.kind is MediaCall.DENOISING:
        params = trajectory_params(call, state=state)
        start_step, step_count = int(params.start_step), int(params.step_count)
        if start_step != calls.require_progress(request).flow_step:
            raise invalid_descriptor(
                "denoising does not begin at the selected request step"
            )
        if step_count != 1:
            raise invalid_descriptor(
                "video denoising requires one selected numerical step"
            )

        result = model_runner.run_denoising(
            model_runner.media_builder.bind(
                numerical_shape, slot, trajectory.schedules, start_step
            ),
            trajectory.schedules,
            state=slot,
            slot=request.request.request_pool_idx,
            input_key=numerical_shape,
        )
        if result.stats is None:
            raise RuntimeError("module output has no execution statistics")
        state.group_forward_stats[completion_group].append(result.stats)
        request.progress = replace(
            calls.require_progress(request),
            flow_step=start_step + step_count,
        )
        if call.outputs:
            if (
                calls.require_progress(request).flow_step
                != media.num_inference_steps
            ):
                raise invalid_descriptor(
                    "final latent products require completed denoising"
                )
            products = transfer.publish_tensors(
                call,
                result.values,
                completion_group,
                tensor_store=tensor_store,
                publication_transports=publication_transports,
                state=state,
            )

    elif call.kind in {MediaCall.VIDEO_DECODING, MediaCall.AUDIO_DECODING}:
        params = decode_range(call, state=state)
        inputs = call.inputs
        if len(inputs) != 1:
            raise invalid_descriptor(
                "media reconstruction requires one Tensor input"
            )
        read = tensor_store.consume(
            inputs[0],
            consumer_call_id=call.call_id,
            device=model_runner.call_devices(call)[0],
        )
        request.device_reads.append(read)
        if read.region is not None or read.tensor is None:
            raise invalid_descriptor(
                "media reconstruction requires complete input coverage"
            )

        cursor, count = params.cursor, params.max_units
        track = (
            MediaTrack.AUDIO
            if call.kind is MediaCall.AUDIO_DECODING
            else MediaTrack.VIDEO
        )
        # Each rank of a distributed decoder binding owns one media unit of
        # the declared range, on either track.
        samples = (
            None
            if track is MediaTrack.VIDEO
            else audio_samples(model_runner, media.num_frames)
        )
        windows = (
            model_runner.video_decoder.frame_slices(media.num_frames)
            if samples is None
            else audio_unit_windows(model_runner, call.component, samples)
        )
        binding = model_runner.bindings[call.component]
        position = binding.config.ranks.index(binding.process_group.global_rank)
        if position >= count or cursor < 0 or cursor + count > len(windows):
            raise invalid_descriptor(
                "media decoder assignment exceeds its media unit range"
            )
        unit = cursor + position
        window = windows[unit]

        if track is MediaTrack.VIDEO:
            decoded = model_runner.run_module(
                call.component,
                (read.tensor,),
                method="decode",
                size=media.num_frames,
                frames=(window,),
                num_frames=(media.num_frames,),
            )
            if decoded.stats is None:
                raise RuntimeError("module output has no execution statistics")
            state.group_forward_stats[completion_group].append(decoded.stats)
            if len(decoded.values) != 1 or decoded.values[0].shape[0] != 1:
                raise invalid_descriptor(
                    "video decoding reconstructs exactly one media unit"
                )

            # The decoded window becomes an RGB media unit on this rank,
            # cross-faded with the neighbouring unit's tail, and that unit
            # is the product a host rank encodes.
            decoder = model_runner.video_decoder
            layout = decoder.output_layout(media.num_frames)["video"]
            unit_outputs = (
                TensorOutput(
                    decoded.values[0],
                    replace(
                        layout,
                        local_slice=(
                            slice(unit, unit + 1),
                            *layout.local_slice[1:],
                        ),
                    ),
                ),
            )
            processed = model_runner.run_module(
                call.component,
                unit_outputs,
                method="forward",
                size=media.num_frames,
                frames=(window,),
                num_frames=(media.num_frames,),
                state=slot,
                unit_count=count,
            )
            state.group_forward_stats[completion_group].append(processed.stats)
            # Concatenation borrows one flat byte span; the product row is
            # the unit's frames at the output raster, and a unit shorter than
            # the longest fills its row's leading frames.
            frames = window.stop - window.start
            value = concatenate_views(processed.values).view(
                1,
                frames,
                decoder.frame_size.height,
                decoder.frame_size.width,
                3,
            )
            longest = max(
                span.stop - span.start
                for span in decoder.frame_slices(media.num_frames)
            )
            if frames < longest:
                row = value.new_zeros((1, longest, *value.shape[2:]))
                row[:, :frames].copy_(value)
                value = row
            values = (value,)
        else:
            audio_decoder = model_runner.component(call.kind)
            result = model_runner.run_module(
                call.component,
                (read.tensor,),
                method="decode",
                size=audio_decoder.latent_frames(samples),
                frames=(window,),
                num_samples=(samples,),
            )
            if result.stats is None:
                raise RuntimeError("module output has no execution statistics")
            state.group_forward_stats[completion_group].append(result.stats)
            if len(result.values) != 1:
                raise invalid_descriptor(
                    "media decoder must return one numerical tensor"
                )
            values = result.values
        # Decoded media units are host products: a host rank's encoder reads
        # them in place from the segment this rank publishes, over the host
        # mechanism of its edges.
        products = transfer.publish_tensors(
            call,
            values,
            completion_group,
            tensor_store=tensor_store,
            publication_transports=publication_transports,
            state=state,
            host=True,
        )
    else:
        raise invalid_descriptor(f"unsupported video call {call.kind!r}")

    request.status = CallStatus.OK
    # Reconstruction calls consume products without advancing the diffusion
    # trajectory. Keep their progress absent; denoising retains the step
    # already projected above through the same completion boundary.
    request.progress = calls.execution_runtime(request, None)
    request.finish_flags = FinishFlags()
    request.product_generations = calls.output_generations(call)
    request.products = products
    return request


def _request_label(call: Call) -> str:
    """Format a stable request and call label for media work."""
    key = call.request_key
    return f"{key.engine_id}:{key.request_id}:{key.request_epoch}"


__all__ = [
    "decode_range",
    "execute",
    "trajectory_params",
    "validate_batch",
]
