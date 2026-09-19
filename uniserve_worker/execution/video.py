"""Bounded execution of numerical video capabilities and media artifact.

actions.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import TYPE_CHECKING

import torch

from uniserve.media import video
from uniserve.model import AudioDecoder, VideoDecoder, VideoPostprocessor
from uniserve.profiling import profile_range
from uniserve.tensors import TensorOutput, concatenate_views
from uniserve_worker.foundation.errors import (
    invalid_descriptor,
    unsupported_setup,
)
from uniserve_worker.media.buffers import MediaBuffers
from uniserve_worker.media.mux import (
    AvMuxConfig,
    MediaEncoder,
    MediaMux,
    read_encoded_unit,
    require_media_codecs,
)
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
    PipelineStage,
)
from uniserve_worker.protocol.identity import CallId
from uniserve_worker.protocol.output import FinishFlags
from uniserve_worker.runtime.host_lane import HostTask

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


def require_video_codecs() -> None:
    """Verify the output owner's configured video and audio encoders."""
    require_media_codecs("libx264", "aac")


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
        media.num_decode_chunks != len(windows)
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

    if kind in {PipelineStage.LATENT_PREPARATION, PipelineStage.DENOISING}:
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

    if kind is PipelineStage.VIDEO_DECODING:
        return {}, runner.prepare_module(
            call.entry, size.num_frames, method="decode"
        )

    if kind is PipelineStage.AUDIO_DECODING:
        decoder = runner.component(kind)
        frames = decoder.latent_frames(audio_samples(runner, size.num_frames))
        return {}, runner.prepare_module(call.entry, frames, method="decode")

    if kind is PipelineStage.VIDEO_ENCODING:
        component = runner.component(kind)
        if "video_overlap" not in trajectory.tensors:
            if storage is None:
                raise RuntimeError(
                    "video assembly requires reserved overlap storage"
                )
            trajectory.tensors["video_overlap"] = storage.view(
                component.state_buffers(size.num_frames)
            )
        return trajectory.tensors["video_overlap"], runner.prepare_module(
            call.entry, size.num_frames, method="forward"
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

    A denoising graph is captured per request slot and per ladder step, so a
    request that first meets its shape would otherwise pay one capture per step
    on its own path. Capture is collective across the component's ranks and
    belongs here, where warmup holds them in lockstep. Samples are unchanged
    when this returns; a shape the deployment does not declare still serves and
    captures on first use.
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
        for slot, buffers in enumerate(storage, start=1):
            views = buffers.view(builder.buffers(size))
            # Capture records kernel launches over these addresses; the values
            # it reads are irrelevant, and the first real request stages its
            # own seeded noise and conditioning before replay.
            with context.activate():
                views["text_condition"].zero_()
                for name in builder.denoiser.modalities:
                    views[name].zero_()

            for index in range(builder.num_steps):
                denoising.capture(
                    builder.bind(size, views, schedules, index),
                    schedules,
                    state=views,
                    slot=slot,
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
    for (name, _, method), (binding, call) in runner._module_entries.items():
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
        for (name, _, _), (_, call) in runner._module_entries.items()
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
                segments = (
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
                    segments,
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


def create_media_resources(
    runner: ModelRunner,
    *,
    rank: int,
    worker_info,
    state_slots: int,
    unresolved_window: int,
):
    """Reserve this rank's media capture, encoder and assembly resources.

    Every rank that reconstructs media units encodes them on its own host lane
    and needs a capture ring; only the muxer rank assembles the artifact.
    """
    from ..bootstrap.components import MUXER_COMPONENT

    muxer = runner.bindings.get(MUXER_COMPONENT)
    assembles = (
        muxer is not None
        and muxer.owns
        and rank == worker_info.output_rank(MUXER_COMPONENT)
    )
    reconstructs = any(
        isinstance(call.module, VideoPostprocessor)
        for _, call in runner._module_entries.values()
    )
    if not assembles and not reconstructs:
        return None, None
    require_video_codecs()

    decoder = runner.video_decoder
    output = runner.video_postprocessor
    audio = runner.audio_decoder
    frames = runner.media_builder.maximum.num_frames
    # A capture slot holds one media unit, which is what one encode call
    # converts, rather than the whole timeline.
    unit = max(
        window.stop - window.start for window in decoder.frame_slices(frames)
    )
    return (
        MediaMux(rank=rank) if assembles else None,
        MediaBuffers(
            state_slots=state_slots,
            unresolved_window=unresolved_window,
            max_video_frames_per_round=unit,
            video=video.Config(frames, decoder.frame_size),
            frame_rate=output.frame_rate,
            audio_rate=audio.sample_rate,
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
        if call.kind is not PipelineStage.LATENT_PREPARATION:
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
    media_mux: MediaMux | None,
    publication_transports: Mapping[str, Transport],
    request_pool: RequestPool,
    model_runner: ModelRunner,
) -> PendingOutput:
    """Land one ready video action without constructing a model input row."""
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

    mux = media_mux
    # Media units are encoded on the ranks that reconstruct them; only the
    # muxer rank encodes the audio track and assembles the artifact.
    if (
        call.kind in {PipelineStage.AUDIO_ENCODING, PipelineStage.MUXING}
        and mux is None
    ):
        raise unsupported_setup("artifact assembly has no muxer resources")
    from . import transfer

    tasks: tuple[HostTask, ...] = ()
    products: tuple[TensorPublication, ...] = ()
    if call.kind is PipelineStage.LATENT_PREPARATION:
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

    elif call.kind is PipelineStage.DENOISING:
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
        request.projected_progress = replace(
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

    elif call.kind in {
        PipelineStage.VIDEO_DECODING,
        PipelineStage.AUDIO_DECODING,
        PipelineStage.VIDEO_ENCODING,
        PipelineStage.AUDIO_ENCODING,
    }:
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
        if (
            read.region is not None
            and call.kind is not PipelineStage.VIDEO_ENCODING
        ):
            # Video encoding consumes the media unit this rank decoded, which is
            # a region of the round's product; every other media call consumes a
            # complete input.
            raise invalid_descriptor(
                "media reconstruction requires complete input coverage"
            )

        cursor, count = params.cursor, params.max_units
        track = (
            MediaTrack.AUDIO
            if call.kind
            in {PipelineStage.AUDIO_DECODING, PipelineStage.AUDIO_ENCODING}
            else MediaTrack.VIDEO
        )
        if call.kind in {
            PipelineStage.VIDEO_DECODING,
            PipelineStage.AUDIO_DECODING,
        }:
            # Each rank of a distributed decoder binding owns one media unit
            # of the declared range, on either track.
            samples = (
                None
                if track is MediaTrack.VIDEO
                else audio_samples(model_runner, media.num_frames)
            )
            windows = (
                model_runner.video_decoder.frame_slices(media.num_frames)
                if samples is None
                else audio_unit_windows(model_runner, call.entry, samples)
            )
            binding = model_runner.bindings[call.entry]
            position = binding.config.ranks.index(
                binding.process_group.global_rank
            )
            if position >= count or cursor < 0 or cursor + count > len(windows):
                raise invalid_descriptor(
                    "media decoder assignment exceeds its media unit range"
                )
            window = windows[cursor + position]
            if read.tensor is None:
                raise invalid_descriptor(
                    "media decoder has no complete numerical input"
                )

            if track is MediaTrack.VIDEO:
                result = model_runner.run_module(
                    call.entry,
                    (read.tensor,),
                    method="decode",
                    size=media.num_frames,
                    frames=(window,),
                    num_frames=(media.num_frames,),
                )
            else:
                decoder = model_runner.component(call.kind)
                result = model_runner.run_module(
                    call.entry,
                    (read.tensor,),
                    method="decode",
                    size=decoder.latent_frames(samples),
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
            products = transfer.publish_tensors(
                call,
                result.values,
                completion_group,
                tensor_store=tensor_store,
                publication_transports=publication_transports,
                state=state,
            )
        elif call.kind is PipelineStage.VIDEO_ENCODING:
            # The rank that decoded this media unit converts it to RGB on its
            # device lane and encodes it on its host lane, so its input is the
            # shard it published rather than the whole round.
            binding = model_runner.bindings[call.entry]
            position = binding.config.ranks.index(
                binding.process_group.global_rank
            )
            decoder = model_runner.video_decoder
            windows = decoder.frame_slices(media.num_frames)
            unit = cursor + position
            if position >= count or unit >= len(windows):
                raise invalid_descriptor(
                    "media unit assignment exceeds its media unit range"
                )
            if read.tensor is None or read.tensor.shape[0] != 1:
                raise invalid_descriptor(
                    "video encoding reconstructs exactly one media unit"
                )

            reservation = (
                request.completion_tasks[0]
                if request.completion_tasks
                else None
            )
            ring_lease = request.media_lease
            if reservation is None or ring_lease is None:
                raise RuntimeError(
                    "media output has no reserved capture storage"
                )

            layout = decoder.output_layout(media.num_frames)["video"]
            segments = (
                TensorOutput(
                    read.tensor[0].unsqueeze(0),
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
                call.entry,
                segments,
                method="forward",
                size=media.num_frames,
                frames=(windows[unit],),
                num_frames=(media.num_frames,),
                state=slot,
                unit_count=count,
            )
            state.group_forward_stats[completion_group].append(processed.stats)
            # Concatenation borrows one flat byte span. Restore the RGB raster
            # axes before handing its pinned capture to the encoder.
            value = concatenate_views(processed.values).view(
                -1, decoder.frame_size.height, decoder.frame_size.width, 3
            )

            with profile_range(
                f"uniserve.video.decode_copy "
                f"request={_request_label(call)} "
                f"step={call.call_id.batch_id} "
                f"op={call.call_id.request_index} "
                f"kind={track.value}"
            ):
                capture = state.group_buffers[
                    completion_group
                ].capture_bytes_into(value, ring_lease.storage)
            try:
                tasks = (
                    MediaEncoder(rank=model_runner.worker_config.rank).unit(
                        call.request_key,
                        config=mux_config(model_runner, media),
                        unit_index=unit,
                        frames=capture,
                        output=state.group_buffers[completion_group],
                        reservation=reservation,
                        ring_lease=ring_lease,
                        call_id=call.call_id,
                    ),
                )
            except BaseException:
                ring_lease.defer_until_ready(
                    state.group_buffers[completion_group].completion_future()
                )
                raise
            # The encoded unit is the product the muxer assembles. Its row is
            # published with this batch and filled when the encode completes;
            # the muxer's call is scheduled only after every encode round has
            # completed, so no rank can read the row before its bytes land.
            row = transfer.reserved_unit_row(
                call, completion_group, state=state
            )
            request.encoded_unit_row = row[0]
            products = transfer.publish_tensors(
                call,
                (row,),
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
                call.request_key,
                video=video.Config(media.num_frames, decoder.frame_size),
                frame_rate=model_runner.video_postprocessor.frame_rate,
                audio_rate=model_runner.audio_decoder.sample_rate,
                video_unit_frames=tuple(
                    window.stop - window.start for window in windows
                ),
            )
            reservation = (
                request.completion_tasks[0]
                if request.completion_tasks
                else None
            )
            ring_lease = request.media_lease
            if reservation is None or ring_lease is None:
                raise RuntimeError(
                    "media output has no reserved capture storage"
                )
            if read.region is not None:
                raise invalid_descriptor(
                    "audio encoding requires the complete sample timeline"
                )
            value = read.tensor.view(torch.uint8)
            with profile_range(
                f"uniserve.video.decode_copy "
                f"request={_request_label(call)} "
                f"step={call.call_id.batch_id} "
                f"op={call.call_id.request_index} "
                f"kind={track.value}"
            ):
                capture = state.group_buffers[
                    completion_group
                ].capture_bytes_into(value, ring_lease.storage)
            try:
                tasks = (
                    mux.audio(
                        call.request_key,
                        capture,
                        state.group_buffers[completion_group],
                        reservation,
                        ring_lease,
                        call.call_id,
                    ),
                )
            except BaseException:
                ring_lease.defer_until_ready(
                    state.group_buffers[completion_group].completion_future()
                )
                raise
    else:
        reservation = (
            request.completion_tasks[0] if request.completion_tasks else None
        )
        if reservation is None or mux is None:
            raise RuntimeError("media finalization has no reserved CPU task")
        if not call.inputs:
            raise invalid_descriptor(
                "artifact assembly requires its encoded media units"
            )
        # One product per encode round, in media unit order.
        units: list[bytes] = []
        for product in call.inputs:
            read = tensor_store.consume(
                product,
                consumer_call_id=call.call_id,
                device=model_runner.call_devices(call)[0],
            )
            request.device_reads.append(read)
            if read.region is not None or read.tensor is None:
                raise invalid_descriptor(
                    "artifact assembly requires every encoded media unit"
                )
            rows = read.tensor.to("cpu")
            units.extend(read_encoded_unit(row) for row in rows.unbind(0))
        tasks = (
            mux.finalize_artifact(
                call.request_key,
                tuple(units),
                reservation,
                call.call_id,
            ),
        )

    # Configured HostTask now owns the media lease through its final CPU read.
    request.media_lease = None
    request.status = CallStatus.OK
    # Reconstruction and mux calls consume products without advancing the
    # diffusion trajectory. Keep their progress absent; denoising retains the
    # step already projected above through the same completion boundary.
    request.projected_progress = calls.execution_runtime(request, None)
    request.finish_flags = FinishFlags()
    request.product_generations = calls.output_generations(call)
    request.completion_tasks = tasks
    request.products = products
    return request


def _request_label(call: Call) -> str:
    """Format a stable request and call label for media work."""
    key = call.request_key
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
        raise unsupported_setup("call requires the media output owner's ring")
    return ring
