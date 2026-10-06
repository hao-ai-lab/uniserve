"""Numerical video denoising, reconstruction and startup preparation.

Rust owns request views, host preparation, latent-bank access and result
export. These helpers assemble numerical inputs and evaluate the same
model operations during serving and startup.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import torch

from uniserve.media import video
from uniserve.model import AudioDecoder, VideoDecoder, VideoPostprocessor
from uniserve.runtime.cuda_graph import CUDAGraphError
from uniserve.tensors import TensorOutput, concatenate_views
from uniserve_worker.errors import invalid_descriptor, unsupported_setup
from uniserve_worker.execution.diffusion_state import DiffusionState
from uniserve_worker.media.mux import AvMuxConfig
from uniserve_worker.protocol.batch import NewRequest

if TYPE_CHECKING:
    from uniserve.runtime.tensor_buffers import TensorBuffers
    from uniserve_worker.execution.model_executor import ModelExecutor
    from uniserve_worker.model_executor.output import ExecutionOutput


def video_shape(runner: ModelExecutor, admission: NewRequest):
    """Resolve a video admission into its exact numerical size.

    The size holds the admitted prompt length, the request's conditions and
    its presented vision spans; it is not rounded up to the layout it
    occupies. Raises ``invalid_descriptor`` when the admission is not a
    video request's, when the rank lacks a media builder or video decoder,
    when the builder or decoder rejects the frame count, prompt length or
    conditions (a size beyond the worker's capacity included), or when the
    admission's media unit count or step count disagrees with the model.
    """
    from uniserve_worker.execution.conditions import library_conditions

    builder = runner.media_builder
    decoder = runner.video_decoder
    media, video = admission.diffusion, admission.video
    if media is None or video is None:
        raise invalid_descriptor("video call has no admitted media dimensions")
    if builder is None or decoder is None:
        raise invalid_descriptor(
            "video input requires denoising and reconstruction capabilities"
        )

    try:
        size = builder.size(
            media.num_frames,
            len(admission.prompt_token_ids),
            media.canvas,
            conditions=library_conditions(video),
            vision_spans=video.vision_spans(),
        )
        windows = decoder.frame_slices(size.num_frames)
    except ValueError as error:
        raise invalid_descriptor(str(error)) from error

    if (
        media.video_units != len(windows)
        or media.num_inference_steps != builder.num_steps
    ):
        raise invalid_descriptor("video request has invalid computation bounds")
    return size


def audio_samples(runner: ModelExecutor, num_frames: int) -> int:
    """Return the sample count of the audio track generated with the video.

    The model's decoder defines the track from its latent timeline; the
    muxer aligns it to the video duration.
    """
    return runner.audio_decoder.track_samples(
        num_frames, runner.video_postprocessor.frame_rate
    )


def audio_unit_count(runner: ModelExecutor, entry: str) -> int:
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
    runner: ModelExecutor, entry: str, num_samples: int
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


def assigned_units(
    runner: ModelExecutor, entry: str, cursor: int, count: int, total: int
) -> range:
    """Return this rank's media units of one decode round.

    The round covers units ``[cursor, cursor + count)`` of a request's
    ``total``; ``ComponentBinding.media_units`` deals them to the component's
    ranks. Raises ``invalid_descriptor`` when the round lies outside the
    request's units or assigns this rank none.
    """
    if cursor < 0 or count < 1 or cursor + count > total:
        raise invalid_descriptor(
            "media decoder assignment exceeds its media unit range"
        )
    units = runner.bindings[entry].media_units(cursor, count)
    if not units:
        raise invalid_descriptor(
            "media decoder assignment exceeds its media unit range"
        )
    return units


def decode_audio(
    runner: ModelExecutor,
    entry: str,
    latent: torch.Tensor,
    num_samples: int,
    *,
    cursor: int,
    count: int,
) -> ExecutionOutput:
    """Reconstruct this rank's audio media units of one decode round.

    ``latent`` is the request's complete audio latent timeline and
    ``num_samples`` its exact duration. The rank's units form a contiguous
    run (see ``assigned_units``), and the decoder reconstructs each unit
    within its own halo in one call, so a round warmed at startup and served
    later share one input signature.

    Returns:
        The rank's samples as one tensor, the concatenation of its units in
        order, which is the sample-major region
        ``ModelExecutor.output_layout`` reserves for the rank, together with
        the call's execution statistics.

    Raises:
        WorkerError: ``invalid_descriptor`` when the round assigns this rank
            no unit or lies outside the request's units, or the decoder does
            not return one tensor per unit; ``unsupported_setup`` from
            ``audio_unit_windows``.
    """
    windows = audio_unit_windows(runner, entry, num_samples)
    units = assigned_units(runner, entry, cursor, count, len(windows))
    result = runner.run_module(
        entry,
        (latent,) * len(units),
        method="decode",
        size=runner.audio_decoder.latent_frames(num_samples),
        frames=tuple(windows[unit] for unit in units),
        num_samples=(num_samples,) * len(units),
    )
    if len(result.values) != len(units):
        raise invalid_descriptor(
            "audio decoder must return one tensor per media unit"
        )
    if len(units) == 1:
        return result

    # Consecutive units' samples abut, so their concatenation is the rank's
    # contiguous span of the track.
    return result.replace(
        values=(torch.cat(result.values),), vocabularies=(), layouts=()
    )


def _prepare_placeholder(builder, size, views, samples, diffusion, layout):
    """Fill a slot and samples with a request's inputs for startup calls.

    Warmup and capture run the numerical calls a request of ``size`` runs in
    ``layout``, with a fixed seed and no prompt. Their results are
    discarded, and the first real request prepares its own draw, tables and
    conditioning.
    """
    entry = diffusion.layout(layout)
    builder.prepare_request(size, views, seed=0, layout=layout)
    copies = builder.initialize(
        size,
        views,
        samples,
        constants=entry.constants,
        workspace=entry.workspace,
        layout=layout,
    )
    with diffusion.execution.context.activate():
        for destination, source in copies:
            destination.copy_(source)
        views["text_condition"].zero_()


def denoising_inputs(runner, storage, schedules, layout, initialize):
    """Bind scratch request tensors for one layout's warmup or capture.

    Slot one is unowned during startup. Only warmup evaluates the numerical
    call, so capture needs the same views without filling placeholder values.
    """
    builder, diffusion = runner.media_builder, runner.diffusion
    size = builder.size(
        layout.num_frames,
        min(layout.num_text_tokens, builder.max_text_tokens),
        layout.canvas,
    )
    views = storage.view(builder.layout_buffers(layout))
    samples = builder.sample_views(size, diffusion.samples, layout=layout)
    if initialize:
        _prepare_placeholder(builder, size, views, samples, diffusion, layout)
    return diffusion.bind(
        layout,
        tuple(
            builder.bind(size, views, samples, schedules, index, layout=layout)
            for index in range(builder.num_steps)
        ),
        schedules,
        state=views,
        slot=1,
        pages=builder.slot_pages(1),
    )


def prepare_denoising(
    runner: ModelExecutor, storage: tuple[TensorBuffers, ...]
) -> None:
    """Make every capacity layout's steps ready before the worker serves.

    Each layout of ``MediaBuilder.layouts`` is prepared, largest first, so
    the first allocates the storage every other layout shares. A placeholder
    request filling the layout's text capacity on slot one runs one eager
    step, which prepares the layout's kernels, plans and scratch; once every
    layout is warm, a capturing runner captures one graph per layout. A
    graph serves every solver step, since the step index, timesteps and
    schedules are inputs a replay copies in; every request slot, since it
    reaches a slot's storage and pages through device indices; and every
    prompt length the layout holds, since the lengths differ only in the
    state a graph gathers from the slot.
    Collective across the component's ranks, which hold each other in
    lockstep here. Serving can prepare additional eager layouts and never
    captures.

    Raises:
        RuntimeError: The request storage or input builder is missing, or
            the layouts' resident storage does not fit the device, with the
            layout count and the settings that bound it.
    """
    builder = runner.media_builder
    if not runner.denoises:
        return
    if builder is None or not storage:
        raise RuntimeError(
            "denoising preparation requires its input builder and request "
            "storage"
        )

    try:
        runner.batch_runners.prepare_denoising(runner, storage)
    except (CUDAGraphError, torch.OutOfMemoryError) as error:
        raise RuntimeError(
            f"the denoiser's {len(builder.layouts())} capacity layouts "
            f"({len(builder.canvases)} canvases x "
            f"{len(builder.frame_counts)} frame counts x "
            f"{len(builder.text_capacities)} text capacities) do not fit "
            "this device: lower --max-video-seconds or --max-model-len, use "
            "fewer --video-text-capacities, raise --mem-fraction-static, or "
            f"serve with --graph-policy off ({error})"
        ) from error


def decode_video_unit(
    runner: ModelExecutor,
    entry: str,
    latent: torch.Tensor,
    frames: slice,
    size: video.Config,
):
    """Reconstruct one media unit of a video's complete packed latent.

    The unit's window is unpacked eagerly on the caller's stream, which
    depends on the video's frame count, and decoded at its segment, whose
    prepared context and captured graph every video of that raster shares.

    Returns:
        The decoder call's ``ExecutionOutput``: one native segment, leading
        with a unit axis of one, and the call's statistics.
    """
    decoder = runner.video_decoder
    segment = decoder.segment(size, frames)
    config = decoder.window_input(segment)
    window = torch.empty(config.shape, dtype=config.dtype, device=latent.device)
    decoder.unpack_latents(latent, frames, size, out=window)
    return runner.run_module(
        entry,
        (window,),
        method="decode",
        size=segment,
        segments=(segment,),
    )


@torch.inference_mode()
def warmup_decoders(runner: ModelExecutor) -> None:
    """Prepare and capture reconstruction at every admitted size.

    A video window's decode depends on its segment alone, so the video
    decoder prepares and captures each segment its admitted videos decode
    to once: the frame count only changes how a window is unpacked, which
    runs eagerly. The audio decoder's prepared context and graph follow the
    track's latent frames, so it warms every admitted duration and the units
    its ranks decode together. Every size's context is prepared before the
    first capture into the decoder's shared graph pool
    (``ModelExecutor.prepare_module``), the largest first.

    Raises:
        RuntimeError: The decoders' prepared contexts and graphs do not fit
            the device, with the settings that bound them.
    """
    builder = runner.media_builder
    frames = tuple(reversed(builder.frame_counts))
    decoder = runner.video_decoder
    # Canvases come in decreasing generated rows and frame counts in
    # decreasing order, so the first segments are the largest.
    segments = (
        ()
        if decoder is None
        else tuple(
            dict.fromkeys(
                decoder.segment(video.Config(num_frames, canvas), window)
                for canvas in builder.canvases
                for num_frames in frames
                for window in decoder.frame_slices(num_frames)
            )
        )
    )
    try:
        for name, binding, call in runner.batch_runners.calls():
            if call.entry_point.method != "decode":
                continue
            module = call.module
            if isinstance(module, VideoDecoder):
                for segment in segments:
                    runner.prepare_module(name, segment, method="decode")
                for segment in segments:
                    config = module.window_input(segment)
                    window = torch.zeros(
                        config.shape, dtype=config.dtype, device=binding.device
                    )
                    runner.run_module(
                        name,
                        (window,),
                        method="decode",
                        size=segment,
                        segments=(segment,),
                    )
            elif isinstance(module, AudioDecoder):
                # The audio track does not depend on the canvas.
                canvas = builder.maximum.canvas
                for num_frames in frames:
                    runner.prepare_module(
                        name,
                        module.latent_frames(audio_samples(runner, num_frames)),
                        method="decode",
                    )
                for num_frames in frames:
                    size = builder.size(
                        num_frames, builder.max_text_tokens, canvas
                    )
                    shape = builder.denoiser.latent_shape("audio", size)
                    latent = torch.zeros(
                        shape, dtype=torch.float32, device=binding.device
                    )
                    # The engine hands a request's audio units to the decoder
                    # in one round, which serving decodes through the same
                    # call.
                    decode_audio(
                        runner,
                        name,
                        latent,
                        audio_samples(runner, num_frames),
                        cursor=0,
                        count=audio_unit_count(runner, name),
                    )
    except (CUDAGraphError, torch.OutOfMemoryError) as error:
        raise RuntimeError(
            f"the decoders ({len(segments)} video segments, {len(frames)} "
            "audio durations) do not fit this device: lower "
            "--max-video-seconds, raise --mem-fraction-static, or serve with "
            f"--graph-policy off ({error})"
        ) from error


@torch.inference_mode()
def warmup_postprocess(
    runner: ModelExecutor, storage: tuple[TensorBuffers, ...]
) -> None:
    """Exercise real output windows through the public numerical interface.

    One post-processing call converts the media unit this rank reconstructed,
    so its prepared context follows the frame count and canvas, and every
    admitted size is prepared. Every rank of the ring prepares each size even
    where it holds no unit in a round, because preparing a context binds the
    ring's communication resources and that binding spans the whole ring.
    """
    entries = [
        (name, call)
        for name, _, call in runner.batch_runners.calls()
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
    position = binding.config.ranks.index(binding.process_group.rank)
    units_per_round = len(binding.config.ranks)
    sizes = (
        video.Config(frames, canvas)
        for canvas in builder.canvases
        for frames in reversed(builder.frame_counts)
    )
    for output in sizes:
        frames = output.num_frames
        runner.prepare_module(name, output, method="forward")
        windows = decoder.frame_slices(frames)
        layout = decoder.output_layout(output)["video"]
        state = storage[0].view(call.module.state_buffers(output))
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
                    size=output,
                    frames=(windows[unit],),
                    sizes=(output,),
                    state=state,
                    unit_count=count,
                )
            cursor += count


def mux_config(runner: ModelExecutor, media) -> AvMuxConfig:
    """Return the container settings one request's media units encode under."""
    decoder = runner.video_decoder
    windows = decoder.frame_slices(media.num_frames)
    return AvMuxConfig(
        width=media.width,
        height=media.height,
        frame_count=media.num_frames,
        frame_rate=runner.video_postprocessor.frame_rate,
        audio_rate=runner.audio_decoder.sample_rate,
        video_unit_frames=tuple(
            window.stop - window.start for window in windows
        ),
    )


def open_state(runner: ModelExecutor, size) -> DiffusionState:
    """Open a video request's diffusion state on the fixed schedule."""
    builder = runner.media_builder
    return DiffusionState.open(
        builder.denoiser,
        size,
        steps=builder.num_steps,
        shift=None,
        device=runner.worker_config.device,
    )


def initialize_video(
    runner, size, views, context, samples, encoded, conditions
):
    """Initialize borrowed samples and retain text/condition features on stream.

    ``conditions`` follows the numerical denoiser's request order. The
    caller has joined any host noise preparation before entering this call.
    """
    builder = runner.media_builder
    copies = builder.initialize(
        size,
        views,
        builder.sample_views(size, samples),
        constants=context.constants,
        workspace=context.workspace,
    )
    result = None
    with runner.preparing_inputs(copies):
        if "conditioning" in runner.encoder_kinds:
            result = runner.encode_conditioning(encoded)
            if len(result.values) != 1:
                raise invalid_descriptor(
                    "conditioning computation must return one Tensor"
                )
            builder.store_conditioning(size, views, result.values[0])

        if conditions:
            builder.encode_conditions(size, views, conditions)

    return result


def bind_video(runner, trajectory, views, slot: int, pages: tuple[int, ...]):
    """Bind numerical step inputs over a request's borrowed slot views."""
    builder = runner.media_builder
    size = trajectory.size
    diffusion = runner.diffusion
    samples = builder.sample_views(size, diffusion.samples)
    return diffusion.bind(
        builder.layout(size),
        tuple(
            builder.bind(size, views, samples, trajectory.schedules, index)
            for index in range(builder.num_steps)
        ),
        trajectory.schedules,
        state=views,
        slot=slot,
        pages=pages,
    )


def reconstruct_video(runner, component, latent, frames, size, views, count):
    """Decode and cross-fade one unit into its padded RGB output row.

    A short final unit occupies the leading frames of the same row shape
    as the longest unit. The padding remains zero for its host consumer.
    """
    decoded = decode_video_unit(runner, component, latent, frames, size)
    if len(decoded.values) != 1 or decoded.values[0].shape[0] != 1:
        raise invalid_descriptor(
            "video decoding reconstructs exactly one media unit"
        )

    decoder = runner.video_decoder
    processed = runner.run_module(
        component,
        (decoder.place(decoded.values[0], frames, size),),
        method="forward",
        size=size,
        frames=(frames,),
        sizes=(size,),
        state=views,
        unit_count=count,
    )
    length = frames.stop - frames.start
    value = concatenate_views(processed.values).view(
        1, length, size.frame.height, size.frame.width, 3
    )
    longest = max(
        span.stop - span.start for span in decoder.frame_slices(size.num_frames)
    )
    if length < longest:
        row = value.new_zeros((1, longest, *value.shape[2:]))
        row[:, :length].copy_(value)
        value = row

    return decoded, processed.replace(
        values=(value,), vocabularies=(), layouts=()
    )
