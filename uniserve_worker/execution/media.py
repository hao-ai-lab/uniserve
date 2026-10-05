"""Device execution of a standalone video denoiser's request calls.

``uniserve_worker.execution.schedule`` dispatches here the video calls of a
worker whose model has a ``VideoPostprocessor``: latent preparation, one
denoising step per call, and video and audio decode rounds. A video decode
round also converts its media unit to RGB and publishes it as a host product;
``uniserve_worker.execution.host_media`` encodes and muxes those products.

Each request keeps a ``DiffusionState`` whose ``SlotLadder`` holds views of
its request slot (the denoising state: tables, noise draws and retained
conditioning; or the video overlap state on a video decoding rank) and its
bound ladder. The solver samples live in the worker's ``LatentPool``:
preparation writes them to bank one of the request's pages, each step gathers
the committed bank and writes its successor to the other bank, and the batch
commit publishes each result through the ``LatentUpdate`` the call leaves on
its output.

The module also holds the startup passes ``ModelExecutor.warmup`` runs for
these capabilities, and ``begin_noise``, which the batch ``Executor`` calls
when it applies a diffusion request's start command.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping
from dataclasses import replace
from typing import TYPE_CHECKING

import torch

from uniserve.media import video
from uniserve.model import AudioDecoder, VideoDecoder, VideoPostprocessor
from uniserve.runtime.cuda_graph import CUDAGraphError
from uniserve.tensors import TensorOutput, concatenate_views
from uniserve_worker._uniserve_ipc import BatchState
from uniserve_worker.errors import invalid_descriptor, unsupported_setup
from uniserve_worker.execution.diffusion_state import (
    DiffusionState,
    SlotLadder,
)
from uniserve_worker.execution.output import PendingOutput
from uniserve_worker.media.mux import AvMuxConfig
from uniserve_worker.protocol.batch import (
    DecodeRange,
    MediaTrack,
    NewRequest,
    TensorPublication,
)
from uniserve_worker.protocol.call import Call, MediaCall
from uniserve_worker.storage.latent_pool import LatentUpdate

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from uniserve.runtime.tensor_buffers import TensorBuffers
    from uniserve_worker.execution.model_executor import ModelExecutor
    from uniserve_worker.execution.request import Request, RequestPool
    from uniserve_worker.model_executor.output import ExecutionOutput
    from uniserve_worker.storage.tensor_store import TensorStore
    from uniserve_worker.transport.interface import Transport


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
    return replace(
        result, values=(torch.cat(result.values),), vocabularies=(), layouts=()
    )


def slot_ladder(trajectory: DiffusionState) -> SlotLadder:
    """Return the slot state of a standalone denoiser's request."""
    if trajectory.slot is None:
        raise invalid_descriptor("video diffusion requires request slot state")
    return trajectory.slot


def prepare_call(
    runner: ModelExecutor, trajectory: DiffusionState, call, storage
):
    """Bind request state and prepare the exact call's runtime-owned resources.

    Only request state survives in the trajectory. Runners own their
    constants and workspace and may be retired after their dependent graphs
    drain.

    Returns the request's slot views for the call and the prepared
    resources whose constants and workspace it uses: the ``"denoising"``
    views and the request layout's ``LayoutEntry`` for preparation and
    denoising, the ``"video_overlap"`` views and the post-processor's context
    for a video decode round, whose decoder context the round's segment
    selects when it decodes (``decode_video_unit``), and no views for an
    audio decode round. Other kinds return ``({}, None)``. Slot views
    are taken from ``storage`` on first use and kept in the trajectory.
    Raises ``invalid_descriptor`` when the trajectory has no slot state, and
    ``RuntimeError`` when a preparation or denoising call reaches a rank that
    does not denoise, or when views must be taken and ``storage`` is ``None``;
    errors of ``prepare_module`` and ``component`` propagate.
    """
    kind, size, slot = call.kind, trajectory.size, slot_ladder(trajectory)

    if kind in {MediaCall.LATENT_PREPARATION, MediaCall.DENOISING}:
        if not runner.denoises:
            raise RuntimeError("rank does not own denoising execution")
        if "denoising" not in slot.tensors:
            if storage is None:
                raise RuntimeError(
                    "denoising requires reserved request storage"
                )
            slot.tensors["denoising"] = storage.view(
                runner.media_builder.buffers(size)
            )
        layout = runner.media_builder.layout(size)
        return slot.tensors["denoising"], runner.diffusion_layout(layout)

    if kind is MediaCall.VIDEO_DECODING:
        # A decode round also converts its media unit to RGB, cross-faded with
        # the neighbouring unit's tail held in the request's overlap state.
        postprocessor = runner.video_postprocessor
        output = video.Config(size.num_frames, size.canvas)
        if "video_overlap" not in slot.tensors:
            if storage is None:
                raise RuntimeError(
                    "video reconstruction requires reserved overlap storage"
                )
            slot.tensors["video_overlap"] = storage.view(
                postprocessor.state_buffers(output)
            )
        return slot.tensors["video_overlap"], runner.prepare_module(
            call.component, output, method="forward"
        ).context

    if kind is MediaCall.AUDIO_DECODING:
        decoder = runner.component(kind)
        frames = decoder.latent_frames(audio_samples(runner, size.num_frames))
        return {}, runner.prepare_module(
            call.component, frames, method="decode"
        ).context

    return {}, None


def _stage_placeholder(builder, size, views, samples, diffusion, layout):
    """Fill a slot and samples with a request's inputs for startup calls.

    Warmup and capture run the numerical calls a request of ``size`` runs in
    ``layout``, with a fixed seed and no prompt. Their results are
    discarded, and the first real request stages its own draw, tables and
    conditioning.
    """
    entry = diffusion.layout(layout)
    builder.stage_request(size, views, seed=0, layout=layout)
    staged = builder.initialize(
        size,
        views,
        samples,
        constants=entry.constants,
        workspace=entry.workspace,
        layout=layout,
    )
    with diffusion.context.activate():
        for destination, source in staged:
            destination.copy_(source)
        views["text_condition"].zero_()


@torch.inference_mode()
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
    lockstep here; serving never prepares a layout or captures.

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

    started = time.perf_counter()
    try:
        layouts = runner.prepare_layouts()
        prepared = time.perf_counter()
        diffusion = runner.diffusion
        schedules = open_state(runner, builder.maximum).schedules

        def ladder(layout, *, staged):
            # A placeholder request filling the layout's text capacity on
            # slot one. ``storage[0]`` is slot one's tensors; no request
            # owns a slot while this pass runs. Capture records the step
            # without evaluating it, so only the warm step needs the
            # placeholder's inputs staged.
            size = builder.size(
                layout.num_frames,
                min(layout.num_text_tokens, builder.max_text_tokens),
                layout.canvas,
            )
            views = storage[0].view(builder.layout_buffers(layout))
            samples = builder.sample_views(
                size, diffusion.samples, layout=layout
            )
            if staged:
                _stage_placeholder(
                    builder, size, views, samples, diffusion, layout
                )
            return diffusion.bind(
                layout,
                tuple(
                    builder.bind(
                        size, views, samples, schedules, index, layout=layout
                    )
                    for index in range(builder.num_steps)
                ),
                schedules,
                state=views,
                slot=1,
                pages=builder.slot_pages(1),
            )

        # Every layout warms before any captures. A warm step allocates the
        # layout's persistent plans and scratch from the runner's pool, and
        # the captured steps of every layout share that pool's free blocks
        # as their intermediates; a persistent allocation made after a
        # capture could land in a block an earlier graph rewrites. Each
        # layout's warm step and captures also leave storage outside the
        # pool (loaded modules, graph executables), so the budget is checked
        # per layout: an overrun is refused at the layout that causes it,
        # before the device itself runs out. The layout bounding the
        # condition capacity warms first, so the scratch every layout
        # borrows already holds the largest step, that of a request with
        # conditions evaluating eagerly in a layout of its own; it is never
        # captured.
        maximum = builder.maximum_layout
        warm = layouts if maximum in layouts else (maximum, *layouts)
        for layout in warm:
            diffusion.warmup(ladder(layout, staged=True))
            runner.graph_storage.check()
        warmed = time.perf_counter()
        if diffusion.captures:
            # One graph per layout evaluates every solver step.
            for layout in layouts:
                diffusion.capture(ladder(layout, staged=False))
                runner.graph_storage.check()
        resident = sum(runner.graph_storage.pool_bytes().values())
        finished = time.perf_counter()
        logger.info(
            "prepared %d denoiser layouts (%d canvases x %d frame counts x "
            "%d text capacities, %d step graphs) in %.1f s (contexts %.1f s, "
            "warm steps %.1f s, capture %.1f s); graph storage %.2f GiB",
            len(layouts),
            len(builder.canvases),
            len(builder.frame_counts),
            len(builder.text_capacities),
            len(layouts) if diffusion.captures else 0,
            finished - started,
            prepared - started,
            warmed - prepared,
            finished - warmed,
            resident / 2**30,
        )
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
        for (name, _, method), (binding, call) in runner._module_calls.items():
            if method != "decode":
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


def sample_generation(step: int) -> int:
    """Return the pool generation of a trajectory at an accepted step.

    A standalone denoiser's samples are never published from the pool, and
    one request slot holds one trajectory, so its accepted step count
    versions it; zero stays the empty slot's generation.
    """
    return int(step) + 1


def sample_update(
    slot: int, params, *, step: int, previous: int | None
) -> LatentUpdate:
    """Describe the pool publication of a trajectory's successor.

    ``previous`` is the committed step the successor advances, or ``None``
    for the prepared trajectory at step zero.
    """
    return LatentUpdate(
        int(slot),
        params=params,
        expected_generation=0
        if previous is None
        else sample_generation(previous),
        expected_step=0 if previous is None else int(previous),
        generation=sample_generation(step),
        step=int(step),
    )


def trajectory_params(call: Call, *, state: BatchState):
    """Return the unique latent trajectory params assigned to a call."""
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
    """Return the unique reconstruction params assigned to a call."""
    selected = tuple(
        params
        for params in state.batch.decode_ranges
        if params.request_key == call.request_key
        and params.call_id == call.call_id
    )
    if len(selected) != 1:
        raise invalid_descriptor("video decode call has no exact decode params")
    return selected[0]


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


def video_state(runner: ModelExecutor, request: Request) -> DiffusionState:
    """Return the admitted video request's state, creating it on first use."""
    size = video_shape(runner, request.admission)
    trajectory = request.diffusion
    if trajectory is None:
        trajectory = open_state(runner, size)
        request.diffusion = trajectory
    if (
        not isinstance(trajectory, DiffusionState)
        or trajectory.slot is None
        or trajectory.size != size
    ):
        raise invalid_descriptor(
            "video request changed its admitted numerical dimensions"
        )
    return trajectory


def begin_noise(
    runner: ModelExecutor, request: Request, request_pool: RequestPool
) -> None:
    """Stage an admitted video request's host inputs off the service thread.

    The seeded draw and the request's state tables depend only on the seed
    and the admitted size, so they are prepared on the rank's noise thread
    while the service thread launches other device work, this request's text
    encoding or another request's denoising steps, and latent preparation
    waits for them. A rank that does not denoise stages none.
    """
    media = request.admission.diffusion
    if media is None or runner.noise_draws is None or not runner.state_buffers:
        return
    trajectory = video_state(runner, request)
    slot = slot_ladder(trajectory)
    if "denoising" not in slot.tensors:
        slot.tensors["denoising"] = request_pool.storage.tensors(
            request.request_pool_idx
        ).view(runner.media_builder.buffers(trajectory.size))
    slot.staging = runner.noise_draws.reserve().submit(
        runner.media_builder.stage_request,
        trajectory.size,
        slot.tensors["denoising"],
        seed=media.seed,
    )


def execute(
    call: Call,
    *,
    state: BatchState,
    tensor_store: TensorStore,
    publication_transports: Mapping[str, Transport],
    request_pool: RequestPool,
    model_runner: ModelExecutor,
) -> PendingOutput:
    """Land one ready video call without constructing a model input row.

    Latent preparation, denoising steps and decode rounds run here on the
    device; a video decode round also converts its media unit to RGB and
    publishes it as a host product for the host ranks that encode it.

    Preparation and each denoising step leave a ``LatentUpdate`` on the
    returned output, which the batch commit applies to the latent pool; a
    denoising step also advances the output's ``flow_step``. Nothing here
    commits request progress. Returns the call's ``PendingOutput`` with an
    ``OK`` status and its publications. Raises ``invalid_descriptor`` when
    the call, its inputs, parameters or progress disagree with the admitted
    request or this rank's model, and ``RuntimeError`` when the rank lacks
    the call's storage or capability or a module returns no statistics;
    errors of the executor, tensor store and latent pool calls propagate.
    """
    if model_runner.video_postprocessor is None:
        raise invalid_descriptor("video execution requires a video model")
    request = state.pending_output(call.request_key.request_id)
    media = request.request.admission.diffusion
    if media is None:
        raise invalid_descriptor("video call has no admitted media dimensions")
    numerical_shape = video_shape(model_runner, request.request.admission)

    trajectory = video_state(model_runner, request.request)

    slot, context = prepare_call(
        model_runner,
        trajectory,
        call,
        request_pool.storage.tensors(request.request.request_pool_idx)
        if model_runner.state_buffers
        else None,
    )

    from uniserve_worker.execution import transfer

    builder = model_runner.media_builder
    pool = model_runner.latent_pool
    slot_index = request.request.request_pool_idx
    products: tuple[TensorPublication, ...] = ()
    if call.kind is MediaCall.LATENT_PREPARATION:
        # Batch preparation validated the call's pages and interval.
        params = trajectory_params(call, state=state)
        assert pool is not None
        # The text features, then a conditioned request's condition latents:
        # every visual round's rows, then every audio track's
        # (``conditions.condition_latents``).
        reads = []
        for product in call.inputs:
            read = tensor_store.consume(
                product,
                consumer_call_id=call.call_id,
                device=model_runner.call_devices(call)[0],
            )
            request.device_reads.append(read)
            if read.region is not None:
                raise invalid_descriptor(
                    "video preparation requires complete input coverage"
                )
            reads.append(read)
        video_inputs = request.request.admission.video
        conditioned = video_inputs is not None and bool(video_inputs.conditions)
        if not reads or (len(reads) > 1) != conditioned:
            raise invalid_descriptor(
                "video preparation reads its conditioning and its conditions"
            )
        conditioning = reads[0]

        # Without a staging task from ``begin_noise``, the seeded draw and
        # the request's tables are staged here on the service thread.
        encoded = conditioning.tensor
        staging = slot_ladder(trajectory).staging
        if staging is None:
            builder.stage_request(numerical_shape, slot, seed=media.seed)
        else:
            staging.result()

        # The initial samples are copied straight into the request's pages
        # of the bank a fresh trajectory starts in, overlapping the
        # conditioning encoder; the batch commit publishes them at step zero.
        bank = pool.initial_bank(
            slot_index, params.page_table, latent_units=params.latent_units
        )
        initial = builder.initialize(
            numerical_shape,
            slot,
            builder.sample_views(
                numerical_shape, pool.bank_view(bank, params.page_table)
            ),
            constants=context.constants,
            workspace=context.workspace,
        )
        with model_runner.preparing_inputs(initial):
            if "conditioning" in model_runner.encoder_kinds:
                if encoded is None:
                    raise invalid_descriptor(
                        "conditioning module has no input Tensor"
                    )
                result = model_runner.encode_conditioning(encoded)
                if result.stats is None:
                    raise RuntimeError(
                        "module output has no execution statistics"
                    )
                state.forward_stats.append(result.stats)
                if len(result.values) != 1:
                    raise invalid_descriptor(
                        "conditioning computation must return one Tensor"
                    )
                builder.store_conditioning(
                    numerical_shape, slot, result.values[0]
                )
            if conditioned:
                from uniserve_worker.execution.conditions import (
                    condition_latents,
                )

                assert video_inputs is not None
                if any(read.tensor is None for read in reads[1:]):
                    raise invalid_descriptor(
                        "condition latents have no input Tensor"
                    )
                # The condition rows follow the stored prompt in the
                # retained conditioning, written on the same stream.
                builder.encode_conditions(
                    numerical_shape,
                    slot,
                    condition_latents(
                        video_inputs, tuple(read.tensor for read in reads[1:])
                    ),
                )
        request.latent.update = sample_update(
            slot_index, params, step=0, previous=None
        )

    elif call.kind is MediaCall.DENOISING:
        params = trajectory_params(call, state=state)
        start_step, step_count = int(params.start_step), int(params.step_count)
        if start_step != request.progress.flow_step:
            raise invalid_descriptor(
                "denoising does not begin at the selected request step"
            )
        if step_count != 1:
            raise invalid_descriptor(
                "video denoising requires one selected numerical step"
            )

        # ``step_banks`` checks the slot's committed trajectory against
        # ``start_step``, its generation and the call's parameters, and names
        # the bank holding it; the runner writes the successor to the other.
        assert pool is not None
        source, _ = pool.step_banks(
            slot_index,
            params.page_table,
            step=start_step,
            generation=sample_generation(start_step),
            latent_units=params.latent_units,
            height=params.height,
            width=params.width,
        )

        # The ladder is bound over the request's slot views and the
        # runner's samples once, in the request's capacity layout, and
        # replayed by every later step.
        slot_state = slot_ladder(trajectory)
        layout = builder.layout(numerical_shape)
        diffusion = model_runner.diffusion
        if slot_state.ladder is None or not diffusion.binds(slot_state.ladder):
            samples = builder.sample_views(numerical_shape, diffusion.samples)
            slot_state.ladder = diffusion.bind(
                layout,
                tuple(
                    builder.bind(
                        numerical_shape,
                        slot,
                        samples,
                        trajectory.schedules,
                        index,
                    )
                    for index in range(builder.num_steps)
                ),
                trajectory.schedules,
                state=slot,
                slot=slot_index,
                pages=params.page_table,
            )
        result = model_runner.run_denoising(
            slot_state.ladder, start_step, source
        )
        if result.stats is None:
            raise RuntimeError("module output has no execution statistics")
        state.forward_stats.append(result.stats)
        request.latent.update = sample_update(
            slot_index,
            params,
            step=start_step + step_count,
            previous=start_step,
        )

        # Only the call that completes denoising may declare products, and
        # it publishes the step's result, the final samples.
        if call.outputs:
            if start_step + step_count != media.num_inference_steps:
                raise invalid_descriptor(
                    "final latent products require completed denoising"
                )
            products = transfer.publish_tensors(
                call,
                result.values,
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
        # Each rank of a distributed decoder binding reconstructs its
        # contiguous run of the declared range's media units, on either
        # track (``ComponentBinding.media_units``).
        if track is MediaTrack.VIDEO:
            windows = model_runner.video_decoder.frame_slices(media.num_frames)
            # A video decoder places one native unit per rank
            # (``validate_components``), so its run is a single unit.
            unit = assigned_units(
                model_runner, call.component, cursor, count, len(windows)
            ).start
            window = windows[unit]
            output = video.Config(media.num_frames, media.canvas)
            decoded = decode_video_unit(
                model_runner, call.component, read.tensor, window, output
            )
            if decoded.stats is None:
                raise RuntimeError("module output has no execution statistics")
            state.forward_stats.append(decoded.stats)
            if len(decoded.values) != 1 or decoded.values[0].shape[0] != 1:
                raise invalid_descriptor(
                    "video decoding reconstructs exactly one media unit"
                )

            # The decoded window becomes an RGB media unit on this rank,
            # cross-faded with the neighbouring unit's tail, and that unit
            # is the product a host rank encodes.
            decoder = model_runner.video_decoder
            unit_outputs = (decoder.place(decoded.values[0], window, output),)
            processed = model_runner.run_module(
                call.component,
                unit_outputs,
                method="forward",
                size=output,
                frames=(window,),
                sizes=(output,),
                state=slot,
                unit_count=count,
            )
            state.forward_stats.append(processed.stats)
            # Concatenation borrows one flat byte span; the product row is
            # the unit's frames at the output raster, and a unit shorter than
            # the longest fills its row's leading frames.
            frames = window.stop - window.start
            value = concatenate_views(processed.values).view(
                1, frames, media.height, media.width, 3
            )
            longest = max(
                span.stop - span.start
                for span in decoder.frame_slices(media.num_frames)
            )
            if frames < longest:
                row = value.new_zeros((1, longest, *value.shape[2:]))
                row[:, :frames].copy_(value)
                value = row
            values: tuple[torch.Tensor, ...] = (value,)
        else:
            result = decode_audio(
                model_runner,
                call.component,
                read.tensor,
                audio_samples(model_runner, media.num_frames),
                cursor=cursor,
                count=count,
            )
            if result.stats is None:
                raise RuntimeError("module output has no execution statistics")
            state.forward_stats.append(result.stats)
            values = result.values
        # Decoded media units are host products: a host rank's encoder reads
        # them in place from the segment this rank publishes, over the host
        # mechanism of its edges.
        products = transfer.publish_tensors(
            call,
            values,
            tensor_store=tensor_store,
            publication_transports=publication_transports,
            state=state,
            host=True,
        )
    else:
        raise invalid_descriptor(f"unsupported video call {call.kind!r}")

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
]
