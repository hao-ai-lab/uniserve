"""Condition products of a video request: their layouts and encodings.

A conditioned video request (``fl2va``, ``ref2va``) runs three calls before
latent preparation, each publishing products the engine reserved at
admission with the request's exact leading extents:

- ``MediaReading`` (``uniserve_worker.execution.media_reader``) decodes the
  condition media on a host rank: ``condition_pixels``, the RGB24 frames the
  video encoder encodes; ``condition_samples``, the PCM the audio encoder
  encodes; and ``vision_pixels``, the vision encoder's packed patch rows.
- ``VisionEncoding`` (``encode_vision``) turns the patch rows into
  ``vision_features``, one row per vision placeholder of the presentation,
  which text encoding splices into the prompt (``vision_inputs``).
- ``LatentEncoding`` (``encode_latents``) turns the pixels into
  ``condition_video_latents`` in rounds of temporal units, each round
  covering the next units its component's ranks encode together, and the
  PCM into ``condition_audio_latents`` in one call.

Every product concatenates its conditions in request order. The server
presents the vision blocks in request order too, so the vision features are
in the order of the presentation's placeholders. Latent preparation reads
the text features and every condition latent (``condition_latents``).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import torch

from uniserve.model import (
    AudioEncoder,
    Condition,
    VideoEncoder,
    VisionInput,
)
from uniserve.tensors import OutputLayout
from uniserve_worker.errors import invalid_descriptor
from uniserve_worker.protocol.call import MediaCall
from uniserve_worker.protocol.video import VideoAdmission
from uniserve_worker.storage.tensor_store import device_product_storage

if TYPE_CHECKING:
    from uniserve_worker._uniserve_ipc import BatchState
    from uniserve_worker.execution.model_executor import ModelExecutor
    from uniserve_worker.execution.output import PendingOutput
    from uniserve_worker.model_executor.component_binding import (
        ComponentBinding,
    )
    from uniserve_worker.model_executor.output import ExecutionOutput
    from uniserve_worker.protocol.batch import DecodeRange
    from uniserve_worker.protocol.call import Call
    from uniserve_worker.protocol.tensor import OutputInfo
    from uniserve_worker.storage.tensor_store import TensorStore
    from uniserve_worker.transport.interface import Transport

#: RGB24 pixels of every visual condition, ``[pixels, 3]`` uint8.
CONDITION_PIXELS = "condition_pixels"
#: Model-rate stereo PCM of every audio track, ``[samples, 2]`` FP32.
CONDITION_SAMPLES = "condition_samples"
#: The vision encoder's packed patch rows of every vision block.
VISION_PIXELS = "vision_pixels"
#: The vision encoder's rows, one per vision placeholder.
VISION_FEATURES = "vision_features"
#: Denoiser rows of every visual condition.
CONDITION_VIDEO_LATENTS = "condition_video_latents"
#: Denoiser rows of every audio track.
CONDITION_AUDIO_LATENTS = "condition_audio_latents"

#: Products whose extents follow a request's conditions.
CONDITION_PRODUCTS = frozenset(
    {
        CONDITION_PIXELS,
        CONDITION_SAMPLES,
        VISION_PIXELS,
        VISION_FEATURES,
        CONDITION_VIDEO_LATENTS,
        CONDITION_AUDIO_LATENTS,
    }
)


def video_admission(request: PendingOutput) -> VideoAdmission:
    """Return a conditioned video request's admission.

    Raises:
        WorkerError: ``invalid_descriptor`` for a request without conditions.
    """
    video = request.request.admission.video
    if video is None or not video.conditions:
        raise invalid_descriptor("a condition call requires conditions")
    return video


def library_conditions(video: VideoAdmission) -> tuple[Condition, ...]:
    """Describe a request's conditions as the denoiser sizes them."""
    return tuple(
        Condition(
            condition.role,
            condition.pixels,
            0 if condition.audio is None else condition.audio.samples,
        )
        for condition in video.conditions
    )


def condition_units(video: VideoAdmission) -> tuple[tuple[int, int], ...]:
    """List the visual encoding units as ``(condition, unit)`` pairs.

    The units of every condition with pixels follow each other in request
    order; a round of latent encoding covers a run of them.
    """
    return tuple(
        (index, unit)
        for index, condition in enumerate(video.conditions)
        for unit in range(len(condition.latent_units))
    )


def _unit_rows(video: VideoAdmission) -> tuple[int, ...]:
    """Denoiser rows of every visual encoding unit, in encoding order."""
    return tuple(
        rows
        for condition in video.conditions
        for rows in condition.latent_units
    )


def _round_units(video: VideoAdmission, decode: DecodeRange) -> range:
    """The visual units a latent encoding round covers.

    Raises:
        WorkerError: ``invalid_descriptor`` when the round lies outside the
            request's units.
    """
    total = len(_unit_rows(video))
    cursor, count = int(decode.cursor), int(decode.max_units)
    if cursor < 0 or count < 1 or cursor + count > total:
        raise invalid_descriptor(
            "a latent encoding round exceeds the request's condition units"
        )
    return range(cursor, cursor + count)


def condition_encoder(
    call: Call, outputs: Mapping[str, tuple[OutputInfo, ...]]
) -> type[VideoEncoder] | type[AudioEncoder] | None:
    """Return the condition encoder a latent encoding call runs.

    A visual round publishes ``condition_video_latents`` and runs the video
    encoder; the audio call publishes ``condition_audio_latents`` and runs
    the audio encoder. None for a latent encoding call of another product,
    such as an image's.
    """
    declared = outputs.get(call.component, ())
    names = {
        declared[output.output_index].name
        for output in call.outputs
        if output.output_index < len(declared)
    }
    if CONDITION_VIDEO_LATENTS in names:
        return VideoEncoder
    if CONDITION_AUDIO_LATENTS in names:
        return AudioEncoder
    return None


def condition_layout(
    info: OutputInfo,
    video: VideoAdmission,
    decode: DecodeRange | None,
    binding: ComponentBinding | None,
) -> OutputLayout | None:
    """Lay out one request's condition product on this rank.

    The leading axis holds the request's rows of the product (see the module
    docstring); the trailing extents and dtype are the declared ones. A
    visual latent round's product holds only the round's units, and a rank
    of a temporally distributed latent encoder publishes the rows of the
    units ``ComponentBinding.media_units`` deals it within the round. Returns
    None for a rank that publishes none of the product.

    Raises:
        WorkerError: ``invalid_descriptor`` when a visual latent round names
            no units, or units outside the request's.
    """
    conditions = video.conditions
    name = info.name
    if name == CONDITION_PIXELS:
        rows = sum(condition.pixel_bytes // 3 for condition in conditions)
    elif name == CONDITION_SAMPLES:
        rows = sum(
            condition.audio.samples
            for condition in conditions
            if condition.audio is not None
        )
    elif name == VISION_PIXELS:
        rows = sum(
            condition.vision.patches
            for condition in conditions
            if condition.vision is not None
        )
    elif name == VISION_FEATURES:
        rows = sum(
            condition.vision.tokens
            for condition in conditions
            if condition.vision is not None
        )
    elif name == CONDITION_AUDIO_LATENTS:
        rows = sum(condition.audio_rows for condition in conditions)
    elif name != CONDITION_VIDEO_LATENTS:
        raise invalid_descriptor(f"{name} is not a condition product")
    else:
        if decode is None:
            raise invalid_descriptor(
                "a visual latent round requires its unit range"
            )
        covered = _round_units(video, decode)
        rows = sum(_unit_rows(video)[covered.start : covered.stop])
    local = slice(0, rows)

    if name in (CONDITION_VIDEO_LATENTS, CONDITION_AUDIO_LATENTS):
        # Units are dealt to a distributed encoder's ranks from the round's
        # first unit; the audio call is one unit, the first rank's.
        units = (
            _unit_rows(video)[covered.start : covered.stop]
            if name == CONDITION_VIDEO_LATENTS
            else (rows,)
        )
        run = (
            range(len(units))
            if binding is None
            else binding.media_units(0, len(units))
        )
        if not run:
            return None
        local = slice(sum(units[: run.start]), sum(units[: run.stop]))

    trailing = tuple(int(dim.extent) for dim in info.shape_bound.dims[1:])
    shape = (rows, *trailing)
    return OutputLayout(
        shape,
        getattr(torch, device_product_storage(info.dtype)[0]),
        (local, *(slice(0, extent) for extent in trailing)),
        variable_axes=(0,),
        value_range=(0, 255) if name == CONDITION_PIXELS else None,
    )


def _consume_whole(
    call: Call,
    index: int,
    *,
    state: BatchState,
    tensor_store: TensorStore,
    model_runner: ModelExecutor,
) -> torch.Tensor:
    """Read one of a call's inputs whole on the call's compute device.

    Raises:
        WorkerError: ``invalid_descriptor`` when the input is missing or
            arrives only in part.
    """
    if index >= len(call.inputs):
        raise invalid_descriptor("a condition call lacks one of its inputs")
    request = state.pending_output(call.request_key.request_id)
    read = tensor_store.consume(
        call.inputs[index],
        consumer_call_id=call.call_id,
        device=model_runner.call_devices(call)[0],
    )
    request.device_reads.append(read)
    if read.region is not None or read.tensor is None:
        raise invalid_descriptor("a condition call requires its whole input")
    return read.tensor


def _outcome(
    request: PendingOutput,
    result: ExecutionOutput,
    state: BatchState,
) -> PendingOutput:
    """Record a condition call's numerical execution statistics."""
    if result.stats is None:
        raise RuntimeError("module output has no execution statistics")
    state.forward_stats.append(result.stats)
    return request


def encode_vision(
    call: Call,
    *,
    state: BatchState,
    tensor_store: TensorStore,
    export_transports: Mapping[str, Transport],
    model_runner: ModelExecutor,
) -> PendingOutput:
    """Encode a request's vision blocks into one row per placeholder.

    The call reads ``vision_pixels``, one packed sample per condition the
    conditioner reads, and publishes ``vision_features``: every sample's
    rows, in request order.

    Raises:
        WorkerError: ``invalid_descriptor`` when the inputs or outputs
            disagree with the request's conditions.
    """
    request = state.pending_output(call.request_key.request_id)
    video = video_admission(request)
    if len(call.outputs) != 1:
        raise invalid_descriptor(
            "vision encoding publishes one product of the request's blocks"
        )
    patches = _consume_whole(
        call,
        0,
        state=state,
        tensor_store=tensor_store,
        model_runner=model_runner,
    )
    features, result = vision_features(video, patches, model_runner)
    state.export_tensors(
        call.request_key.request_id,
        (features,),
        tensor_store,
        export_transports,
    )
    return _outcome(request, result, state)


def vision_features(
    video: VideoAdmission, patches: torch.Tensor, model_runner: ModelExecutor
) -> tuple[torch.Tensor, ExecutionOutput]:
    """Encode a request's vision blocks into one row per placeholder.

    ``patches`` are the request's ``vision_pixels``: the conditioner's packed
    patch rows of every condition it reads, in request order. Returns their
    features, one row per vision placeholder in request order, and the
    encoder's output.

    Raises:
        WorkerError: ``invalid_descriptor`` when the request reads no vision
            block, or the patch rows or features disagree with its blocks.
    """
    views = tuple(
        condition.vision
        for condition in video.conditions
        if condition.vision is not None
    )
    if not views or patches.shape[0] != sum(view.patches for view in views):
        raise invalid_descriptor(
            "vision patch rows disagree with the request's blocks"
        )

    # One packed sample per condition: the processor's patch rows of its
    # (time, height, width) grid, which the encoder also takes as a [1, 3]
    # device tensor.
    samples = patches.split([view.patches for view in views])
    grids = tuple(
        torch.tensor([view.grid], dtype=torch.int64, device=patches.device)
        for view in views
    )
    result = model_runner.encode_vision(
        VisionInput(samples, grids, tuple(view.grid for view in views))
    )
    features = torch.cat(result.values)
    if features.shape[0] != sum(view.tokens for view in views):
        raise invalid_descriptor(
            "vision features disagree with the request's placeholders"
        )
    return features, result


def vision_grids(
    video: VideoAdmission,
) -> dict[str, tuple[tuple[int, int, int], ...]]:
    """Return the patch grids of a request's vision blocks by modality.

    The result holds ``image_grids`` and ``video_grids`` in request order,
    as ``ModelExecutor.encode_text`` takes them; a video's grid covers all
    of its blocks, and an image is one block.
    """
    views = tuple(
        (condition.vision, condition.video is not None)
        for condition in video.conditions
        if condition.vision is not None
    )
    return {
        "image_grids": tuple(view.grid for view, moving in views if not moving),
        "video_grids": tuple(view.grid for view, moving in views if moving),
    }


def vision_inputs(
    call: Call,
    *,
    state: BatchState,
    tensor_store: TensorStore,
    model_runner: ModelExecutor,
) -> dict[str, object]:
    """Return text encoding's vision inputs for a conditioned prompt.

    A text encoding call of a request with vision blocks reads their
    ``vision_features``. The result holds them as ``visual`` with the
    blocks' grids (``vision_grids``), the keywords
    ``ModelExecutor.encode_text`` takes; it is empty for a call without
    inputs.

    Raises:
        WorkerError: ``invalid_descriptor`` when the call reads features of
            a request without vision blocks, or more than one input.
    """
    if not call.inputs:
        return {}
    request = state.pending_output(call.request_key.request_id)
    video = video_admission(request)
    if len(call.inputs) != 1:
        raise invalid_descriptor("text encoding reads one vision product")
    visual = _consume_whole(
        call,
        0,
        state=state,
        tensor_store=tensor_store,
        model_runner=model_runner,
    )
    return {"visual": visual, **vision_grids(video)}


def _condition_offsets(video: VideoAdmission) -> tuple[int, ...]:
    """Each condition's first pixel in ``condition_pixels``."""
    offsets, cursor = [], 0
    for condition in video.conditions:
        offsets.append(cursor)
        cursor += condition.pixel_bytes // 3
    return (*offsets, cursor)


def encode_units(
    video: VideoAdmission,
    run: range,
    pixels: torch.Tensor,
    model_runner: ModelExecutor,
) -> tuple[torch.Tensor, ExecutionOutput]:
    """Encode a run of a request's visual units into their rows.

    ``run`` indexes ``condition_units``; ``pixels`` is the request's
    ``condition_pixels``. Returns the units' rows concatenated in unit order
    and the encoder's output, one result per unit.

    Raises:
        WorkerError: ``invalid_descriptor`` when the pixels or the encoded
            rows disagree with the request's conditions.
    """
    offsets = _condition_offsets(video)
    if pixels.shape != (offsets[-1], 3):
        raise invalid_descriptor(
            "condition pixels disagree with the request's conditions"
        )

    encoder = model_runner.component(
        MediaCall.LATENT_ENCODING, capability_type=VideoEncoder
    )
    units = condition_units(video)
    values, frames, counts, planned = [], [], [], []
    for index, unit in (units[position] for position in run):
        condition = video.conditions[index]
        size = condition.pixels
        if size is None:
            raise invalid_descriptor("a visual unit belongs to no pixels")
        # A unit's frames are whole [height, width, 3] rasters of its
        # condition's pixels, frame-major.
        window = encoder.frame_slices(size.num_frames)[unit]
        area = size.frame.height * size.frame.width
        values.append(
            pixels[
                offsets[index] + window.start * area : offsets[index]
                + window.stop * area
            ].view(
                window.stop - window.start,
                size.frame.height,
                size.frame.width,
                3,
            )
        )
        frames.append(window)
        counts.append(size.num_frames)
        planned.append(condition.latent_units[unit])

    result = model_runner.run_encoder(
        "video_condition",
        *values,
        frames=tuple(frames),
        num_frames=tuple(counts),
    )
    if [value.shape[0] for value in result.values] != planned:
        raise invalid_descriptor(
            "encoded condition units disagree with their planned rows"
        )
    return torch.cat(result.values), result


def encode_tracks(
    video: VideoAdmission,
    samples: torch.Tensor,
    model_runner: ModelExecutor,
) -> tuple[torch.Tensor, ExecutionOutput]:
    """Encode every audio track of a request into its rows.

    ``samples`` is the request's ``condition_samples``. Returns every
    track's rows in request order and the encoder's output, one result per
    track.

    Raises:
        WorkerError: ``invalid_descriptor`` when the samples or the encoded
            rows disagree with the request's audio tracks.
    """
    tracks = tuple(
        condition
        for condition in video.conditions
        if condition.audio is not None
    )
    lengths = [track.audio.samples for track in tracks]
    if samples.shape != (sum(lengths), 2):
        raise invalid_descriptor(
            "condition samples disagree with the request's audio tracks"
        )
    result = model_runner.run_encoder(
        "audio_condition", *samples.split(lengths)
    )
    if [value.shape[0] for value in result.values] != [
        track.audio_rows for track in tracks
    ]:
        raise invalid_descriptor(
            "encoded audio tracks disagree with their planned rows"
        )
    return torch.cat(result.values), result


def encode_latents(
    call: Call,
    *,
    state: BatchState,
    tensor_store: TensorStore,
    export_transports: Mapping[str, Transport],
    model_runner: ModelExecutor,
) -> PendingOutput:
    """Encode one round of a request's condition latents.

    A visual round reads ``condition_pixels`` and publishes the rows of the
    units this rank takes of the round; the audio call reads
    ``condition_samples`` and publishes every track's rows. The call's
    decode range names the round's units, counted from the request's first
    visual unit, or the audio call's one unit.

    Raises:
        WorkerError: ``invalid_descriptor`` when the inputs, the round or
            the encoded rows disagree with the request's conditions.
    """
    from uniserve_worker.execution.media import decode_range

    request = state.pending_output(call.request_key.request_id)
    video = video_admission(request)
    encoder = condition_encoder(call, model_runner.outputs)
    if encoder is None or len(call.outputs) != 1:
        raise invalid_descriptor(
            "latent encoding publishes one condition latent product"
        )
    source = _consume_whole(
        call,
        0,
        state=state,
        tensor_store=tensor_store,
        model_runner=model_runner,
    )
    if encoder is VideoEncoder:
        # This rank encodes its share of the round's units, which follow
        # the earlier rounds' units.
        covered = _round_units(video, decode_range(call, state=state))
        run = model_runner.bindings[call.component].media_units(
            covered.start, len(covered)
        )
        if not run:
            raise invalid_descriptor(
                "a latent encoding round assigns this rank no unit"
            )
        rows, result = encode_units(video, run, source, model_runner)
    else:
        rows, result = encode_tracks(video, source, model_runner)
    state.export_tensors(
        call.request_key.request_id,
        (rows,),
        tensor_store,
        export_transports,
    )
    return _outcome(request, result, state)


def condition_latents(
    video: VideoAdmission, reads: Sequence[torch.Tensor]
) -> tuple[torch.Tensor, ...]:
    """Split latent preparation's condition reads into per-condition latents.

    ``reads`` are the visual rounds' rows in round order, then, when any
    condition carries audio, every track's rows. The result lists each
    condition's latents in request order, its visual rows before its audio
    rows, as ``VideoDenoiser.encode_conditions`` takes them.

    Raises:
        WorkerError: ``invalid_descriptor`` when the reads disagree with the
            request's planned rows.
    """
    audio = any(condition.audio_rows for condition in video.conditions)
    rounds = tuple(reads[:-1] if audio else reads)
    visual = torch.cat(rounds) if rounds else None
    tracks = reads[-1] if audio else None
    video_rows = [condition.video_rows for condition in video.conditions]
    audio_rows = [condition.audio_rows for condition in video.conditions]
    if (0 if visual is None else visual.shape[0]) != sum(video_rows) or (
        0 if tracks is None else tracks.shape[0]
    ) != sum(audio_rows):
        raise invalid_descriptor(
            "condition latents disagree with the request's planned rows"
        )

    pixels = () if visual is None else visual.split(video_rows)
    sounds = () if tracks is None else tracks.split(audio_rows)
    result = []
    for index, condition in enumerate(video.conditions):
        if condition.video_rows:
            result.append(pixels[index])
        if condition.audio_rows:
            result.append(sounds[index])
    return tuple(result)


__all__ = [
    "CONDITION_AUDIO_LATENTS",
    "CONDITION_PIXELS",
    "CONDITION_PRODUCTS",
    "CONDITION_SAMPLES",
    "CONDITION_VIDEO_LATENTS",
    "VISION_FEATURES",
    "VISION_PIXELS",
    "condition_encoder",
    "condition_latents",
    "condition_layout",
    "condition_units",
    "encode_latents",
    "encode_tracks",
    "encode_units",
    "encode_vision",
    "library_conditions",
    "video_admission",
    "vision_features",
    "vision_grids",
    "vision_inputs",
]
