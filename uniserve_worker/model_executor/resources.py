"""Size per-request media storage and the layouts of call products.

These functions derive, from a rank's bound capabilities and its
``MediaBuilder``, the request-slot fields a media rank reserves and the
``OutputLayout`` of each product a call publishes to other components.
Worker bootstrap (``bootstrap.capacity``, ``bootstrap.report`` and
``bootstrap.outputs``) and ``ModelExecutor`` consume them.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch

from uniserve.media import image, video
from uniserve.model import (
    AudioDecoder,
    Denoiser,
    TextEncoder,
    VideoDecoder,
    VideoDenoiser,
    VideoPostprocessor,
)
from uniserve.tensors import BufferConfig, OutputLayout
from uniserve_worker.config.execution import WorkerConfig
from uniserve_worker.model_executor.component_binding import (
    Call,
    ComponentBinding,
)
from uniserve_worker.model_executor.media_inputs import MediaBuilder, bound


def media_state_buffers(
    bindings: Mapping[str, ComponentBinding],
    builder: MediaBuilder | None,
) -> dict[str, BufferConfig]:
    """Reserve per-request state and transfer staging on participating ranks.

    A rank that denoises holds the denoiser's tables, conditioning and host
    staging, and a rank that post-processes video holds its overlap state.
    The denoiser's samples live in the latent pool instead. Returns an empty
    mapping without a media builder.

    Raises:
        ValueError: Two calls declare different fields under one name.
    """
    if builder is None:
        return {}

    result: dict[str, BufferConfig] = {}
    for binding in bindings.values():
        for call in binding.calls:
            if (
                isinstance(call.module, Denoiser)
                and call.entry_point.method == "forward"
            ):
                fields = builder.capacity_buffers()
            elif isinstance(call.module, VideoPostprocessor):
                fields = _postprocess_state(call.module, builder)
            else:
                continue

            for name, field in fields.items():
                if name in result and result[name] != field:
                    raise ValueError(
                        f"media request fields disagree about {name!r}"
                    )
                result[name] = field
    return result


def _postprocess_state(
    postprocessor: VideoPostprocessor, builder: MediaBuilder
) -> dict[str, BufferConfig]:
    """Bound the post-processor's overlap state over every served raster."""
    result: dict[str, BufferConfig] = {}
    for frame in builder.frame_sizes:
        size = video.Config(builder.max_frames, frame)
        for name, field in postprocessor.state_buffers(size).items():
            result[name] = (
                bound(result[name], field) if name in result else field
            )
    return result


def largest_frame(decoder: VideoDecoder) -> image.Config:
    """Return the decoder's raster with the most pixels, the first among equals.

    A declared video product is sized at it: every served raster fits its
    pixel count, which the engine checks when it reserves a request's rows.
    """
    return max(
        decoder.frame_sizes, key=lambda frame: frame.height * frame.width
    )


def holds_samples(
    state_buffers: Mapping[str, BufferConfig], builder: MediaBuilder | None
) -> bool:
    """Whether a rank's request storage holds a standalone denoiser's state.

    Such a rank advances the denoiser's samples, and its latent pool holds
    them.
    """
    return builder is not None and set(builder.capacity_buffers()) <= set(
        state_buffers
    )


def decoded_units_layout(
    decoder: VideoDecoder,
    num_frames: int,
    frame: image.Config | None = None,
) -> OutputLayout:
    """Describe a video decoding round's product: RGB media units.

    Each row holds one media unit's frames at ``frame``, by default the
    decoder's largest raster, which bounds every request's rows; a unit
    shorter than the longest fills its row's leading frames, and the unit
    division names how many. The rows are host products a host rank's
    encoder reads in place.
    """
    windows = decoder.frame_slices(num_frames)
    frames = max(window.stop - window.start for window in windows)
    frame = largest_frame(decoder) if frame is None else frame
    shape = (len(windows), frames, frame.height, frame.width, 3)
    return OutputLayout(
        shape,
        torch.uint8,
        tuple(slice(0, extent) for extent in shape),
        variable_axes=(0,),
        value_range=(0, 255),
    )


def encoded_units_layout(
    decoder: VideoDecoder, num_frames: int
) -> OutputLayout:
    """Describe a video encoding round's product: framed encoded unit rows.

    An encoded unit's length is not known when its row is reserved, so a row
    is bounded by the largest unit at any of the decoder's rasters and
    carries its own length. The row's extent is static, so every request
    reserves the same row whatever raster it selects.
    """
    from uniserve_worker.media.mux import encoded_unit_bytes

    windows = decoder.frame_slices(num_frames)
    frames = max(window.stop - window.start for window in windows)
    row = max(
        encoded_unit_bytes(frames, size.height, size.width)
        for size in decoder.frame_sizes
    )
    return OutputLayout(
        (len(windows), row),
        torch.uint8,
        (slice(0, len(windows)), slice(0, row)),
        variable_axes=(0,),
    )


def output_layouts(
    config: WorkerConfig,
    call: Call,
    *,
    builder: MediaBuilder | None = None,
    clock: VideoPostprocessor | None = None,
    frames: int | None = None,
    frame: image.Config | None = None,
    prompt_tokens: int | None = None,
) -> Mapping[str, OutputLayout]:
    """Describe the products one call publishes, keyed by product name.

    ``frames``, ``frame`` and ``prompt_tokens`` size the layout for one request.
    They default to the admitted maxima, the largest raster for a video
    decoder's product and the first served raster otherwise. ``clock`` is the
    video post-processor whose frame rate relates audio samples to video frames;
    with a clock, a video decoder's product is its RGB media units
    (``decoded_units_layout``) rather than its own declared layout.

    The mapping is empty for a video post-processor, for a call that is
    neither a text encoder's ``encode`` nor a denoiser, video decoder or
    audio decoder call, for a video decoder with neither ``frames`` nor a
    builder, and for a denoiser or audio decoder without a builder.

    Raises:
        ValueError: For a denoiser or audio decoder, the requested frames
            or prompt exceed the builder's capacity; a denoiser on a media
            timeline is not a ``VideoDenoiser``; or an audio decoder has no
            ``clock``.
    """
    component = call.module
    if (
        isinstance(component, TextEncoder)
        and call.entry_point.method == "encode"
    ):
        return component.output_layout(
            config.max_sequence_tokens
            if prompt_tokens is None
            else prompt_tokens,
            getattr(torch, config.model_dtype),
        )

    if isinstance(component, VideoPostprocessor):
        # The post-processor's RGB media units are the decoding call's
        # product, declared with the decoder below.
        return {}

    if not isinstance(component, (Denoiser, VideoDecoder, AudioDecoder)):
        return {}
    if isinstance(component, VideoDecoder):
        if frames is None:
            if builder is None:
                # Standalone decoders have no serving timeline bound. Their
                # caller supplies the exact frame range with the invocation.
                return {}
            count = builder.max_frames
        else:
            count = frames
        if clock is None:
            return component.output_layout(
                video.Config(
                    count, largest_frame(component) if frame is None else frame
                )
            )
        return {"video": decoded_units_layout(component, count, frame)}

    if builder is None:
        # Image execution returns its features and decoded raster through the
        # token/image protocol rather than persistent inter-component products.
        return {}

    size = builder.size(
        builder.max_frames if frames is None else frames,
        builder.frame_sizes[0] if frame is None else frame,
        config.max_sequence_tokens if prompt_tokens is None else prompt_tokens,
    )
    if isinstance(component, Denoiser):
        # Only a video denoiser shares the media timeline the builder sizes.
        # Its rows on this rank follow the capacity layout the request
        # evaluates in.
        if not isinstance(component, VideoDenoiser):
            raise ValueError("a media timeline's denoiser is a video denoiser")
        return component.output_layout(builder.layout(size))

    # The product holds the whole decoded track, which the decoder defines
    # from the latent timeline generated with the video's frames.
    if clock is None:
        raise ValueError("audio output layout requires its media clock")
    samples = component.track_samples(size.num_frames, clock.frame_rate)
    return component.output_layout(samples)
