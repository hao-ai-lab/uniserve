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
    AudioEncoder,
    Denoiser,
    PatchEncoder,
    TextEncoder,
    VideoDecoder,
    VideoDenoiser,
    VideoEncoder,
    VideoPostprocessor,
)
from uniserve.tensors import BufferConfig, OutputLayout
from uniserve_worker.config.execution import WorkerConfig
from uniserve_worker.model_executor.component_binding import (
    Call,
    ComponentBinding,
)
from uniserve_worker.model_executor.media_inputs import MediaBuilder, envelope


def bounding_layout(layouts: tuple[OutputLayout, ...]) -> OutputLayout:
    """Combine one product's layouts at several sizes into a bound.

    Equal layouts bound themselves. Otherwise the result takes each
    dimension's largest extent and covers it whole; its variable axes are
    the layouts' own, and a request's exact extents come from its admission.

    Raises:
        ValueError: The layouts disagree on rank, dtype, variable axes or
            value range.
    """
    first = layouts[0]
    if all(layout == first for layout in layouts):
        return first
    if any(
        len(layout.shape) != len(first.shape)
        or layout.dtype != first.dtype
        or layout.variable_axes != first.variable_axes
        or layout.value_range != first.value_range
        for layout in layouts
    ):
        raise ValueError("bounded layouts must describe one product")
    shape = tuple(
        max(extents) for extents in zip(*(layout.shape for layout in layouts))
    )
    return OutputLayout(
        shape,
        first.dtype,
        tuple(slice(0, extent) for extent in shape),
        variable_axes=first.variable_axes,
        value_range=first.value_range,
    )


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
                # The overlap of the longest video at every admitted canvas.
                fields = envelope(
                    tuple(
                        call.module.state_buffers(size)
                        for size in builder.video_sizes()
                    )
                )
            else:
                continue

            for name, field in fields.items():
                if name in result and result[name] != field:
                    raise ValueError(
                        f"media request fields disagree about {name!r}"
                    )
                result[name] = field
    return result


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


#: The ``(height, width)`` axes of ``decoded_units_layout``, which hold the
#: request's canvas.
DECODED_UNITS_RASTER_AXES = (2, 3)


def decoded_units_layout(
    decoder: VideoDecoder, size: video.Config
) -> OutputLayout:
    """Describe a video decoding round's product: RGB media units.

    The product is ``[units, frames, height, width, 3]`` uint8. Each row
    holds one media unit's frames at the output raster
    (``DECODED_UNITS_RASTER_AXES``); a unit shorter than the longest fills
    its row's leading frames, and the unit division names how many. The rows
    are host products a host rank's encoder reads in place.
    """
    windows = decoder.frame_slices(size.num_frames)
    frames = max(window.stop - window.start for window in windows)
    shape = (len(windows), frames, size.frame.height, size.frame.width, 3)
    return OutputLayout(
        shape,
        torch.uint8,
        tuple(slice(0, extent) for extent in shape),
        variable_axes=(0,),
        value_range=(0, 255),
    )


def encoded_units_layout(
    decoder: VideoDecoder, size: video.Config
) -> OutputLayout:
    """Describe a video encoding round's product: framed encoded unit rows.

    An encoded unit's length is not known when its row is reserved, so a row
    is bounded by the largest unit of ``size`` and carries its own length.
    """
    from uniserve_worker.media.mux import encoded_unit_bytes

    windows = decoder.frame_slices(size.num_frames)
    row = encoded_unit_bytes(
        max(window.stop - window.start for window in windows),
        size.frame.height,
        size.frame.width,
    )
    return OutputLayout(
        (len(windows), row),
        torch.uint8,
        (slice(0, len(windows)), slice(0, row)),
        variable_axes=(0,),
    )


def _rows(shape: tuple[int, ...], dtype: torch.dtype) -> OutputLayout:
    """A product of rows whose count varies by request."""
    return OutputLayout(
        shape,
        dtype,
        tuple(slice(0, extent) for extent in shape),
        variable_axes=(0,),
    )


def condition_media_layouts(
    config: WorkerConfig,
    *,
    video_encoder: VideoEncoder,
    audio_encoder: AudioEncoder,
    vision: PatchEncoder,
    frame_counts: tuple[int, ...],
) -> dict[str, OutputLayout]:
    """Describe the media reader's products, bounded for every request.

    A request's condition rows are at most ``config.max_condition_rows`` and
    its presentation at most ``config.max_sequence_tokens`` tokens, which
    bound what it reads:

    - ``condition_pixels``: ``[pixels, 3]`` RGB24 of every visual condition.
      A condition's pixels are its rows times the pixels per row of its
      frame count, one of ``frame_counts``, so the bound is the condition
      rows times the largest such ratio, read from the video encoder's own
      layout (it does not depend on the raster).
    - ``condition_samples``: ``[samples, 2]`` FP32 PCM of every audio track.
      A track's stereo rows are twice its latent frames, each frame
      ``latent_rate`` samples.
    - ``vision_pixels``: the vision encoder's packed patch rows of every
      vision block, whose tokens all lie in the presentation.
    """
    rows = config.max_condition_rows
    # A 32-pixel square is the smallest raster whose rows are whole latent
    # patches, so its rows per frame count the rows of one patch.
    patch = image.Config(32, 32)
    pixels_per_row = max(
        -(
            -frames
            * patch.height
            * patch.width
            // video_encoder.output_layout(video.Config(frames, patch))[
                "video"
            ].shape[0]
        )
        for frames in frame_counts
    )
    return {
        "condition_pixels": _rows((rows * pixels_per_row, 3), torch.uint8),
        "condition_samples": _rows(
            (rows // 2 * audio_encoder.latent_rate, 2), torch.float32
        ),
        "vision_pixels": vision.pixels_layout(config.max_sequence_tokens),
    }


def output_layouts(
    config: WorkerConfig,
    call: Call,
    *,
    builder: MediaBuilder | None = None,
    clock: VideoPostprocessor | None = None,
    frames: int | None = None,
    canvas: image.Config | None = None,
    prompt_tokens: int | None = None,
) -> Mapping[str, OutputLayout]:
    """Describe the products one call publishes, keyed by product name.

    ``frames``, ``canvas`` and ``prompt_tokens`` size the layout for one
    request; without ``frames`` and ``canvas`` the layout bounds every
    admitted size (``bounding_layout`` over the longest video at each
    admitted canvas), and the prompt defaults to the admitted maximum.
    ``clock`` is the video post-processor whose frame rate relates audio
    samples to video frames; with a clock, a video decoder's product is its
    RGB media units (``decoded_units_layout``) rather than its own declared
    layout.

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

    # Condition encoders publish products only on a deployment that serves
    # conditions. A request's vision tokens lie within its presentation, and
    # its condition rows within the condition capacity; the request's own
    # extents come from its admission.
    condition_encoder = isinstance(component, (VideoEncoder, AudioEncoder)) or (
        isinstance(component, PatchEncoder) and builder is not None
    )
    if condition_encoder and call.entry_point.method == "encode":
        if config.max_condition_rows == 0:
            return {}
        if isinstance(component, PatchEncoder):
            return component.features_layout(config.max_sequence_tokens)
        # One row's width, from the encoder's own layout of a small input.
        if isinstance(component, VideoEncoder):
            sample = component.output_layout(
                video.Config(1, image.Config(32, 32))
            )["video"]
            return {
                "video": _rows(
                    (config.max_condition_rows, sample.shape[1]), sample.dtype
                )
            }
        sample = component.output_layout(component.latent_rate)["audio"]
        return {
            "audio": _rows(
                (config.max_condition_rows, sample.shape[1]), sample.dtype
            )
        }

    if isinstance(component, VideoPostprocessor):
        # The post-processor's RGB media units are the decoding call's
        # product, declared with the decoder below.
        return {}

    if not isinstance(component, (Denoiser, VideoDecoder, AudioDecoder)):
        return {}
    if (frames is None) != (canvas is None):
        raise ValueError("a request's layout names its frames and canvas")
    if frames is None:
        if builder is None:
            # Standalone decoders have no serving timeline bound, and image
            # execution returns its features and decoded raster through the
            # token/image protocol rather than persistent products.
            return {}
        bounds = tuple(
            output_layouts(
                config,
                call,
                builder=builder,
                clock=clock,
                frames=size.num_frames,
                canvas=size.frame,
                prompt_tokens=prompt_tokens,
            )
            for size in builder.video_sizes()
        )
        return {
            name: bounding_layout(tuple(layouts[name] for layouts in bounds))
            for name in bounds[0]
        }
    if isinstance(component, VideoDecoder):
        output = video.Config(frames, canvas)
        if clock is None:
            return component.output_layout(output)
        return {"video": decoded_units_layout(component, output)}

    if builder is None:
        # Image execution returns its features and decoded raster through the
        # token/image protocol rather than persistent inter-component products.
        return {}

    size = builder.size(
        frames,
        config.max_sequence_tokens if prompt_tokens is None else prompt_tokens,
        canvas,
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
