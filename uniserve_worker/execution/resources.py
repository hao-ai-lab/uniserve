"""Query numerical result layouts and reserve caller-owned media input.

storage.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn

from uniserve.model import (
    AudioDecoder,
    Denoiser,
    TextEncoder,
    VideoDecoder,
    VideoPostprocessor,
)
from uniserve.tensors import BufferConfig, OutputLayout

from ..config import WorkerConfig
from .component_binding import Call, ComponentBinding


def media_state_buffers(
    model: nn.Module,
    bindings: Mapping[str, ComponentBinding],
    config: WorkerConfig,
) -> dict[str, BufferConfig]:
    """Reserve resident samples and transfer staging only on participating.

    ranks.
    """
    from ..bootstrap.inputs import media_builder

    builder = media_builder(model, config)
    if builder is None:
        return {}

    result = {}
    for binding in bindings.values():
        for call in binding.calls:
            if (
                isinstance(call.module, Denoiser)
                and call.entry_point.method == "forward"
            ):
                fields = builder.capacity_buffers()
            elif isinstance(call.module, VideoPostprocessor):
                fields = call.module.state_buffers(builder.maximum.num_frames)
            else:
                continue

            for name, field in fields.items():
                if name in result and result[name] != field:
                    raise ValueError(
                        f"media request fields disagree about {name!r}"
                    )
                result[name] = field
    return result


def decoded_units_layout(model: nn.Module, num_frames: int) -> OutputLayout:
    """Describe a video decoding round's product: RGB media units.

    Each row holds one media unit's frames at the output raster; a unit
    shorter than the longest fills its row's leading frames, and the unit
    division names how many. The rows are host products a host rank's
    encoder reads in place.
    """
    from ..bootstrap.inputs import capability

    decoder = capability(model, VideoDecoder)
    windows = decoder.frame_slices(num_frames)
    frames = max(window.stop - window.start for window in windows)
    shape = (
        len(windows),
        frames,
        decoder.frame_size.height,
        decoder.frame_size.width,
        3,
    )
    return OutputLayout(
        shape,
        torch.uint8,
        tuple(slice(0, extent) for extent in shape),
        variable_axes=(0,),
        value_range=(0, 255),
    )


def encoded_units_layout(model: nn.Module, num_frames: int) -> OutputLayout:
    """Describe a video encoding round's product: framed encoded unit rows.

    An encoded unit's length is not known when its row is reserved, so a row
    is bounded by the largest unit and carries its own length.
    """
    from ..bootstrap.inputs import capability
    from ..media.mux import encoded_unit_bytes

    decoder = capability(model, VideoDecoder)
    windows = decoder.frame_slices(num_frames)
    row = encoded_unit_bytes(
        max(window.stop - window.start for window in windows),
        decoder.frame_size.height,
        decoder.frame_size.width,
    )
    return OutputLayout(
        (len(windows), row),
        torch.uint8,
        (slice(0, len(windows)), slice(0, row)),
        variable_axes=(0,),
    )


def output_layouts(
    model: nn.Module,
    config: WorkerConfig,
    call: Call,
    *,
    frames: int | None = None,
    prompt_tokens: int | None = None,
) -> Mapping[str, OutputLayout]:
    """Describe complete persistent tensor products for one capability call."""
    from ..bootstrap.inputs import capability, media_builder

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
        builder = media_builder(model, config)
        count = builder.maximum.num_frames if frames is None else frames
        return {"video": decoded_units_layout(model, count)}

    builder = media_builder(model, config)
    if builder is None:
        # Image execution returns its features and decoded raster through the
        # token/image protocol rather than persistent inter-component products.
        return {}

    size = builder.size(
        builder.maximum.num_frames if frames is None else frames,
        config.max_sequence_tokens if prompt_tokens is None else prompt_tokens,
    )
    if isinstance(component, Denoiser):
        return component.output_layout(size)

    # Audio length follows from the video frame count at the declared rates.
    clock = capability(model, VideoPostprocessor)
    samples = round(size.num_frames * component.sample_rate / clock.frame_rate)
    return component.output_layout(samples)
