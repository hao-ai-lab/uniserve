"""Translate numerical output layouts into bounded protocol products.

Models describe each output as an ``OutputLayout`` (global shape, torch dtype,
this rank's slice, variable axes). The worker reports products to the engine
as ``OutputInfo`` values in ``ComponentInfo.outputs``, with a protocol
``DType`` and a ``ShapeBound``. This module owns that translation, the
protocol names of media products, and the ``encoded_units`` product the host
video codec publishes. ``uniserve_worker.bootstrap.capacity`` and
``uniserve_worker.bootstrap.report`` also size product storage from the result.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from types import MappingProxyType

import torch
from torch import nn

from uniserve.model import (
    AudioDecoder,
    AudioEncoder,
    Denoiser,
    PatchEncoder,
    VideoDecoder,
    VideoEncoder,
    VideoPostprocessor,
)
from uniserve.tensors import OutputLayout
from uniserve_worker.config.execution import WorkerConfig
from uniserve_worker.model_executor.resources import output_layouts
from uniserve_worker.protocol.tensor import (
    DeviceDim,
    DType,
    OutputInfo,
    ShapeBound,
    StaticDim,
)
from uniserve_worker.protocol.video import VideoTask

# Torch dtypes a product may carry on the wire. An output in any other dtype
# is refused by ``resolve_outputs``.
_DTYPES = {
    torch.uint8: DType.U8,
    torch.int16: DType.I16,
    torch.int32: DType.I32,
    torch.int64: DType.I64,
    torch.float16: DType.F16,
    torch.bfloat16: DType.BF16,
    torch.float32: DType.F32,
}


def product_name(module: nn.Module, name: str) -> str:
    """Name a numerical modality by its downstream protocol use.

    A denoiser's ``video``/``audio`` outputs are latents, a video decoder's
    ``video`` output is its decoded media units, an audio decoder's
    ``audio`` output is audio samples, a vision encoder's ``features`` are
    vision features, and the condition encoders' outputs are condition
    latents. Every other name passes through.
    """
    if isinstance(module, Denoiser):
        return {"video": "video_latents", "audio": "audio_latents"}.get(
            name, name
        )
    if isinstance(module, PatchEncoder) and name == "features":
        return "vision_features"
    if isinstance(module, VideoEncoder) and name == "video":
        return "condition_video_latents"
    if isinstance(module, AudioEncoder) and name == "audio":
        return "condition_audio_latents"
    if isinstance(module, VideoDecoder) and name == "video":
        return "video_units"
    if isinstance(module, AudioDecoder) and name == "audio":
        return "audio_samples"
    return name


def resolve_outputs(
    model: nn.Module, config: WorkerConfig
) -> Mapping[str, tuple[OutputInfo, ...]]:
    """Resolve global product bounds from the worker's admitted numerical sizes.

    Local shards retain their global allocation bound so remote consumers can
    assemble them. Wire dtypes and persistent product names belong here; model
    output layouts retain only their numerical representation and placement.

    Every component ``describe_components`` reports for ``model`` is
    resolved, not only the components this rank holds. Condition products
    are declared only where the deployment's video denoiser executes a
    conditioned task and ``config`` grants condition capacity. The video
    decoder's RGB media units name their raster axes, which each request
    binds to its canvas; the video codec's encoded rows are bounded by the
    largest unit of every admitted canvas, the row every request uses.

    Returns:
        A read-only mapping from component name to its products. Components
        that publish nothing, such as the muxer, are omitted.

    Raises:
        ValueError: An output's dtype has no protocol ``DType``, an output
            has more than one variable axis, the video codec component is
            present but the model has no ``MediaBuilder`` or no
            ``VideoDecoder``, or the media reader serves conditions for a
            model without both condition encoders and a vision encoder.
            Errors from ``media_builder``, the capability lookups,
            ``describe_components`` and ``output_layouts`` propagate.
    """
    from uniserve_worker.bootstrap.components import (
        MEDIA_READER_COMPONENT,
        VIDEO_CODEC_COMPONENT,
        describe_components,
    )
    from uniserve_worker.bootstrap.inputs import (
        capability,
        executed_video_tasks,
        media_builder,
        video_denoiser,
    )
    from uniserve_worker.model_executor.resources import (
        DECODED_UNITS_RASTER_AXES,
        bounding_layout,
        condition_media_layouts,
        encoded_units_layout,
    )

    # Only a conditioned task carries conditions, so a deployment whose
    # denoiser executes none provisions no condition product, whatever its
    # condition capacity.
    denoiser = video_denoiser(model, config)
    if config.max_condition_rows and (
        denoiser is None
        or not set(executed_video_tasks(denoiser)) - {VideoTask.T2VA}
    ):
        config = replace(config, max_condition_rows=0)

    builder = media_builder(model, config)
    decoder = capability(model, VideoDecoder)
    clock = capability(model, VideoPostprocessor)
    result = {}
    for component, calls in describe_components(model).items():
        outputs = []
        # The encoded media units are not the output of any numerical module.
        layouts: list[tuple[nn.Module | None, str, OutputLayout]] = [
            (call.module, name, layout)
            for call in calls
            for name, layout in output_layouts(
                config,
                call,
                builder=builder,
                clock=clock,
            ).items()
        ]
        if component == VIDEO_CODEC_COMPONENT:
            if builder is None or decoder is None:
                raise ValueError("video encoding requires a decoder timeline")
            layouts.append(
                (
                    None,
                    "encoded_units",
                    bounding_layout(
                        tuple(
                            encoded_units_layout(decoder, size)
                            for size in builder.video_sizes()
                        )
                    ),
                )
            )
        # The media reader's products are the condition media it decodes,
        # declared only where the deployment serves conditions.
        if component == MEDIA_READER_COMPONENT and config.max_condition_rows:
            encoders = (
                capability(model, VideoEncoder),
                capability(model, AudioEncoder),
                capability(model, PatchEncoder),
            )
            if builder is None or any(value is None for value in encoders):
                raise ValueError(
                    "media reading requires the condition encoders and a "
                    "video timeline"
                )
            video_encoder, audio_encoder, vision = encoders
            # A condition encodes one frame (an image) or a reference
            # video's leading frames, a count the denoiser generates.
            denoiser = builder.denoiser
            frame_counts = [1, denoiser.legal_frame_count(1)]
            while (
                count := denoiser.legal_frame_count(frame_counts[-1] + 1)
            ) <= builder.maximum.num_frames:
                frame_counts.append(count)
            layouts.extend(
                (None, name, layout)
                for name, layout in condition_media_layouts(
                    config,
                    video_encoder=video_encoder,
                    audio_encoder=audio_encoder,
                    vision=vision,
                    frame_counts=tuple(frame_counts),
                ).items()
            )
        for module, name, layout in layouts:
            dtype = _DTYPES.get(layout.dtype)
            if dtype is None:
                raise ValueError(
                    f"result {component}.{name} has no protocol dtype"
                )
            if len(layout.variable_axes) > 1:
                raise ValueError(
                    f"result {component}.{name} exceeds the protocol "
                    "dynamic axes"
                )
            # A variable axis becomes a device-sized bound; every other
            # extent is a static protocol dimension. ``ShapeBound`` admits at
            # most one ``DeviceDim``, which the check above reports by name.
            # Decoded media units hold each request's own raster, at most the
            # largest admitted one.
            raster = (
                DECODED_UNITS_RASTER_AXES
                if isinstance(module, VideoDecoder)
                and name == "video"
                and clock is not None
                else None
            )
            outputs.append(
                OutputInfo(
                    name if module is None else product_name(module, name),
                    dtype,
                    ShapeBound(
                        tuple(
                            DeviceDim(extent)
                            if axis in layout.variable_axes
                            else StaticDim(extent)
                            for axis, extent in enumerate(layout.shape)
                        )
                    ),
                    raster_axes=raster,
                )
            )
        if outputs:
            result[component] = tuple(outputs)

    return MappingProxyType(result)
