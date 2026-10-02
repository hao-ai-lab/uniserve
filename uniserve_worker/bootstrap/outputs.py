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
from types import MappingProxyType

import torch
from torch import nn

from uniserve.model import (
    AudioDecoder,
    Denoiser,
    VideoDecoder,
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
    ``video`` output is its decoded media units, and an audio decoder's
    ``audio`` output is audio samples. Every other name passes through.
    """
    if isinstance(module, Denoiser):
        return {"video": "video_latents", "audio": "audio_latents"}.get(
            name, name
        )
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
    resolved, not only the components this rank holds.

    Returns:
        A read-only mapping from component name to its products. Components
        that publish nothing, such as the muxer, are omitted.

    Raises:
        ValueError: An output's dtype has no protocol ``DType``, an output
            has more than one variable axis, or the video codec component is
            present but the model has no ``MediaBuilder`` or no
            ``VideoDecoder``. Errors from ``media_builder``, the capability
            lookups, ``describe_components`` and ``output_layouts`` propagate.
    """
    from uniserve_worker.bootstrap.components import (
        VIDEO_CODEC_COMPONENT,
        describe_components,
    )
    from uniserve_worker.bootstrap.inputs import capability, media_builder
    from uniserve_worker.model_executor.resources import (
        bounding_layout,
        encoded_units_layout,
    )

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
                )
            )
        if outputs:
            result[component] = tuple(outputs)

    return MappingProxyType(result)
