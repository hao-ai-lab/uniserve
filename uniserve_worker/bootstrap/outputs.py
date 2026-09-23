"""Translate numerical output layouts into bounded protocol products."""

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
    """Name a numerical modality by its downstream protocol use."""
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
    """
    from uniserve_worker.bootstrap.components import (
        VIDEO_ENCODER_COMPONENT,
        describe_components,
    )
    from uniserve_worker.bootstrap.inputs import capability, media_builder
    from uniserve_worker.model_executor.resources import encoded_units_layout

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
        if component == VIDEO_ENCODER_COMPONENT:
            if builder is None or decoder is None:
                raise ValueError("video encoding requires a decoder timeline")
            layouts.append(
                (
                    None,
                    "encoded_units",
                    encoded_units_layout(decoder, builder.maximum.num_frames),
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
            # extent is a static protocol dimension.
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
