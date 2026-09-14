"""Translate numerical output requirements into bounded protocol products."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

import torch

from uniserve.model.model import Model

from ..execution.resources import output_layouts
from ..protocol.batch import DeviceDim, DType, OutputInfo, ShapeBound, StaticDim

_DTYPES = {
    torch.uint8: DType.U8,
    torch.int16: DType.I16,
    torch.int32: DType.I32,
    torch.int64: DType.I64,
    torch.float16: DType.F16,
    torch.bfloat16: DType.BF16,
    torch.float32: DType.F32,
}

# The media protocol names persistent products by their downstream use. The
# numerical capabilities name modalities, independently of publication/storage.
_PRODUCT_NAMES = {
    "forward_diffusion": {"video": "video_latents", "audio": "audio_latents"},
    "decode:video": {"video": "video_segments"},
    "decode:audio": {"audio": "audio_samples"},
}


def resolve_outputs(model: Model) -> Mapping[str, tuple[OutputInfo, ...]]:
    """Resolve complete logical result bounds before placement and allocation.

    Models declare maximum numerical input shapes and the resulting tensor
    requirements. Only this boundary selects wire dtypes, dynamic extent
    descriptors, and persistent product names. Local output regions do not
    reduce a product's global capacity: remote readers can assemble its shards.
    Unsupported wire representations fail before any product is allocated.
    """

    from uniserve_models.catalog import entry_paths

    result = {}
    calls = model.component_calls(model.config)
    for entry, path in entry_paths(type(model), model.config).items():
        outputs = []
        try:
            component = model.get_submodule(path)
        except AttributeError:
            continue
        numerical = [
            (call.method, name, layout)
            for call in calls
            if call.component == path
            for name, layout in output_layouts(model, call.method, component).items()
        ]
        for call, name, layout in numerical:
            dtype = _DTYPES.get(layout.dtype)
            if dtype is None:
                raise ValueError(f"result {entry}.{name} has no protocol dtype")
            if len(layout.variable_axes) > 1:
                raise ValueError(f"result {entry}.{name} exceeds the protocol's dynamic axes")
            outputs.append(
                OutputInfo(
                    _PRODUCT_NAMES.get(call, {}).get(name, name),
                    dtype,
                    ShapeBound(
                        tuple(
                            DeviceDim(extent) if axis in layout.variable_axes else StaticDim(extent)
                            for axis, extent in enumerate(layout.shape)
                        )
                    ),
                )
            )
        if outputs:
            result[entry] = tuple(outputs)
    return MappingProxyType(result)
