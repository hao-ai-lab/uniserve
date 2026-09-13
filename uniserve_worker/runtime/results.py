"""Translate numerical output requirements into bounded protocol products."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

import torch

from ..modeling.components import Call
from ..modeling.model import Model
from ..protocol.batch import DeviceDim, DType, ShapeBound, StaticDim, TensorSpec

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
    Call.DIFFUSION: {"video": "video_latents", "audio": "audio_latents"},
    Call.DECODE_VIDEO: {"video": "video_segments"},
    Call.DECODE_AUDIO: {"audio": "audio_samples"},
}


def resolve_outputs(model: Model) -> Mapping[str, tuple[TensorSpec, ...]]:
    """Resolve complete logical result bounds before placement and allocation.

    Models declare maximum numerical input shapes and the resulting tensor
    requirements. Only this boundary selects wire dtypes, dynamic extent
    descriptors, and persistent product names. Local output regions do not
    reduce a product's global capacity: remote readers can assemble its shards.
    Unsupported wire representations fail before any product is allocated.
    """

    result = {}
    for entry, (call, shape) in model.output_shapes.items():
        outputs = []
        for name, schema in model.tensor_specs(call, shape).outputs.items():
            dtype = _DTYPES.get(schema.dtype)
            if dtype is None:
                raise ValueError(f"result {entry}.{name} has no protocol dtype")
            if len(schema.variable_axes) > 1:
                raise ValueError(f"result {entry}.{name} exceeds the protocol's dynamic axes")
            outputs.append(
                TensorSpec(
                    _PRODUCT_NAMES.get(call, {}).get(name, name),
                    dtype,
                    ShapeBound(
                        tuple(
                            DeviceDim(extent) if axis in schema.variable_axes else StaticDim(extent)
                            for axis, extent in enumerate(schema.shape)
                        )
                    ),
                )
            )
        if not outputs:
            raise ValueError(f"result entry {entry!r} declares no numerical outputs")
        result[entry] = tuple(outputs)
    return MappingProxyType(result)
