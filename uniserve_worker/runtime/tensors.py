"""Runtime allocation and binding of a model's declared numerical tensors."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import TYPE_CHECKING

import torch

from ..modeling.components import Call
from ..modeling.geometry import MediaShape, Shape, TextShape
from ..modeling.resources import TensorNeeds
from ..modeling.resources import TensorSchema as NumericalTensor
from ..modeling.tensors import TensorViews
from ..modeling.video import VideoMixin
from ..nn.video_attention import VideoAttention
from .tensor_buffers import TensorBuffers, TensorSchema

if TYPE_CHECKING:
    from ..execution.model_entry import ModelEntry
    from ..modeling.model import Model


@dataclass(frozen=True, slots=True)
class TensorResources:
    """Resolved per-request and per-invocation backing, owned by runtime callers."""

    state: Mapping[str, TensorSchema]
    scratch: Mapping[str, TensorSchema]

    def __post_init__(self) -> None:
        object.__setattr__(self, "state", MappingProxyType(dict(self.state)))
        object.__setattr__(self, "scratch", MappingProxyType(dict(self.scratch)))


def media_calls(model: Model, bindings: Mapping[str, ModelEntry]) -> tuple[tuple[Call, Shape], ...]:
    """Resolve maximum media geometry for the locally participating components."""

    if not isinstance(model, VideoMixin):
        return ()
    maximum = model.output_capacity
    shape = MediaShape(
        maximum.height,
        maximum.width,
        frames=maximum.frame_count,
        prompt_tokens=model.text_max_tokens,
    )
    calls: list[tuple[Call, Shape]] = []
    for binding in bindings.values():
        for call in binding.local_calls:
            geometry = (
                TextShape(model.text_max_tokens)
                if call.call in {Call.ENCODE_TEXT, Call.ENCODE_CONDITIONING}
                else shape
            )
            calls.append((call.call, geometry))
    return tuple(calls)


def _storage_schema(
    requirements: Iterable[Mapping[str, NumericalTensor]],
) -> dict[str, TensorSchema]:
    """Bound packed selections and merge compatible, serialized tensor views."""

    schema: dict[str, TensorSchema] = {}
    for fields in requirements:
        for name, requirement in fields.items():
            extents = list(requirement.shape)
            partition = requirement.partition
            if partition is not None:
                extents[partition.axis] = min(partition.elements, partition.rows // partition.parts)
            field = TensorSchema(
                tuple(extents),
                requirement.dtype,
                memory="pinned" if requirement.domain == "host" else "device",
            )
            previous = schema.get(name)
            if previous is not None:
                if (
                    previous.dtype != field.dtype
                    or previous.memory != field.memory
                    or len(previous.shape) != len(field.shape)
                ):
                    raise ValueError(f"tensor {name!r} has incompatible call representations")
                field = replace(
                    field,
                    shape=tuple(
                        max(a, b) for a, b in zip(previous.shape, field.shape, strict=True)
                    ),
                )
            schema[name] = field
    return schema


def resolve_resources(model: Model, calls: Iterable[tuple[Call, Shape]]) -> TensorResources:
    """Resolve state and scratch before the caller allocates their backing.

    Each request needs independent state. Scratch can be shared only by calls
    serialized through their final readers. Packed-axis declarations bound
    selections that move between ranks as request geometry changes. Runtime
    selects physical pinning and collective registration; models receive only
    the numerical views borrowed from the eventual allocations.
    """

    needs: tuple[TensorNeeds, ...] = tuple(model.tensor_specs(call, shape) for call, shape in calls)
    state = _storage_schema(item.state for item in needs)
    schema = _storage_schema(item.scratch for item in needs)
    # These buffers are public video-attention operands. Their registration
    # follows the shared layer's actual sequence exchanges, not model identity.
    for layer in model.modules():
        if not isinstance(layer, VideoAttention):
            continue
        groups = {
            "attention_workspace": (
                layer.to_qkvg.sequence_group
                if layer.projected_head and layer.sequence_size > 1
                else None
            ),
            "attention_output": (
                layer.parallel_attention.ulysses_group
                if layer.context_size == 1 and layer.sequence_size > 1
                else None
            ),
        }
        for name, group in groups.items():
            if group is None or name not in schema:
                continue
            field = schema[name]
            if field.memory == "pinned" or (field.group is not None and field.group != group):
                raise ValueError(f"scratch {name!r} cannot share incompatible collective bindings")
            schema[name] = replace(field, memory="symmetric", group=group)
    return TensorResources(state, schema)


def stage_tensor(source: torch.Tensor, target: torch.Tensor) -> None:
    """Copy a produced value into request storage with zero-filled padding.

    Source and target must not alias. The caller owns their delivery order and
    retains both allocations through the copy; the model receives only the
    resulting numerical view, without a staging or request-storage owner.
    """

    if (
        source.dtype != target.dtype
        or source.ndim != target.ndim
        or any(
            extent > capacity for extent, capacity in zip(source.shape, target.shape, strict=True)
        )
    ):
        raise ValueError("produced tensor does not fit its declared request representation")
    target.zero_()
    target[tuple(slice(0, extent) for extent in source.shape)].copy_(source)


@torch.inference_mode()
def prepare_constants(
    model: Model, call: Call, shape: Shape, *, device: torch.device | str
) -> TensorViews:
    """Return owned constant tensors after the model fills borrowed views.

    Host-domain tensors remain on CPU. The caller chooses the device for device
    representations and retains these allocations through every graph and reader
    using them. Geometry caching and cross-stream delivery belong to that caller.
    """

    schemas = model.tensor_specs(call, shape).constants
    views = MappingProxyType(
        {
            name: torch.empty(
                schema.shape,
                dtype=schema.dtype,
                device="cpu" if schema.domain == "host" else device,
            )
            for name, schema in schemas.items()
        }
    )
    model.prepare_metadata(call, shape, out=views)
    return views


def bind_state(
    model: Model,
    call: Call,
    shape: Shape,
    storage: TensorBuffers | None,
) -> TensorViews:
    """Borrow exactly declared state views from caller-owned request capacity.

    The model supplies numerical shape, dtype, and host/device representation;
    it never receives the storage owner. The returned views share its backing
    and remain valid until that owner retires or reuses the request state.
    """

    schemas = model.tensor_specs(call, shape).state
    if not schemas:
        return MappingProxyType({})
    if storage is None:
        raise ValueError("stateful computation requires bound request storage")
    for name, schema in schemas.items():
        value = storage.capacity.get(name)
        if value is None or value.dtype != schema.dtype:
            raise ValueError(f"state {name!r} has no backing with the declared dtype")
        if schema.domain == "host" and value.device.type != "cpu":
            raise ValueError(f"state {name!r} requires host representation")
    shapes = {name: schema.shape for name, schema in schemas.items()}
    # Different valid prompt lengths can share one padded tensor geometry.
    # Retain one set of views for that representation, independent of content.
    return storage.bind((call, tuple(shapes.items())), shapes)


def bind_scratch(
    model: Model,
    call: Call,
    shape: Shape,
    storage: TensorBuffers,
) -> TensorViews:
    """Bind a call's declared scratch to resident numerical views.

    Capacity allocation precedes this operation; exact views are compact
    prefixes of their backing. Public attention layers borrow buffers from
    separate execution scopes, retaining their mapped page geometry.
    The caller caches and retains the returned views through dependent graphs.
    """

    backing = storage.capacity
    views: dict[str, torch.Tensor] = {}
    for name, schema in model.tensor_specs(call, shape).scratch.items():
        value = backing.get(name)
        if value is None or value.dtype != schema.dtype:
            raise ValueError(f"scratch {name!r} has no backing with the declared dtype")
        if schema.domain == "host" and value.device.type != "cpu":
            raise ValueError(f"scratch {name!r} requires host representation")
        if value.ndim != len(schema.shape) or any(
            extent > capacity for extent, capacity in zip(schema.shape, value.shape, strict=True)
        ):
            raise ValueError(f"scratch {name!r} exceeds resident capacity")
        if not value.is_contiguous():
            raise ValueError(f"scratch {name!r} requires contiguous backing")
        elements = schema.nbytes // schema.dtype.itemsize
        views[name] = value.reshape(-1)[:elements].view(schema.shape)
    return MappingProxyType(views)
