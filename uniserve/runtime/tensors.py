"""Runtime allocation and binding of a model's declared numerical tensors."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import replace
from types import MappingProxyType
from typing import TYPE_CHECKING, Protocol, TypeVar

import torch

from uniserve.distributed.mesh import Communicator
from uniserve.model.tensors import TensorViews
from uniserve.nn.video_attention import VideoAttention
from uniserve.runtime.tensor_buffers import TensorBuffers
from uniserve.tensors import BufferConfig

if TYPE_CHECKING:
    from uniserve.model.model import Model


SizeT = TypeVar("SizeT", contravariant=True)


class _Constants(Protocol[SizeT]):
    """Numerical constant preparation, independent of allocation ownership."""

    def constant_buffers(self, size: SizeT) -> Mapping[str, BufferConfig]: ...

    def prepare_constants(self, size: SizeT, *, out: TensorViews) -> None: ...


def merge_buffers(
    requirements: Iterable[Mapping[str, BufferConfig]],
) -> dict[str, BufferConfig]:
    """Merge numerical capacity bounds for serialized computations."""

    schema: dict[str, BufferConfig] = {}
    for fields in requirements:
        for name, requirement in fields.items():
            field = replace(
                requirement,
                shape=requirement.capacity_shape or requirement.shape,
                capacity_shape=None,
            )
            previous = schema.get(name)
            if previous is not None:
                if (
                    previous.dtype != field.dtype
                    or previous.host != field.host
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


def workspace_groups(
    model: Model, fields: Mapping[str, BufferConfig]
) -> Mapping[str, Communicator]:
    """Bind workspace names to the shared layers' actual communication groups."""

    bindings: dict[str, Communicator] = {}
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
            if group is None or name not in fields:
                continue
            previous = bindings.get(name)
            if fields[name].host or (previous is not None and previous != group):
                raise ValueError(
                    f"workspace {name!r} cannot share incompatible collective bindings"
                )
            bindings[name] = group
    return bindings


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
    component: _Constants[SizeT], size: SizeT, *, device: torch.device | str
) -> TensorViews:
    """Return owned constant tensors after the model fills borrowed views.

    Host-domain tensors remain on CPU. The caller chooses the device for device
    representations and retains these allocations through every graph and reader
    using them. Geometry caching and cross-stream delivery belong to that caller.
    """

    schemas = component.constant_buffers(size)
    views = MappingProxyType(
        {
            name: torch.empty(
                schema.shape,
                dtype=schema.dtype,
                device="cpu" if schema.host else device,
            )
            for name, schema in schemas.items()
        }
    )
    component.prepare_constants(size, out=views)
    return views


def bind_state(
    configs: Mapping[str, BufferConfig],
    storage: TensorBuffers | None,
) -> TensorViews:
    """Borrow exactly declared state views from caller-owned request capacity.

    The model supplies numerical shape, dtype, and host/device representation;
    it never receives the storage owner. The returned views share its backing
    and remain valid until that owner retires or reuses the request state.
    """

    schemas = configs
    if not schemas:
        return MappingProxyType({})
    if storage is None:
        raise ValueError("stateful computation requires bound request storage")
    for name, schema in schemas.items():
        value = storage.capacity.get(name)
        if value is None or value.dtype != schema.dtype:
            raise ValueError(f"state {name!r} has no backing with the declared dtype")
        if schema.host and value.device.type != "cpu":
            raise ValueError(f"state {name!r} requires host representation")
    shapes = {name: schema.shape for name, schema in schemas.items()}
    # Different valid prompt lengths can share one padded tensor geometry.
    # Retain one set of views for that representation, independent of content.
    return storage.bind(tuple(shapes.items()), shapes)


def bind_scratch(
    configs: Mapping[str, BufferConfig],
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
    for name, schema in configs.items():
        value = backing.get(name)
        if value is None or value.dtype != schema.dtype:
            raise ValueError(f"scratch {name!r} has no backing with the declared dtype")
        if schema.host and value.device.type != "cpu":
            raise ValueError(f"scratch {name!r} requires host representation")
        if value.ndim != len(schema.shape) or any(
            extent > capacity for extent, capacity in zip(schema.shape, value.shape, strict=True)
        ):
            raise ValueError(f"scratch {name!r} exceeds resident capacity")
        if not value.is_contiguous():
            raise ValueError(f"scratch {name!r} requires contiguous backing")
        elements = math.prod(schema.shape)
        views[name] = value.reshape(-1)[:elements].view(schema.shape)
    return MappingProxyType(views)
