"""Logical tensor descriptions used by worker calls.

The engine declares every cross-call tensor as a `TensorRef`: a request-scoped
identity plus a `DType` and a `ShapeBound` that fix its maximum size before any
worker runs. The actual shape travels separately with the published transfer.
`OutputInfo` is the same bounded representation before request binding, as a
component advertises its results in `ComponentInfo`. These records mirror the
Rust `uniserve_worker_ipc` tensor types. The PyO3 transport
(`crates/worker-ipc-py`) constructs `TensorRef`, `ShapeBound`, `StaticDim`,
and `DeviceDim` positionally, so their field order is part of that contract.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TypeAlias

from uniserve_worker.errors import invalid_descriptor
from uniserve_worker.protocol import identity
from uniserve_worker.protocol.validation import (
    _enum,
    _map,
    _seq,
    _str,
    _tagged,
    _uint,
)


class DType(StrEnum):
    """Wire-stable element dtypes supported by scheduler descriptors.

    Members are matched by string value. The declaration order differs from
    the Rust `DType` discriminant order and carries no meaning.
    """

    U8 = "u8"
    I32 = "i32"
    I16 = "i16"
    I64 = "i64"
    F16 = "f16"
    BF16 = "bf16"
    F32 = "f32"

    @property
    def element_bytes(self) -> int:
        """Storage width of one element in bytes."""
        return {
            DType.U8: 1,
            DType.I32: 4,
            DType.I16: 2,
            DType.I64: 8,
            DType.F16: 2,
            DType.BF16: 2,
            DType.F32: 4,
        }[self]


@dataclass(frozen=True, slots=True)
class StaticDim:
    """Fixes one tensor dimension to an exact extent."""

    extent: int


@dataclass(frozen=True, slots=True)
class DeviceDim:
    """Bounds one tensor dimension whose live extent is chosen on the device.

    Storage and `ShapeBound.max_elements` use `bound`.
    """

    bound: int


DimBound: TypeAlias = StaticDim | DeviceDim


@dataclass(frozen=True, slots=True)
class ShapeBound:
    """The maximum shape a tensor product may take.

    A bound is host-static except for at most one `DeviceDim`. An empty bound
    denotes a scalar. A bound consisting of a single `DeviceDim` denotes flat
    capacity: the tensor may have any rank as long as its element count fits.
    """

    dims: tuple[DimBound, ...] = ()

    def __post_init__(self) -> None:
        """Reject more than one `DeviceDim` and any non-positive extent."""
        device_dims = sum(1 for dim in self.dims if isinstance(dim, DeviceDim))
        if device_dims > 1:
            raise invalid_descriptor(
                "a shape bound carries more than one device-actual dimension"
            )
        if any(
            (dim.extent if isinstance(dim, StaticDim) else dim.bound) < 1
            for dim in self.dims
        ):
            raise invalid_descriptor("a shape bound contains a zero extent")

    @property
    def max_elements(self) -> int:
        """Multiply static extents and the maximum device-selected extent."""
        elements = 1
        for dim in self.dims:
            elements *= dim.extent if isinstance(dim, StaticDim) else dim.bound
        return elements

    def contains_shape(self, shape: tuple[int, ...]) -> bool:
        """Return whether a concrete tensor shape fits this bound.

        Any non-positive extent fails. A scalar bound accepts any shape with
        exactly one element. A flat-capacity bound accepts any rank whose
        element count is at most its bound. Otherwise the ranks must match,
        static extents must be equal, and the device extent must not exceed
        its bound.
        """
        if any(extent < 1 for extent in shape):
            return False
        elements = math.prod(shape)
        if not self.dims:
            return elements == 1
        if len(self.dims) == 1 and isinstance(self.dims[0], DeviceDim):
            return elements <= self.dims[0].bound
        return len(shape) == len(self.dims) and all(
            extent == bound.extent
            if isinstance(bound, StaticDim)
            else extent <= bound.bound
            for extent, bound in zip(shape, self.dims, strict=True)
        )

    @classmethod
    def from_mapping(
        cls, value: object, where: str = "shape_bound"
    ) -> ShapeBound:
        """Parse tagged ``static`` and ``device`` dimension variants."""
        data = _map(value, where)
        dims: list[DimBound] = []
        for index, item in enumerate(
            _seq(data.get("dims", ()), f"{where}.dims")
        ):
            kind, payload = _tagged(item, f"{where}.dims[{index}]")
            if kind == "static":
                dims.append(
                    StaticDim(_uint(payload, f"{where}.dims[{index}].value"))
                )
            elif kind == "device":
                inner = _map(payload, f"{where}.dims[{index}].value")
                dims.append(
                    DeviceDim(
                        _uint(
                            inner.get("max"), f"{where}.dims[{index}].value.max"
                        )
                    )
                )
            else:
                raise invalid_descriptor(
                    f"{where}.dims[{index}] has unknown variant {kind!r}"
                )
        return cls(tuple(dims))

    def to_mapping(self) -> dict[str, object]:
        """Serialize ordered dimension bounds into tagged wire variants."""
        return {"dims": [_dim_to_mapping(dim) for dim in self.dims]}


def _dim_to_mapping(dim: DimBound) -> dict[str, object]:
    """Encode a static or device dimension bound as a tagged wire variant."""
    if isinstance(dim, StaticDim):
        return {"kind": "static", "value": dim.extent}
    return {"kind": "device", "value": {"max": dim.bound}}


@dataclass(frozen=True, slots=True)
class OutputInfo:
    """A component's named tensor result, before request and storage binding.

    `ComponentInfo.outputs` lists these for each loaded component.
    """

    name: str
    dtype: DType
    shape_bound: ShapeBound

    def __post_init__(self) -> None:
        if not self.name:
            raise invalid_descriptor("tensor result must have a name")

    @property
    def max_bytes(self) -> int:
        """Maximum storage in bytes: bound element count times dtype width."""
        return self.shape_bound.max_elements * self.dtype.element_bytes

    @classmethod
    def from_mapping(
        cls, value: object, where: str = "output_info"
    ) -> OutputInfo:
        """Parse a named result description with its dtype and shape bound."""
        data = _map(value, where)
        return cls(
            name=_str(data.get("name"), f"{where}.name"),
            dtype=DType(_str(data.get("dtype"), f"{where}.dtype")),
            shape_bound=ShapeBound.from_mapping(
                data.get("shape_bound"), f"{where}.shape_bound"
            ),
        )

    def to_mapping(self) -> dict[str, object]:
        """Serialize the result description for IPC."""
        return {
            "name": self.name,
            "dtype": self.dtype.value,
            "shape_bound": self.shape_bound.to_mapping(),
        }


@dataclass(frozen=True, slots=True)
class TensorRef:
    """Identifies tensor storage independently of its role in a computation.

    Carries the identity (owning request, producing call, output index,
    allocation generation) and the bounded representation (dtype and
    `ShapeBound`), but not the actual shape or physical location.
    """

    request_key: identity.RequestKey
    producer_call_id: identity.CallId
    output_index: int
    generation: int
    dtype: DType
    shape_bound: ShapeBound

    _buffer_id: identity.BufferId | None = field(
        default=None, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        """Require a positive generation and re-check the shape bound."""
        if self.generation < 1:
            raise invalid_descriptor(
                "product reference has no logical generation"
            )
        self.shape_bound.__post_init__()

    @property
    def buffer_id(self) -> identity.BufferId:
        """Return the `BufferId` that keys this tensor's storage.

        The `BufferId` is derived on first access and memoized in the
        ``_buffer_id`` slot, so every consumer of this reference shares one
        instance. The slot's field flags keep it out of the constructor and
        equality, and `to_mapping` does not emit it.
        """
        buffer_id = self._buffer_id
        if buffer_id is None:
            buffer_id = identity.BufferId(
                owner=self.request_key,
                producer_call_id=self.producer_call_id,
                output_index=self.output_index,
                generation=self.generation,
            )
            object.__setattr__(self, "_buffer_id", buffer_id)
        return buffer_id

    @property
    def max_bytes(self) -> int:
        """Maximum storage in bytes: bound element count times dtype width.

        `Batch.validate` requires each persistent output's buffer allocation
        to be at least this large.
        """
        return self.shape_bound.max_elements * self.dtype.element_bytes

    @classmethod
    def from_mapping(
        cls, value: object, where: str = "tensor_ref"
    ) -> TensorRef:
        """Parse and validate a typed logical product and its storage bounds."""
        data = _map(value, where)
        return cls(
            request_key=identity.RequestKey.from_mapping(
                data.get("request_key"), f"{where}.request_key"
            ),
            producer_call_id=identity.CallId.from_mapping(
                data.get("producer_call_id"), f"{where}.producer_call_id"
            ),
            output_index=_uint(
                data.get("output_index"), f"{where}.output_index"
            ),
            generation=_uint(data.get("generation"), f"{where}.generation"),
            dtype=_enum(DType, data.get("dtype"), f"{where}.dtype"),
            shape_bound=ShapeBound.from_mapping(
                data.get("shape_bound"), f"{where}.shape_bound"
            ),
        )

    def to_mapping(self) -> dict[str, object]:
        """Serialize the complete logical product description for IPC."""
        return {
            "request_key": self.request_key.to_mapping(),
            "producer_call_id": self.producer_call_id.to_mapping(),
            "output_index": self.output_index,
            "generation": self.generation,
            "dtype": self.dtype.value,
            "shape_bound": self.shape_bound.to_mapping(),
        }
