"""Logical tensor descriptions used by worker call kinds."""

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
    """Defines wire-stable scalar dtypes supported by scheduler descriptors."""

    U8 = "u8"
    I32 = "i32"
    I16 = "i16"
    I64 = "i64"
    F16 = "f16"
    BF16 = "bf16"
    F32 = "f32"

    @property
    def element_bytes(self) -> int:
        """Width of one scalar in the logical representation."""
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
    """Bounds one tensor dimension.

    The live extent is selected on the device.
    """

    bound: int


DimBound: TypeAlias = StaticDim | DeviceDim


@dataclass(frozen=True, slots=True)
class ShapeBound:
    """Defines the maximum physical tensor shape allowed for a product."""

    dims: tuple[DimBound, ...] = ()

    def __post_init__(self) -> None:
        """Normalize dimensions.

        Rejects empty or non-positive shape bounds.
        """
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
        """Check tensor bounds.

        A single dynamic dimension denotes flat capacity.
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
        """Parse static and device-selected dimension bounds.

        Reads the bounds from the wire schema.
        """
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
    """Encode a static or symbolic dimension bound for the wire format."""
    if isinstance(dim, StaticDim):
        return {"kind": "static", "value": dim.extent}
    return {"kind": "device", "value": {"max": dim.bound}}


@dataclass(frozen=True, slots=True)
class OutputInfo:
    """Name and bounded representation of a component result.

    Applies before request binding.
    """

    name: str
    dtype: DType
    shape_bound: ShapeBound

    def __post_init__(self) -> None:
        if not self.name:
            raise invalid_descriptor("tensor result must have a name")

    @property
    def max_bytes(self) -> int:
        """Maximum physical storage required by this Tensor result."""
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
    """Identifies tensor storage independently of its role in a computation."""

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
        """Validate allocation generation and bounded tensor capacity."""
        if self.generation < 1:
            raise invalid_descriptor(
                "product reference has no logical generation"
            )
        self.shape_bound.__post_init__()

    @property
    def buffer_id(self) -> identity.BufferId:
        """Borrow the immutable storage identity.

        The identity is shared by this reference's consumers.
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
        """Return the maximum physical bytes allowed.

        The bound derives from this product’s shape and dtype.
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
