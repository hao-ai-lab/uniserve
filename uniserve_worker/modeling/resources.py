"""Mathematical tensor requirements without backing or execution policy."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from math import prod
from types import MappingProxyType
from typing import Literal

import torch

from ..transfer.layout import TensorRegion


@dataclass(frozen=True, slots=True)
class PackedAxis:
    """A selected axis inside an evenly partitioned packed sequence.

    ``elements`` counts this selection in the global sequence, ``rows`` counts
    the complete packed sequence, and ``parts`` is its mathematical partition
    count. The selected local extent can vary nonmonotonically as other packed
    sequences grow. Runtime can bound that extent using both global counts,
    without treating the largest request's local selection as a capacity.
    """

    axis: int
    elements: int
    rows: int
    parts: int

    def __post_init__(self) -> None:
        if self.axis < 0 or self.parts < 1 or not 0 <= self.elements <= self.rows:
            raise ValueError("packed tensor axis has invalid selection or partition geometry")
        if self.rows % self.parts:
            raise ValueError("packed tensor rows must divide their mathematical partitions")


@dataclass(frozen=True, slots=True)
class TensorAlias:
    """A returned tensor shares backing with one explicit numerical operand.

    This identifies storage lifetime, not ownership or an implicit output
    argument. Shape and logical region remain properties of the returned view.
    """

    scope: Literal["state", "scratch"]
    name: str

    def __post_init__(self) -> None:
        if self.scope not in {"state", "scratch"} or not self.name:
            raise ValueError("tensor alias must name an explicit state or scratch operand")


@dataclass(frozen=True, slots=True)
class TensorSchema:
    """A tensor's numerical shape, representation domain, and logical region.

    Host representation identifies CPU numerical inputs such as seeded normal
    draws and index arrays. Pinning, device ordinals, symmetric memory, capacity
    counts, and resource retirement are choices of the public resource binding.
    ``variable_axes`` identifies extents that may vary with the numerical input;
    other axes remain exact when this schema is used as a bounded result.
    """

    shape: tuple[int, ...]
    dtype: torch.dtype
    domain: Literal["host", "device"] = "device"
    region: TensorRegion | None = None
    partition: PackedAxis | None = None
    variable_axes: tuple[int, ...] = ()
    alias: TensorAlias | None = None

    def __post_init__(self) -> None:
        if any(extent < 0 for extent in self.shape):
            raise ValueError("tensor schema extents cannot be negative")
        if self.domain not in {"host", "device"}:
            raise ValueError("tensor representation requires a host or device domain")
        if len(set(self.variable_axes)) != len(self.variable_axes) or any(
            axis < 0 or axis >= len(self.shape) for axis in self.variable_axes
        ):
            raise ValueError("variable tensor axes must be distinct axes of the logical shape")
        if self.region is not None and not self.region.within(self.shape):
            raise ValueError("tensor region must lie within its complete logical shape")
        if self.partition is not None:
            part = self.partition
            if part.axis >= len(self.shape) or self.shape[part.axis] > min(
                part.elements, part.rows // part.parts
            ):
                raise ValueError("tensor extent exceeds its packed selection geometry")

    @property
    def nbytes(self) -> int:
        """Return numerical payload bytes before physical alignment or replication."""

        return prod(self.shape) * self.dtype.itemsize

    def validate(
        self,
        value: torch.Tensor,
        *,
        state: Mapping[str, torch.Tensor],
        scratch: Mapping[str, torch.Tensor],
        name: str = "value",
    ) -> None:
        """Check one borrowed or returned view without reading device values."""

        shape = self.shape if self.region is None else self.region.shape
        if value.dtype != self.dtype or value.ndim != len(shape):
            raise ValueError(f"output {name!r} has an incompatible tensor representation")
        if any(
            not 0 <= extent <= expected if axis in self.variable_axes else extent != expected
            for axis, (extent, expected) in enumerate(zip(value.shape, shape, strict=True))
        ):
            raise ValueError(f"output {name!r} disagrees with its numerical shape")
        if self.domain == "host" and value.device.type != "cpu":
            raise ValueError(f"output {name!r} requires host representation")
        alias = self.alias
        if alias is not None:
            operands = state if alias.scope == "state" else scratch
            source = operands.get(alias.name)
            if (
                source is None
                or source.device != value.device
                or source.untyped_storage().data_ptr() != value.untyped_storage().data_ptr()
            ):
                raise ValueError(f"output {name!r} does not borrow its declared operand")


@dataclass(frozen=True, slots=True)
class TensorNeeds:
    """Immutable declarations for one numerical call and its borrowed state.

    Constants are written by metadata preparation and then read-only. State
    survives numerical calls; scratch is local to one live invocation. Outputs
    describe delivered numerical values, whose reader lifetime belongs to the
    caller. Names are local to each mapping.
    """

    constants: Mapping[str, TensorSchema] = field(default_factory=dict)
    state: Mapping[str, TensorSchema] = field(default_factory=dict)
    scratch: Mapping[str, TensorSchema] = field(default_factory=dict)
    outputs: Mapping[str, TensorSchema] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("constants", "state", "scratch", "outputs"):
            values = getattr(self, name)
            if any(not key for key in values):
                raise ValueError("numerical tensor requirements need nonempty names")
            object.__setattr__(self, name, MappingProxyType(dict(values)))
        for name, output in self.outputs.items():
            alias = output.alias
            if alias is None:
                continue
            operands = self.state if alias.scope == "state" else self.scratch
            source = operands.get(alias.name)
            if source is None or source.dtype != output.dtype or source.domain != output.domain:
                raise ValueError(f"output {name!r} aliases an incompatible numerical operand")
