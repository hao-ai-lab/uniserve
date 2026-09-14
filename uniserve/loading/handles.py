"""Lazy checkpoint tensor metadata, slicing, and scoped materialization caches."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import load as load_safetensors

__all__ = [
    "PrefixedWeightHandle",
    "PtFileWeightHandle",
    "SafetensorFileWeightHandle",
    "SafetensorWeightHandle",
    "TensorWeightHandle",
    "WeightHandle",
    "safetensor_dtype",
    "weight_handle_materialization",
]


@dataclass(slots=True)
class _MaterializationCache:
    """Per-load-scope cache owning open readers and materialized checkpoint states."""

    stack: ExitStack
    safetensors: dict[tuple[Path, bool], Any] = field(default_factory=dict)
    pt: dict[Path, dict[str, torch.Tensor]] = field(default_factory=dict)


_MATERIALIZATION_CACHE: ContextVar[_MaterializationCache | None] = ContextVar(
    "uniserve_weight_handle_materialization_cache",
    default=None,
)


class WeightHandle(ABC):
    """Lazy, sliceable access to one checkpoint tensor."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Identify the tensor by its checkpoint key."""

        raise NotImplementedError

    @property
    @abstractmethod
    def shape(self) -> tuple[int, ...]:
        """Describe the complete checkpoint tensor without materializing it."""

        raise NotImplementedError

    @property
    @abstractmethod
    def dtype(self) -> torch.dtype:
        """Describe the stored element type without materializing the tensor."""

        raise NotImplementedError

    @abstractmethod
    def full(self) -> torch.Tensor:
        """Materialize the complete checkpoint tensor on CPU."""

        raise NotImplementedError

    @abstractmethod
    def narrow(self, dim: int, start: int, length: int) -> torch.Tensor:
        """Materialize a bounds-checked slice along one dimension."""

        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class TensorWeightHandle(WeightHandle):
    """Sliceable handle for an already materialized tensor."""

    name: str = field()
    tensor: torch.Tensor = field()

    @property
    def shape(self) -> tuple[int, ...]:
        """Read shape metadata from the resident tensor."""

        return tuple(int(value) for value in self.tensor.shape)

    @property
    def dtype(self) -> torch.dtype:
        """Read element type metadata from the resident tensor."""

        return self.tensor.dtype

    def full(self) -> torch.Tensor:
        """Expose the resident tensor without copying it."""

        return self.tensor

    def narrow(self, dim: int, start: int, length: int) -> torch.Tensor:
        """Expose a view of the requested resident tensor slice."""

        return self.tensor.narrow(int(dim), int(start), int(length))


@dataclass(frozen=True, slots=True)
class SafetensorWeightHandle(WeightHandle):
    """Lazy, sliceable handle for one tensor in a safetensors file."""

    name: str = field()
    _slice: Any = field()
    shape: tuple[int, ...] = field()
    dtype: torch.dtype = field()

    @classmethod
    def from_slice(cls, name: str, value: Any) -> "SafetensorWeightHandle":
        """Wrap an already opened safetensors slice without materializing its payload."""

        return cls(
            name=str(name),
            _slice=value,
            shape=tuple(int(size) for size in value.get_shape()),
            dtype=safetensor_dtype(str(value.get_dtype())),
        )

    def full(self) -> torch.Tensor:
        """Materialize the entire tensor through the open safetensors slice."""

        if not self.shape:
            return self._slice[...]
        return self._slice[:]

    def narrow(self, dim: int, start: int, length: int) -> torch.Tensor:
        """Materialize a bounds-checked slice along one tensor dimension."""

        axis = int(dim)
        begin = int(start)
        extent = int(length)
        if axis < 0:
            axis += len(self.shape)
        if not 0 <= axis < len(self.shape):
            raise IndexError(f"tensor {self.name!r} has no dimension {dim}")
        if begin < 0 or extent < 0 or begin + extent > self.shape[axis]:
            raise IndexError(
                f"tensor {self.name!r} slice [{begin}:{begin + extent}] exceeds dimension "
                f"{axis} of length {self.shape[axis]}"
            )
        selection = [slice(None)] * len(self.shape)
        selection[axis] = slice(begin, begin + extent)
        return self._slice[tuple(selection)]


@dataclass(frozen=True, slots=True)
class SafetensorFileWeightHandle(WeightHandle):
    """Durable safetensors metadata whose payload is opened only for materialization."""

    name: str = field()
    path: Path = field()
    shape: tuple[int, ...] = field()
    dtype: torch.dtype = field()
    mmap: bool = True

    def full(self) -> torch.Tensor:
        """Materialize the complete tensor from its safetensors file."""

        value = self._value()
        if isinstance(value, torch.Tensor):
            return value
        return value[...] if not self.shape else value[:]

    def narrow(self, dim: int, start: int, length: int) -> torch.Tensor:
        """Read a bounds-checked tensor slice directly from its safetensors file."""

        axis, begin, extent = _validated_slice(self.name, self.shape, dim, start, length)
        value = self._value()
        if isinstance(value, torch.Tensor):
            return value.narrow(axis, begin, extent)
        selection = [slice(None)] * len(self.shape)
        selection[axis] = slice(begin, begin + extent)
        return value[tuple(selection)]

    def _value(self) -> Any:
        """Open or reuse the backing file and return this tensor's lazy or resident value."""

        cache = _MATERIALIZATION_CACHE.get()

        # Memory-mapped readers remain open for the enclosing materialization scope.
        if self.mmap:
            if cache is None:
                with safe_open(self.path, framework="pt", device="cpu") as checkpoint:
                    return checkpoint.get_tensor(self.name)
            key = (self.path, True)
            checkpoint = cache.safetensors.get(key)
            if checkpoint is None:
                checkpoint = cache.stack.enter_context(
                    safe_open(self.path, framework="pt", device="cpu")
                )
                cache.safetensors[key] = checkpoint
            return checkpoint.get_slice(self.name)

        # Byte-loaded safetensors states are also reused within the scope because
        # decoding the entire file for each parameter would multiply host memory I/O.
        state: Any
        if cache is None:
            state = load_safetensors(self.path.read_bytes())
        else:
            key = (self.path, False)
            state = cache.safetensors.get(key)
            if state is None:
                state = load_safetensors(self.path.read_bytes())
                cache.safetensors[key] = state
        return state[self.name]


@dataclass(frozen=True, slots=True)
class PtFileWeightHandle(WeightHandle):
    """Durable PT tensor metadata backed by a per-load-unit file cache."""

    name: str = field()
    path: Path = field()
    shape: tuple[int, ...] = field()
    dtype: torch.dtype = field()

    def full(self) -> torch.Tensor:
        """Materialize the named tensor from its cached PyTorch state file."""

        return self._tensor()

    def narrow(self, dim: int, start: int, length: int) -> torch.Tensor:
        """Materialize a validated slice from the cached PyTorch state file."""

        axis, begin, extent = _validated_slice(self.name, self.shape, dim, start, length)
        return self._tensor().narrow(axis, begin, extent)

    def _tensor(self) -> torch.Tensor:
        """Return the named tensor from the scope cache or a freshly decoded PT state."""

        cache = _MATERIALIZATION_CACHE.get()
        if cache is not None and self.path in cache.pt:
            state = cache.pt[self.path]
        else:
            state = _load_pt_state(self.path)
            if cache is not None:
                cache.pt[self.path] = state
        try:
            return state[self.name]
        except KeyError as error:
            raise KeyError(f"checkpoint file {self.path} has no tensor {self.name!r}") from error


@dataclass(frozen=True, slots=True)
class PrefixedWeightHandle(WeightHandle):
    """Logical namespace view over a source checkpoint handle."""

    prefix: str
    source: WeightHandle

    @property
    def name(self) -> str:
        """Prefix the source checkpoint key with this logical namespace."""

        return f"{self.prefix}{self.source.name}"

    @property
    def shape(self) -> tuple[int, ...]:
        """Preserve the source tensor's complete shape metadata."""

        return self.source.shape

    @property
    def dtype(self) -> torch.dtype:
        """Preserve the source tensor's stored element type."""

        return self.source.dtype

    def full(self) -> torch.Tensor:
        """Materialize the complete source tensor under its prefixed identity."""

        return self.source.full()

    def narrow(self, dim: int, start: int, length: int) -> torch.Tensor:
        """Materialize a source slice while preserving the prefixed identity."""

        return self.source.narrow(dim, start, length)


@contextmanager
def weight_handle_materialization() -> Iterator[None]:
    """Share opened shard state while one module subtree is materialized."""

    with ExitStack() as stack:
        token = _MATERIALIZATION_CACHE.set(_MaterializationCache(stack))
        try:
            yield
        finally:
            _MATERIALIZATION_CACHE.reset(token)


def _load_pt_state(path: Path) -> dict[str, torch.Tensor]:
    """Load a PT checkpoint and normalize an optional ``state_dict`` envelope."""

    value = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(value, dict) and isinstance(value.get("state_dict"), dict):
        value = value["state_dict"]
    if not isinstance(value, dict):
        raise TypeError(f"checkpoint weight file {path} does not contain a tensor mapping")
    return {str(name): tensor for name, tensor in value.items() if isinstance(tensor, torch.Tensor)}


def _validated_slice(
    name: str,
    shape: tuple[int, ...],
    dim: int,
    start: int,
    length: int,
) -> tuple[int, int, int]:
    """Normalize a slice axis and reject intervals outside the declared tensor shape."""

    axis = int(dim)
    begin = int(start)
    extent = int(length)
    if axis < 0:
        axis += len(shape)
    if not 0 <= axis < len(shape):
        raise IndexError(f"tensor {name!r} has no dimension {dim}")
    if begin < 0 or extent < 0 or begin + extent > shape[axis]:
        raise IndexError(
            f"tensor {name!r} slice [{begin}:{begin + extent}] exceeds dimension "
            f"{axis} of length {shape[axis]}"
        )
    return axis, begin, extent


def safetensor_dtype(name: str) -> torch.dtype:
    """Map a safetensors dtype name to its PyTorch dtype."""

    table = {
        "BOOL": torch.bool,
        "F8_E4M3": torch.float8_e4m3fn,
        "F8_E4M3FNUZ": torch.float8_e4m3fnuz,
        "F8_E5M2": torch.float8_e5m2,
        "F8_E5M2FNUZ": torch.float8_e5m2fnuz,
        "BF16": torch.bfloat16,
        "C64": torch.complex64,
        "F16": torch.float16,
        "F32": torch.float32,
        "F64": torch.float64,
        "I8": torch.int8,
        "I16": torch.int16,
        "I32": torch.int32,
        "I64": torch.int64,
        "U8": torch.uint8,
        "U16": torch.uint16,
        "U32": torch.uint32,
        "U64": torch.uint64,
    }
    try:
        return table[name]
    except KeyError as error:
        raise TypeError(f"unsupported safetensors dtype {name!r}") from error
