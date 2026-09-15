"""Borrowed prefix state and complete-block numerical operations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import ClassVar, Literal

import torch

from uniserve.quantization import QuantizedTensor, Quantizer
from uniserve.tensors import BufferConfig


def _blocks(blocks: tuple[int, ...], count: int) -> None:
    if not isinstance(blocks, tuple) or any(
        type(block) is not int or not 0 <= block < count for block in blocks
    ):
        raise ValueError("state block indices must lie within the backing")


@dataclass(frozen=True)
class State:
    """Borrow named state tensors and a separate initialization bit per block.

    Physical fields retain a leading block dimension, including each encoded
    tensor's backing. Transfer views borrow storage; their caller must retain
    the state owner until all asynchronous readers finish.
    """

    tensors: Mapping[str, torch.Tensor]
    initialized: Mapping[str, torch.Tensor]
    block_size: int

    def __post_init__(self) -> None:
        if type(self.block_size) is not int or self.block_size < 1:
            raise ValueError("state block size must be positive")
        if not self.tensors or set(self.tensors) != set(self.initialized):
            raise ValueError("each state field requires its own initialization flags")
        if any(tensor.ndim == 0 for tensor in self.tensors.values()):
            raise ValueError("state tensors must retain a block dimension")

        count = next(iter(self.tensors.values())).shape[0]
        for name, tensor in self.tensors.items():
            if not name or "." in name or tensor.ndim == 0 or tensor.shape[0] != count:
                raise ValueError("state fields must name tensors with the same block count")
            flag = self.initialized[name]
            if flag.shape != (count,) or flag.dtype != torch.bool:
                raise ValueError("initialization flags must be one boolean per block")
            buffers = (
                tensor.buffers().values() if isinstance(tensor, QuantizedTensor) else (tensor,)
            )
            if any(value.ndim == 0 or value.shape[0] != count for value in buffers):
                raise ValueError("each state backing must retain the leading block dimension")

        object.__setattr__(self, "tensors", MappingProxyType(dict(self.tensors)))
        object.__setattr__(self, "initialized", MappingProxyType(dict(self.initialized)))

    def _storage(self) -> Mapping[str, torch.Tensor]:
        """Flatten each field into its physical buffers plus its flag vector."""

        result = {}
        for name, tensor in self.tensors.items():
            fields = tensor.buffers() if isinstance(tensor, QuantizedTensor) else {"values": tensor}
            result.update((f"{name}.{field}", value) for field, value in fields.items())
            result[f"{name}.initialized"] = self.initialized[name]
        return result

    def copy_blocks(
        self, source: tuple[int, ...] | torch.Tensor, target: tuple[int, ...] | torch.Tensor
    ) -> None:
        """Copy complete blocks using all source values from before this call.

        Indices are either two tuples or equally sized int64 device vectors.
        Paired -1 vector entries leave storage unchanged; other addresses must
        lie within the backing and valid targets must be distinct. Device
        vectors stay on their backing device, including during graph replay.
        """

        count = next(iter(self.tensors.values())).shape[0]
        storage = tuple(
            # One-byte floating storage includes formats without index_copy
            # support. Its integer view preserves the exact encoded bytes.
            tensor.view(torch.uint8)
            if tensor.dtype.is_floating_point and tensor.element_size() == 1
            else tensor
            for tensor in self._storage().values()
        )

        vector = isinstance(source, torch.Tensor) or isinstance(target, torch.Tensor)
        if vector:
            if (
                not isinstance(source, torch.Tensor)
                or not isinstance(target, torch.Tensor)
                or source.ndim != 1
                or target.shape != source.shape
                or source.dtype != torch.int64
                or target.dtype != torch.int64
                or source.device != target.device
                or any(tensor.device != source.device for tensor in storage)
            ):
                raise ValueError(
                    "block copy vectors must be matching int64 indices on the backing device"
                )
            if not source.numel():
                return

            valid = ((source == -1) & (target == -1)) | (
                (source >= 0) & (source < count) & (target >= 0) & (target < count)
            )
            ordered = target.sort().values
            unique = ((ordered[1:] < 0) | (ordered[1:] != ordered[:-1])).all()
            if source.is_cuda:
                torch._assert_async(valid.all(), "state block indices must lie within the backing")
                torch._assert_async(unique, "block copy targets must be unique")
            elif not bool(valid.all()) or not bool(unique):
                raise ValueError("block copies require valid addresses and unique targets")
            selections = {source.device: (source.clamp(0, max(0, count - 1)), target)}
        else:
            _blocks(source, count)
            _blocks(target, count)
            if len(source) != len(target) or len(set(target)) != len(target):
                raise ValueError("block copies require equally sized sources and unique targets")
            if not source:
                return
            selections = {
                device: (
                    torch.tensor(source, dtype=torch.int64, device=device),
                    torch.tensor(target, dtype=torch.int64, device=device),
                )
                for device in {tensor.device for tensor in storage}
            }
        if not count:
            return

        # Take every field's snapshot before writing any target, including
        # when distinct named fields borrow overlapping physical storage.
        snapshots = tuple(
            tensor.index_select(0, selections[tensor.device][0]) for tensor in storage
        )
        for tensor, snapshot in zip(storage, snapshots, strict=True):
            indices = selections[tensor.device][1]
            if vector and tensor.is_cuda:
                from ._copy import scatter_blocks

                scatter_blocks(tensor, indices, snapshot)
            elif vector:
                active = indices >= 0
                tensor.index_copy_(0, indices[active], snapshot[active])
            else:
                tensor.index_copy_(0, indices, snapshot)

    def transfer_views(self, blocks: tuple[int, ...]) -> Mapping[str, tuple[torch.Tensor, ...]]:
        """Borrow complete encoded fields and flags in the requested block order."""

        _blocks(blocks, next(iter(self.tensors.values())).shape[0])
        return MappingProxyType(
            {
                name: tuple(tensor[block : block + 1] for block in blocks)
                for name, tensor in self._storage().items()
            }
        )


@dataclass(frozen=True)
class StateConfig(ABC):
    """A numerical state layout; allocation and prefix policy belong to callers."""

    indexing: ClassVar[Literal["tokens", "states"]]

    @abstractmethod
    def buffers(
        self,
        *,
        num_blocks: int,
        block_size: int,
        dtype: torch.dtype | None,
        quantizer: Quantizer | None,
    ) -> Mapping[str, BufferConfig]:
        """Declare all ordinary backing fields, including initialization bits."""
        raise NotImplementedError

    @abstractmethod
    def bind(
        self,
        tensors: Mapping[str, torch.Tensor],
        *,
        block_size: int,
        dtype: torch.dtype | None,
        quantizer: Quantizer | None,
    ) -> State:
        """Validate and borrow supplied backing without allocation or conversion."""
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class Config:
    """Organize resident state layouts by their numerical module paths."""

    layers: Mapping[str, StateConfig]

    def __post_init__(self) -> None:
        if any(
            not isinstance(name, str) or not name or any(not part for part in name.split("."))
            for name in self.layers
        ):
            raise ValueError("state names must be nonempty module paths")
        if any(not isinstance(value, StateConfig) for value in self.layers.values()):
            raise TypeError("each layer must supply a StateConfig")
        object.__setattr__(self, "layers", MappingProxyType(dict(self.layers)))
