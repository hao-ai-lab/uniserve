"""Owned or externally borrowed backing for compact numerical tensor views."""

from __future__ import annotations

from collections.abc import Mapping
from math import prod
from types import MappingProxyType
from typing import Self

import torch

from uniserve.distributed.mesh import Communicator
from uniserve.runtime._peer_memory import SymmetricMemory, allocate_symmetric_memory
from uniserve.tensors import BufferConfig


class TensorBuffers:
    """Retain tensor backing and peer mappings independently of numerical views.

    ``allocate`` owns newly allocated storage; ``from_tensors`` retains borrowed
    external tensors. Both keep their views live until the caller retires all
    numerical readers and closes this object. Closing does not mutate backing.
    """

    def __init__(self) -> None:
        self._tensors: dict[str, torch.Tensor] = {}
        self._views: dict[tuple[tuple[str, BufferConfig], ...], Mapping[str, torch.Tensor]] = {}
        self._peers: dict[str, tuple[torch.Tensor, ...]] = {}
        self._symmetric: list[SymmetricMemory] = []
        self._closed = False

    @classmethod
    def from_tensors(cls, tensors: Mapping[str, torch.Tensor]) -> TensorBuffers:
        """Borrow contiguous named backing without copying or changing values."""

        if any(
            not isinstance(name, str) or not name or not tensor.is_contiguous()
            for name, tensor in tensors.items()
        ):
            raise ValueError("tensor backing requires nonempty names and contiguous storage")
        result = cls()
        result._tensors = dict(tensors)
        return result

    @classmethod
    def allocate(
        cls,
        configs: Mapping[str, BufferConfig],
        *,
        device: torch.device | str,
        pin_memory: bool = False,
        symmetric: Mapping[str, Communicator] | None = None,
    ) -> TensorBuffers:
        """Allocate capacity, with optional host pinning and ordered peer views.

        Initialization belongs to the numerical caller. The caller must retain
        this owner through asynchronous transfers and graph use of its backing.
        """

        device = torch.device(device)
        symmetric = {} if symmetric is None else symmetric
        if symmetric.keys() - configs.keys():
            raise ValueError("symmetric allocations must name declared buffer fields")
        for name, group in symmetric.items():
            if configs[name].host or group.device != device:
                raise ValueError("symmetric storage requires the group's assigned device")

        result = cls()
        for name, config in configs.items():
            shape = config.capacity_shape if config.capacity_shape is not None else config.shape
            if name in symmetric:
                allocation = allocate_symmetric_memory(symmetric[name], shape, dtype=config.dtype)
                result._symmetric.append(allocation)
                result._tensors[name] = allocation.local
                result._peers[name] = allocation.peers
            else:
                result._tensors[name] = torch.empty(
                    shape,
                    dtype=config.dtype,
                    device="cpu" if config.host else device,
                    pin_memory=config.host and pin_memory,
                )
        return result

    def view(self, configs: Mapping[str, BufferConfig]) -> Mapping[str, torch.Tensor]:
        """Borrow compact prefixes matching shape, dtype and host requirements.

        Each requested dimension must fit its backing dimension. Smaller views
        compact their leading elements; they are not strided rectangular crops.
        Equivalent requests reuse views without changing any numerical values.
        """

        if self._closed:
            raise RuntimeError("tensor buffers are closed")

        key = tuple(configs.items())
        cached = self._views.get(key)
        if cached is not None:
            return cached

        views = {}
        for name, config in configs.items():
            value = self._tensors.get(name)
            if value is None or value.dtype != config.dtype:
                raise ValueError(f"tensor {name!r} has no backing with the declared dtype")
            if config.host and value.device.type != "cpu":
                raise ValueError(f"tensor {name!r} requires host representation")
            if value.ndim != len(config.shape) or any(
                extent > capacity
                for extent, capacity in zip(config.shape, value.shape, strict=True)
            ):
                raise ValueError(f"tensor {name!r} exceeds resident capacity")
            views[name] = value.reshape(-1)[: prod(config.shape)].view(config.shape)
        result = MappingProxyType(views)
        self._views[key] = result
        return result

    def peers(self, name: str) -> tuple[torch.Tensor, ...]:
        """Borrow peer views in the allocation communicator's logical order."""

        if self._closed:
            raise RuntimeError("tensor buffers are closed")
        return self._peers[name]

    def close(self) -> None:
        """Release references after the caller retires all borrowed uses."""

        self._views.clear()
        self._peers.clear()
        self._tensors.clear()
        self._symmetric.clear()
        self._closed = True

    def __enter__(self) -> Self:
        if self._closed:
            raise RuntimeError("tensor buffers are closed")
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()
