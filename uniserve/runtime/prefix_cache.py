"""Ownership of storage for heterogeneous prefix state layouts."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Self

import torch

from uniserve.cache import Config, State
from uniserve.cache.state import _blocks
from uniserve.quantization import QuantizedTensor, Quantizer

from ._block_fill import BlockFill
from .tensor_buffers import TensorBuffers


class PrefixCache:
    """Allocate per-layer state; callers own block assignment and retirement.

    Layers may use different block sizes and budgets. No layer, attention
    method, or numerical input receives this owning object.
    """

    def __init__(
        self,
        config: Config,
        *,
        num_blocks: int | Mapping[str, int],
        block_size: int | Mapping[str, int],
        device: torch.device | str,
        dtype: torch.dtype | None = None,
        quantization: Mapping[str, Quantizer | None] | None = None,
    ) -> None:
        self.config = config
        self.device = torch.device(device)
        self._states: dict[str, State] = {}
        self._backing: dict[str, TensorBuffers] = {}
        self._fills: dict[str, BlockFill] = {}
        if quantization is not None and set(quantization) - set(config.layers):
            raise ValueError("cache quantization names must identify resident state layers")
        for parameter in (num_blocks, block_size):
            if isinstance(parameter, Mapping) and set(parameter) != set(config.layers):
                raise ValueError("per-layer block allocation must cover every state layer")
        for name, layout in config.layers.items():
            count = num_blocks[name] if isinstance(num_blocks, Mapping) else num_blocks
            size = block_size[name] if isinstance(block_size, Mapping) else block_size
            quantizer = None if quantization is None else quantization.get(name)
            requirements = layout.buffers(
                num_blocks=count, block_size=size, dtype=dtype, quantizer=quantizer
            )
            allocation = TensorBuffers.allocate(requirements, device=self.device)
            tensors = allocation.view(requirements)
            for tensor in tensors.values():
                tensor.zero_()
            state = layout.bind(tensors, block_size=size, dtype=dtype, quantizer=quantizer)
            self._backing[name] = allocation
            self._states[name] = state
            fields, values = [], []
            for field, tensor in state.tensors.items():
                buffers = (
                    tensor.buffers() if isinstance(tensor, QuantizedTensor) else {"values": tensor}
                )
                for key, backing in buffers.items():
                    value = int(key in {"scale", "tensor_scale"})
                    if value:
                        backing.fill_(1)
                    # Zero bytes cover encoded values and metadata without
                    # depending on arithmetic support for their storage dtype.
                    fields.append(backing if value else backing.view(torch.uint8))
                    values.append(value)
                fields.append(state.initialized[field])
                values.append(0)
            self._fills[name] = BlockFill(tuple(fields), tuple(values))

    def state(self, name: str) -> State:
        """Borrow one layer's state while retaining this owner through its use."""

        return self._states[name]

    def zero_blocks(self, name: str, blocks: tuple[int, ...]) -> None:
        """Reset caller-selected blocks and their encoding initialization state."""

        state = self.state(name)
        _blocks(blocks, next(iter(state.tensors.values())).shape[0])
        ranges = []
        for block in sorted(set(blocks)):
            if ranges and ranges[-1][1] == block:
                ranges[-1] = (ranges[-1][0], block + 1)
            else:
                ranges.append((block, block + 1))
        fill = self._fills[name]
        for start, stop in ranges:
            fill(start, stop)

    def mark_initialized(
        self, name: str, blocks: tuple[int, ...], *, fields: tuple[str, ...]
    ) -> None:
        """Commit externally transferred values and scales for specified fields."""

        state = self.state(name)
        _blocks(blocks, next(iter(state.tensors.values())).shape[0])
        if any(field not in state.initialized for field in fields):
            raise ValueError("initialized fields must belong to the selected state")
        for field in fields:
            for block in blocks:
                state.initialized[field][block] = True

    def close(self) -> None:
        """Release owner references after the caller has retired borrowed uses."""

        self._states.clear()
        self._fills.clear()
        for allocation in self._backing.values():
            allocation.close()
        self._backing.clear()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()
