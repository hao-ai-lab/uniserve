"""Ownership of storage for heterogeneous prefix state layouts."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Self

import torch

from uniserve.cache import Config, State, StateConfig
from uniserve.cache.state import _blocks
from uniserve.quantization import QuantizedTensor, Quantizer
from uniserve.tensors import BufferConfig

from ._block_fill import BlockFill
from .tensor_buffers import TensorBuffers


class PrefixCache:
    """Allocate per-layer state; callers own block assignment and retirement.

    Layers may use different block sizes and budgets. Consecutive layers whose
    backing requirements agree share one allocation per buffer, stacked on a
    leading layer axis, so one buffer of a whole run of layers is addressable
    as a single strided tensor (see `layer_stacks`). No layer, attention
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
        self._backing: list[TensorBuffers] = []
        # Each run of layers sharing an allocation, in configuration order,
        # with its buffers as [layers in run, *buffer shape] views.
        self._stacks: list[
            tuple[tuple[str, ...], Mapping[str, torch.Tensor]]
        ] = []
        self._fills: dict[str, BlockFill] = {}

        if quantization is not None and set(quantization) - set(config.layers):
            raise ValueError(
                "cache quantization names must identify resident state layers"
            )
        for parameter in (num_blocks, block_size):
            if isinstance(parameter, Mapping) and set(parameter) != set(
                config.layers
            ):
                raise ValueError(
                    "per-layer block allocation must cover every state layer"
                )

        # Group maximal runs of consecutive layers whose backing requirements
        # agree. Each member keeps its own layout, block size and quantizer
        # for binding; only the physical allocation is shared.
        runs: list[
            tuple[
                Mapping[str, BufferConfig],
                list[tuple[str, StateConfig, int, Quantizer | None]],
            ]
        ] = []
        for name, layout in config.layers.items():
            count = (
                num_blocks[name]
                if isinstance(num_blocks, Mapping)
                else num_blocks
            )
            size = (
                block_size[name]
                if isinstance(block_size, Mapping)
                else block_size
            )
            quantizer = None if quantization is None else quantization.get(name)

            requirements = layout.buffers(
                num_blocks=count,
                block_size=size,
                dtype=dtype,
                quantizer=quantizer,
            )
            member = (name, layout, size, quantizer)
            if runs and runs[-1][0] == requirements:
                runs[-1][1].append(member)
            else:
                runs.append((requirements, [member]))

        for requirements, members in runs:
            # A run's buffers are [layers in run, *layer shape]: every layer
            # of one buffer shares an allocation at a uniform layer stride.
            stacked = {
                buffer: BufferConfig(
                    (len(members), *requirement.shape),
                    requirement.dtype,
                    None
                    if requirement.capacity_shape is None
                    else (len(members), *requirement.capacity_shape),
                    requirement.host,
                )
                for buffer, requirement in requirements.items()
            }
            allocation = TensorBuffers.allocate(stacked, device=self.device)
            self._backing.append(allocation)
            stacks = allocation.view(stacked)
            for tensor in stacks.values():
                tensor.zero_()
            self._stacks.append(
                (tuple(name for name, _, _, _ in members), stacks)
            )

            for index, (name, layout, size, quantizer) in enumerate(members):
                # A leading-axis slice is a contiguous view of one layer.
                state = layout.bind(
                    {buffer: stack[index] for buffer, stack in stacks.items()},
                    block_size=size,
                    dtype=dtype,
                    quantizer=quantizer,
                )
                self._states[name] = state

                # Block reset values: multiplicative scales restart at one,
                # while encoded values, metadata, and initialized flags
                # restart at zero.
                fields, values = [], []
                for field, tensor in state.tensors.items():
                    buffers = (
                        tensor.buffers()
                        if isinstance(tensor, QuantizedTensor)
                        else {"values": tensor}
                    )
                    for key, backing in buffers.items():
                        value = int(key in {"scale", "tensor_scale"})
                        if value:
                            backing.fill_(1)
                        # Zero bytes cover encoded values and metadata
                        # without depending on arithmetic support for their
                        # storage dtype.
                        fields.append(
                            backing if value else backing.view(torch.uint8)
                        )
                        values.append(value)
                    fields.append(state.initialized[field])
                    values.append(0)
                self._fills[name] = BlockFill(tuple(fields), tuple(values))

    def state(self, name: str) -> State:
        """Borrow one layer's state.

        Borrow one layer's state while retaining this owner through its use.
        """
        return self._states[name]

    def layer_stacks(
        self, buffer: str
    ) -> tuple[tuple[tuple[str, ...], torch.Tensor], ...]:
        """Borrow one backing buffer of every layer, grouped by allocation.

        `buffer` names a backing field the layers' state configs declare, such
        as ``"key.values"``. Returns ``(names, stack)`` pairs that cover every
        state layer in configuration order: `names` are consecutive layers
        sharing one allocation, and `stack` is ``[len(names), *buffer shape]``
        with its leading axis in the order of `names`. Writes through a
        layer's state are visible in its stack slice and vice versa. The views
        remain valid until this owner closes; the caller retains the owner
        through every asynchronous reader.

        Raises:
            ValueError: A state layer does not declare `buffer`.
        """
        if any(buffer not in stacks for _, stacks in self._stacks):
            raise ValueError(
                f"every state layer must declare backing buffer {buffer!r}"
            )
        return tuple((names, stacks[buffer]) for names, stacks in self._stacks)

    def zero_blocks(self, name: str, blocks: tuple[int, ...]) -> None:
        """Reset caller-selected blocks.

        Reset caller-selected blocks and their encoding initialization
        state.
        """
        state = self.state(name)
        _blocks(blocks, next(iter(state.tensors.values())).shape[0])

        # Adjacent blocks merge into one fill range per contiguous run.
        ranges: list[tuple[int, int]] = []
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
        """Commit externally transferred fields.

        Commit externally transferred values and scales for specified
        fields.
        """
        state = self.state(name)
        _blocks(blocks, next(iter(state.tensors.values())).shape[0])
        if any(field not in state.initialized for field in fields):
            raise ValueError(
                "initialized fields must belong to the selected state"
            )

        for field in fields:
            for block in blocks:
                state.initialized[field][block] = True

    def close(self) -> None:
        """Release owner references.

        Release owner references after the caller has retired borrowed uses.
        """
        self._states.clear()
        self._fills.clear()
        self._stacks.clear()
        for allocation in self._backing:
            allocation.close()
        self._backing.clear()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()
