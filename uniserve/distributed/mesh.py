"""Logical rank topology and borrowed numerical communication interfaces."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from math import prod
from types import MappingProxyType
from typing import Any, Literal

import torch
import torch.distributed as dist

from uniserve.runtime import communication


def divide(numerator: int, denominator: int) -> int:
    """Return an exact integer quotient.

    After validating divisibility and sign.
    """
    if denominator <= 0:
        raise ValueError("denominator must be positive")
    if numerator % denominator:
        raise ValueError(f"{numerator} is not divisible by {denominator}")
    return numerator // denominator


def _dimension(value: torch.Tensor, dim: int) -> int:
    if type(dim) is not int or not -value.ndim <= dim < value.ndim:
        raise ValueError("collective dimension is outside the input tensor")
    return dim % value.ndim


def _destination(
    value: torch.Tensor, shape: tuple[int, ...], out: torch.Tensor | None
) -> torch.Tensor:
    if out is None:
        return torch.empty(shape, dtype=value.dtype, device=value.device)
    if (
        tuple(out.shape) != shape
        or out.dtype != value.dtype
        or out.device != value.device
    ):
        raise ValueError("collective output must match shape, dtype and device")
    return out


@dataclass(frozen=True)
class Communicator:
    """Tensor communication with group-local roots and ordered logical members.

    The distributed runtime supplies the backend and owns its lifetime. Torch
    sorts process-group ranks; this interface preserves worker_config order even
    when it differs from backend order. Singleton calls need no backend.
    """

    ranks: tuple[int, ...] = (0,)
    rank: int = 0
    name: str = "local"
    device: torch.device = torch.device("cpu")
    _group: Any = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if not self.ranks or len(set(self.ranks)) != len(self.ranks):
            raise ValueError(
                f"group {self.name!r} requires unique non-empty membership"
            )
        if any(type(rank) is not int or rank < 0 for rank in self.ranks):
            raise ValueError(
                f"group {self.name!r} ranks must be nonnegative integers"
            )
        if type(self.rank) is not int or not 0 <= self.rank < len(self.ranks):
            raise ValueError(
                f"local rank {self.rank} is outside group {self.name!r}: "
                f"{self.ranks}"
            )

    @property
    def size(self) -> int:
        return len(self.ranks)

    @property
    def global_rank(self) -> int:
        return self.ranks[self.rank]

    def _require(self):
        """Return the borrowed backend group of a multi-rank communicator."""
        if self._group is None or not dist.is_initialized():
            raise RuntimeError(
                f"group {self.name!r} requires initialized distributed "
                "communication"
            )
        return self._group

    @property
    def _backend_name(self) -> str | None:
        """Expose a non-owning collective identity.

        For tensor-layout metadata.
        """
        return None if self.size == 1 else self._require().group_name

    def _peer(self, peer: int) -> int:
        if not 0 <= peer < self.size:
            raise ValueError(
                f"peer {peer} is outside group {self.name!r} of size "
                f"{self.size}"
            )
        return self.ranks[peer]

    @property
    def backend_order(self) -> tuple[int, ...]:
        """Map each ascending backend position to its logical member index.

        Backend collectives place member contributions in ascending global
        rank order; logical order is this communicator's ``ranks``.
        """
        return tuple(self.ranks.index(rank) for rank in sorted(self.ranks))

    def start_all_gather(
        self, output: torch.Tensor, value: torch.Tensor
    ) -> communication.Transfer:
        """Publish ``value`` into its backend-order slot of flat ``output``.

        ``output`` holds ``size`` equal contributions ordered by
        :attr:`backend_order`; read it after the returned transfer's
        ``wait``. Both tensors stay live until then.
        """
        if output.numel() != value.numel() * self.size:
            raise ValueError(
                "all-gather output size must equal input size times group size"
            )
        return communication.start_all_gather(output, value, self._require())

    def start_all_to_all(
        self, output: torch.Tensor, value: torch.Tensor
    ) -> communication.Transfer:
        """Publish equal backend-ordered peer payloads of ``value``.

        Leading-axis partitions follow :attr:`backend_order`; read ``output``
        after the returned transfer's ``wait``.
        """
        if output.shape != value.shape:
            raise ValueError("exchange output must match the input payloads")
        return communication.start_all_to_all(output, value, self._require())

    def all_reduce(
        self,
        value: torch.Tensor,
        *,
        op: Literal["sum", "min", "max"] = "sum",
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Reduce in place, or into matching output storage when supplied."""
        if op not in ("sum", "min", "max"):
            raise ValueError(f"unsupported reduction {op!r}")

        result = (
            value
            if out is None
            else _destination(value, tuple(value.shape), out)
        )
        if result is not value:
            result.copy_(value)
        if self.size > 1:
            communication.all_reduce(result, op, self._require())
        return result

    def all_gather(
        self,
        value: torch.Tensor,
        *,
        dim: int = 0,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Concatenate equal tensors in logical membership order."""
        dim = _dimension(value, dim)
        shape = list(value.shape)
        shape[dim] *= self.size
        result = _destination(value, tuple(shape), out)

        if self.size == 1:
            return result.copy_(value)
        if dim == 0 and result.is_contiguous():
            self.all_gather_into(result, value.contiguous())
            return result

        # General path: gather into a leading member axis, then move each
        # backend-ordered chunk to its logical position along dim.
        storage = torch.empty(
            (self.size, *value.shape), dtype=value.dtype, device=value.device
        )
        chunks = list(storage.unbind(0))
        communication.all_gather(storage, value.contiguous(), self._require())
        for backend_rank, logical_rank in enumerate(self.backend_order):
            result.narrow(
                dim, logical_rank * value.shape[dim], value.shape[dim]
            ).copy_(chunks[backend_rank])
        return result

    def gather(
        self,
        value: torch.Tensor,
        *,
        dst: int,
        dim: int = 0,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        """Concatenate tensors on the destination's group-local rank."""
        self._peer(dst)
        dim = _dimension(value, dim)
        if self.rank != dst and out is not None:
            raise ValueError(
                "only the gather destination may provide output storage"
            )

        shape = list(value.shape)
        shape[dim] *= self.size
        result = (
            _destination(value, tuple(shape), out) if self.rank == dst else None
        )

        # A contiguous dim-0 destination can receive the leading member axis
        # directly; other layouts stage through a scratch buffer and copy out.
        if dim == 0 and (result is None or result.is_contiguous()):
            storage = (
                None if result is None else result.view(self.size, *value.shape)
            )
        else:
            storage = (
                None
                if result is None
                else torch.empty(
                    (self.size, *value.shape),
                    dtype=value.dtype,
                    device=value.device,
                )
            )
        self._gather_into_tensor(storage, value.contiguous(), dst=dst)
        if (
            result is not None
            and storage is not None
            and storage.data_ptr() != result.data_ptr()
        ):
            for index, chunk in enumerate(storage.unbind(0)):
                result.narrow(
                    dim, index * value.shape[dim], value.shape[dim]
                ).copy_(chunk)
        return result

    def all_to_all(
        self,
        value: torch.Tensor,
        *,
        input_splits: tuple[int, ...],
        output_splits: tuple[int, ...],
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Exchange leading-axis partitions in logical membership order."""
        for splits in (input_splits, output_splits):
            if (
                not isinstance(splits, tuple)
                or len(splits) != self.size
                or any(type(count) is not int or count < 0 for count in splits)
            ):
                raise ValueError(
                    "all-to-all requires one nonnegative count per member"
                )
        if value.ndim == 0 or sum(input_splits) != value.shape[0]:
            raise ValueError(
                "all-to-all counts must cover the input leading axis"
            )

        result = _destination(
            value, (sum(output_splits), *value.shape[1:]), out
        )
        target = (
            result
            if result.is_contiguous()
            else torch.empty_like(result).contiguous()
        )
        self._all_to_all_single_into(
            target, value.contiguous(), output_splits, input_splits
        )
        return result if target is result else result.copy_(target)

    def all_gather_into(
        self, output: torch.Tensor, input: torch.Tensor
    ) -> None:
        """Gather equal contributions into flat ``output`` in logical order."""
        if output.numel() != input.numel() * self.size:
            raise ValueError(
                "all-gather output size must equal input size times group size"
            )
        if self.size == 1:
            output.reshape(-1).copy_(input.reshape(-1))
            return

        name = self._require().group_name
        if tuple(sorted(self.ranks)) == self.ranks:
            communication.all_gather_into_tensor(output, input, name)
        else:
            # The backend delivers chunks in ascending-rank order; copy each
            # into its logical member slot when the orders disagree.
            scratch = torch.empty_like(output)
            communication.all_gather_into_tensor(scratch, input, name)
            sources = scratch.reshape(self.size, *input.shape)
            targets = output.reshape(self.size, *input.shape)
            for backend_rank, logical_rank in enumerate(self.backend_order):
                targets[logical_rank].copy_(sources[backend_rank])

    def _all_to_all_single_into(
        self,
        output: torch.Tensor,
        input: torch.Tensor,
        output_splits: tuple[int, ...] | list[int],
        input_splits: tuple[int, ...] | list[int],
    ) -> None:
        for tensor, splits in ((output, output_splits), (input, input_splits)):
            if len(splits) != self.size or any(size < 0 for size in splits):
                raise ValueError(
                    "all-to-all requires one nonnegative row count per group "
                    "member"
                )
            if sum(splits) != tensor.shape[0]:
                raise ValueError("all-to-all row counts must cover the tensor")
        if self.size == 1:
            output.copy_(input)
            return

        name = self._require().group_name
        order = self.backend_order
        if tuple(sorted(self.ranks)) == self.ranks:
            communication.all_to_all_single_into(
                output, input, list(output_splits), list(input_splits), name
            )
            return

        # Repack splits from logical member order into backend rank order,
        # exchange, then scatter the received chunks back to logical order.
        send = input.split(tuple(input_splits), dim=0)
        packed = torch.cat([send[index] for index in order], dim=0)
        received = torch.empty_like(output)
        backend_output_splits = [output_splits[index] for index in order]
        communication.all_to_all_single_into(
            received,
            packed,
            backend_output_splits,
            [input_splits[index] for index in order],
            name,
        )
        targets = output.split(tuple(output_splits), dim=0)
        for index, chunk in zip(
            order, received.split(backend_output_splits, dim=0)
        ):
            targets[index].copy_(chunk)

    def _gather_into_tensor(
        self, output: torch.Tensor | None, input: torch.Tensor, *, dst: int
    ) -> None:
        global_dst = self._peer(dst)
        if self.rank == dst:
            expected = (self.size, *input.shape)
            if output is None or tuple(output.shape) != expected:
                raise ValueError(f"gather output must have shape {expected}")
            if output.device != input.device or output.dtype != input.dtype:
                raise ValueError(
                    "gather output must match input dtype and device"
                )
            chunks = list(output.unbind(0))
            gather_list = [chunks[index] for index in self.backend_order]
        else:
            if output is not None:
                raise ValueError(
                    "only the gather destination may provide output storage"
                )
            gather_list = None

        if self.size == 1:
            assert output is not None
            output[0].copy_(input)
            return
        communication.gather(gather_list, input, global_dst, self._require())

    def broadcast(
        self, value: torch.Tensor, *, src: int, out: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Broadcast from a group-local source into matching tensor storage."""
        global_src = self._peer(src)
        result = (
            value
            if out is None
            else _destination(value, tuple(value.shape), out)
        )
        if result is not value:
            result.copy_(value)
        if self.size > 1:
            communication.broadcast(result, global_src, self._require())
        return result

    def reduce_scatter(
        self,
        value: torch.Tensor,
        *,
        dim: int = 0,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Sum equal partitions.

        Return the local logical member's partition.
        """
        dim = _dimension(value, dim)
        shape = list(value.shape)
        shape[dim] = divide(shape[dim], self.size)
        result = _destination(value, tuple(shape), out)

        if self.size == 1:
            return result.copy_(value)

        # reduce_scatter_tensor consumes a flat concatenation of partitions in
        # ascending backend rank order, one movedim-packed chunk per member.
        chunks = value.chunk(self.size, dim=dim)
        packed = torch.cat(
            [chunks[index].movedim(dim, 0) for index in self.backend_order],
            dim=0,
        )
        output = result.movedim(dim, 0)
        if not output.is_contiguous():
            output = torch.empty_like(
                output, memory_format=torch.contiguous_format
            )

        communication.reduce_scatter(
            output, packed.contiguous(), self._require()
        )
        return result.copy_(output.movedim(0, dim))

    def send(self, value: torch.Tensor, *, dst: int) -> None:
        """Send a tensor's logical bytes.

        Including dtypes unsupported by NCCL.
        """
        payload = value.contiguous().reshape(-1).view(torch.uint8)
        communication.send(payload, self._peer(dst), self._require())

    def recv(self, *, src: int, out: torch.Tensor) -> torch.Tensor:
        """Receive bytes into caller-owned storage.

        With the agreed shape and dtype.
        """
        value = out
        storage = (
            value
            if value.is_contiguous()
            else torch.empty_like(value).contiguous()
        )
        communication.recv(
            storage.reshape(-1).view(torch.uint8),
            self._peer(src),
            self._require(),
        )
        if storage is not value:
            value.copy_(storage)
        return value

    def send_recv(
        self,
        value: torch.Tensor,
        *,
        dst: int,
        src: int,
        out: torch.Tensor,
    ) -> torch.Tensor:
        """Exchange tensor bytes with group-local peers.

        In one batched P2P launch. Callers keep send storage live until
        stream completion and supply contiguous receive storage with the
        peer's agreed tensor shape.
        """
        output = out
        global_dst, global_src = self._peer(dst), self._peer(src)
        if not output.is_contiguous():
            raise ValueError(
                "point-to-point exchange requires contiguous output storage"
            )

        if self.size == 1:
            return output.copy_(value)
        communication.send_recv(
            value.contiguous(),
            output,
            global_dst,
            global_src,
            self._require().group_name,
        )
        return output


@dataclass(frozen=True, kw_only=True)
class DeviceMesh:
    """An ordered mathematical topology borrowing any bound communicators.

    Every process may inspect the topology. A process outside ``ranks`` cannot
    borrow a local communicator or select its local submesh. Group creation,
    streams and communication resource lifetimes belong to ProcessGroups.
    """

    ranks: tuple[int, ...]
    shape: tuple[int, ...]
    axes: tuple[str, ...]
    rank: int
    _groups: Mapping[tuple[str, ...], Communicator] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )
    _device: torch.device = field(
        default=torch.device("cpu"), init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if (
            not isinstance(self.ranks, tuple)
            or not self.ranks
            or len(set(self.ranks)) != len(self.ranks)
            or any(type(rank) is not int or rank < 0 for rank in self.ranks)
            or type(self.rank) is not int
            or self.rank < 0
        ):
            raise ValueError("mesh ranks must be unique nonnegative integers")
        if (
            not isinstance(self.shape, tuple)
            or any(type(size) is not int or size < 1 for size in self.shape)
            or prod(self.shape) != len(self.ranks)
        ):
            raise ValueError("mesh shape product must equal its membership")
        if (
            not isinstance(self.axes, tuple)
            or len(self.axes) != len(self.shape)
            or len(set(self.axes)) != len(self.axes)
            or any(not isinstance(axis, str) or not axis for axis in self.axes)
        ):
            raise ValueError("mesh axes must uniquely name each dimension")
        object.__setattr__(self, "_groups", MappingProxyType({}))

    def coordinate(self, rank: int) -> tuple[int, ...]:
        """Return a member's coordinates in the declared rank ordering."""
        index = self.ranks.index(rank)
        coordinates = []
        for size in reversed(self.shape):
            coordinates.append(index % size)
            index //= size
        return tuple(reversed(coordinates))

    def _selection(self, axes: str | tuple[str, ...]) -> tuple[str, ...]:
        selected = (axes,) if isinstance(axes, str) else axes
        if (
            not isinstance(selected, tuple)
            or len(set(selected)) != len(selected)
            or any(axis not in self.axes for axis in selected)
        ):
            raise ValueError(
                f"unknown or repeated mesh axes {axes!r}; declared: {self.axes}"
            )
        return selected

    def size(self, axes: str | tuple[str, ...]) -> int:
        selected = self._selection(axes)
        return prod(self.shape[self.axes.index(axis)] for axis in selected)

    def members(self, axes: tuple[str, ...]) -> tuple[tuple[int, ...], ...]:
        """Enumerate fibers in topology order.

        Ordering each by selected axes.
        """
        selected = self._selection(axes)
        varying = tuple(self.axes.index(axis) for axis in selected)
        fixed = tuple(
            index
            for index, axis in enumerate(self.axes)
            if axis not in selected
        )
        fibers = {}
        for rank in self.ranks:
            coordinate = self.coordinate(rank)
            key = tuple(coordinate[index] for index in fixed)
            fibers.setdefault(key, []).append(
                (tuple(coordinate[index] for index in varying), rank)
            )
        return tuple(
            tuple(rank for _, rank in sorted(fiber))
            for fiber in fibers.values()
        )

    def get_group(self, axes: str | tuple[str, ...]) -> Communicator:
        selected = self._selection(axes)
        if self.rank not in self.ranks:
            raise ValueError(
                "nonparticipating rank cannot borrow a mesh communicator"
            )
        if selected in self._groups:
            return self._groups[selected]

        members = next(
            fiber for fiber in self.members(selected) if self.rank in fiber
        )
        # The same axis set selected in a different order shares one backend
        # group; only the logical member ordering of the descriptor changes.
        for axes, group in self._groups.items():
            if set(axes) == set(selected) or set(group.ranks) == set(members):
                return Communicator(
                    members,
                    members.index(self.rank),
                    ".".join(selected) or "local",
                    self._device,
                    group._group,
                )

        # An unbound descriptor remains useful for mathematical queries. Any
        # multi-rank numerical communication still requires runtime binding.
        return Communicator(
            ranks=members,
            rank=members.index(self.rank),
            name=".".join(selected) or "local",
            device=self._device,
        )

    def submesh(self, axes: tuple[str, ...]) -> DeviceMesh:
        selected = self._selection(axes)
        if self.rank not in self.ranks:
            raise ValueError("nonparticipating rank has no local submesh")
        members = next(
            fiber for fiber in self.members(selected) if self.rank in fiber
        )
        result = DeviceMesh(
            ranks=members,
            shape=tuple(self.shape[self.axes.index(axis)] for axis in selected),
            axes=selected,
            rank=self.rank,
        )
        object.__setattr__(result, "_device", self._device)
        object.__setattr__(
            result,
            "_groups",
            MappingProxyType(
                {
                    axes: group
                    for axes, group in self._groups.items()
                    if all(axis in selected for axis in axes)
                }
            ),
        )
        return result
