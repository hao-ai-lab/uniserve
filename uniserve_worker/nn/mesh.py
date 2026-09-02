"""Device mesh and transport boundaries for model parallelism."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, TypeAlias

import torch

from ..profiling import profile_range


@torch.library.custom_op(
    "uniserve_worker::symmetric_memory_fence",
    mutates_args=("output",),
)
def _symmetric_memory_fence_custom(
    input: torch.Tensor,
    output: torch.Tensor,
) -> None:
    work = torch.distributed.all_gather_into_tensor(output, input, async_op=True)
    work.block_current_stream()


@_symmetric_memory_fence_custom.register_fake
def _symmetric_memory_fence_custom_fake(
    input: torch.Tensor,
    output: torch.Tensor,
) -> None:
    del input, output


@torch.library.custom_op(
    "uniserve_worker::all_to_all_single_into",
    mutates_args=("output",),
)
def _all_to_all_single_into_custom(
    output: torch.Tensor,
    input: torch.Tensor,
    output_splits: list[int],
    input_splits: list[int],
) -> None:
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    with profile_range(f"uniserve.h3.collective kind=all_to_all rank={rank}"):
        work = torch.distributed.all_to_all_single(
            output,
            input,
            output_split_sizes=output_splits,
            input_split_sizes=input_splits,
            async_op=True,
        )
        work.block_current_stream()


@_all_to_all_single_into_custom.register_fake
def _all_to_all_single_into_custom_fake(
    output: torch.Tensor,
    input: torch.Tensor,
    output_splits: list[int],
    input_splits: list[int],
) -> None:
    del output, input, output_splits, input_splits


@torch.library.custom_op(
    "uniserve_worker::all_gather_into_tensor",
    mutates_args=("output",),
)
def _all_gather_into_tensor_custom(
    output: torch.Tensor,
    input: torch.Tensor,
) -> None:
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    with profile_range(f"uniserve.h3.collective kind=all_gather rank={rank}"):
        work = torch.distributed.all_gather_into_tensor(
            output,
            input,
            async_op=True,
        )
        work.block_current_stream()


@_all_gather_into_tensor_custom.register_fake
def _all_gather_into_tensor_custom_fake(
    output: torch.Tensor,
    input: torch.Tensor,
) -> None:
    del output, input


@torch.library.custom_op(
    "uniserve_worker::all_reduce_max",
    mutates_args=("value",),
)
def _all_reduce_max_custom(value: torch.Tensor) -> None:
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    with profile_range(f"uniserve.h3.collective kind=all_reduce_max rank={rank}"):
        work = torch.distributed.all_reduce(
            value,
            op=torch.distributed.ReduceOp.MAX,
            async_op=True,
        )
        work.block_current_stream()


@_all_reduce_max_custom.register_fake
def _all_reduce_max_custom_fake(value: torch.Tensor) -> None:
    del value


__all__ = [
    "divide",
    "AxisTransport",
    "CollectiveTransport",
    "LocalP2PTransport",
    "MeshAxis",
    "DeviceMesh",
    "TensorParallel",
    "SymmetricMemoryWorkspace",
]


def divide(numerator: int, denominator: int) -> int:
    """Exact integer division, raising on a non-divisible or non-positive denom."""
    numerator = int(numerator)
    denominator = int(denominator)
    if denominator <= 0:
        raise ValueError("denominator must be positive")
    if numerator % denominator != 0:
        raise ValueError(f"{numerator} is not divisible by {denominator}")
    return numerator // denominator


@dataclass(frozen=True)
class CollectiveTransport:
    """A ``torch.distributed`` process group bound to one mesh axis.

    ``group`` is ``None`` for the default world group. ``size`` and ``coord``
    identify this process within the axis.
    """

    axis: str
    _size: int
    _coord: int
    group: Any = None

    @property
    def size(self) -> int:
        return int(self._size)

    @property
    def coord(self) -> int:
        return int(self._coord)

    def _require(self) -> Any:
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            raise RuntimeError(
                f"mesh axis {self.axis!r} size>1 requires an initialized "
                "torch.distributed collective"
            )
        if self.group is None:
            world = int(torch.distributed.get_world_size())
            if world != self.size:
                raise RuntimeError(
                    f"mesh axis {self.axis!r} requires an explicit process group when "
                    f"the default world size ({world}) differs from axis size ({self.size})"
                )
        return self.group

    def all_reduce(self, t: torch.Tensor) -> torch.Tensor:
        group = self._require()
        torch.distributed.all_reduce(t, group=group)
        return t

    def all_reduce_max(self, value: torch.Tensor) -> torch.Tensor:
        group = self._require()
        if group is None:
            _all_reduce_max_custom(value)
            return value
        work = torch.distributed.all_reduce(
            value,
            op=torch.distributed.ReduceOp.MAX,
            group=group,
            async_op=True,
        )
        work.block_current_stream()
        return value

    def all_gather(self, t: torch.Tensor, dim: int) -> torch.Tensor:
        group = self._require()
        chunks = [torch.empty_like(t) for _ in range(self.size)]
        torch.distributed.all_gather(chunks, t.contiguous(), group=group)
        return torch.cat(chunks, dim=dim)

    def all_to_all_single_into(
        self,
        output: torch.Tensor,
        input: torch.Tensor,
        output_splits: tuple[int, ...] | list[int],
        input_splits: tuple[int, ...] | list[int],
    ) -> Any:
        group = self._require()
        if group is None:
            _all_to_all_single_into_custom(
                output,
                input,
                list(output_splits),
                list(input_splits),
            )
            return None
        work = torch.distributed.all_to_all_single(
            output,
            input,
            output_split_sizes=list(output_splits),
            input_split_sizes=list(input_splits),
            group=group,
            async_op=True,
        )
        work.block_current_stream()
        return work

    def all_gather_into_tensor(self, output: torch.Tensor, input: torch.Tensor) -> Any:
        group = self._require()
        if group is None:
            _all_gather_into_tensor_custom(output, input)
            return None
        work = torch.distributed.all_gather_into_tensor(
            output,
            input,
            group=group,
            async_op=True,
        )
        work.block_current_stream()
        return work

    def gather_into_tensor(
        self,
        output: torch.Tensor | None,
        input: torch.Tensor,
        *,
        dst: int,
    ) -> Any:
        group = self._require()
        if not 0 <= int(dst) < self.size:
            raise ValueError(f"gather destination {dst} is outside axis {self.axis!r}")
        if self.coord == int(dst):
            expected = (self.size, *input.shape)
            if output is None or tuple(output.shape) != expected:
                raise ValueError(f"gather output must have shape {expected}")
            if output.dtype != input.dtype or output.device != input.device:
                raise ValueError("gather output must match the input dtype and device")
            gather_list = list(output.unbind(0))
        else:
            if output is not None:
                raise ValueError("only the gather destination may provide output storage")
            gather_list = None
        global_dst = (
            int(dst) if group is None else int(torch.distributed.get_global_rank(group, int(dst)))
        )
        work = torch.distributed.gather(
            input,
            gather_list=gather_list,
            dst=global_dst,
            group=group,
            async_op=True,
        )
        work.block_current_stream()
        return work

    def broadcast(self, t: torch.Tensor, *, src: int) -> torch.Tensor:
        group = self._require()
        torch.distributed.broadcast(t, src=src, group=group)
        return t


@dataclass(frozen=True)
class SymmetricMemoryWorkspace:
    """One mesh-axis rendezvous allocation and its stream-ordered fence."""

    axis: str
    rank: int
    size: int
    local: torch.Tensor
    peers: tuple[torch.Tensor, ...]
    handle: Any
    group: Any = None

    def fence(self, input: torch.Tensor, output: torch.Tensor) -> None:
        if tuple(input.shape) != (1,) or tuple(output.shape) != (self.size,):
            raise ValueError("symmetric-memory fence buffers do not match the mesh axis")
        if self.size == 1:
            output.copy_(input)
            return
        if self.group is None:
            _symmetric_memory_fence_custom(input, output)
            return
        work = torch.distributed.all_gather_into_tensor(
            output,
            input,
            group=self.group,
            async_op=True,
        )
        work.block_current_stream()


@dataclass(frozen=True)
class LocalP2PTransport:
    """Coordinates are CUDA devices inside one process.

    ``devices[i]`` is the device for coordinate ``i``; ``coord`` is this mesh
    view's home coordinate (the device shared/primary modules live on).
    Movement is a direct device-to-device copy. This transport exposes no
    collective surface, so routing axes cannot be used for reductions.
    """

    axis: str
    devices: tuple[torch.device, ...]
    _coord: int

    @property
    def size(self) -> int:
        return len(self.devices)

    @property
    def coord(self) -> int:
        return int(self._coord)

    def device(self, coord: int) -> torch.device:
        """The CUDA device backing ``coord`` (in-process tower routing seam)."""
        return self.devices[int(coord)]

    def copy_to(self, t: torch.Tensor, *, coord: int, non_blocking: bool = True) -> torch.Tensor:
        return t.to(self.devices[int(coord)], non_blocking=non_blocking)

    def broadcast(self, t: torch.Tensor, *, src: int) -> torch.Tensor:
        if int(src) < 0 or int(src) >= self.size:
            raise ValueError(f"broadcast source {src} is outside axis {self.axis!r}")
        return t.to(self.devices[self.coord]) if t.device != self.devices[self.coord] else t


AxisTransport: TypeAlias = CollectiveTransport | LocalP2PTransport


@dataclass(frozen=True)
class MeshAxis:
    """One parallelism dimension of the mesh.

    ``parent`` marks a *sub-factorization* axis (e.g. ``attn_tp`` carved out of
    ``tp``) rather than an independent product dimension; this models the real
    topology of production systems where finer axes divide a coarser one.
    """

    name: str
    size: int
    coord: int
    transport: AxisTransport | None = None
    parent: str | None = None

    def __post_init__(self) -> None:
        if self.size <= 0:
            raise ValueError(f"mesh axis {self.name!r} size must be positive")
        if self.coord < 0 or self.coord >= self.size:
            raise ValueError(
                f"mesh axis {self.name!r} coord must satisfy 0 <= coord < size "
                f"(coord={self.coord}, size={self.size})"
            )
        # Load-time sharding can use axis coordinates without a live transport.


@dataclass(frozen=True)
class DeviceMesh:
    """The process-wide parallelism topology: named axes + this rank's local device.

    The degenerate mesh (no non-trivial axis) reproduces a single-rank,
    single-device worker exactly.
    """

    axes: Mapping[str, MeshAxis] = field(default_factory=dict)
    local_device: torch.device = field(default_factory=lambda: torch.device("cpu"))
    _symmetric_workspaces: dict[tuple[object, ...], SymmetricMemoryWorkspace] = field(
        default_factory=dict,
        repr=False,
        compare=False,
    )

    def axis(self, name: str) -> MeshAxis | None:
        return self.axes.get(name)

    def size(self, name: str) -> int:
        ax = self.axes.get(name)
        return int(ax.size) if ax is not None else 1

    def coord(self, name: str) -> int:
        ax = self.axes.get(name)
        return int(ax.coord) if ax is not None else 0

    def is_trivial(self, name: str) -> bool:
        return self.size(name) <= 1

    def transport(self, name: str) -> AxisTransport:
        ax = self.axes.get(name)
        if ax is None or ax.transport is None:
            raise RuntimeError(f"mesh axis {name!r} has no transport")
        return ax.transport

    def all_reduce(self, tensor: torch.Tensor, group: str = "tp") -> torch.Tensor:
        """Reduce one tensor in place on a named collective axis."""

        if self.is_trivial(group):
            return tensor
        transport = self.transport(group)
        if not isinstance(transport, CollectiveTransport):
            raise RuntimeError(f"mesh axis {group!r} does not support collectives")
        return transport.all_reduce(tensor)

    def symmetric_memory(
        self,
        shape: tuple[int, ...],
        *,
        dtype: torch.dtype,
        group: str = "sp",
        name: str = "workspace",
    ) -> SymmetricMemoryWorkspace:
        """Allocate or retrieve one symmetric-memory workspace on a collective axis."""

        if self.is_trivial(group):
            key = (group, name, tuple(int(value) for value in shape), dtype, self.local_device)
            cached = self._symmetric_workspaces.get(key)
            if cached is not None:
                return cached
            local = torch.empty(shape, dtype=dtype, device=self.local_device)
            workspace = SymmetricMemoryWorkspace(
                axis=group,
                rank=0,
                size=1,
                local=local,
                peers=(local,),
                handle=None,
                group=None,
            )
            self._symmetric_workspaces[key] = workspace
            return workspace
        transport = self.transport(group)
        if not isinstance(transport, CollectiveTransport):
            raise RuntimeError(f"mesh axis {group!r} does not support symmetric memory")
        process_group = transport._require()
        key = (group, name, tuple(int(value) for value in shape), dtype, self.local_device)
        cached = self._symmetric_workspaces.get(key)
        if cached is not None:
            return cached
        import torch.distributed._symmetric_memory as symm_mem

        backend = symm_mem.get_backend(self.local_device)
        if backend != "NCCL":
            symm_mem.set_backend("NCCL")
        local = symm_mem.empty(shape, dtype=dtype, device=self.local_device)
        rendezvous_group = (
            torch.distributed.group.WORLD if process_group is None else process_group
        )
        handle = symm_mem.rendezvous(local, rendezvous_group)
        peers = tuple(
            handle.get_buffer(rank, shape, dtype) for rank in range(transport.size)
        )
        workspace = SymmetricMemoryWorkspace(
            axis=group,
            rank=transport.coord,
            size=transport.size,
            local=local,
            peers=peers,
            handle=handle,
            group=process_group,
        )
        self._symmetric_workspaces[key] = workspace
        return workspace

    def all_reduce_max(self, tensor: torch.Tensor, group: str = "tp") -> torch.Tensor:
        """Reduce one tensor with MAX in place on a named collective axis."""

        if self.is_trivial(group):
            return tensor
        transport = self.transport(group)
        if not isinstance(transport, CollectiveTransport):
            raise RuntimeError(f"mesh axis {group!r} does not support collectives")
        return transport.all_reduce_max(tensor)

    def all_to_all_single_into(
        self,
        output: torch.Tensor,
        input: torch.Tensor,
        output_splits: tuple[int, ...] | list[int],
        input_splits: tuple[int, ...] | list[int],
        group: str = "sp",
    ) -> Any:
        """Enqueue one caller-buffered all-to-all on a named mesh axis."""

        if self.is_trivial(group):
            output.copy_(input, non_blocking=input.device.type == "cuda")
            return None
        transport = self.transport(group)
        if not isinstance(transport, CollectiveTransport):
            raise RuntimeError(f"mesh axis {group!r} does not support collectives")
        return transport.all_to_all_single_into(
            output,
            input,
            output_splits,
            input_splits,
        )

    def all_gather_into_tensor(
        self,
        output: torch.Tensor,
        input: torch.Tensor,
        group: str = "sp",
    ) -> Any:
        """Enqueue one caller-buffered all-gather on a named mesh axis."""

        if self.is_trivial(group):
            output.copy_(input, non_blocking=input.device.type == "cuda")
            return None
        transport = self.transport(group)
        if not isinstance(transport, CollectiveTransport):
            raise RuntimeError(f"mesh axis {group!r} does not support collectives")
        return transport.all_gather_into_tensor(output, input)

    def gather_into_tensor(
        self,
        output: torch.Tensor | None,
        input: torch.Tensor,
        *,
        dst: int,
        group: str = "sp",
    ) -> Any:
        """Enqueue one caller-buffered gather on a named mesh axis."""

        if self.is_trivial(group):
            if int(dst) != 0 or output is None or tuple(output.shape) != (1, *input.shape):
                raise ValueError("single-rank gather requires destination zero output storage")
            output[0].copy_(input, non_blocking=input.device.type == "cuda")
            return None
        transport = self.transport(group)
        if not isinstance(transport, CollectiveTransport):
            raise RuntimeError(f"mesh axis {group!r} does not support collectives")
        return transport.gather_into_tensor(output, input, dst=dst)

    def broadcast(
        self,
        tensor: torch.Tensor,
        *,
        src: int,
        group: str = "tp",
    ) -> torch.Tensor:
        """Broadcast one caller-owned tensor on a named mesh axis."""

        if self.is_trivial(group):
            return tensor
        transport = self.transport(group)
        return transport.broadcast(tensor, src=src)

    @property
    def tp_size(self) -> int:
        return self.size("tp")

    @property
    def tp_rank(self) -> int:
        return self.coord("tp")

    @classmethod
    def trivial(cls, device: torch.device | str = "cpu") -> "DeviceMesh":
        return cls(axes={}, local_device=torch.device(device))

    @classmethod
    def of(cls, *axes: MeshAxis, device: torch.device | str = "cpu") -> "DeviceMesh":
        return cls(
            axes={ax.name: ax for ax in axes if ax.size > 1},
            local_device=torch.device(device),
        )


@dataclass(frozen=True, slots=True)
class TensorParallel:
    """Transport-free tensor-parallel coordinates used during layer construction."""

    rank: int
    size: int

    def __post_init__(self) -> None:
        if self.size <= 0:
            raise ValueError("tensor-parallel size must be positive")
        if self.rank < 0 or self.rank >= self.size:
            raise ValueError("tensor-parallel rank must satisfy 0 <= rank < size")

    @classmethod
    def from_mesh(cls, mesh: DeviceMesh) -> "TensorParallel":
        return cls(rank=mesh.tp_rank, size=mesh.tp_size)
