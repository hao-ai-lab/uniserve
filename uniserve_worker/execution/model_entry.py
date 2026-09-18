"""Resolved numerical module bindings and their resident input storage."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeAlias

import torch

from uniserve.distributed.mesh import Communicator, DeviceMesh
from uniserve.model import EntryPoint
from uniserve.runtime import CUDAStream

from ..bootstrap.config import ComponentConfig
from ..protocol.call import CallKind
from ..protocol.tensor import OutputInfo

if TYPE_CHECKING:
    from .batch import ExecutionOutput
    from .input_buffers import InputBuffers
from uniserve.runtime import ExecutionContext

TensorOutput: TypeAlias = torch.Tensor | tuple[torch.Tensor, ...]


def capture_required(
    missing: bool, groups: tuple[Communicator, ...], device: torch.device
) -> bool:
    """Coordinate first-use work over the actual numerical communication.

    groups.
    """
    if not groups:
        return missing
    decision = torch.tensor(int(missing), dtype=torch.int32, device=device)
    for group in groups:
        group.all_reduce(decision, op="max")
    return bool(decision.item())


@dataclass(frozen=True, slots=True)
class Call:
    """A borrowed capability method and its numerical participation groups."""

    path: str
    module: torch.nn.Module
    entry: EntryPoint
    groups: tuple[Communicator, ...] = ()

    @property
    def forward(self) -> Callable[..., Any]:
        return getattr(self.module, self.entry.method)


@dataclass(slots=True, eq=False)
class ModelEntry:
    """An entry's placement, local callable and resident execution addresses.

    All ranks retain placement for routing and product sizing. Only members
    have a mesh; their callable is attached after checkpoint materialization.
    Physical stream bindings share the immutable configuration and mesh.
    """

    name: str
    config: ComponentConfig
    process_group: Communicator
    mesh: DeviceMesh | None
    device: torch.device
    # The ranks this component's media units are distributed over, ordered by
    # unit. Only a distributed entry has one, and only on its own members.
    units: Communicator | None = None
    forward: Callable[..., TensorOutput | ExecutionOutput] | None = None
    groups: tuple[Communicator, ...] = ()
    outputs: tuple[OutputInfo, ...] = ()
    call_kinds: tuple[CallKind, ...] = ()
    component: str = ""
    calls: tuple[Call, ...] = ()
    context: ExecutionContext | None = None
    input_buffers: InputBuffers | None = None
    cuda_stream: CUDAStream | None = None

    def __post_init__(self) -> None:
        if any(
            rank not in self.process_group.ranks for rank in self.config.ranks
        ):
            raise ValueError(
                f"entry {self.name} members lie outside its Worker"
            )

        if self.config.distribution is None:
            if (self.mesh is not None) != self.owns:
                raise ValueError(f"entry {self.name} requires its local mesh")
            if self.mesh is not None and (
                self.mesh.ranks != self.config.ranks
                or tuple(zip(self.mesh.axes, self.mesh.shape))
                != self.config.parallel_config.dimensions
            ):
                raise ValueError(
                    f"entry {self.name} mesh disagrees with configuration"
                )

        if self.mesh is not None:
            if not self.groups:
                # Deduplicate the mesh's per-axis groups by their member ranks.
                self.groups = tuple(
                    {
                        group.ranks: group
                        for axes in self.mesh._groups
                        for group in (self.mesh.get_group(axes),)
                    }.values()
                )

    @property
    def owns(self) -> bool:
        """Whether this process executes the configured entry."""
        return self.process_group.global_rank in self.config.ranks

    @property
    def input_ranks(self) -> tuple[int, ...]:
        """First pipeline-stage input members in configured order."""
        config = self.config
        if config.distribution is not None:
            return config.ranks

        width = (
            config.parallel_config.world_size
            // config.parallel_config.pipeline_parallel_size
        )
        return config.ranks[:width]

    @property
    def output_ranks(self) -> tuple[int, ...]:
        """Final pipeline-stage members with tensor replicas counted once."""
        config = self.config
        if config.distribution is not None:
            return config.ranks

        # Only tp coordinate 0 on the last pipeline stage owns each replica.
        mesh = DeviceMesh(
            ranks=config.ranks,
            rank=config.ranks[0],
            shape=tuple(size for _, size in config.parallel_config.dimensions),
            axes=tuple(axis for axis, _ in config.parallel_config.dimensions),
        )
        axes = mesh.axes
        return tuple(
            rank
            for rank in config.ranks
            if mesh.coordinate(rank)[axes.index("tp")] == 0
            and mesh.coordinate(rank)[axes.index("pp")]
            == config.parallel_config.pipeline_parallel_size - 1
        )
