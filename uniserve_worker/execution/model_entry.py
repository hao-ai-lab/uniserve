"""Resolved numerical module bindings and their resident input geometry."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, TypeAlias

import torch

from ..nn.mesh import Communicator, DeviceMesh
from ..nn.parallel import ComponentConfig
from ..protocol.batch import Computation, TensorSpec
from ..runtime.collectives import NcclCommunicator
from .cuda_stream import CudaStream

if TYPE_CHECKING:
    from .forward_batch import ForwardOutput
    from .input_buffers import InputBuffers
from .cuda_graph import CudaGraph

TensorOutput: TypeAlias = torch.Tensor | tuple[torch.Tensor, ...]
TensorSignature: TypeAlias = tuple[tuple[tuple[int, ...], torch.dtype, tuple[int, ...]], ...]


def tensor_signature(inputs: tuple[torch.Tensor, ...]) -> TensorSignature:
    """Include layout as well as shape in a tensor-only computation's identity."""

    return tuple((tuple(value.shape), value.dtype, tuple(value.stride())) for value in inputs)


def capture_required(missing: bool, groups: tuple[Communicator, ...], device: torch.device) -> bool:
    """Coordinate first-use work over the actual numerical communication groups."""

    if not groups:
        return missing
    decision = torch.tensor(int(missing), dtype=torch.int32, device=device)
    for group in groups:
        group.all_reduce_max(decision)
    return bool(decision.item())


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
    forward: Callable[..., TensorOutput | ForwardOutput] | None = None
    groups: tuple[Communicator, ...] = ()
    output_schema: tuple[TensorSpec, ...] = ()
    computations: tuple[Computation, ...] = ()
    input_buffers: InputBuffers | None = None
    cuda_stream: CudaStream | None = None
    collectives: dict[str, NcclCommunicator] = field(default_factory=dict)
    fixed_inputs: tuple[torch.Tensor, ...] | None = None
    signature: TensorSignature | None = None
    graph: CudaGraph[TensorOutput] | None = None

    def __post_init__(self) -> None:
        if any(rank not in self.process_group.ranks for rank in self.config.ranks):
            raise ValueError(f"entry {self.name} members lie outside its Worker")
        if self.config.distribution is None:
            if (self.mesh is not None) != self.owns:
                raise ValueError(f"entry {self.name} requires its local mesh")
            if self.mesh is not None and (
                self.mesh.ranks != self.config.ranks
                or self.mesh.parallel_config != self.config.parallel_config
            ):
                raise ValueError(f"entry {self.name} mesh disagrees with configuration")
        if self.mesh is not None:
            if not self.groups:
                self.groups = tuple(
                    {group.name: group for group in self.mesh.groups.values()}.values()
                )

    @property
    def owns(self) -> bool:
        """Whether this process executes the configured entry."""
        return self.process_group.rank in self.config.ranks

    @property
    def input_ranks(self) -> tuple[int, ...]:
        """First pipeline-stage input members in configured order."""
        config = self.config
        if config.distribution is not None:
            return config.ranks
        width = config.parallel_config.world_size // config.parallel_config.pipeline_parallel_size
        return config.ranks[:width]

    @property
    def output_ranks(self) -> tuple[int, ...]:
        """Final pipeline-stage members with tensor replicas counted once."""
        config = self.config
        if config.distribution is not None:
            return config.ranks
        geometry = DeviceMesh(
            config.ranks, config.ranks[0], config.parallel_config, self.process_group.device
        )
        axes = tuple(name for name, _ in geometry.dimensions)
        return tuple(
            rank
            for rank in config.ranks
            if geometry.get_coordinate(rank)[axes.index("tp")] == 0
            and geometry.get_coordinate(rank)[axes.index("pp")]
            == config.parallel_config.pipeline_parallel_size - 1
        )
