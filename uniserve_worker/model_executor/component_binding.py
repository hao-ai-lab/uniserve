"""Resolved component placement and borrowed numerical calls.

A ``ComponentBinding`` records where one model component runs across the
Worker's ranks. ``uniserve_worker.bootstrap.distributed`` builds one per
configured component (``ModelExecutor`` builds rank-local ones when none are
supplied), and ``uniserve_worker.bootstrap.components.bind_components``
attaches the component's ``Call`` values on member ranks. Runners and lanes
borrow those calls; they do not own the modules or communicators.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

from uniserve.distributed.mesh import Communicator, DeviceMesh
from uniserve.model import EntryPoint
from uniserve_worker.config.deployment import ComponentConfig
from uniserve_worker.protocol.call import CallKind


@dataclass(frozen=True, slots=True)
class Call:
    """A borrowed capability method and its numerical participation groups.

    ``groups`` lists the communicators the call exchanges tensors in on this
    rank; ``bind_components`` fills it from the module's communicators, the
    entry point's declared communication axes and, for a temporally
    distributed ``VideoPostprocessor``, the component's unit ring.
    """

    path: str
    module: torch.nn.Module
    entry_point: EntryPoint
    groups: tuple[Communicator, ...] = ()

    @property
    def forward(self) -> Callable[..., Any]:
        """The bound entry-point method, resolved on the module per access."""
        return getattr(self.module, self.entry_point.method)


@dataclass(slots=True, eq=False)
class ComponentBinding:
    """A component's placement and borrowed numerical calls.

    All ranks retain placement for routing and product sizing. Only members
    have a mesh; ``bind_components`` attaches their calls and call kinds
    after the model is loaded. Lanes reference these calls without copying
    the component's topology.
    """

    name: str
    config: ComponentConfig
    process_group: Communicator
    mesh: DeviceMesh | None
    device: torch.device
    # The ranks this component's media units are distributed over, ordered by
    # unit. Only a distributed component has one, and only on its own members.
    units: Communicator | None = None
    groups: tuple[Communicator, ...] = ()
    call_kinds: tuple[CallKind, ...] = ()
    calls: tuple[Call, ...] = ()

    def __post_init__(self) -> None:
        if any(
            rank not in self.process_group.ranks for rank in self.config.ranks
        ):
            raise ValueError(
                f"component {self.name} members lie outside its Worker"
            )

        # Mesh agreement is checked only without a distribution: a
        # ``temporal_units`` member's mesh is that rank alone, which differs
        # from the configured ranks whenever there are several.
        if self.config.distribution is None:
            if (self.mesh is not None) != self.owns:
                raise ValueError(
                    f"component {self.name} requires its local mesh"
                )
            if self.mesh is not None and (
                self.mesh.ranks != self.config.ranks
                or tuple(zip(self.mesh.axes, self.mesh.shape))
                != self.config.parallel_config.dimensions
            ):
                raise ValueError(
                    f"component {self.name} mesh disagrees with configuration"
                )

    @property
    def owns(self) -> bool:
        """Whether this process executes the configured component."""
        return self.process_group.global_rank in self.config.ranks

    @property
    def communicators(self) -> tuple[Communicator, ...]:
        """Every group this rank exchanges tensors in for this component.

        A temporally distributed component's numerical mesh is this rank
        alone, so its mesh groups state no exchange. The ranks holding
        consecutive media units still exchange their overlap over the unit
        ring, which is that component's only collective and belongs with the
        mesh's groups wherever participation decides behavior.
        """
        if self.units is None:
            return self.groups
        return (*self.groups, self.units)

    @property
    def input_ranks(self) -> tuple[int, ...]:
        """First pipeline-stage input members in configured order."""
        config = self.config
        if config.distribution is not None:
            return config.ranks

        # ``pp`` is the outermost axis of ``ParallelConfig.dimensions`` and
        # ranks map onto the mesh in row-major order, so stage 0 is the
        # leading ``world_size / pipeline_parallel_size`` ranks.
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

        # Only tp coordinate 0 on the last pipeline stage owns each replica;
        # other axes (context, Ulysses) keep all their ranks. The mesh is an
        # unbound descriptor used only for coordinate arithmetic, so any
        # member serves as its ``rank``.
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
