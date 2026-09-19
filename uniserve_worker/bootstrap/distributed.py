"""Construct process worlds and component fibers in canonical rank order."""

from __future__ import annotations

from collections.abc import Mapping

from uniserve.distributed.mesh import DeviceMesh
from uniserve.runtime.process_groups import ProcessGroups

from ..execution.component_binding import ComponentBinding
from .config import ComponentConfig


def initialize_components(
    groups: ProcessGroups, components: Mapping[str, ComponentConfig]
) -> dict[str, ComponentBinding]:
    """Bind declared components to their local meshes and process world."""
    meshes = {}
    rings = {}
    for name, component in sorted(components.items()):
        if component.distribution is not None:
            # Media units are independent local invocations, so the numerical
            # mesh is this rank alone. The ranks holding consecutive units still
            # exchange the overlap between them, over a ring every rank creates
            # in the same order because group creation spans the process world.
            ring = groups.bind(
                DeviceMesh(
                    ranks=component.ranks,
                    shape=(len(component.ranks),),
                    axes=("units",),
                    rank=groups.rank,
                ),
                device=groups.device,
            )
            if groups.rank not in component.ranks:
                continue
            rings[name] = ring.get_group("units")
            ranks = (groups.rank,)
        else:
            ranks = component.ranks
        topology = DeviceMesh(
            ranks=ranks,
            shape=tuple(
                size for _, size in component.parallel_config.dimensions
            ),
            axes=tuple(
                axis for axis, _ in component.parallel_config.dimensions
            ),
            rank=groups.rank,
        )
        bound = groups.bind(topology, device=groups.device)
        if groups.rank in ranks:
            meshes[name] = bound

    if not components or not any(
        groups.rank in component.ranks for component in components.values()
    ):
        raise ValueError("rank has no configured component")

    return {
        name: ComponentBinding(
            name,
            config,
            groups.process_group,
            meshes.get(name),
            groups.device,
            units=rings.get(name),
        )
        for name, config in components.items()
    }
