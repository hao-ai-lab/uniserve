"""Construct process worlds and component fibers in canonical rank order."""

from __future__ import annotations

from collections.abc import Mapping

from uniserve.distributed.mesh import DeviceMesh
from uniserve.runtime.process_groups import ProcessGroups

from ..execution.model_entry import ModelEntry
from .config import ComponentConfig


def initialize_entries(
    groups: ProcessGroups, entries: Mapping[str, ComponentConfig]
) -> dict[str, ModelEntry]:
    """Bind declared entries to their local meshes and shared process world."""
    meshes = {}
    rings = {}
    for name, entry in sorted(entries.items()):
        if entry.distribution is not None:
            # Media units are independent local invocations, so the numerical
            # mesh is this rank alone. The ranks holding consecutive units still
            # exchange the overlap between them, over a ring every rank creates
            # in the same order because group creation spans the process world.
            ring = groups.bind(
                DeviceMesh(
                    ranks=entry.ranks,
                    shape=(len(entry.ranks),),
                    axes=("units",),
                    rank=groups.rank,
                ),
                device=groups.device,
            )
            if groups.rank not in entry.ranks:
                continue
            rings[name] = ring.get_group("units")
            ranks = (groups.rank,)
        else:
            ranks = entry.ranks
        topology = DeviceMesh(
            ranks=ranks,
            shape=tuple(size for _, size in entry.parallel_config.dimensions),
            axes=tuple(axis for axis, _ in entry.parallel_config.dimensions),
            rank=groups.rank,
        )
        bound = groups.bind(topology, device=groups.device)
        if groups.rank in ranks:
            meshes[name] = bound

    if not entries or not any(
        groups.rank in entry.ranks for entry in entries.values()
    ):
        raise ValueError("rank has no configured computation entry")

    return {
        name: ModelEntry(
            name,
            config,
            groups.process_group,
            meshes.get(name),
            groups.device,
            units=rings.get(name),
        )
        for name, config in entries.items()
    }
