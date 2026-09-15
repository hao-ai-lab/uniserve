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
    for name, entry in sorted(entries.items()):
        # Temporal units are independent local invocations. All ranks still
        # bind every distributed component in a common creation order.
        if entry.distribution is not None:
            if groups.rank not in entry.ranks:
                continue
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

    if not entries or not any(groups.rank in entry.ranks for entry in entries.values()):
        raise ValueError("rank has no configured computation entry")

    return {
        name: ModelEntry(name, config, groups.process_group, meshes.get(name), groups.device)
        for name, config in entries.items()
    }
