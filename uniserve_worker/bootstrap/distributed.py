"""Construct process worlds and component fibers in canonical rank order."""

from __future__ import annotations

from collections.abc import Mapping

from uniserve.distributed.mesh import DeviceMesh
from uniserve.distributed.process_groups import ProcessGroups, initialize_model_parallel
from uniserve_worker.config import ComponentConfig

from ..execution.model_entry import ModelEntry


def initialize_entries(
    groups: ProcessGroups, entries: Mapping[str, ComponentConfig]
) -> dict[str, ModelEntry]:
    """Bind declared entries to their local meshes and shared process world."""

    meshes = initialize_model_parallel(
        groups,
        {
            name: (entry.ranks, entry.parallel_config)
            for name, entry in entries.items()
            if entry.distribution is None
        },
    )

    # Temporal distribution partitions work across independent local models;
    # its width is an entry parameter, not a model-parallel mesh dimension.
    for name, entry in entries.items():
        if entry.distribution is not None and groups.rank in entry.ranks:
            meshes[name] = DeviceMesh(
                (groups.rank,), groups.rank, entry.parallel_config, groups.local_device
            )

    if not entries or not any(groups.rank in entry.ranks for entry in entries.values()):
        raise ValueError("rank has no configured computation entry")
    return {
        name: ModelEntry(name, config, groups.process_group, meshes.get(name), groups.local_device)
        for name, config in entries.items()
    }
