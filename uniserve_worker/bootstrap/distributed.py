"""Construct process worlds and component fibers in canonical rank order."""

from __future__ import annotations

import os
from collections.abc import Mapping

import torch
import torch.distributed as dist

from ..execution.model_entry import ModelEntry
from ..foundation.errors import distributed_setup_error
from ..nn.mesh import Communicator, DeviceMesh
from ..nn.parallel import ComponentConfig, ParallelConfig
from ..runtime.process_groups import ProcessGroups


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


def initialize_process_groups(
    *,
    rank: int,
    local_rank: int,
    world_size: int,
    device: str,
    backend: str | None = None,
    init_method: str | None = None,
) -> ProcessGroups:
    """Select the rank's device and join or create its physical process world.

    The returned ProcessGroups owns any group created here. The caller closes it
    after all dependent execution resources retire, or transfers that obligation
    to the constructed Worker. Its context manager closes on every scope exit.
    """

    if world_size < 1 or not 0 <= rank < world_size or local_rank < 0:
        raise distributed_setup_error("launch rank must satisfy 0 <= rank < positive world_size")
    local_device = torch.device(device)
    if local_device.type == "cuda":
        index = local_device.index if local_device.index is not None else local_rank
        local_device = torch.device("cuda", index)
        if not torch.cuda.is_available() or index >= torch.cuda.device_count():
            raise distributed_setup_error(
                f"cuda device {local_device} is outside the {torch.cuda.device_count()} visible CUDA device(s)"
            )
        torch.cuda.set_device(local_device)
    backend = backend or ("nccl" if local_device.type == "cuda" else "gloo")
    environment = ProcessGroups(rank, world_size, local_device, backend)
    if dist.is_initialized():
        if (dist.get_rank(), dist.get_world_size()) != (rank, world_size):
            raise distributed_setup_error(
                "existing process world disagrees with worker launch rank/world_size"
            )
        if dist.get_backend() != backend:
            raise distributed_setup_error(
                "existing process world disagrees with worker launch backend"
            )
    elif world_size > 1:
        if not init_method:
            port = os.environ.get("MASTER_PORT")
            if not port:
                raise distributed_setup_error(
                    "MASTER_PORT or --distributed-init-method is required"
                )
            init_method = f"tcp://{os.environ.get('MASTER_ADDR', '127.0.0.1')}:{port}"
        dist.init_process_group(
            backend=backend,
            init_method=init_method,
            rank=rank,
            world_size=world_size,
            pg_options=_group_options(backend),
            device_id=local_device if backend == "nccl" else None,
        )
        environment._groups.append(dist.group.WORLD)
    return environment


def _group_options(backend: str):
    if backend != "nccl":
        return None
    options = dist.ProcessGroupNCCL.Options()
    options.use_pg_for_symm_mem_rendezvous = True
    # NCCL checks topology, driver, symmetric windows and collective kind for
    # each request. Zero-CTA is a preference; unsupported requests retain its
    # kernel algorithm. Logical axis names do not describe these capabilities.
    # These CUDA 13 fields are exported by the installed PyTorch extension;
    # its NCCLConfig stubs do not yet declare them.
    options.config.cta_policy = dist.ProcessGroupNCCL.NCCL_CTA_POLICY_ZERO  # type: ignore[attr-defined]
    return options


def initialize_model_parallel(
    environment: ProcessGroups,
    components: Mapping[str, tuple[tuple[int, ...], ParallelConfig]],
) -> dict[str, DeviceMesh]:
    """Construct every component fiber in canonical order on every process.

    Nonmembers participate in group construction but receive no local component
    mesh. Component membership is supplied by worker_config, never inferred from
    the process-world size.
    """

    # Validate all geometry before entering collective group creation so a bad
    # component cannot leave another rank waiting in a later rendezvous.
    layouts = {}
    for name, (ranks, config) in sorted(components.items()):
        if not name or not ranks or any(rank >= environment.world_size for rank in ranks):
            raise distributed_setup_error(
                f"component {name!r} has invalid process membership {ranks}"
            )
        layouts[name] = DeviceMesh(ranks, ranks[0], config, environment.local_device)
    meshes = {}
    for component, layout in layouts.items():
        groups = {}
        names = tuple(name for name, _ in layout.dimensions) + ("sp",)
        if "cp_row" in names:
            names += ("cp",)
        # Identical fibers of a component share a communicator, while logical
        # names remain distinct for resource and graph identity.
        process_groups = {}
        for name in names:
            for members in layout.group_members(name):
                backend_members = tuple(sorted(members))
                if len(members) > 1 and backend_members not in process_groups:
                    group = dist.new_group(
                        ranks=list(backend_members),
                        backend=environment.backend,
                        pg_options=_group_options(environment.backend),
                        device_id=environment.local_device
                        if environment.backend == "nccl"
                        else None,
                    )
                    process_groups[backend_members] = group
                    if environment.rank in members:
                        environment._groups.append(group)
                if environment.rank in members:
                    groups[name] = Communicator(
                        members,
                        environment.rank,
                        f"{component}.{name}:{','.join(map(str, members))}",
                        environment.local_device,
                        process_groups.get(backend_members),
                    )
        if environment.rank in layout.ranks:
            meshes[component] = DeviceMesh(
                layout.ranks,
                environment.rank,
                layout.parallel_config,
                environment.local_device,
                groups,
            )
    return meshes
