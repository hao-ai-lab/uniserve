"""Process-world initialization and component communicator/resource ownership."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Any, Mapping

import torch
import torch.distributed as dist

from ..foundation.errors import distributed_setup_error
from ..nn.mesh import DeviceMesh, GroupCoordinator, PeerTensorWorkspace, SymmetricMemoryWorkspace
from ..nn.parallel import ParallelConfig


@dataclass
class DistributedEnvironment:
    """Own group resources until dependent runners and captured graphs retire."""

    rank: int
    world_size: int
    local_device: torch.device
    backend: str
    _groups: list[Any] = field(default_factory=list, repr=False)
    _workspaces: dict[tuple[object, ...], SymmetricMemoryWorkspace] = field(
        default_factory=dict, repr=False
    )
    _peer_tensors: dict[tuple[object, ...], PeerTensorWorkspace] = field(
        default_factory=dict, repr=False
    )

    @property
    def process_group(self) -> GroupCoordinator:
        """Bind component transfers to the instance's ordered physical ranks."""

        return GroupCoordinator(
            tuple(range(self.world_size)),
            self.rank,
            "instance",
            self.local_device,
            dist.group.WORLD if dist.is_initialized() else None,
            self.symmetric_memory,
            self.peer_tensor,
        )

    def symmetric_memory(
        self,
        group: GroupCoordinator,
        shape: tuple[int, ...],
        *,
        dtype: torch.dtype,
        name: str,
        layout: tuple[object, ...],
    ) -> SymmetricMemoryWorkspace:
        """Own stable peer allocations keyed by membership and execution layout."""

        key = (group.name, group.ranks, layout, name, shape, dtype, self.local_device)
        if key in self._workspaces:
            return self._workspaces[key]
        peers: tuple[torch.Tensor, ...]
        if group.world_size == 1:
            local = torch.empty(shape, dtype=dtype, device=self.local_device)
            handle = None
            peers = (local,)
        else:
            if self.backend != "nccl":
                raise distributed_setup_error("symmetric peer memory requires the NCCL backend")
            import torch.distributed._symmetric_memory as symm_mem

            if symm_mem.get_backend(self.local_device) != "NCCL":
                symm_mem.set_backend("NCCL")
            local = symm_mem.empty(shape, dtype=dtype, device=self.local_device)
            handle = symm_mem.rendezvous(local, group._require())
            backend_ranks = sorted(group.ranks)
            peers = tuple(
                handle.get_buffer(backend_ranks.index(rank), shape, dtype) for rank in group.ranks
            )
        workspace = SymmetricMemoryWorkspace(group, local, peers, handle)
        self._workspaces[key] = workspace
        return workspace

    def peer_tensor(
        self,
        group: GroupCoordinator,
        shape: tuple[int, ...],
        *,
        dtype: torch.dtype,
        name: str,
        row_multiple: int,
    ) -> PeerTensorWorkspace:
        """Map ordered CUDA peer allocations without replicating tensor data."""

        from uniserve_kernel.peer_memory import allocation_granularity

        from .peer_memory import allocate_peer_tensor

        if not shape or any(size < 1 for size in shape) or row_multiple < 1:
            raise ValueError("peer tensor extents and row alignment must be positive")
        key = (group.name, group.ranks, name, shape, dtype, row_multiple, self.local_device)
        if key in self._peer_tensors:
            return self._peer_tensors[key]
        element_bytes = torch.empty((), dtype=dtype).element_size()
        row_bytes = math.prod(shape[1:]) * element_bytes
        granularity = allocation_granularity(self.local_device)
        aligned_rows = math.lcm(row_multiple, granularity // math.gcd(row_bytes, granularity))
        capacity = ((shape[0] + aligned_rows - 1) // aligned_rows) * aligned_rows
        global_tensor = allocate_peer_tensor(
            group,
            (capacity, *shape[1:]),
            dtype=dtype,
        )
        local = global_tensor.narrow(0, group.rank_in_group * capacity, capacity)
        workspace = PeerTensorWorkspace(group, local, global_tensor)
        self._peer_tensors[key] = workspace
        return workspace

    def close(self) -> None:
        """Release peer allocations and groups after the caller retires runners."""

        if self.local_device.type == "cuda":
            torch.cuda.synchronize(self.local_device)
        self._peer_tensors.clear()
        self._workspaces.clear()
        for group in reversed(self._groups):
            dist.destroy_process_group(group)
        self._groups.clear()


def init_distributed_environment(
    *,
    rank: int,
    local_rank: int,
    world_size: int,
    device: str,
    backend: str | None = None,
    init_method: str | None = None,
) -> DistributedEnvironment:
    """Initialize physical launch information without assigning model-parallel degrees."""

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
    environment = DistributedEnvironment(rank, world_size, local_device, backend)
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
    return environment


def _group_options(backend: str):
    if backend != "nccl":
        return None
    options = dist.ProcessGroupNCCL.Options()
    options.use_pg_for_symm_mem_rendezvous = True
    return options


def initialize_model_parallel(
    environment: DistributedEnvironment,
    components: Mapping[str, tuple[tuple[int, ...], ParallelConfig]],
) -> dict[str, DeviceMesh]:
    """Construct every component fiber in canonical order on every process.

    Nonmembers participate in group construction but receive no local component
    mesh. Component membership is supplied by deployment, never inferred from
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
                    )
                    process_groups[backend_members] = group
                    if environment.rank in members:
                        environment._groups.append(group)
                if environment.rank in members:
                    groups[name] = GroupCoordinator(
                        members,
                        environment.rank,
                        f"{component}.{name}:{','.join(map(str, members))}",
                        environment.local_device,
                        process_groups.get(backend_members),
                        environment.symmetric_memory,
                        environment.peer_tensor,
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
