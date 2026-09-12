"""Process-world initialization and component communicator/resource ownership."""

from __future__ import annotations

import math
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
from types import TracebackType
from typing import TYPE_CHECKING, Any, Mapping, Self

import torch
import torch.distributed as dist

from ..foundation.errors import distributed_setup_error
from ..foundation.resources import close_resources
from ..nn.mesh import (
    Communicator,
    DeviceMesh,
    EntryBindings,
    PeerTensorWorkspace,
    SymmetricMemoryWorkspace,
)
from ..nn.parallel import ComponentConfig, ParallelConfig
from ..nn.parallel_attention import AttentionContextGeometry, AttentionContextWorkspace

if TYPE_CHECKING:
    from .collectives import PeerSumReduction


@dataclass
class DistributedEnvironment:
    """Own collective resources until dependent runners and captured graphs retire.

    This includes a default process group created during initialization. A
    pre-existing default group belongs to the caller and is never destroyed here.
    """

    rank: int
    world_size: int
    local_device: torch.device
    backend: str
    _groups: list[Any] = field(default_factory=list, repr=False)
    _sum_groups: dict[Any, Communicator] = field(default_factory=dict, repr=False)
    _workspaces: dict[tuple[object, ...], SymmetricMemoryWorkspace] = field(
        default_factory=dict, repr=False
    )
    _peer_tensors: dict[tuple[object, ...], PeerTensorWorkspace] = field(
        default_factory=dict, repr=False
    )

    def __enter__(self) -> Self:
        """Enter a scope owning the groups and distributed storage created here."""

        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        try:
            self.close()
        except BaseException as cleanup_error:
            if exc_value is None:
                raise
            exc_value.add_note(f"Resource cleanup also failed: {cleanup_error!r}")

    @property
    def process_group(self) -> Communicator:
        """Bind component transfers to the instance's ordered physical ranks."""

        return Communicator(
            tuple(range(self.world_size)),
            self.rank,
            "instance",
            self.local_device,
            dist.group.WORLD if dist.is_initialized() else None,
        )

    def initialize_entries(self, entries: Mapping[str, ComponentConfig]) -> EntryBindings:
        """Bind declared entries to their local meshes and shared process world."""

        meshes = initialize_model_parallel(
            self,
            {
                name: (entry.ranks, entry.parallel_config)
                for name, entry in entries.items()
                if entry.distribution is None
            },
        )

        # Temporal distribution partitions work across independent local models;
        # its width is an entry parameter, not a model-parallel mesh dimension.
        for name, entry in entries.items():
            if entry.distribution is not None and self.rank in entry.ranks:
                meshes[name] = DeviceMesh(
                    (self.rank,), self.rank, entry.parallel_config, self.local_device
                )

        return EntryBindings(dict(entries), meshes, self.process_group)

    def symmetric_memory(
        self,
        group: Communicator,
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
        group: Communicator,
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

    def attention_context(self, geometry: AttentionContextGeometry) -> AttentionContextWorkspace:
        """Allocate context K/V and fences using their actual physical row capacity."""

        group, rows = geometry.group, geometry.rows
        shape = (rows, geometry.heads, geometry.head_dim)
        if geometry.mapped:
            keys = self.peer_tensor(
                group,
                shape,
                dtype=geometry.dtype,
                name="attention_keys",
                row_multiple=geometry.block_size,
            )
            values = self.peer_tensor(
                group,
                shape,
                dtype=geometry.dtype,
                name="attention_values",
                row_multiple=geometry.block_size,
            )
            key, value = keys.global_tensor, values.global_tensor
            local_key, local_value = keys.local, values.local
        else:
            key = torch.empty(
                (rows * group.world_size, *shape[1:]), dtype=geometry.dtype, device=group.device
            )
            value = torch.empty_like(key)
            begin = group.rank_in_group * rows
            local_key, local_value = key[begin : begin + rows], value[begin : begin + rows]
        return AttentionContextWorkspace(
            key,
            value,
            local_key,
            local_value,
            torch.empty(
                key.shape[0] // geometry.block_size, dtype=torch.int32, device=group.device
            ),
            torch.zeros(1, dtype=torch.int32, device=group.device),
            torch.empty(group.world_size, dtype=torch.int32, device=group.device),
        )

    def stream_collectives(self, stream: torch.cuda.Stream):
        """Allocate independent communication resources for one computation stream."""

        from .collectives import NcclStreamCollectives

        bindings = {}
        try:
            for group in self._groups:
                if dist.get_backend(group) == "nccl":
                    bindings[group.group_name] = NcclStreamCollectives(group, stream)
        except BaseException as error:
            for binding in reversed(tuple(bindings.values())):
                try:
                    binding.close()
                except BaseException as cleanup_error:
                    error.add_note(f"collective binding cleanup failed: {cleanup_error!r}")
            raise
        return bindings

    def sum_reductions(self) -> dict[Any, PeerSumReduction]:
        """Allocate collective scratch for one serialized full-device execution scope.

        The runner invokes this before variable memory pools are sized and owns
        the returned workspaces until all of its graph executables retire.
        """

        from .collectives import PeerSumReduction, supports_peer_reduction

        bindings: dict[Any, PeerSumReduction] = {}
        try:
            for process_group, group in self._sum_groups.items():
                if supports_peer_reduction(group):
                    bindings[process_group] = PeerSumReduction(group)
            return bindings
        except Exception:
            for reduction in reversed(tuple(bindings.values())):
                reduction.close()
            raise

    def close(self) -> None:
        """Release all owned resources after the caller retires runners.

        Attempt every release even if device synchronization or a group teardown
        fails. Component groups retire before the default world they depend on.
        """

        actions: list[Callable[[], object]] = []
        if self.local_device.type == "cuda":
            actions.append(partial(torch.cuda.synchronize, self.local_device))
        actions.extend((self._peer_tensors.clear, self._workspaces.clear, self._sum_groups.clear))
        actions.extend(
            partial(dist.destroy_process_group, group) for group in reversed(self._groups)
        )
        self._groups.clear()
        close_resources(*actions)


def init_distributed_environment(
    *,
    rank: int,
    local_rank: int,
    world_size: int,
    device: str,
    backend: str | None = None,
    init_method: str | None = None,
) -> DistributedEnvironment:
    """Select the rank's device and join or create its physical process world.

    The returned environment owns any process group it creates; callers must
    close it after all dependent execution resources have retired. Used with
    `with`, it releases resources on construction failure and retains them on
    success for the constructed owner's lifetime.
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
    environment: DistributedEnvironment,
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
                    if name == "tp" and len(members) > 1:
                        environment._sum_groups[process_groups[backend_members]] = groups[name]
        if environment.rank in layout.ranks:
            meshes[component] = DeviceMesh(
                layout.ranks,
                environment.rank,
                layout.parallel_config,
                environment.local_device,
                groups,
            )
    return meshes
