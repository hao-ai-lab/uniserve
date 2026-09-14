"""Construction and ownership of explicitly configured process groups."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from functools import partial
from types import TracebackType
from typing import Any, Self

import torch
import torch.distributed as dist

from uniserve.distributed.mesh import Communicator, DeviceMesh
from uniserve.distributed.parallel import ParallelConfig
from uniserve.runtime.resources import close_resources


@dataclass
class ProcessGroups:
    """Keep owned subgroup handles and borrow any pre-existing default world."""

    rank: int
    world_size: int
    local_device: torch.device
    backend: str
    _groups: list[Any] = field(default_factory=list, repr=False)

    def __enter__(self) -> Self:
        """Enter a scope owning the process groups created here."""

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

    def close(self) -> None:
        """Destroy owned process groups after all communication consumers retire.

        Attempt every release even if device synchronization or a group teardown
        fails. Component groups retire before the default world they depend on.
        """

        actions: list[Callable[[], object]] = []
        if self.local_device.type == "cuda":
            actions.append(partial(torch.cuda.synchronize, self.local_device))
        actions.extend(
            partial(dist.destroy_process_group, group) for group in reversed(self._groups)
        )
        self._groups.clear()
        close_resources(*actions)


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
        raise ValueError("launch rank must satisfy 0 <= rank < positive world_size")
    local_device = torch.device(device)
    if local_device.type == "cuda":
        index = local_device.index if local_device.index is not None else local_rank
        local_device = torch.device("cuda", index)
        if not torch.cuda.is_available() or index >= torch.cuda.device_count():
            raise ValueError(
                f"cuda device {local_device} is outside the {torch.cuda.device_count()} visible CUDA device(s)"
            )
        torch.cuda.set_device(local_device)
    backend = backend or ("nccl" if local_device.type == "cuda" else "gloo")
    environment = ProcessGroups(rank, world_size, local_device, backend)
    if dist.is_initialized():
        if (dist.get_rank(), dist.get_world_size()) != (rank, world_size):
            raise ValueError("existing process world disagrees with supplied rank/world_size")
        if dist.get_backend() != backend:
            raise ValueError("existing process world disagrees with supplied backend")
    elif world_size > 1:
        if not init_method:
            port = os.environ.get("MASTER_PORT")
            if not port:
                raise ValueError("MASTER_PORT or an explicit init_method is required")
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
    mesh. Component membership is supplied explicitly, never inferred from
    the process-world size.
    """

    # Validate all geometry before entering collective group creation so a bad
    # component cannot leave another rank waiting in a later rendezvous.
    layouts = {}
    for name, (ranks, config) in sorted(components.items()):
        if not name or not ranks or any(rank >= environment.world_size for rank in ranks):
            raise ValueError(f"component {name!r} has invalid process membership {ranks}")
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
