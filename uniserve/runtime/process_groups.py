"""Construction and ownership of explicitly configured process groups."""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
from itertools import combinations
from types import MappingProxyType, TracebackType
from typing import Any, Self

import torch
import torch.distributed as dist

from uniserve.distributed.mesh import Communicator, DeviceMesh
from uniserve.runtime.resources import close_resources


@dataclass
class ProcessGroups:
    """Keep owned subgroup handles and borrow any pre-existing default world."""

    rank: int
    world_size: int
    device: torch.device
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
            self.device,
            dist.group.WORLD if dist.is_initialized() else None,
        )

    def bind(self, mesh: DeviceMesh, *, device: torch.device | str) -> DeviceMesh:
        """Bind topology fibers in the same order on every process.

        Nonmembers participate in group creation and retain only the topology.
        Equivalent fibers of this binding share one backend handle; each mesh
        binding owns its own ordering domain for independent model components.
        """

        if mesh.rank != self.rank or any(rank >= self.world_size for rank in mesh.ranks):
            raise ValueError("mesh ranks disagree with the process world")

        device = torch.device(device)
        handles = {}
        groups = {}
        for width in range(len(mesh.axes) + 1):
            for axes in combinations(mesh.axes, width):
                for members in mesh.members(axes):
                    ordered = tuple(sorted(members))
                    if len(members) > 1 and ordered not in handles:
                        if not dist.is_initialized():
                            raise RuntimeError(
                                "multi-rank mesh binding requires an initialized process world"
                            )
                        handle = dist.new_group(
                            ranks=list(ordered),
                            backend=self.backend,
                            pg_options=_group_options(self.backend),
                            device_id=device if self.backend == "nccl" else None,
                        )
                        handles[ordered] = handle
                        if self.rank in members:
                            self._groups.append(handle)
                    if self.rank in members:
                        groups[axes] = Communicator(
                            members,
                            members.index(self.rank),
                            ".".join(axes) or "local",
                            device,
                            handles.get(ordered),
                        )

        result = DeviceMesh(ranks=mesh.ranks, shape=mesh.shape, axes=mesh.axes, rank=mesh.rank)
        object.__setattr__(result, "_device", device)
        object.__setattr__(result, "_groups", MappingProxyType(groups))
        return result

    def close(self) -> None:
        """Destroy owned process groups after all communication consumers retire.

        Attempt every release even if device synchronization or a group teardown
        fails. Component groups retire before the default world they depend on.
        """

        actions: list[Callable[[], object]] = []
        if self.device.type == "cuda":
            actions.append(partial(torch.cuda.synchronize, self.device))
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
    device: torch.device | str,
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
