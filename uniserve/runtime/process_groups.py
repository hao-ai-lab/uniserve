"""Construction and ownership of explicitly configured process groups."""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from functools import partial
from types import MappingProxyType, TracebackType
from typing import Any, Self

import torch
import torch.distributed as dist
from torch.distributed.constants import (
    default_pg_nccl_timeout,
    default_pg_timeout,
)

from uniserve.distributed.mesh import Communicator, DeviceMesh
from uniserve.runtime.resources import close_resources


@dataclass(frozen=True)
class Rendezvous:
    """The TCP store a process world forms at.

    Rank 0 serves the store and every rank connects to it at ``host:port``.
    ``listen_fd`` is a socket, already bound and listening at that port, that
    rank 0 serves the store on, so the port is held from whoever reserved it
    until rank 0 exits and no other process can take it meanwhile. Without
    one, rank 0 binds the port itself. Only rank 0 takes a socket; it owns the
    descriptor from then on.
    """

    host: str
    port: int
    listen_fd: int | None = None


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
            # Destroying an owned group can wait on peers this rank cannot
            # observe, so an exception retains owned groups. Without owned
            # groups there is nothing whose retirement needs another rank.
            self.close(aborted=exc_value is not None and bool(self._groups))
        except BaseException as cleanup_error:
            if exc_value is None:
                raise
            exc_value.add_note(
                f"Resource cleanup also failed: {cleanup_error!r}"
            )

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

    def bind(
        self,
        mesh: DeviceMesh,
        *,
        device: torch.device | str,
        axes: Iterable[tuple[str, ...]] | None = None,
    ) -> DeviceMesh:
        """Bind topology fibers in the same order on every process.

        Nonmembers participate in group creation and retain only the topology.
        Equivalent fibers of this binding share one backend handle; each mesh
        binding owns its own ordering domain for independent model components.
        ``axes`` names the required fibers. By default only individual axes
        are bound; models supply their numerical communication requirements.
        """
        if mesh.rank != self.rank or any(
            rank >= self.world_size for rank in mesh.ranks
        ):
            raise ValueError("mesh ranks disagree with the process world")

        device = torch.device(device)
        handles = {}
        groups = {}
        selections = (
            tuple((axis,) for axis in mesh.axes)
            if axes is None
            else tuple(axes)
        )
        for selection in sorted(set(selections)):
            for members in mesh.members(selection):
                ordered = tuple(sorted(members))
                if len(members) > 1 and ordered not in handles:
                    if not dist.is_initialized():
                        raise RuntimeError(
                            "multi-rank mesh binding requires an "
                            "initialized process world"
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
                    groups[selection] = Communicator(
                        members,
                        members.index(self.rank),
                        ".".join(selection) or "local",
                        device,
                        handles.get(ordered),
                    )

        result = DeviceMesh(
            ranks=mesh.ranks, shape=mesh.shape, axes=mesh.axes, rank=mesh.rank
        )
        object.__setattr__(result, "_device", device)
        object.__setattr__(result, "_groups", MappingProxyType(groups))
        return result

    def close(self, *, aborted: bool = False) -> None:
        """Destroy owned process groups.

        Destroy owned process groups after all communication consumers
        retire.

        Attempt every release even if device synchronization or a group teardown
        fails. Component groups retire before the default world they depend on.

        ``aborted`` retains the groups without destroying them, for a rank
        releasing after a failure. Destroying a group and synchronizing the
        device both wait on ranks that are still serving, so on that path they
        would replace a reported failure with a stall. The process that owns
        the groups is leaving, and its exit releases them.
        """
        actions: list[Callable[[], object]] = []
        if aborted:
            from .resources import retain_until_exit

            retain_until_exit(self)
            return
        if not self._groups:
            return
        if self.device.type == "cuda":
            actions.append(partial(torch.cuda.synchronize, self.device))
        actions.extend(
            partial(dist.destroy_process_group, group)
            for group in reversed(self._groups)
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
    rendezvous: Rendezvous | None = None,
) -> ProcessGroups:
    """Select the rank's device and join or create its physical process world.

    A new world forms at ``rendezvous`` or through the torch ``init_method``
    URL, which are exclusive; without either, ``MASTER_ADDR`` and
    ``MASTER_PORT`` name a TCP init method. Both are ignored when the process
    already has a world, which is borrowed.

    The returned ProcessGroups owns any group created here. The caller closes it
    after all dependent execution resources retire, or transfers that obligation
    to the constructed Worker. Its context manager closes on every scope exit.
    """
    if world_size < 1 or not 0 <= rank < world_size or local_rank < 0:
        raise ValueError(
            "launch rank must satisfy 0 <= rank < positive world_size"
        )
    if init_method is not None and rendezvous is not None:
        raise ValueError("init_method and rendezvous are exclusive")
    if rendezvous is not None and rendezvous.listen_fd is not None and rank:
        raise ValueError("only rank 0 serves the rendezvous store")

    local_device = torch.device(device)
    if local_device.type == "cuda":
        index = (
            local_device.index if local_device.index is not None else local_rank
        )
        local_device = torch.device("cuda", index)
        if not torch.cuda.is_available() or index >= torch.cuda.device_count():
            raise ValueError(
                f"cuda device {local_device} is outside the "
                f"{torch.cuda.device_count()} visible CUDA device(s)"
            )
        torch.cuda.set_device(local_device)

    backend = backend or ("nccl" if local_device.type == "cuda" else "gloo")
    environment = ProcessGroups(rank, world_size, local_device, backend)

    if dist.is_initialized():
        if (dist.get_rank(), dist.get_world_size()) != (rank, world_size):
            raise ValueError(
                "existing process world disagrees with supplied rank/world_size"
            )
        if dist.get_backend() != backend:
            raise ValueError(
                "existing process world disagrees with supplied backend"
            )
    elif world_size > 1 and rendezvous is not None:
        dist.init_process_group(
            backend=backend,
            store=_rendezvous_store(rendezvous, rank, world_size, backend),
            rank=rank,
            world_size=world_size,
            pg_options=_group_options(backend),
            device_id=local_device if backend == "nccl" else None,
        )
        environment._groups.append(dist.group.WORLD)
    elif world_size > 1:
        if not init_method:
            port = os.environ.get("MASTER_PORT")
            if not port:
                raise ValueError(
                    "MASTER_PORT or an explicit init_method is required"
                )
            init_method = (
                f"tcp://{os.environ.get('MASTER_ADDR', '127.0.0.1')}:{port}"
            )
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


def _rendezvous_store(
    rendezvous: Rendezvous, rank: int, world_size: int, backend: str
) -> dist.TCPStore:
    """Serve the world's store on rank 0 and connect to it on every rank.

    The store keeps the timeout ``init_process_group`` gives a store it creates
    from an init method for this backend, which bounds how long ranks wait for
    one another to join and to publish communicator identifiers.
    """
    timeout = default_pg_timeout
    if backend == "nccl" and default_pg_nccl_timeout is not None:
        timeout = default_pg_nccl_timeout

    if rendezvous.listen_fd is not None:
        # The store takes the socket over. A process this rank later starts
        # must not hold the store's port open after the rank exits, so the
        # descriptor is not passed on again.
        os.set_inheritable(rendezvous.listen_fd, False)
    return dist.TCPStore(
        rendezvous.host,
        rendezvous.port,
        world_size,
        is_master=rank == 0,
        timeout=timeout,
        master_listen_fd=rendezvous.listen_fd,
    )


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
