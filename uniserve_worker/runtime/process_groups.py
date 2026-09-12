"""Ownership of process groups created for one worker."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
from types import TracebackType
from typing import Any, Self

import torch
import torch.distributed as dist

from ..foundation.resources import close_resources
from ..nn.mesh import Communicator


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
