"""Resolved component placement and borrowed numerical calls.

A ``ComponentBinding`` records where one model component runs across the
Worker's ranks. ``uniserve_worker.bootstrap.distributed`` builds one per
configured component (``ModelExecutor`` builds rank-local ones when none are
supplied), and ``uniserve_worker.bootstrap.components.bind_components``
attaches the component's ``Call`` values on member ranks. Runners and lanes
borrow those calls; they do not own the modules or communicators.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

from uniserve.distributed.mesh import Communicator
from uniserve.model import EntryPoint
from uniserve_worker._uniserve_ipc import ComponentBinding as ComponentBinding


@dataclass(frozen=True, slots=True)
class Call:
    """A borrowed capability method and its numerical participation groups.

    ``groups`` lists the communicators the call exchanges tensors in on this
    rank; ``bind_components`` fills it from the module's communicators, the
    entry point's declared communication axes and, for a temporally
    distributed ``VideoPostprocessor``, the component's unit ring.
    """

    path: str
    module: torch.nn.Module
    entry_point: EntryPoint
    groups: tuple[Communicator, ...] = ()

    @property
    def forward(self) -> Callable[..., Any]:
        """The bound entry-point method, resolved on the module per access."""
        return getattr(self.module, self.entry_point.method)
