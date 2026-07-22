"""System owner of worker transfer: one transport, one tower-handoff choice.

One :class:`Mover` exists per worker and is composed by the worker process,
never by a model. It owns the two transfer decisions a deployment makes:

* **The worker Transport.** The single register-once byte transport
  (:mod:`uniserve_worker.runtime.transfer`) is built lazily exactly once and
  shared by every consumer on the worker — tower crossings and the
  deferred-sampling tensor store — so per-buffer registration happens once no
  matter how many edges ride it.
* **The und↔gen tower handoff.** The crossing implementation is selected by
  the deployment edge: an in-process edge gets
  :class:`~uniserve_worker.runtime.tower_handoff.LocalP2PTowerHandoff` (NVLink
  peer copy with record_ready/wait_ready barriers, trivial on one device); a
  cross-process edge gets
  :class:`~uniserve_worker.runtime.tower_handoff.DataPlaneTowerHandoff` over
  the worker Transport. Models contribute only the tower geometry declaration
  (a ``TowerBinding`` resolver) and never see transport objects.

Placement and barrier primitives (``reshard_kv_snapshot`` /
``wait_kv_snapshot_ready``) are pure functions in
:mod:`uniserve_worker.runtime.tower_kv`; the selected handoffs reference them.
"""

from __future__ import annotations

from collections.abc import Callable

from ..foundation.errors import capability_mismatch
from .tower_handoff import (
    DataPlaneTowerHandoff,
    LocalP2PTowerHandoff,
    TowerBinding,
    TowerHandoff,
)
from .transfer import Transport, make_transport

__all__ = ["Mover"]

# Backends that cannot carry a cross-process GPU KV crossing: same-process
# ("", "local") or host-snapshot ("shm") transports.
_IN_PROCESS_BACKENDS = frozenset({"", "local", "shm"})


class Mover:
    """Per-worker transfer authority: the Transport and the tower-handoff selection."""

    def __init__(self, *, transfer_backend: str, cross_process: bool = False) -> None:
        self.transfer_backend = str(transfer_backend)
        self.cross_process = bool(cross_process)
        if self.cross_process and self.transfer_backend in _IN_PROCESS_BACKENDS:
            raise capability_mismatch(
                "a cross-process deployment edge requires a cross-process "
                f"data-plane transport, got {self.transfer_backend!r}"
            )
        self._transport: Transport | None = None

    @property
    def transport(self) -> Transport:
        """The worker's single register-once transport, built lazily once."""
        if self._transport is None:
            self._transport = make_transport(self.transfer_backend)
        return self._transport

    def tower_handoff(self, bind: Callable[[], TowerBinding]) -> TowerHandoff:
        """Select the und↔gen crossing for this worker's deployment edge.

        ``bind`` is the family's live tower geometry declaration; the returned
        handoff resolves it fresh per crossing.
        """
        if self.cross_process:
            return DataPlaneTowerHandoff(data_plane=self.transport, bind=bind)
        return LocalP2PTowerHandoff(bind)

    def close(self) -> None:
        """Tear down the owned transport (idempotent)."""
        if self._transport is not None:
            self._transport.close()
            self._transport = None
