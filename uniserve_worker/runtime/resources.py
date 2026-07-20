"""Worker-side resource residency and pressure reporting.

Tracks physical residency per resource class and per request so host logical
leases can be cross-checked. GPU-free: counts/units only, never tensors.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Iterable

from ..contracts.caps import RESOURCE_CLASSES
from ..foundation.errors import capability_mismatch, resource_lease_violation

__all__ = [
    "VALID_CLASSES",
    "ResourceRuntime",
]

logger = logging.getLogger(__name__)

VALID_CLASSES = tuple(sorted(RESOURCE_CLASSES))


class ResourceRuntime:
    """Per-class acquire/release ledger and pressure snapshots."""

    def __init__(self, classes, *, totals: dict):
        requested = [str(c) for c in classes]
        self.classes = [c for c in requested if c in VALID_CLASSES]
        unknown = [c for c in requested if c not in VALID_CLASSES]
        if unknown:
            logger.warning(
                "ignoring unknown resource classes not managed by this worker",
                extra={"classes": unknown, "valid": list(VALID_CLASSES)},
            )
        self.totals = {str(k): max(0, int(v)) for k, v in dict(totals).items()}
        # class -> {request_id -> units}
        self._resident: dict[str, dict[int, int]] = {c: defaultdict(int) for c in self.classes}
        missing = [cls for cls in self.classes if cls not in self.totals]
        if missing:
            logger.warning(
                "resource classes missing explicit totals; using zero capacity",
                extra={"classes": missing},
            )
            for cls in missing:
                self.totals[cls] = 0

    def acquire(self, cls: str, req_id: int, units: int) -> None:
        if cls not in self._resident:
            raise capability_mismatch(f"resource class {cls!r} not managed by this worker")
        amount = int(units)
        if amount <= 0:
            return
        total = int(self.totals.get(cls, 0))
        used = self.used(cls)
        if used + amount > total:
            raise resource_lease_violation(
                f"resource class {cls!r} would exceed worker capacity ({used}+{amount}>{total})",
                req_id=int(req_id),
                details={"class": cls, "used": used, "requested": amount, "total": total},
            )
        self._resident[cls][int(req_id)] += amount

    def release_class(self, cls: str, req_id: int) -> int:
        freed = self._resident.get(cls, {}).pop(int(req_id), 0)
        return freed

    def release_request(self, req_id: int) -> int:
        """Drop all residency for a request (mirror of the host lease release)."""
        rid = int(req_id)
        freed = 0
        for table in self._resident.values():
            freed += table.pop(rid, 0)
        return freed

    def snapshot_requests(self, request_ids: Iterable[int]) -> dict[str, dict[int, int]]:
        """Capture the exact ledger entries touched by one execution step."""
        ids = {int(request_id) for request_id in request_ids}
        return {
            cls: {
                request_id: int(units) for request_id, units in table.items() if request_id in ids
            }
            for cls, table in self._resident.items()
        }

    def restore_requests(
        self,
        request_ids: Iterable[int],
        snapshot: dict[str, dict[int, int]],
    ) -> None:
        """Restore touched ledger entries without changing unrelated requests."""
        ids = {int(request_id) for request_id in request_ids}
        for cls, table in self._resident.items():
            for request_id in ids:
                table.pop(request_id, None)
            for request_id, units in snapshot.get(cls, {}).items():
                if int(units) > 0:
                    table[int(request_id)] = int(units)

    def used(self, cls: str) -> int:
        return sum(self._resident.get(cls, {}).values())

    def residency(self) -> list[dict]:
        out = []
        for cls, table in self._resident.items():
            for req_id, units in table.items():
                if units:
                    out.append({"class": cls, "request_id": req_id, "units": units})
        return out

    def pressure(self) -> list[dict]:
        snap = []
        for cls in self.classes:
            used = self.used(cls)
            total = int(self.totals.get(cls, 0))
            free = max(0, total - used) if total else 0
            # ``evictable`` is reserved on the wire; not tracked here yet.
            snap.append({"class": cls, "total": total, "used": used, "evictable": 0, "free": free})
        return snap

    def total_active(self) -> int:
        return sum(self.used(c) for c in self.classes)
