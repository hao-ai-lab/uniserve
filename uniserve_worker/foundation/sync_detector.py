"""Runtime forbidden-synchronization detector for the steady-state request path.

`specs/decode-runtime.md` (Zero-blocking execution) forbids, between the first
request submission and the terminal public commit, live device scalar
conversion, live device-to-host list conversion, and accelerator/stream/event
synchronization on a request thread. This detector installs hooks over exactly
those operations for the duration of a guarded region and observes each one that
runs against live device state. Host-to-device transfers of host-known metadata
are the sanctioned direction and are not detected.

In `enforce` mode the first detected operation raises; otherwise detections are
counted and exposed through the metrics surface. Formal qualification runs the
request path under the detector and requires a zero count.

The detector is inert unless activated through `UNISERVE_SYNC_DETECT`, so
production hot paths are unchanged until a qualification run enables it.
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterator

from .env import flag_from_value

try:  # torch is optional for CPU-only control-plane tests.
    import torch
except Exception:  # pragma: no cover - exercised only in torch-free envs.
    torch = None  # type: ignore[assignment]

__all__ = [
    "ForbiddenSyncError",
    "SyncDetector",
    "sync_detection_active",
    "sync_detection_enforced",
    "sync_detector",
]

_DETECT_ENV = "UNISERVE_SYNC_DETECT"
_ENFORCE_ENV = "UNISERVE_SYNC_DETECT_ENFORCE"
_MAX_RECORDS = 64

# Device->host observations forbidden on a live device tensor. `numpy` and `cpu`
# are also legitimate on a host tensor (delivery of an already-ready product),
# so they are detected only when the receiver is resident on the accelerator.
_DEVICE_OBSERVATIONS = ("item", "tolist", "numpy", "cpu")


class ForbiddenSyncError(RuntimeError):
    """Raised in enforce mode when a forbidden synchronization is observed."""


class SyncDetector:
    """Counts forbidden device observations/synchronizations in guarded regions."""

    def __init__(self) -> None:
        self._detections = 0
        self._records: list[str] = []

    @property
    def detections(self) -> int:
        return self._detections

    def records(self) -> tuple[str, ...]:
        return tuple(self._records)

    def reset(self) -> None:
        self._detections = 0
        self._records.clear()

    def _flag(self, operation: str, label: str, *, enforce: bool) -> None:
        self._detections += 1
        detail = f"{label}: {operation}" if label else operation
        if len(self._records) < _MAX_RECORDS:
            self._records.append(detail)
        if enforce:
            raise ForbiddenSyncError(f"forbidden steady-state synchronization: {detail}")

    @contextmanager
    def guard(self, label: str = "", *, enforce: bool = False) -> Iterator[None]:
        """Detect forbidden device observations/synchronizations in the region."""
        if torch is None or not torch.cuda.is_available():
            yield
            return
        restores: list[tuple[object, str, object]] = []

        def hook_tensor_observation(name: str) -> None:
            original = getattr(torch.Tensor, name)

            def wrapper(tensor, *args, **kwargs):
                if bool(getattr(tensor, "is_cuda", False)):
                    self._flag(f".{name}()", label, enforce=enforce)
                return original(tensor, *args, **kwargs)

            setattr(torch.Tensor, name, wrapper)
            restores.append((torch.Tensor, name, original))

        def hook_synchronize(owner: object, name: str, operation: str) -> None:
            original = getattr(owner, name)

            def wrapper(*args, **kwargs):
                self._flag(operation, label, enforce=enforce)
                return original(*args, **kwargs)

            setattr(owner, name, wrapper)
            restores.append((owner, name, original))

        for name in _DEVICE_OBSERVATIONS:
            hook_tensor_observation(name)
        hook_synchronize(torch.cuda, "synchronize", "torch.cuda.synchronize()")
        hook_synchronize(torch.cuda.Stream, "synchronize", "Stream.synchronize()")
        hook_synchronize(torch.cuda.Event, "synchronize", "Event.synchronize()")
        try:
            yield
        finally:
            for owner, name, original in reversed(restores):
                setattr(owner, name, original)


_DETECTOR = SyncDetector()


def sync_detector() -> SyncDetector:
    """Return the process-wide detector accumulating steady-state detections."""
    return _DETECTOR


def sync_detection_active() -> bool:
    """Whether the steady-state request path should run under the detector."""
    return flag_from_value(os.environ.get(_DETECT_ENV))


def sync_detection_enforced() -> bool:
    """Whether a detection should raise rather than be counted."""
    return flag_from_value(os.environ.get(_ENFORCE_ENV))
