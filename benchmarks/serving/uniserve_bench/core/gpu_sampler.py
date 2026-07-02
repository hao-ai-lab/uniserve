"""Harness-side GPU memory sampling.

Polls ``nvidia-smi`` on a background thread during the timed region and keeps
the peak per-GPU and summed used memory. Client-side by design: both backends
under comparison are measured with the identical instrument, independent of
what (if anything) each server self-reports.
"""
from __future__ import annotations

import shutil
import subprocess
import threading
from dataclasses import dataclass, field
from typing import Any


def _read_used_mib() -> list[int]:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    if out.returncode != 0:
        return []
    return [int(line.strip()) for line in out.stdout.splitlines() if line.strip().isdigit()]


@dataclass
class GpuMemorySampler:
    """Background peak-memory sampler; a no-op when nvidia-smi is unavailable."""

    interval_s: float = 0.5
    peak_per_gpu_mib: list[int] = field(default_factory=list)
    peak_total_mib: int = 0
    samples: int = 0
    _stop: threading.Event = field(default_factory=threading.Event)
    _thread: threading.Thread | None = None

    @property
    def available(self) -> bool:
        return shutil.which("nvidia-smi") is not None

    def start(self) -> None:
        if not self.available:
            return
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=15)
            self._thread = None

    def _sample_once(self) -> None:
        try:
            used = _read_used_mib()
        except Exception:  # noqa: BLE001 - sampling is best-effort context.
            return
        if not used:
            return
        if len(self.peak_per_gpu_mib) < len(used):
            self.peak_per_gpu_mib.extend([0] * (len(used) - len(self.peak_per_gpu_mib)))
        for index, value in enumerate(used):
            self.peak_per_gpu_mib[index] = max(self.peak_per_gpu_mib[index], value)
        self.peak_total_mib = max(self.peak_total_mib, sum(used))
        self.samples += 1

    def _loop(self) -> None:
        self._sample_once()
        while not self._stop.wait(self.interval_s):
            self._sample_once()

    def summary(self) -> dict[str, Any] | None:
        if self.samples == 0:
            return None
        return {
            "peak_per_gpu_mib": list(self.peak_per_gpu_mib),
            "peak_total_mib": int(self.peak_total_mib),
            "peak_single_gpu_mib": max(self.peak_per_gpu_mib, default=0),
            "samples": int(self.samples),
            "interval_s": self.interval_s,
        }
