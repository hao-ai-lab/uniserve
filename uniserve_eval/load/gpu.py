"""Samples host-visible NVIDIA GPU memory and utilization telemetry."""

from __future__ import annotations

import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Any


def _read_gpu_rows() -> list[dict[str, int]]:
    """Read one numeric telemetry row per GPU from nvidia-smi."""
    out = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    if out.returncode != 0:
        return []
    rows: list[dict[str, int]] = []
    for line in out.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 3:
            continue
        try:
            rows.append(
                {
                    "index": int(parts[0]),
                    "memory_used_mib": int(parts[1]),
                    "utilization_gpu_pct": int(parts[2]),
                }
            )
        except ValueError:
            continue
    return rows


@dataclass
class GpuMemorySampler:
    """Collects periodic GPU telemetry and aggregate peaks on a background thread."""  # noqa: E501

    interval_s: float = 0.5
    peak_per_gpu_mib: list[int] = field(default_factory=list)
    peak_utilization_gpu_pct: list[int] = field(default_factory=list)
    utilization_total_pct: int = 0
    peak_total_mib: int = 0
    samples: int = 0
    sample_records: list[dict[str, Any]] = field(default_factory=list)
    _stop: threading.Event = field(default_factory=threading.Event)
    _thread: threading.Thread | None = None

    @property
    def available(self) -> bool:
        """Report whether the nvidia-smi executable is available."""
        return shutil.which("nvidia-smi") is not None

    def start(self) -> None:
        """Start background sampling when NVIDIA telemetry is available."""
        if not self.available:
            return
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop sampling and join the background thread."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=15)
            self._thread = None

    def _sample_once(self) -> None:
        """Record one telemetry snapshot and update aggregate peaks."""
        try:
            rows = _read_gpu_rows()
        except Exception:
            return
        if not rows:
            return
        used = [row["memory_used_mib"] for row in rows]
        util = [row["utilization_gpu_pct"] for row in rows]
        if len(self.peak_per_gpu_mib) < len(used):
            self.peak_per_gpu_mib.extend(
                [0] * (len(used) - len(self.peak_per_gpu_mib))
            )
        if len(self.peak_utilization_gpu_pct) < len(util):
            self.peak_utilization_gpu_pct.extend(
                [0] * (len(util) - len(self.peak_utilization_gpu_pct))
            )
        for index, value in enumerate(used):
            self.peak_per_gpu_mib[index] = max(
                self.peak_per_gpu_mib[index], value
            )
        for index, value in enumerate(util):
            self.peak_utilization_gpu_pct[index] = max(
                self.peak_utilization_gpu_pct[index],
                value,
            )
        self.peak_total_mib = max(self.peak_total_mib, sum(used))
        self.utilization_total_pct += sum(util)
        self.samples += 1
        self.sample_records.append({"time": time.time(), "gpus": rows})

    def _loop(self) -> None:
        """Sample immediately and then at the configured interval."""
        self._sample_once()
        while not self._stop.wait(self.interval_s):
            self._sample_once()

    def summary(self) -> dict[str, Any] | None:
        """Return aggregate telemetry, or ``None`` when no sample succeeded."""
        if self.samples == 0:
            return None
        return {
            "peak_per_gpu_mib": list(self.peak_per_gpu_mib),
            "peak_total_mib": int(self.peak_total_mib),
            "peak_single_gpu_mib": max(self.peak_per_gpu_mib, default=0),
            "peak_utilization_gpu_pct": list(self.peak_utilization_gpu_pct),
            "mean_total_utilization_gpu_pct": self.utilization_total_pct
            / self.samples,
            "samples": int(self.samples),
            "interval_s": self.interval_s,
        }
