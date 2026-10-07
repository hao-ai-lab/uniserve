"""PyTorch profiler backend.

The native worker owns capture windows, trace paths and export lifetime.
These functions expose the installed PyTorch profiler and CUDA profiler API.
"""

from __future__ import annotations

import logging

from uniserve_worker._uniserve_ipc import (
    WorkerProfiler,
    timing_events_enabled,
)

__all__ = ["WorkerProfiler", "timing_events_enabled"]

logger = logging.getLogger(__name__)


def _create_profiler(activities, with_stack, record_shapes):
    import torch

    selected = []
    if "CPU" in activities:
        selected.append(torch.profiler.ProfilerActivity.CPU)
    if "GPU" in activities:
        if torch.cuda.is_available():
            selected.append(torch.profiler.ProfilerActivity.CUDA)
        else:
            logger.warning(
                "UNISERVE_PROFILE_ACTIVITIES requested GPU but CUDA is "
                "unavailable"
            )
    if not selected:
        return None
    return torch.profiler.profile(
        activities=selected,
        with_stack=with_stack,
        record_shapes=record_shapes,
        acc_events=True,
    )


def _cuda_profiler(start):
    import torch

    if not torch.cuda.is_available():
        if start:
            logger.warning("CUDA profiler requested but CUDA is unavailable")
        return
    if start:
        torch.cuda.cudart().cudaProfilerStart()
    else:
        torch.cuda.cudart().cudaProfilerStop()
