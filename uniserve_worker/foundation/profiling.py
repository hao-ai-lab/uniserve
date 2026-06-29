"""Lightweight profiling-span primitive shared by the hot path.

``profile_range`` emits a torch-profiler span and/or NVTX range when profiling
is active and is a no-op otherwise, so production hot paths in the execution
drivers stay unchanged unless a profiling run asks for traces. The stateful
per-step capture controller built on top of this lives in ``server/profiler.py``.
"""
from __future__ import annotations

import os
from contextlib import ExitStack, contextmanager, nullcontext
from typing import Iterator

from .env import flag_from_value

try:  # torch is an optional import for CPU-only control-plane tests.
    import torch
except Exception:  # pragma: no cover - exercised only in torch-free envs.
    torch = None  # type: ignore[assignment]

__all__ = ["profile_range"]

_PROFILE_ACTIVITIES_ENV = "UNISERVE_PROFILE_ACTIVITIES"
_PROFILE_NVTX_ENV = "UNISERVE_PROFILE_NVTX"
_NVTX_ENV = "UNISERVE_NVTX"
_CUDA_PROFILER_ENV = "UNISERVE_CUDA_PROFILER"

_NULL_CONTEXT = nullcontext()


def profile_range(debug_name: str):
    """Emit a torch-profiler span and/or NVTX range when profiling is active."""
    record = _torch_profiler_enabled()
    nvtx = _nvtx_ranges_enabled()
    if not record and not nvtx:
        return _NULL_CONTEXT
    return _profile_range_impl(debug_name, record=record, nvtx=nvtx)


@contextmanager
def _profile_range_impl(debug_name: str, *, record: bool, nvtx: bool) -> Iterator[None]:
    with ExitStack() as stack:
        if record and torch is not None:
            stack.enter_context(torch.profiler.record_function(debug_name))
        if nvtx and torch is not None:
            torch.cuda.nvtx.range_push(debug_name)
            stack.callback(torch.cuda.nvtx.range_pop)
        yield


def _torch_profiler_enabled() -> bool:
    if torch is None:
        return False
    enabled = getattr(torch.autograd, "_profiler_enabled", None)
    return bool(enabled()) if callable(enabled) else False


def _nvtx_ranges_enabled() -> bool:
    if torch is None:
        return False
    if not torch.cuda.is_available():
        return False
    env = os.environ
    if flag_from_value(env.get(_PROFILE_NVTX_ENV)) or flag_from_value(env.get(_NVTX_ENV)):
        return True
    activities = _parse_activities(env.get(_PROFILE_ACTIVITIES_ENV, ""))
    return "CUDA_PROFILER" in activities or flag_from_value(env.get(_CUDA_PROFILER_ENV))


def _parse_activities(raw: str | None) -> tuple[str, ...]:
    values = []
    for piece in (raw or "").replace(",", " ").split():
        value = piece.strip().upper()
        if value == "CUDA":
            value = "GPU"
        if value in {"CPU", "GPU", "CUDA_PROFILER"} and value not in values:
            values.append(value)
    return tuple(values)
