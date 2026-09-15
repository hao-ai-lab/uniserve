"""Optional PyTorch and NVTX spans around numerical and infrastructure calls."""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager, nullcontext
from typing import Any

from .env import flag_from_value

torch: Any | None
try:
    import torch as _torch_module
except ImportError:
    torch = None
else:
    torch = _torch_module

_NVTX_ENV = "UNISERVE_NVTX"
_NULL_CONTEXT = nullcontext()


def profile_range(debug_name: str):
    """Emit a torch-profiler span and/or NVTX range when profiling is active."""
    record = _torch_profiler_enabled()
    nvtx = _nvtx_ranges_enabled()
    if not record and not nvtx:
        return _NULL_CONTEXT
    return _profile_range_impl(debug_name, record=record, nvtx=nvtx)


@contextmanager
def _profile_range_impl(
    debug_name: str, *, record: bool, nvtx: bool
) -> Iterator[None]:
    """Enter configured record-function and NVTX ranges.

    Around one code region.
    """
    with ExitStack() as stack:
        if record and torch is not None:
            stack.enter_context(torch.profiler.record_function(debug_name))
        if nvtx and torch is not None:
            torch.cuda.nvtx.range_push(debug_name)
            stack.callback(torch.cuda.nvtx.range_pop)
        yield


def _torch_profiler_enabled() -> bool:
    """Return whether PyTorch autograd profiling is currently active."""
    if torch is None:
        return False
    enabled = getattr(torch.autograd, "_profiler_enabled", None)
    return bool(enabled()) if callable(enabled) else False


def _nvtx_ranges_enabled() -> bool:
    """Return whether environment policy enables NVTX range emission."""
    if torch is None:
        return False
    if not torch.cuda.is_available():
        return False
    return flag_from_value(os.environ.get(_NVTX_ENV))
