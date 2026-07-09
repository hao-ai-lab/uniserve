"""Runner-owned mixed forward execution seam."""
from __future__ import annotations

import inspect
from collections.abc import Mapping, Sequence
from typing import Any

from ..contracts.forward_mode import ForwardMode
from ..foundation.errors import capability_mismatch, invalid_descriptor
from ..runtime.request_state import RequestStateTable

__all__ = ["ForwardDriver"]


class ForwardDriver:
    """Dispatch one admitted mixed group through the model seam.

    Production interleaved generation models implement ``run_forward``.
    Once the admission router selects this path, a missing or malformed hook is a
    loud capability error rather than a per-mode fallback.
    """

    def can_run(self, model: Any, fb: Any) -> bool:
        hook = getattr(model, "run_forward", None)
        if not callable(hook):
            return False
        modes = tuple(getattr(fb, "op_modes", ()))
        if not modes:
            return False
        mode_set = set(modes)
        has_text = bool(mode_set & {ForwardMode.EXTEND, ForwardMode.DECODE})
        if not has_text:
            return False
        if mode_set & {ForwardMode.DENOISE, ForwardMode.COMMIT}:
            return True
        return {ForwardMode.EXTEND, ForwardMode.DECODE} <= mode_set <= {
            ForwardMode.EXTEND,
            ForwardMode.DECODE,
        }

    def step(
        self,
        fb: Any,
        group: Sequence[tuple[int, Mapping[str, Any]]],
        request_states: RequestStateTable,
        model: Any,
        *,
        defer_text_cpu_results: bool = False,
    ) -> list[Any]:
        hook = getattr(model, "run_forward", None)
        if not callable(hook):
            raise capability_mismatch("model does not implement run_forward")
        kwargs: dict[str, Any] = {"request_states": request_states, "group": list(group)}
        if _accepts_deferred_text_cpu_results(hook):
            kwargs["defer_text_cpu_results"] = bool(defer_text_cpu_results)
        result = hook(fb, **kwargs)
        if not isinstance(result, Sequence) or isinstance(result, (str, bytes, bytearray)):
            raise invalid_descriptor("run_forward must return one result per mixed op")
        if len(result) != len(group):
            raise invalid_descriptor("run_forward returned the wrong number of results")
        return list(result)


def _accepts_deferred_text_cpu_results(hook: Any) -> bool:
    try:
        signature = inspect.signature(hook)
    except (TypeError, ValueError):
        return False
    for parameter in signature.parameters.values():
        if parameter.kind is inspect.Parameter.VAR_KEYWORD:
            return True
    return "defer_text_cpu_results" in signature.parameters
