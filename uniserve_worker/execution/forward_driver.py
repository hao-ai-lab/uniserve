"""Runner-owned mixed text+denoise forward execution seam."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ..contracts.forward_mode import ForwardMode
from ..foundation.errors import capability_mismatch, invalid_descriptor
from ..runtime.request_state import RequestStateTable

__all__ = ["ForwardDriver"]


class ForwardDriver:
    """Dispatch one admitted mixed text+denoise group through the model seam.

    Production interleaved generation models implement ``run_forward``.
    Once the admission router selects this path, a missing or malformed hook is a
    loud capability error rather than a per-mode fallback.
    """

    def can_run(self, model: Any, fb: Any) -> bool:
        hook = getattr(model, "run_forward", None)
        if not callable(hook):
            return False
        modes = tuple(getattr(fb, "op_modes", ()))
        return bool(modes) and any(m is ForwardMode.DECODE for m in modes) and any(
            m is ForwardMode.DENOISE for m in modes
        )

    def step(
        self,
        fb: Any,
        group: Sequence[tuple[int, Mapping[str, Any]]],
        request_states: RequestStateTable,
        model: Any,
    ) -> list[Any]:
        hook = getattr(model, "run_forward", None)
        if not callable(hook):
            raise capability_mismatch("model does not implement run_forward")
        result = hook(fb, request_states=request_states, group=list(group))
        if not isinstance(result, Sequence) or isinstance(result, (str, bytes, bytearray)):
            raise invalid_descriptor("run_forward must return one result per mixed op")
        if len(result) != len(group):
            raise invalid_descriptor("run_forward returned the wrong number of results")
        return list(result)
