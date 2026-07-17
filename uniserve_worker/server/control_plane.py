"""Worker control-plane dispatch."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..contracts.caps import CONTROL_KINDS
from ..foundation.errors import scheduler_bug, unsupported_control
from .controls import CONTROL_SPECS

__all__ = [
    "ControlPlane",
]


class ControlPlane:
    """Owns supported-control gating and worker argument shaping."""

    def __init__(self, worker: Any, supported_controls: set[str]) -> None:
        self.worker = worker
        self.supported_controls = set(supported_controls)
        self.specs = dict(CONTROL_SPECS)

    def supported_kinds(self) -> frozenset[str]:
        return frozenset(kind for kind in self.specs if kind in self.supported_controls)

    def handle(
        self,
        kind: str,
        request: Mapping[str, Any],
    ) -> dict[str, Any]:
        if kind not in CONTROL_KINDS:
            raise scheduler_bug(f"unknown control kind: {kind!r}")
        if kind not in self.supported_controls:
            raise unsupported_control(kind)
        spec = self.specs[kind]
        fn = getattr(self.worker, spec.method)
        fn(**spec.build_kwargs(kind, request))
        return {"kind": "ok"}
