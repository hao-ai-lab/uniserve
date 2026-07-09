"""Runtime state views consumed by forward planning and postprocessing."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

__all__ = ["ForwardRuntimeStateView"]


@dataclass(frozen=True)
class ForwardRuntimeStateView:
    request_states: Any = None
    residency: Any = None
    cache_handles: Mapping[int, Any] = field(default_factory=dict)
    latent_handles: Mapping[int, Any] = field(default_factory=dict)
    branch_caches: Mapping[int, Any] = field(default_factory=dict)
    scratch: Any = None
    tensor_store: Any = None
