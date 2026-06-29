"""No-op attention backend test double.

``StubAttentionBackend`` satisfies the attention-backend contract (the base
``AttentionBackend`` Protocol plus the capability-gated paged / varlen /
visible-end entry points) with zero-returning forward methods. It replaces the
per-test-file ``FakePagedBackend`` / ``FakeVarlenBackend`` stubs with one
canonical double whose advertised :class:`AttentionCapabilities` and recorded
call counts are configurable.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from uniserve_worker.backends.attention.base import AttentionCapabilities

__all__ = [
    "StubAttentionBackend",
]


@dataclass
class StubAttentionBackend:
    """Configurable no-op attention backend.

    The advertised capabilities default to a fully-capable backend so a single
    instance can stand in on paged, varlen, and visible-end dispatch paths; pass
    a narrower :class:`AttentionCapabilities` to model a capability-restricted
    backend. Every forward method returns zeros shaped like its query input and
    increments :attr:`calls`, and the most recent keyword arguments are retained
    on :attr:`last_kwargs` for assertions.
    """

    name: str = "stub_attention"
    caps: AttentionCapabilities = field(
        default_factory=lambda: AttentionCapabilities(
            segment_batched_cfg=True,
            mixed_mode=True,
            paged_kv=True,
            varlen_attention=True,
            varlen_paged_kv=True,
            visible_end=True,
            tree_verify=True,
        )
    )
    calls: int = 0
    last_kwargs: dict | None = None

    def capabilities(self) -> AttentionCapabilities:
        return self.caps

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        causal: bool,
        scale: float | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del k, v, causal, scale, attn_mask
        self.calls += 1
        return torch.zeros_like(q)

    def forward_paged(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        del k_cache, v_cache
        self.calls += 1
        self.last_kwargs = kwargs
        return torch.zeros_like(q)

    def forward_varlen(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        del k, v
        self.calls += 1
        self.last_kwargs = kwargs
        return torch.zeros_like(q)

    def forward_visible_end(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        del k, v
        self.calls += 1
        self.last_kwargs = kwargs
        return torch.zeros_like(q)
