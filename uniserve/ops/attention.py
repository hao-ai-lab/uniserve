"""Typed attention-request bridge to startup-selected backend instances.

Attention providers own their execution resources and eligibility contract;
these functions keep model-facing code independent of backend class details.
"""

from __future__ import annotations

import torch

from uniserve.attention.base import AttentionBackend
from uniserve.ops.requests import AttentionReq


def can_run_attention(provider: AttentionBackend, req: AttentionReq) -> bool:
    """Report whether ``provider`` can execute the concrete attention request."""

    return provider.can_run(req)


def run_attention(provider: AttentionBackend, req: AttentionReq) -> torch.Tensor:
    """Execute ``req`` through an already selected attention provider."""

    return provider.run(req)
