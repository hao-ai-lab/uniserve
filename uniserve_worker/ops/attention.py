"""Attention backend dispatch."""

from __future__ import annotations

import torch

from ..backends.attention.base import AttentionBackend
from .requests import AttentionReq


def can_run_attention(provider: AttentionBackend, req: AttentionReq) -> bool:
    return provider.can_run(req)


def run_attention(provider: AttentionBackend, req: AttentionReq) -> torch.Tensor:
    return provider.run(req)
