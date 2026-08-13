"""Shared mixture-of-experts primitives."""

from __future__ import annotations

import torch
import torch.nn as nn

from ..execution.forward_batch import MeshView

__all__ = [
    "TopK",
    "FusedMoE",
]


class TopK(nn.Module):
    def __init__(self, k: int, *, renormalize: bool = True) -> None:
        super().__init__()
        if k <= 0:
            raise ValueError("k must be positive")
        self.k = k
        self.renormalize = renormalize

    def forward(self, scores: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        probs = torch.softmax(scores, dim=-1, dtype=torch.float32)
        weights, ids = torch.topk(probs, self.k, dim=-1)
        if self.renormalize:
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(
                torch.finfo(weights.dtype).eps
            )
        return weights, ids


class FusedMoE(nn.Module):
    """Reference per-expert dense-masked MoE dispatch.

    Despite the name, this performs no kernel fusion or token grouping: it loops
    over experts, evaluates each expert on every token, and accumulates the
    outputs scaled by a per-token routing gate that is zero wherever the router
    did not select that expert. It is the deterministic correctness floor that
    model code builds on.
    """

    def __init__(
        self,
        experts: list[nn.Module] | nn.ModuleList,
        top_k: int = 1,
        *,
        norm_topk_prob: bool = True,
    ) -> None:
        super().__init__()
        self.experts = nn.ModuleList(experts)
        self.topk = TopK(top_k, renormalize=norm_topk_prob)

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        mesh: MeshView,
    ) -> torch.Tensor:
        original_shape = hidden_states.shape
        flat = hidden_states.reshape(-1, original_shape[-1])
        flat_logits = router_logits.reshape(-1, router_logits.shape[-1])
        weights, expert_ids = self.topk(flat_logits)
        weights = weights.to(flat.dtype)
        out = torch.zeros_like(flat)
        for expert_idx, expert in enumerate(self.experts):
            # Per-token gate for this expert: the routed weight where the router
            # selected it, zero everywhere else. Selection is one-hot across the
            # top-k axis, so the masked sum recovers exactly that weight.
            gate = (weights * (expert_ids == expert_idx)).sum(dim=-1)
            expert_out = expert(flat, mesh)
            out += expert_out * gate.unsqueeze(-1).to(expert_out.dtype)
        return out.reshape(original_shape)
