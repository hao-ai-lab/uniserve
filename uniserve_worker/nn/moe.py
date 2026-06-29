"""Shared mixture-of-experts primitives."""
from __future__ import annotations

import torch
import torch.nn as nn

__all__ = [
    'TopK',
    'FusedMoE',
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
    """Reference per-expert masked-loop MoE dispatch.

    Despite the name, this performs no kernel fusion or token grouping: it loops
    over experts, gathers each expert's routed tokens via a boolean mask, and
    accumulates the weighted expert outputs back into place. It is the
    deterministic correctness floor that model code builds on.
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

    def forward(self, hidden_states: torch.Tensor, router_logits: torch.Tensor) -> torch.Tensor:
        original_shape = hidden_states.shape
        flat = hidden_states.reshape(-1, original_shape[-1])
        flat_logits = router_logits.reshape(-1, router_logits.shape[-1])
        weights, expert_ids = self.topk(flat_logits)
        weights = weights.to(flat.dtype)
        out = torch.zeros_like(flat)
        for expert_idx, expert in enumerate(self.experts):
            hits = expert_ids == expert_idx
            if not hits.any():
                continue
            token_idx, kth = hits.nonzero(as_tuple=True)
            expert_out = expert(flat[token_idx])
            out[token_idx] += expert_out * weights[token_idx, kth].unsqueeze(-1).to(expert_out.dtype)
        return out.reshape(original_shape)
