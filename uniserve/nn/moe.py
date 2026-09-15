"""Shared mixture-of-experts primitives."""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = [
    "TopK",
    "FusedMoE",
]


class TopK(nn.Module):
    """Selects and renormalizes the highest-scoring experts for each token."""

    def __init__(self, k: int, *, renormalize: bool = True) -> None:
        """Validate the expert count and configure optional probability
        renormalization.
        """  # noqa: D205
        super().__init__()
        if k <= 0:
            raise ValueError("k must be positive")
        self.k = k
        self.renormalize = renormalize

    def forward(
        self, scores: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Select the highest-probability experts per row and optionally
        renormalize weights.
        """  # noqa: D205
        probs = torch.softmax(scores, dim=-1, dtype=torch.float32)
        weights, ids = torch.topk(probs, self.k, dim=-1)
        if self.renormalize:
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(
                torch.finfo(weights.dtype).eps
            )
        return weights, ids


class FusedMoE(nn.Module):
    """Evaluate every expert densely and accumulate top-k per-token
    contributions.

    The implementation preserves exact top-k routing semantics without token
    dispatch: every expert processes every row, so compute scales with the full
    expert count even though unselected outputs receive a zero gate.
    """  # noqa: D205

    def __init__(
        self,
        experts: list[nn.Module] | nn.ModuleList,
        top_k: int = 1,
        *,
        norm_topk_prob: bool = True,
    ) -> None:
        """Register every expert and configure the per-token top-k gate."""
        super().__init__()
        self.experts = nn.ModuleList(experts)
        self.topk = TopK(top_k, renormalize=norm_topk_prob)

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Route flattened rows to top-k experts and accumulate their gated
        outputs.
        """  # noqa: D205
        original_shape = hidden_states.shape
        flat = hidden_states.reshape(-1, original_shape[-1])
        flat_logits = router_logits.reshape(-1, router_logits.shape[-1])

        weights, expert_ids = self.topk(flat_logits)
        weights = weights.to(flat.dtype)
        out = torch.zeros_like(flat)
        for expert_idx, expert in enumerate(self.experts):
            # Summing the one-hot top-k matches yields this expert's scalar gate
            # for every token and zero for tokens routed elsewhere.
            gate = (weights * (expert_ids == expert_idx)).sum(dim=-1)
            expert_out = expert(flat)
            out += expert_out * gate.unsqueeze(-1).to(expert_out.dtype)

        return out.reshape(original_shape)
