"""Logit post-processing helpers."""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = [
    "LogitsProcessor",
    "forced_eos_logits",
]


def forced_eos_logits(
    eos_id: int,
    *,
    device: torch.device | str,
    batch_shape: tuple[int, ...] = (),
) -> torch.Tensor:
    """Synthetic one-hot logits row that forces EOS.

    Used for empty-token text ops, where the contract still requires a logits
    tensor but no model forward runs; sampling any distribution over these
    logits yields ``eos_id``.
    """
    eos = int(eos_id)
    logits = torch.full((*batch_shape, eos + 1), float("-inf"), device=device)
    logits[..., eos] = 0.0
    return logits


class LogitsProcessor(nn.Module):
    """Select and mask already-projected token logits."""

    def forward(
        self,
        logits: torch.Tensor,
        *,
        positions: torch.Tensor | None = None,
        valid_vocab_size: int | None = None,
    ) -> torch.Tensor:
        if positions is not None:
            logits = logits.reshape(-1, logits.shape[-1]).index_select(
                0,
                positions,
            )
        if valid_vocab_size is not None and 0 < int(valid_vocab_size) < int(logits.shape[-1]):
            logits[..., int(valid_vocab_size) :] = float("-inf")
        return logits
