"""Greedy token selection over complete or sharded vocabularies."""

from __future__ import annotations

import torch

from uniserve.model.logits import VocabShard


def greedy(
    logits: torch.Tensor, vocab: VocabShard | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select global maxima with ``torch.max`` tie and NaN behavior.

    A sharded vocabulary exchanges one score/token pair per row. Padded tokens
    never participate, and communicator order defines the global token order.
    """

    if vocab is None:
        return torch.max(logits, dim=-1)
    if logits.ndim != 2 or logits.shape[-1] != vocab.local_slice.stop - vocab.local_slice.start:
        raise ValueError("greedy logits must be rows of the declared vocabulary shard")

    begin = vocab.local_slice.start
    valid = max(0, min(logits.shape[-1], vocab.size - begin))
    if valid:
        values, tokens = torch.max(logits[:, :valid], dim=-1)
        tokens = tokens + begin
    else:
        values = logits.new_full((logits.shape[0],), float("-inf"))
        tokens = torch.full_like(values, vocab.size, dtype=torch.int64)

    if vocab.group.size == 1:
        return values, tokens

    candidates = torch.stack((values.to(torch.float64).view(torch.int64), tokens), dim=-1)
    gathered = vocab.group.all_gather(candidates, dim=0).reshape(vocab.group.size, -1, 2)
    scores = gathered[..., 0].contiguous().view(torch.float64)
    maxima, owners = scores.max(dim=0)
    selected = gathered[..., 1].gather(0, owners.unsqueeze(0)).squeeze(0)
    return maxima.to(logits.dtype), selected
