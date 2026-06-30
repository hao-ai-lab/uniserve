"""Logit post-processing helpers."""
from __future__ import annotations

import torch
import torch.nn as nn

from ..foundation.runtime_config import get_worker_config

__all__ = [
    'LogitsProcessor',
]


def _logits_chunk_size() -> int:
    return get_worker_config().logits_processor_chunk_size


class LogitsProcessor(nn.Module):
    """Project hidden states through an LM head and optionally select positions."""

    def __init__(self, chunk_size: int | None = None) -> None:
        super().__init__()
        # Resolve the chunk size once (env default) so ``forward`` stays free of
        # ``os.environ`` reads on the hot path.
        self.chunk_size = _logits_chunk_size() if chunk_size is None else max(0, int(chunk_size))

    def forward(
        self,
        hidden_states: torch.Tensor,
        lm_head: nn.Module,
        *,
        positions: torch.Tensor | None = None,
        valid_vocab_size: int | None = None,
    ) -> torch.Tensor:
        if positions is not None:
            hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1]).index_select(
                0,
                positions,
            )
        logits = _project_lm_head(hidden_states, lm_head, self.chunk_size)
        if valid_vocab_size is not None and 0 < int(valid_vocab_size) < int(logits.shape[-1]):
            # ``_project_lm_head`` just produced ``logits`` and nothing else
            # aliases it, so mask the out-of-range vocab in place.
            logits[..., int(valid_vocab_size):] = float("-inf")
        return logits


def _project_lm_head(hidden_states: torch.Tensor, lm_head: nn.Module, chunk_size: int) -> torch.Tensor:
    if chunk_size <= 0:
        return lm_head(hidden_states)
    flat = hidden_states.reshape(-1, hidden_states.shape[-1])
    if int(flat.shape[0]) <= chunk_size:
        return lm_head(hidden_states)
    chunks = [
        lm_head(flat[start : start + chunk_size])
        for start in range(0, int(flat.shape[0]), chunk_size)
    ]
    logits = torch.cat(chunks, dim=0)
    return logits.reshape(*hidden_states.shape[:-1], logits.shape[-1])
