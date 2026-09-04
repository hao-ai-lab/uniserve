"""Token-row selection and vocabulary masking for projected logits."""

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
    """Create a minimal logits tensor whose only finite token is ``eos_id``.

    Empty-token text operations still require a sampling input even though no
    model forward is run. Any categorical sampler over the returned tensor
    therefore selects EOS. ``batch_shape`` is prepended to the vocabulary axis.
    """

    eos = int(eos_id)

    # The vocabulary axis needs only enough entries to address EOS. Negative
    # infinity excludes every preceding token under ordinary logit sampling.
    logits = torch.full((*batch_shape, eos + 1), float("-inf"), device=device)
    logits[..., eos] = 0.0
    return logits


class LogitsProcessor(nn.Module):
    """Token-row selector and padded-vocabulary masker for projected logits."""

    def forward(
        self,
        logits: torch.Tensor,
        *,
        positions: torch.Tensor | None = None,
        valid_vocab_size: int | None = None,
    ) -> torch.Tensor:
        """Post-process projected logits for sampling.

        ``positions`` indexes a flattened token axis, typically selecting the
        final token of each sequence. ``valid_vocab_size`` masks any padded
        projection columns in place on the resulting tensor.
        """

        # Selection precedes vocabulary masking so only sampling rows are
        # materialized when callers provide packed sequence positions.
        if positions is not None:
            logits = logits.reshape(-1, logits.shape[-1]).index_select(
                0,
                positions,
            )

        # Tensor-parallel projections may pad their vocabulary width; prevent
        # those storage-only columns from participating in sampling.
        if valid_vocab_size is not None and 0 < int(valid_vocab_size) < int(logits.shape[-1]):
            logits[..., int(valid_vocab_size) :] = float("-inf")

        return logits
