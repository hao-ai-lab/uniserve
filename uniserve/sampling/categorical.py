"""Inverse-CDF selection from row-wise categorical probabilities."""

from __future__ import annotations

import torch


def sample_categorical(
    probabilities: torch.Tensor, draws: torch.Tensor
) -> torch.Tensor:
    """Select one entry per row by inverting its cumulative distribution.

    Each positive-mass entry owns the half-open interval ``[c - p, c)`` of
    the canonical draw range ``[0, 1)``, where ``c`` is its inclusive
    cumulative mass, so a row selects the first positive-mass entry whose
    cumulative mass is strictly greater than its draw. Zero-mass (masked)
    entries own an empty interval and are never selected: a draw of exactly
    0.0 skips leading masked entries, and a draw at or above a float32 total
    that rounds below one selects the last positive-mass entry. This is
    FlashInfer's sampling-from-probabilities boundary rule, and the
    simulator's ``sample_categorical`` in ``uniserve_core::sampling`` applies
    the same rule; ``tests/python/generated/sampling_rng_parity.json`` pins
    it for both.

    The selection uses only tensor operations without host synchronization,
    so it runs under ``torch.compile(fullgraph=True)`` and CUDA graph capture.

    Args:
        probabilities: Floating ``[rows, n]`` non-negative probabilities; a
            row need not sum exactly to one.
        draws: ``[rows]`` uniform draws in ``[0, 1)``.

    Returns:
        ``[rows]`` int64 column indexes. A row with no positive mass, such as
        a NaN row, selects index 0; callers reject such rows through their
        own validity checks.
    """
    columns = probabilities.shape[-1]
    positions = torch.arange(columns, device=probabilities.device)
    positive = probabilities > 0.0
    cumulative = probabilities.cumsum(dim=-1)
    exceeds = positive & (
        cumulative > draws.to(dtype=cumulative.dtype).unsqueeze(-1)
    )

    # Take the lowest qualifying position rather than counting positions at
    # or below the draw: a parallel scan may round the cumulative sum
    # non-monotonically across a zero-mass entry, and a count would then land
    # on that entry.
    first = torch.where(exceeds, positions, columns).amin(dim=-1)
    last = torch.where(positive, positions, 0).amax(dim=-1)
    return torch.where(first < columns, first, last)


__all__ = ["sample_categorical"]
