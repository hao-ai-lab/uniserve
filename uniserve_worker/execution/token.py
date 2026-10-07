"""Numerical token rows, prompt scores and speculative device coordinates."""

from __future__ import annotations

import torch

from uniserve.runtime.device import async_tensor_h2d
from uniserve.sampling import SamplingParams
from uniserve_worker.errors import invalid_descriptor
from uniserve_worker.model_executor.input_batch import TokenRow
from uniserve_worker.sampling import sampler as sampling
from uniserve_worker.sampling.result import LogprobValues, SamplerRow


def token_values(
    tokens: tuple[int, ...],
    position: int,
    *,
    current: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build token and position views from executor-selected coordinates.

    A device continuation stays on device, including when concatenated with
    verification drafts. Positions are host inputs for batch preparation.
    """
    if current is None:
        values = torch.tensor(tokens, dtype=torch.long, device="cpu")
    elif tokens:
        drafts = async_tensor_h2d(
            tokens, dtype=torch.long, device=current.device
        )
        values = torch.cat((current.reshape(1), drafts))
    else:
        values = current.reshape(1)

    positions = torch.arange(
        position, position + values.numel(), dtype=torch.long, device="cpu"
    )
    return values, positions


def next_position(row: TokenRow) -> int:
    """Return the logical position after a context row.

    A row occupies the positions of its temporal axis: a prompt run one per
    token, and a vision block one per feature at consecutive positions, or
    one when its features share a temporal position.
    """
    positions = row.positions
    if positions is None:
        raise RuntimeError("a context row has no positions")
    temporal = positions if positions.ndim == 1 else positions[0]
    return int(temporal.max()) + 1


def prompt_logprobs(
    tokens: torch.Tensor,
    logits: torch.Tensor,
    previous: torch.Tensor | None,
    parameters: SamplingParams,
) -> tuple[torch.Tensor, LogprobValues | None]:
    """Score a prompt run and return the logits for the next run.

    The executor supplies preceding logits for a continued run. Without
    them, the first token has no prediction and is excluded from scoring.
    All score tensors remain on the logits' device until output capture.
    """
    tokens = tokens.reshape(-1).to(device=logits.device, dtype=torch.long)
    if logits.ndim != 2 or int(logits.shape[0]) != int(tokens.numel()):
        raise invalid_descriptor(
            "prompt scoring logits do not align with input tokens"
        )

    if previous is None:
        score_logits, targets = logits[:-1], tokens[1:]
    else:
        previous = previous.reshape(1, -1).to(
            device=logits.device, dtype=logits.dtype
        )
        score_logits = torch.cat((previous, logits[:-1]), dim=0)
        targets = tokens

    retained = logits[-1].detach()
    if targets.numel() == 0:
        return retained, None

    indexes = torch.arange(
        int(targets.numel()), dtype=torch.long, device=score_logits.device
    )
    details = sampling.logprob_details(
        score_logits.float(),
        indexes,
        targets,
        (parameters,) * int(targets.numel()),
    )
    return retained, details


def speculative_positions(
    sampled: SamplerRow, visible: int, position: int, sampling_position: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Offset device-selected acceptance counts from their host coordinates.

    Accepted drafts include a correction or bonus token unless a draft
    finished the request. The selected extent stays on device for queued
    execution; native result materialization resolves it after completion.
    """
    count = sampled.accepted_token_count
    if count is None:
        accepted = sampled.accepted_draft_count
        if accepted is None:
            raise RuntimeError("speculative sampling lost its selected point")
        count = accepted.to(dtype=torch.int32) + 1
    return count + visible, count + position, count + sampling_position
