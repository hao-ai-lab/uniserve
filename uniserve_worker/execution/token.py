"""Numerical token rows, prompt scores and speculative device coordinates."""

from __future__ import annotations

import torch

from uniserve.runtime.device import async_tensor_h2d
from uniserve.sampling import SamplingParams
from uniserve_worker.errors import invalid_descriptor
from uniserve_worker.model_executor.input_batch import TokenRow
from uniserve_worker.protocol.call import ForwardMode
from uniserve_worker.sampling import sampler as sampling
from uniserve_worker.sampling.metadata import TokenSelection
from uniserve_worker.sampling.result import LogprobValues, SamplerRow


def token_row(
    mode: ForwardMode,
    tokens: tuple[int, ...],
    position: int,
    slot: int,
    visible: int,
    selection: TokenSelection,
    *,
    current: torch.Tensor | None = None,
    indexed: bool = False,
    predicate: tuple[torch.Tensor, bool] | None = None,
    force_finish: bool = False,
) -> TokenRow:
    """Build numerical views from executor-selected tokens and coordinates.

    A device continuation stays on device, including when concatenated with
    verification drafts. Indexed decode borrows its inputs from DecodeState
    during batch packing and requires no per-row token or position tensor.
    """
    token_values = position_values = None
    if not indexed:
        if current is None:
            token_values = torch.tensor(tokens, dtype=torch.long)
        elif tokens:
            drafts = async_tensor_h2d(
                tokens, dtype=torch.long, device=current.device
            )
            token_values = torch.cat((current.reshape(1), drafts))
        else:
            token_values = current.reshape(1)

        position_values = torch.arange(
            position, position + token_values.numel(), dtype=torch.long
        )

    return TokenRow(
        forward_mode=mode,
        token_ids=token_values,
        positions=position_values,
        selection=selection,
        request_pool_idx=slot,
        seq_len=visible,
        write_kv=True,
        causal=True,
        request_indexed_decode=indexed,
        decode_predicate=None if predicate is None else predicate[0],
        decode_predicate_tagged=False if predicate is None else predicate[1],
        decode_force_finish=force_finish,
    )


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
