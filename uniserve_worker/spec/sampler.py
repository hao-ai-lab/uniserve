"""SGLang-style target-only speculative sampling for linear draft chains."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch

from ..foundation.errors import invalid_descriptor
from ..nn.sampler import shape_logits_for_sampling, sync_tp_sampled_tokens

__all__ = [
    'SpeculativeSampleResult',
    'SpecSampleConfig',
    'speculative_sample_target_only',
    'accept_chain',
]

@dataclass(frozen=True)
class SpeculativeSampleResult:
    sampled_token_id: int
    num_accepted_tokens: int
    sampled_token_device: torch.Tensor


@dataclass(frozen=True)
class SpecSampleConfig:
    """Explicit accept thresholds, resolved once per sample call."""

    threshold_single: float
    threshold_acc: float

    @classmethod
    def resolve(
        cls,
        *,
        threshold_single: float | None,
        threshold_acc: float | None,
    ) -> "SpecSampleConfig":
        return cls(
            threshold_single=_resolve_threshold(
                threshold_single,
                default=1.0,
            ),
            threshold_acc=_resolve_threshold(
                threshold_acc,
                default=1.0,
            ),
        )


@dataclass(frozen=True)
class _AcceptChainResult:
    accepted: int
    sample_probs: torch.Tensor


def speculative_sample_target_only(
    logits: torch.Tensor,
    draft_token_ids: Sequence[int],
    sampling_params: dict[str, Any],
    *,
    recent: Sequence[int] | None = None,
    allowed: Sequence[int] | None = None,
    suppress: Sequence[int] | None = None,
    threshold_single: float | None = None,
    threshold_acc: float | None = None,
    uniform_samples: torch.Tensor | None = None,
    uniform_sample_for_final: torch.Tensor | None = None,
) -> SpeculativeSampleResult:
    """Accept a linear speculative draft using SGLang target-only semantics.

    SGLang's `tree_speculative_sampling_target_only` kernel accepts a candidate
    from the target distribution, then samples the replacement/bonus token from
    `relu(target_probs - rejected_candidate_probs)`. UniServe's current
    scheduler emits linear draft chains, so this implements the same threshold
    and residual-sampling behavior for one row without requiring the tree CUDA
    kernel.
    """

    draft = _validate_shapes(logits, draft_token_ids)
    rows = len(draft) + 1
    config = SpecSampleConfig.resolve(
        threshold_single=threshold_single,
        threshold_acc=threshold_acc,
    )

    target_probs = _target_probs_for_rows(
        logits[:rows],
        draft,
        sampling_params,
        recent=recent or (),
        allowed=allowed,
        suppress=suppress,
    )
    coins = _uniform_samples(
        uniform_samples,
        shape=(len(draft),),
        device=target_probs.device,
    )
    final_coin = _uniform_samples(
        uniform_sample_for_final,
        shape=(1,),
        device=target_probs.device,
    )[0]

    chain = accept_chain(target_probs, draft, coins, config)
    sampled = _sample_from_probs(chain.sample_probs, final_coin)
    return _tp_payload(sampled, chain.accepted)


def _validate_shapes(
    logits: torch.Tensor,
    draft_token_ids: Sequence[int],
) -> tuple[int, ...]:
    if logits.ndim != 2:
        raise invalid_descriptor("speculative target-only sampler expects logits shaped [rows, vocab]")
    draft = tuple(int(token_id) for token_id in draft_token_ids)
    rows = len(draft) + 1
    if int(logits.shape[0]) < rows:
        raise invalid_descriptor(
            f"speculative target-only sampler needs {rows} logit rows, got {int(logits.shape[0])}"
        )
    vocab = int(logits.shape[-1])
    if vocab <= 0:
        raise invalid_descriptor("speculative target-only sampler requires a non-empty vocabulary")
    for token_id in draft:
        if token_id < 0 or token_id >= vocab:
            raise invalid_descriptor(f"draft token id {token_id} is outside vocabulary size {vocab}")
    return draft


def accept_chain(
    target_probs: torch.Tensor,
    draft: tuple[int, ...],
    coins: torch.Tensor,
    config: SpecSampleConfig,
) -> _AcceptChainResult:
    """Accept the leading draft prefix, picking the residual probs on first reject.

    Decisions match the per-token loop: a position is accepted when
    `threshold_acc <= 0`, or `coin <= target_prob / threshold_acc`, or
    `target_prob >= threshold_single`. ``accepted`` is the length of the leading
    run of acceptances; on the first reject the residual sampling probs come from
    that row with the rejected candidate zeroed. When the whole draft is
    accepted the bonus row is used.
    """

    if config.threshold_acc <= 0.0:
        # Every position is accepted unconditionally; no coin draw needed.
        return _AcceptChainResult(accepted=len(draft), sample_probs=target_probs[len(draft)])

    if len(draft) == 0:
        return _AcceptChainResult(accepted=0, sample_probs=target_probs[0])

    draft_idx = torch.tensor(draft, device=target_probs.device, dtype=torch.long)
    draft_probs = target_probs[: len(draft)].gather(1, draft_idx.unsqueeze(1)).squeeze(1)
    accept_mask = torch.logical_or(
        coins <= draft_probs / config.threshold_acc,
        draft_probs >= config.threshold_single,
    )
    # ``accepted`` is the count of leading acceptances == index of the first
    # rejection. A single cumulative-product reduction replaces the per-token
    # ``.item()`` syncs while reproducing the same prefix decisions.
    leading = torch.cumprod(accept_mask.to(torch.long), dim=0)
    accepted = int(leading.sum().item())

    if accepted == len(draft):
        return _AcceptChainResult(accepted=accepted, sample_probs=target_probs[len(draft)])

    sample_probs = target_probs[accepted].clone()
    sample_probs[draft[accepted]] = 0
    return _AcceptChainResult(accepted=accepted, sample_probs=sample_probs)


def _tp_payload(sampled: torch.Tensor, accepted: int) -> SpeculativeSampleResult:
    payload = torch.cat(
        [
            sampled.new_full((1,), int(accepted), dtype=torch.long),
            sampled.to(dtype=torch.long),
        ]
    )
    payload = sync_tp_sampled_tokens(payload)
    payload_cpu = payload.detach().to("cpu")
    return SpeculativeSampleResult(
        sampled_token_id=int(payload_cpu[1].item()),
        num_accepted_tokens=int(payload_cpu[0].item()),
        sampled_token_device=payload[1:2],
    )


def _target_probs_for_rows(
    logits: torch.Tensor,
    draft: tuple[int, ...],
    sampling_params: dict[str, Any],
    *,
    recent: Sequence[int],
    allowed: Sequence[int] | None,
    suppress: Sequence[int] | None,
) -> torch.Tensor:
    work = logits.float().clone()
    vocab = int(work.shape[-1])
    for row in range(int(work.shape[0])):
        # Match SGLang's relaxed speculative-verify penalties: logits for every
        # verify row receive the same request-level recent-token penalties. The
        # accepted draft prefix is not folded into penalties row by row.
        row_recent = tuple(int(token_id) for token_id in recent)
        shape_logits_for_sampling(
            work[row],
            sampling_params,
            recent=row_recent,
            allowed=allowed,
            suppress=suppress,
            vocab=vocab,
        )
    return torch.softmax(work, dim=-1)


def _sample_from_probs(probs: torch.Tensor, coin: torch.Tensor) -> torch.Tensor:
    work = probs.float().clamp_min(0)
    total = work.sum()
    if bool((total <= 0).detach().to("cpu").item()):
        work = probs.float()
        total = work.sum()
    scaled = coin.to(device=work.device, dtype=work.dtype).clamp(0.0, 1.0) * total
    cdf = torch.cumsum(work, dim=-1)
    idx = torch.searchsorted(cdf, scaled, right=False)
    idx = idx.clamp(max=work.shape[-1] - 1)
    return idx.reshape(1).to(dtype=torch.long)


def _uniform_samples(
    samples: torch.Tensor | None,
    *,
    shape: tuple[int, ...],
    device: torch.device,
) -> torch.Tensor:
    if samples is None:
        return torch.rand(shape, dtype=torch.float32, device=device)
    if tuple(samples.shape) != shape:
        raise invalid_descriptor(
            f"speculative sampler uniform shape mismatch: expected {shape}, got {tuple(samples.shape)}"
        )
    return samples.to(device=device, dtype=torch.float32)


def _resolve_threshold(value: float | None, *, default: float) -> float:
    if value is not None:
        return float(value)
    return default
