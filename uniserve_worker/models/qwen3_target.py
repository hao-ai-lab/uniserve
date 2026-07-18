"""Target Qwen3 family adapter over the dormant unified stack.

Stage 6 family port of ``specs/unified_forward_execution.md``: one adapter,
one resident root, one packed traversal. The shared route-aware decoder root
lives in :mod:`uniserve_worker.nn.target_decoder`; this family file owns the
Qwen3 configuration (one text route) and the family output projection —
greedy sampled tokens for requested output positions and greedy candidate
acceptance for verification spans. Family code owns no cache state,
provider, plan, or graph; attention arrives through the injected shared
seam. Weights arrive through the root constructor (checkpoint reuse is
explicitly permitted); tests prove conformance against an independent dense
recompute, the Stage 11 shape at unit scale.
"""
from __future__ import annotations

from ..contracts.cache_schema import CacheEffect
from ..contracts.residency_batch import ResidencyBatchArrays
from ..contracts.segment_table import GraphCapacity, SegmentTableArrays
from ..execution.transaction import AdapterPayload, AdapterRowOutcome
from ..nn.grouped_routing import WeightOverlayBank
from ..nn.target_decoder import SharedAttention, TargetDecoderConfig, TargetDecoderRoot

__all__ = ["Qwen3Target", "Qwen3TargetConfig", "project_greedy_outcomes"]

Qwen3TargetConfig = TargetDecoderConfig


def project_greedy_outcomes(
    segments: SegmentTableArrays,
    capacity: GraphCapacity,
    payload: AdapterPayload,
    logits,
) -> tuple[AdapterRowOutcome, ...]:
    """Family output projection: greedy sampling and candidate acceptance.

    Rows with persistent sequence segments sample greedily at their final
    position; verification spans accept candidates while each matches the
    prediction from the previous position; rows with only transient or
    read-only segments (denoise, encode, materialize) project no tokens.
    """

    greedy = logits.argmax(dim=-1)
    sampled: dict[int, int | None] = {}
    accepted: dict[int, int] = {}
    for index in range(capacity.segments):
        if not segments.segment_active[index]:
            continue
        row = segments.row_id[index]
        sampled.setdefault(row, None)
        accepted.setdefault(row, 0)
        begin = segments.token_begin[index]
        count = segments.token_count[index]
        if segments.candidate_count[index]:
            matched = 0
            for offset in range(count):
                predicted = int(greedy[begin + offset - 1])
                if predicted == payload.token_ids[begin + offset]:
                    matched += 1
                else:
                    break
            accepted[row] = matched
            sampled[row] = int(greedy[begin + count - 1])
        elif segments.cache_effect[index] == int(CacheEffect.PERSISTENT_APPEND):
            sampled[row] = int(greedy[begin + count - 1])
    outcomes: list[AdapterRowOutcome] = []
    for row in sorted(sampled):
        token = sampled[row]
        outcomes.append(
            AdapterRowOutcome(
                sampled_tokens=(token,) if token is not None else (),
                accepted_candidates=accepted[row],
            )
        )
    return tuple(outcomes)


class Qwen3Target:
    """One resident root, one packed traversal, compact projected outcomes."""

    def __init__(
        self,
        config: TargetDecoderConfig,
        attention: SharedAttention,
        *,
        device: str = "cuda",
        seed: int = 0,
        overlay_bank: WeightOverlayBank | None = None,
    ) -> None:
        if config.routes != 1:
            raise ValueError("Qwen3 registers one text route")
        self.root = TargetDecoderRoot(
            config, attention, device=device, seed=seed, overlay_bank=overlay_bank
        )

    @property
    def config(self) -> TargetDecoderConfig:
        return self.root.config

    @property
    def device(self):
        return self.root.device

    @property
    def embedding(self):
        return self.root.embedding

    @property
    def layers(self):
        return self.root.layers

    @property
    def final_norm(self):
        return self.root.final_norm

    @property
    def lm_head(self):
        return self.root.lm_head

    def forward(
        self,
        segments: SegmentTableArrays,
        residency: ResidencyBatchArrays,
        capacity: GraphCapacity,
        payload: AdapterPayload,
    ) -> tuple[AdapterRowOutcome, ...]:
        logits = self.root.logits(
            segments, residency, capacity, payload.token_ids, payload.positions
        )
        return project_greedy_outcomes(segments, capacity, payload, logits)
