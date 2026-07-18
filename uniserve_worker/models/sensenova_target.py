"""Target SenseNova family adapter over the dormant unified stack.

Stage 6 family port slice: the SenseNova root registers two resident routes
— text understanding (route 0) and generation (route 1) — over the shared
route-aware decoder. Text decode rows, CFG denoise flow rows (transient
overlays on branch sequences), encode rows, and materialize rows traverse
the same root in one packed invocation; route and overlay variation is
segment data, never a model branch or a second forward.

Output projection reuses the shared greedy projection: sequence rows sample,
flow/encode/materialize rows project no tokens (their value is the committed
cache effect or published product). Real checkpoint geometry, the temporal
tower, and image conditioning arrive with the production port; this slice
fixes the composition semantics the conformance matrix scales up.
"""
from __future__ import annotations

from ..contracts.residency_batch import ResidencyBatchArrays
from ..contracts.segment_table import GraphCapacity, SegmentTableArrays
from ..execution.transaction import AdapterPayload, AdapterRowOutcome
from ..nn.grouped_routing import WeightOverlayBank
from ..nn.target_decoder import SharedAttention, TargetDecoderConfig, TargetDecoderRoot
from .qwen3_target import project_greedy_outcomes

__all__ = ["SenseNovaTarget"]

_TEXT_ROUTE = 0
_GENERATION_ROUTE = 1


class SenseNovaTarget:
    """Two-route resident root; one packed traversal for mixed batches."""

    def __init__(
        self,
        config: TargetDecoderConfig,
        attention: SharedAttention,
        *,
        device: str = "cuda",
        seed: int = 0,
        overlay_bank: WeightOverlayBank | None = None,
        dtype=None,
        zero_init: bool = False,
    ) -> None:
        import torch

        dtype = dtype if dtype is not None else torch.float32
        if config.routes != 2:
            raise ValueError(
                "SenseNova registers a text route and a generation route"
            )
        self.root = TargetDecoderRoot(
            config,
            attention,
            device=device,
            seed=seed,
            overlay_bank=overlay_bank,
            dtype=dtype,
            zero_init=zero_init,
        )

    @property
    def config(self) -> TargetDecoderConfig:
        return self.root.config

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

    def raw_logits(
        self,
        segments: SegmentTableArrays,
        residency: ResidencyBatchArrays,
        capacity: GraphCapacity,
        payload: AdapterPayload,
    ):
        """Diagnostic access for conformance tests (never a production seam)."""

        return self.root.logits(
            segments, residency, capacity, payload.token_ids, payload.positions
        )
