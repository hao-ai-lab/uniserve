"""Target BAGEL family adapter over the dormant unified stack.

Stage 6 family port slice: BAGEL registers the understanding route (0) and
the generation route (1) over the shared route-aware decoder, with the
family's marker/generation-body/marker denoise route runs coming from its
cache registration — text-routed image markers surrounding generation-routed
latent tokens share one attention region, exactly the segment shape the
lowering derives. Output projection is the shared greedy projection. Real
checkpoint geometry, VAE/ViT transforms, and the production port arrive at
cutover; this slice fixes the composition semantics.
"""
from __future__ import annotations

from ..contracts.residency_batch import ResidencyBatchArrays
from ..contracts.segment_table import GraphCapacity, SegmentTableArrays
from ..execution.transaction import AdapterPayload, AdapterRowOutcome
from ..nn.grouped_routing import WeightOverlayBank
from ..nn.target_decoder import SharedAttention, TargetDecoderConfig, TargetDecoderRoot
from .qwen3_target import project_greedy_outcomes

__all__ = ["BagelTarget"]


class BagelTarget:
    """Two-route resident root with marker-run denoise regions."""

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
                "BAGEL registers an understanding route and a generation route"
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
