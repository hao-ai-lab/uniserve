"""Immutable attention-provider selection resolved before numerical execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from uniserve.attention.metadata import AttentionMode

if TYPE_CHECKING:
    from uniserve.attention.base import AttentionBackend


@dataclass(frozen=True, slots=True)
class AttentionSelection:
    """Immutable ordered backend set resolved before numerical calls."""

    identity: str
    providers: tuple[AttentionBackend, ...]

    def __post_init__(self) -> None:
        """Require a non-empty ordered set of distinct attention backends."""

        if not self.identity or not self.providers:
            raise ValueError("attention selection requires an identity and providers")
        names = tuple(str(provider.name) for provider in self.providers)
        if len(set(names)) != len(names):
            raise ValueError("attention selection contains duplicate providers")

    def select_provider(
        self,
        mode: AttentionMode,
        *,
        head_dim: int,
        block_size: int,
        device: torch.device,
    ) -> AttentionBackend | None:
        """Resolve the same geometry-bound provider for execution and graph preparation.

        Packed attention prefers providers that consume live segmentation on
        device. Other modes retain startup order; graph eligibility is checked
        against this selected implementation rather than a different candidate.
        """

        graph_options = (True, False) if mode is AttentionMode.PACKED else (False,)
        for cuda_graph in graph_options:
            for provider in self.providers:
                if provider.can_bind(
                    mode,
                    head_dim=head_dim,
                    block_size=block_size,
                    device=device,
                    cuda_graph=cuda_graph,
                ):
                    return provider
        return None
