"""CPU values that move through one execution step."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import torch

from uniserve_worker.execution.sampling import TokenSelection
from uniserve_worker.protocol.identity import ComputationId, RequestKey
from uniserve_worker.protocol.operation import ForwardMode, PipelineStage

OperationIdentity: TypeAlias = tuple[RequestKey, ComputationId]


@dataclass(frozen=True, slots=True)
class ForwardRow:
    """Immutable input views or a request-slot continuation for one model row.

    An indexed decode borrows token and position columns from DecodeState at
    staging time; it carries no redundant per-row tensor views.
    """

    forward_mode: ForwardMode | PipelineStage
    token_ids: torch.Tensor | None = None
    token_embeddings: torch.Tensor | None = None
    token_embedding_mask: torch.Tensor | None = None
    positions: torch.Tensor | None = None
    selection: TokenSelection | None = None
    timestep: torch.Tensor | None = None
    latent: torch.Tensor | None = None
    image_tokens: int = 0
    image_height: int = 0
    image_width: int = 0
    encode_pixels: torch.Tensor | None = None
    encode_grid: torch.Tensor | None = None
    encode_grid_shape: tuple[int, int] | None = None
    request_pool_idx: int = 0
    seq_len: int = 0
    group_id: int = 0
    write_kv: bool = False
    causal: bool = True
    decode_predicate: torch.Tensor | None = None
    decode_predicate_tagged: bool = False
    decode_force_finish: bool = False
    request_indexed_decode: bool = False

    @property
    def query_tokens(self) -> int:
        """Return the live token or image-patch count represented by this.

        row.
        """
        if self.request_indexed_decode:
            return 1
        if self.token_ids is not None:
            return int(self.token_ids.numel())
        if self.latent is not None and self.image_tokens > 0:
            return int(self.image_tokens)
        return 0


__all__ = [
    "ForwardRow",
    "OperationIdentity",
]
