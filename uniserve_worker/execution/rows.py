"""CPU values that move through one execution step."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import torch

from uniserve_worker.execution.sampling import TokenSelection
from uniserve_worker.protocol.call import ForwardMode, MediaCall
from uniserve_worker.protocol.identity import CallId, RequestKey

CallIdentity: TypeAlias = tuple[RequestKey, CallId]


@dataclass(frozen=True, slots=True)
class InputRow:
    """One numerical input and the request slot receiving its output."""

    forward_mode: ForwardMode | MediaCall
    request_pool_idx: int = 0


@dataclass(frozen=True, slots=True)
class AttentionRow(InputRow):
    """Position and cache coordinates of one attention sequence."""

    positions: torch.Tensor | None = None
    seq_len: int = 0
    group_id: int = 0
    write_kv: bool = False
    causal: bool = True


@dataclass(frozen=True, slots=True)
class TokenRow(AttentionRow):
    """Token views or a resident request-slot continuation.

    Indexed decode borrows tokens and positions from DecodeState at staging
    time and carries no duplicate per-row tensor views.
    """

    token_ids: torch.Tensor | None = None
    token_embeddings: torch.Tensor | None = None
    token_embedding_mask: torch.Tensor | None = None
    selection: TokenSelection | None = None
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
        return 0


@dataclass(frozen=True, slots=True, kw_only=True)
class DiffusionRow(AttentionRow):
    """Latent sample, solver time and spatial extent of an image sequence."""

    timestep: torch.Tensor
    latent: torch.Tensor
    image_tokens: int
    image_height: int
    image_width: int

    @property
    def query_tokens(self) -> int:
        return self.image_tokens


@dataclass(frozen=True, slots=True, kw_only=True)
class VisionRow(InputRow):
    """Prepared pixels and optional patch-grid coordinates."""

    encode_pixels: torch.Tensor
    encode_grid: torch.Tensor | None = None
    encode_grid_shape: tuple[int, int] | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class DecodeRow(InputRow):
    """Latent sample and the image dimensions requested from its decoder."""

    latent: torch.Tensor
    image_height: int
    image_width: int


__all__ = [
    "InputRow",
    "AttentionRow",
    "TokenRow",
    "DiffusionRow",
    "VisionRow",
    "DecodeRow",
    "CallIdentity",
]
