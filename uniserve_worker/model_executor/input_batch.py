"""One numerical input view and the aligned rows used to construct it."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

import torch

from uniserve_worker.protocol.call import ForwardMode, MediaCall
from uniserve_worker.sampling.metadata import TokenSelection

InputT = TypeVar("InputT")


@dataclass(frozen=True, slots=True)
class InputBatch(Generic[InputT]):
    """One typed numerical input and the worker's aligned output controls."""

    forward_mode: ForwardMode | MediaCall
    inputs: InputT
    request_pool_indices: torch.Tensor
    token_selections: tuple[TokenSelection, ...] = ()
    decode_force_finish: torch.Tensor | None = None

    @property
    def row_count(self) -> int:
        return self.request_pool_indices.numel()

    def __post_init__(self):
        if self.request_pool_indices.ndim != 1 or self.row_count < 1:
            raise ValueError(
                "execution requires a nonempty vector of request slots"
            )
        if (
            isinstance(self.forward_mode, ForwardMode)
            and len(self.token_selections) != self.row_count
        ):
            raise ValueError(
                "text output selections must align with request slots"
            )
        if self.decode_force_finish is not None and (
            self.decode_force_finish.shape != self.request_pool_indices.shape
            or self.decode_force_finish.dtype != torch.bool
        ):
            raise ValueError(
                "decode completion controls must align with request slots"
            )


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

    @property
    def query_tokens(self) -> int:
        """Return the query length this row appends after its cached prefix."""
        raise NotImplementedError


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
