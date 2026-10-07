"""Borrowed numerical operator bindings for the active execution context.

Each ContextVar maps a layer or weight identity to an operator or storage
binding installed by the surrounding ExecutionContext. The empty defaults
keep standalone numerical calls working: they prepare their own execution
resources instead of borrowing graph-stable ones.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager
from contextvars import ContextVar
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    # The runtime bindings import these layers; their names are needed only
    # to annotate the values an ExecutionContext installs.
    from uniserve.runtime.bindings.attention import (
        AttentionBinding,
        ExchangeBuffers,
    )
    from uniserve.runtime.bindings.matmul import MatmulBinding
    from uniserve.runtime.bindings.moe import MoEBinding
    from uniserve.runtime.bindings.vsa import VsaBinding

# Branch (name, weight id) pairs and the interleaved branch width.
MergedKey = tuple[tuple[tuple[str, int], ...], int | None]

# Borrow ``size`` transport bytes on ``device`` for one chunked exchange.
ChunkStorage = Callable[
    [int, torch.device], AbstractContextManager[torch.Tensor]
]

# weight or module id -> prepared matmul operator
matmul: ContextVar[Mapping[int, MatmulBinding]] = ContextVar(
    "uniserve_matmul_operators", default={}
)

# module id, or branch key -> prepared merged operator
merged_matmul: ContextVar[Mapping[int | MergedKey, MatmulBinding]] = ContextVar(
    "uniserve_merged_matmul_operators", default={}
)

# module id -> reusable gather-storage binding for chunked token exchange
linear_chunks: ContextVar[Mapping[int, ChunkStorage]] = ContextVar(
    "uniserve_linear_chunk_storage", default={}
)

attention: ContextVar[Mapping[int, AttentionBinding]] = ContextVar(
    "uniserve_attention_operators", default={}
)
attention_storage: ContextVar[Mapping[int, ExchangeBuffers]] = ContextVar(
    "uniserve_attention_exchange_storage", default={}
)
vsa: ContextVar[Mapping[int, VsaBinding]] = ContextVar(
    "uniserve_vsa_operators", default={}
)

# FusedMoE module id -> prepared routed-expert operator
moe: ContextVar[Mapping[int, MoEBinding]] = ContextVar(
    "uniserve_moe_operators", default={}
)
