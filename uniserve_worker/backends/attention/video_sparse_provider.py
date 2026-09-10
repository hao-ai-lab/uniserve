"""Device-capability selection and execution ownership for sparse attention."""

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial

import torch

from ...ops.video_sparse_rows import SparseRowExecution
from . import video_sparse_cute, video_sparse_flashinfer, video_sparse_sm100, video_sparse_triton


@dataclass(frozen=True, slots=True)
class SparseAttentionProvider:
    """Bind numerical execution and optional row production to one storage owner.

    Callers serialize calls sharing this provider. Independent execution domains
    resolve distinct providers, so their mutable plans and scratch never alias.
    Compiled code may remain globally cached because it owns no mutable inputs.
    """

    name: str
    execute: Callable[..., torch.Tensor]
    prepare_rows: Callable[..., Callable]
    row_major: bool


def resolve_sparse_provider(device: torch.device) -> SparseAttentionProvider:
    """Resolve the same installed numerical capabilities at startup and execution."""

    if video_sparse_sm100.available(device):
        rows = SparseRowExecution(video_sparse_cute.block_sparse_attention)
        return SparseAttentionProvider(
            "sm100", video_sparse_sm100.block_sparse_attention, rows.prepare, True
        )
    if video_sparse_flashinfer.available(device):
        state = video_sparse_flashinfer.SparseExecutionState()
        return SparseAttentionProvider(
            "flashinfer",
            partial(video_sparse_flashinfer.execute_sparse_attention, state),
            partial(video_sparse_flashinfer.prepare_sparse_attention_rows, state),
            video_sparse_flashinfer.uses_row_major_inputs(device),
        )
    if video_sparse_triton.available(device):
        rows = SparseRowExecution(video_sparse_triton.block_sparse_attention)
        return SparseAttentionProvider(
            "triton", video_sparse_triton.execute_sparse_attention, rows.prepare, True
        )
    raise RuntimeError(f"no installed sparse-attention provider supports {device}") from (
        video_sparse_sm100.import_error()
        or video_sparse_flashinfer.import_error()
        or video_sparse_triton.import_error()
    )
