"""Stable online-softmax combination of independently computed KV segments."""

from __future__ import annotations

import torch

from uniserve_kernels.triton import launchable

try:  # pragma: no cover - availability depends on the serving environment.
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover
    triton = None
    tl = None

if triton is not None:

    @triton.jit
    def _merge_attention_states_kernel(
        first_output_ptr,
        first_lse_ptr,
        second_output_ptr,
        second_lse_ptr,
        output_ptr,
        merged_lse_ptr,
        state_count,
        head_dim: tl.constexpr,
        block_rows: tl.constexpr,
        block_dim: tl.constexpr,
    ):
        """Merge two normalized attention states.

        Merge two independently normalized attention states with
        online-softmax rescaling.

        Rows index independent states (token, head); each row carries one LSE
        scalar plus `head_dim` output values. An LSE of -inf marks an empty
        segment whose output row contributes nothing.
        """
        rows = tl.program_id(0) * block_rows + tl.arange(0, block_rows)
        columns = tl.arange(0, block_dim)
        row_mask = rows < state_count

        first_lse = tl.load(
            first_lse_ptr + rows, mask=row_mask, other=-float("inf")
        )
        second_lse = tl.load(
            second_lse_ptr + rows, mask=row_mask, other=-float("inf")
        )
        maximum = tl.maximum(first_lse, second_lse)
        both_empty = (first_lse == -float("inf")) & (
            second_lse == -float("inf")
        )
        first_scale = tl.exp(first_lse - maximum)
        second_scale = tl.exp(second_lse - maximum)
        denominator = first_scale + second_scale
        first_weight = tl.where(both_empty, 0.0, first_scale / denominator)
        second_weight = tl.where(both_empty, 0.0, second_scale / denominator)
        merged_lse = tl.where(
            both_empty, -float("inf"), maximum + tl.log(denominator)
        )

        offsets = rows[:, None] * head_dim + columns[None, :]
        mask = row_mask[:, None] & (columns[None, :] < head_dim)
        first_output = tl.load(first_output_ptr + offsets, mask=mask, other=0.0)
        second_output = tl.load(
            second_output_ptr + offsets, mask=mask, other=0.0
        )
        merged = (
            first_output.to(tl.float32) * first_weight[:, None]
            + second_output.to(tl.float32) * second_weight[:, None]
        )
        tl.store(output_ptr + offsets, merged, mask=mask)
        tl.store(merged_lse_ptr + rows, merged_lse, mask=row_mask)


def merge_attention_states(
    first_output: torch.Tensor,
    first_lse: torch.Tensor,
    second_output: torch.Tensor,
    second_lse: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Merge independently evaluated KV segments with stable online softmax.

    Each state is an (output [..., head_dim], lse [...]) pair; returns the
    merged pair in the same layouts. Empty segments carry an LSE of -inf.
    """
    if (
        first_output.shape != second_output.shape
        or first_lse.shape != second_lse.shape
    ):
        raise ValueError("attention states must have matching shapes")

    if _triton_merge_eligible(
        first_output, first_lse, second_output, second_lse
    ):
        output = torch.empty_like(first_output)
        merged_lse = torch.empty_like(first_lse)

        state_count = int(first_lse.numel())
        head_dim = int(first_output.shape[-1])
        block_dim = triton.next_power_of_2(head_dim)
        block_rows = max(1, min(8, 1024 // block_dim))
        _merge_attention_states_kernel[(triton.cdiv(state_count, block_rows),)](
            first_output,
            first_lse,
            second_output,
            second_lse,
            output,
            merged_lse,
            state_count,
            head_dim,
            block_rows,
            block_dim,
            num_warps=8,
        )
        return output, merged_lse

    merged_lse = torch.logaddexp(first_lse, second_lse)
    first_weight = torch.exp(first_lse - merged_lse).nan_to_num(0.0)
    second_weight = torch.exp(second_lse - merged_lse).nan_to_num(0.0)

    output = (
        first_output.float() * first_weight.unsqueeze(-1)
        + second_output.float() * second_weight.unsqueeze(-1)
    ).to(first_output.dtype)
    return output, merged_lse


def _triton_merge_eligible(
    first_output: torch.Tensor,
    first_lse: torch.Tensor,
    second_output: torch.Tensor,
    second_lse: torch.Tensor,
) -> bool:
    """Check fused merge kernel eligibility.

    Return whether two attention states satisfy the fused merge kernel
    requirements.
    """
    tensors = (first_output, first_lse, second_output, second_lse)
    return bool(
        triton is not None
        and not torch.is_grad_enabled()
        and first_output.ndim == first_lse.ndim + 1
        and tuple(first_output.shape[:-1]) == tuple(first_lse.shape)
        and int(first_output.shape[-1]) > 0
        and int(first_lse.numel()) > 0
        and all(tensor.is_cuda and tensor.is_contiguous() for tensor in tensors)
        and len({tensor.device for tensor in tensors}) == 1
        and first_output.dtype == second_output.dtype
        and first_lse.dtype == second_lse.dtype
        and first_output.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and first_lse.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and launchable(first_output.device)
    )
