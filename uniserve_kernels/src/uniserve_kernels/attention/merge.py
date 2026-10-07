"""Stable online-softmax combination of independently computed KV segments.

An attention state is a normalized output together with its log-sum-exp
(LSE) over one KV segment. Merging two states over disjoint segments
reproduces attention over their union. Providers report the LSE in different
bases: FlashAttention-4 and the portable SDPA path use the natural logarithm,
while TensorRT-LLM and FlashInfer kernels report base-2 values. Callers state
the base of the states they pass; both states share it. The FlashAttention-4
provider in ``uniserve.runtime.backends.attention.flash_attn_4`` merges its
current window with its paged prefix this way.
"""

from __future__ import annotations

import torch

from uniserve_kernels.triton import require_kernel, unsupported_operands

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
        base2: tl.constexpr,
    ):
        """Merge two independently normalized attention states.

        Rows index independent states (token, head); each row carries one LSE
        scalar plus ``head_dim`` contiguous output values. An LSE of -inf
        marks an empty segment whose output row receives zero weight. Each
        program merges ``block_rows`` rows; outputs combine in FP32 and round
        to the output dtype on store.
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
        # With both LSEs at -inf, ``lse - maximum`` is NaN; such rows get zero
        # weights and a -inf merged LSE instead. One empty side alone already
        # yields a zero weight through exp(-inf).
        both_empty = (first_lse == -float("inf")) & (
            second_lse == -float("inf")
        )
        # Base-2 states weight each side by 2^(lse - max) and add log2 of the
        # denominator; natural-log states use e and ln identically.
        if base2:
            first_scale = tl.exp2(first_lse - maximum)
            second_scale = tl.exp2(second_lse - maximum)
        else:
            first_scale = tl.exp(first_lse - maximum)
            second_scale = tl.exp(second_lse - maximum)
        denominator = first_scale + second_scale
        first_weight = tl.where(both_empty, 0.0, first_scale / denominator)
        second_weight = tl.where(both_empty, 0.0, second_scale / denominator)
        if base2:
            total = tl.log2(denominator)
        else:
            total = tl.log(denominator)
        merged_lse = tl.where(both_empty, -float("inf"), maximum + total)

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
    *,
    base2: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Merge independently evaluated KV segments with stable online softmax.

    Each state is an (output [..., head_dim], lse [...]) pair; returns the
    merged pair in the same layouts, with the output in ``first_output``'s
    dtype. ``base2`` states that both LSEs, and the merged result, are base-2
    logarithms; otherwise they are natural logarithms. Empty segments carry
    an LSE of -inf; when both segments are empty the merged output is zero
    and the merged LSE is -inf. CUDA states run the fused kernel; other
    devices evaluate the same formula with tensor operations. Both paths
    allocate new result tensors.

    Raises:
        ValueError: If the two outputs or the two LSE tensors differ in shape,
            or CUDA states do not fit the kernel (see :func:`unsupported`).
    """
    if (
        first_output.shape != second_output.shape
        or first_lse.shape != second_lse.shape
    ):
        raise ValueError("attention states must have matching shapes")

    if first_output.is_cuda:
        require_kernel(
            "merge_attention_states",
            unsupported(first_output, first_lse, second_output, second_lse),
            first_output=first_output,
            first_lse=first_lse,
            second_output=second_output,
            second_lse=second_lse,
        )
        output = torch.empty_like(first_output)
        merged_lse = torch.empty_like(first_lse)

        state_count = int(first_lse.numel())
        if state_count == 0:
            return output, merged_lse
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
            base2,
            num_warps=8,
        )
        return output, merged_lse

    # nan_to_num zeroes the NaN weights that both-empty rows produce, matching
    # the fused kernel.
    if base2:
        merged_lse = torch.logaddexp2(first_lse, second_lse)
        first_weight = torch.exp2(first_lse - merged_lse).nan_to_num(0.0)
        second_weight = torch.exp2(second_lse - merged_lse).nan_to_num(0.0)
    else:
        merged_lse = torch.logaddexp(first_lse, second_lse)
        first_weight = torch.exp(first_lse - merged_lse).nan_to_num(0.0)
        second_weight = torch.exp(second_lse - merged_lse).nan_to_num(0.0)

    output = (
        first_output.float() * first_weight.unsqueeze(-1)
        + second_output.float() * second_weight.unsqueeze(-1)
    ).to(first_output.dtype)
    return output, merged_lse


def unsupported(
    first_output: torch.Tensor,
    first_lse: torch.Tensor,
    second_output: torch.Tensor,
    second_lse: torch.Tensor,
) -> str | None:
    """Return why the fused merge cannot take two states, or ``None``.

    The kernel addresses rows as ``row * head_dim``, so every tensor must be
    contiguous and the LSE shape must equal the output's leading dimensions.
    Outputs share a floating dtype, as do the LSEs.
    """
    reason = unsupported_operands(
        first_output, first_lse, second_output, second_lse
    )
    if reason is not None:
        return reason
    floating = (torch.float16, torch.bfloat16, torch.float32)
    if (
        first_output.ndim != first_lse.ndim + 1
        or tuple(first_output.shape[:-1]) != tuple(first_lse.shape)
        or int(first_output.shape[-1]) == 0
    ):
        return "outputs are not [..., head_dim] rows over the LSE shape"
    if not all(
        tensor.is_contiguous()
        for tensor in (first_output, first_lse, second_output, second_lse)
    ):
        return "a state tensor is not contiguous"
    if (
        first_output.dtype != second_output.dtype
        or first_lse.dtype != second_lse.dtype
        or first_output.dtype not in floating
        or first_lse.dtype not in floating
    ):
        return "states do not share float16, bfloat16 or float32 dtypes"
    return None
