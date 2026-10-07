"""Numerical updates over native-owned request continuation tensors.

Slot zero is inactive padding. The native owner validates host slots and
chooses fused CUDA or indexed tensor updates. Device coordinates remain
on device, including speculative acceptance counts.
"""

from __future__ import annotations

import torch

from uniserve.runtime.device import async_tensor_h2d
from uniserve_worker._uniserve_ipc import DecodeState
from uniserve_worker.storage import _decode_state as kernels


def warmup(tensors, capacity, width, vocab):
    """Compile fixed-capacity updates without changing real request rows."""
    device = tensors["predicates"].device
    columns = async_tensor_h2d((0, 0, 0, 0), dtype=torch.int64, device=device)
    reset_rows(tensors, columns, 1, width, vocab)
    commit_tokens_cuda(
        tensors,
        tensors["_ones_int64"],
        tensors["future_input_tokens"][:, 0],
        tensors["predicates"],
        0,
        width,
        capacity,
    )


def reset_rows(tensors, columns, count, width, vocab):
    """Reset [4, count] columns: slots, cache, logical and RNG lengths."""
    block_size = 256
    kernels._reset_rows_kernel[
        (count, kernels.triton.cdiv(max(width, vocab), block_size))
    ](
        columns,
        tensors["future_input_tokens"],
        tensors["penalty_counts"],
        tensors["predicates"],
        tensors["logical_lengths"],
        tensors["sampling_positions"],
        tensors["valid_cache_lengths"],
        count,
        continuation_width=width,
        vocab_size=vocab,
        block_size=block_size,
    )


def reset_indexed(tensors, indices, valid, logical, sampling):
    """Reset selected rows, accepting host or device-valued columns."""
    tensors["future_input_tokens"].index_fill_(0, indices, 1)
    tensors["penalty_counts"].index_fill_(0, indices, 0)
    tensors["predicates"].index_fill_(0, indices, False)
    for name, values in (
        ("valid_cache_lengths", valid),
        ("logical_lengths", logical),
        ("sampling_positions", sampling),
    ):
        target = tensors[name]
        if values is None:
            target.index_fill_(0, indices, 0)
            continue
        source = torch.as_tensor(
            values, dtype=target.dtype, device=target.device
        ).reshape(-1)
        if source.numel() != indices.numel():
            raise ValueError("runtime-state reset columns are not aligned")
        target[indices] = source


def device_indices(values, device, capacity):
    """Check device-provided slots asynchronously, with no scalar readback."""
    indices = values.reshape(-1).to(device=device, dtype=torch.long)
    if indices.numel() and indices.device.type == "cuda":
        torch._assert_async(
            torch.all((indices >= 1) & (indices <= capacity)),
            "request-pool index is outside runtime-state capacity",
        )
        if indices.numel() > 1:
            ordered = torch.sort(indices).values
            torch._assert_async(
                torch.all(ordered[1:] != ordered[:-1]),
                "runtime-state mutation repeats a request-pool index",
            )
    return indices


def copy_scalar(target, value):
    """Copy a device scalar or fill a host integer without scalar readback."""
    if isinstance(value, torch.Tensor):
        target.copy_(
            value.reshape(-1)[:1].to(device=target.device, dtype=target.dtype)
        )
    else:
        target.fill_(int(value))


def commit_explicit(tensors, slot, tokens, predicates, logical, sampling):
    """Install a prefill or verification result at its selected coordinates."""
    future = tensors["future_input_tokens"][slot, :1]
    future.copy_(tokens.reshape(-1)[:1])
    future.bitwise_and_((1 << 31) - 1)
    tensors["predicates"][slot : slot + 1].copy_(
        predicates.reshape(-1)[:1].to(torch.bool)
    )
    copy_scalar(tensors["logical_lengths"][slot : slot + 1], logical)
    copy_scalar(tensors["sampling_positions"][slot : slot + 1], sampling)
    return future


def commit_tokens_cuda(
    tensors, indices, tokens, predicates, count, width, capacity
):
    """Advance live decode rows with a capacity-specialized CUDA kernel."""
    kernels._commit_tokens_kernel[(1,)](
        indices,
        tokens,
        predicates,
        tensors["future_input_tokens"],
        tensors["predicates"],
        tensors["logical_lengths"],
        tensors["sampling_positions"],
        tensors["valid_cache_lengths"],
        count=count,
        continuation_width=width,
        index_stride=indices.stride(0),
        token_stride=tokens.stride(0),
        predicate_stride=predicates.stride(0),
        block_size=kernels.triton.next_power_of_2(capacity),
    )


def commit_tokens_torch(
    tensors, indices, tokens, predicates, count, width, capacity
):
    """Advance decode rows with indexed tensor operations."""
    tensors["future_input_tokens"][:, 0].index_copy_(
        0, indices, tokens.to(dtype=torch.int64).bitwise_and((1 << 31) - 1)
    )
    tensors["predicates"].index_copy_(
        0, indices, predicates.to(dtype=torch.bool)
    )
    tensors["logical_lengths"].index_add_(
        0, indices, tensors["_ones_int32"][:count]
    )
    tensors["sampling_positions"].index_add_(
        0, indices, tensors["_ones_int64"][:count]
    )
    tensors["valid_cache_lengths"].index_add_(
        0, indices, tensors["_ones_int32"][:count]
    )


def commit_penalties(tokens, valid, active, rows):
    """Count only active valid tokens, excluding their continuation flag."""
    tokens = tokens.reshape(-1)
    valid, active = valid.reshape(-1), active.reshape(-1)
    for index, counts in rows:
        weight = (valid[index : index + 1] & active[index : index + 1]).to(
            counts.dtype
        )
        selected = (
            tokens[index : index + 1].to(torch.int64).bitwise_and((1 << 31) - 1)
        )
        counts.scatter_add_(0, selected, weight)


__all__ = ["DecodeState"]
