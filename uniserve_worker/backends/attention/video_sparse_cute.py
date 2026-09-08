"""CuTe DSL SM100 block-64 provider for shared video sparse attention."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import torch

_IMPORT_ERROR: BaseException | None = None
_cute: Any | None
_from_dlpack: Any | None
_requires_int64_kv_strides: Any | None
_kernel_type: Any | None
try:  # pragma: no cover - worker_config-only CUDA provider.
    import cuda.bindings.driver as _cuda_driver_module
    import cutlass.cute as _cute_module
    from cudnn.block_sparse_attention._interface import (
        _sm100_blk64_requires_int64_kv_strides as _stride_validator,
    )
    from cutlass.cute.runtime import from_dlpack as _dlpack_converter

    from uniserve_worker.backends.attention.sparse_sm100_kernel import (
        SparseAttentionSm100 as _sm100_kernel_type,
    )
except BaseException as error:  # pragma: no cover
    _IMPORT_ERROR = error
    _cuda_driver = None
    _cute = None
    _from_dlpack = None
    _requires_int64_kv_strides = None
    _kernel_type = None
else:  # pragma: no cover
    _cuda_driver = _cuda_driver_module
    _cute = _cute_module
    _from_dlpack = _dlpack_converter
    _requires_int64_kv_strides = _stride_validator
    _kernel_type = _sm100_kernel_type

_EXECUTORS: dict[tuple[object, ...], Callable[..., None]] = {}
_SOFTMAX_SCALE = 1.0 / math.sqrt(128)


def available(device: torch.device | None = None) -> bool:
    """Return whether the CuTe kernel can compile for the requested SM100 device."""

    if (
        _cute is None
        or _from_dlpack is None
        or _requires_int64_kv_strides is None
        or _kernel_type is None
        or not torch.cuda.is_available()
    ):
        return False
    major, _minor = torch.cuda.get_device_capability(device)
    return major == 10


def import_error() -> BaseException | None:
    """Return the exception that prevented CuTe kernel registration, if any."""

    return _IMPORT_ERROR


def should_use(*, rows: int, prefix_tiles: int) -> bool:
    """Select shapes whose measured provider boundary favors CuTe."""

    return int(rows) <= 65_536 or int(prefix_tiles) <= 64


def _dynamic_tensor(tensor: torch.Tensor, assumed_align: int = 16) -> Any:
    """Wrap a tensor as a CuTe dynamic-layout argument with declared alignment."""

    if _from_dlpack is None:
        raise RuntimeError("CuTe tensor ingress is unavailable") from _IMPORT_ERROR
    return _from_dlpack(
        tensor.detach(),
        assumed_align=assumed_align,
        enable_tvm_ffi=True,
    ).mark_layout_dynamic(leading_dim=tensor.ndim - 1)


def _tensor_abi_key(tensor: torch.Tensor) -> tuple[object, ...]:
    """Describe a tensor's device, dtype, shape, and stride for kernel specialization."""

    leading_dim = tensor.ndim - 1
    return (
        tensor.dtype,
        tensor.ndim,
        leading_dim,
        int(tensor.stride(leading_dim)),
        tuple(stride == 0 for stride in tensor.stride()),
    )


def _compile_key(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    block_indices: torch.Tensor,
    block_counts: torch.Tensor,
    valid_sizes: torch.Tensor,
    use_int64_kv_strides: bool,
    lse: torch.Tensor | None,
    allow_empty_blocks: bool,
) -> tuple[object, ...]:
    """Build the CuTe kernel specialization key from tensor layouts and stride mode."""

    device_index = query.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    return (
        device_index,
        torch.cuda.get_device_capability(query.device),
        torch.bfloat16,
        1,
        int(query.shape[1]),
        128,
        128,
        64,
        256,
        64,
        True,
        False,
        True,
        1,
        bool(use_int64_kv_strides),
        allow_empty_blocks,
        None if lse is None else _tensor_abi_key(lse),
        *(
            _tensor_abi_key(tensor)
            for tensor in (
                query,
                key,
                value,
                output,
                block_indices,
                block_counts,
                valid_sizes,
            )
        ),
    )


def _compile(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    block_indices: torch.Tensor,
    block_counts: torch.Tensor,
    valid_sizes: torch.Tensor,
    use_int64_kv_strides: bool,
    lse: torch.Tensor | None,
    allow_empty_blocks: bool,
) -> Callable[..., None]:
    """Compile and cache the block-sparse CuTe kernel for one concrete tensor geometry."""

    if _cute is None or _kernel_type is None:
        raise RuntimeError("CuTe sparse video attention is unavailable") from _IMPORT_ERROR
    kernel = _kernel_type(
        head_dim=128,
        head_dim_v=128,
        qhead_per_kvhead=1,
        pack_gqa=False,
        m_block_size=64,
        n_block_size=256,
        sparse_block_size=64,
        is_persistent=True,
        use_clc_scheduler=True,
        allow_empty_block_nums=allow_empty_blocks,
        has_block_sizes=True,
        num_splits=1,
        use_int64_kv_strides=use_int64_kv_strides,
    )
    return _cute.compile(
        kernel,
        _dynamic_tensor(query),
        _dynamic_tensor(key),
        _dynamic_tensor(value),
        _dynamic_tensor(output),
        None if lse is None else _dynamic_tensor(lse),
        _SOFTMAX_SCALE,
        _dynamic_tensor(block_indices),
        _dynamic_tensor(valid_sizes),
        0,
        _dynamic_tensor(block_counts),
        None,
        _cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=False),
        options="--enable-tvm-ffi",
    )


def _validate(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    block_indices: torch.Tensor,
    block_counts: torch.Tensor,
    valid_sizes: torch.Tensor,
    lse: torch.Tensor | None,
) -> None:
    """Validate tensor ranks, dtypes, shapes, devices, and block-index bounds."""

    if key.shape != value.shape or key.shape[1:] != query.shape[1:]:
        raise ValueError("CuTe sparse attention requires matching K/V and Q/K head geometry")
    if query.ndim != 3 or query.shape[1] < 1 or query.shape[2] != 128:
        raise ValueError("CuTe sparse attention requires [sequence, heads, 128] Q/K/V")
    if any(t.dtype != torch.bfloat16 for t in (query, key, value)) or not query.is_cuda:
        raise ValueError("CuTe sparse attention requires CUDA BF16 Q/K/V")
    if any(tensor.device != query.device for tensor in (key, value, output)):
        raise ValueError("CuTe sparse attention tensors must share one CUDA device")
    if (
        output.shape != query.shape
        or output.dtype not in (torch.bfloat16, torch.float32)
        or not output.is_contiguous()
    ):
        raise ValueError("CuTe sparse attention output must be contiguous BF16/FP32 with Q's shape")
    if any(tensor.stride(-1) != 1 for tensor in (query, key, value, output)):
        raise ValueError("CuTe sparse attention requires a contiguous head dimension")
    if any(stride % 8 for tensor in (query, key, value, output) for stride in tensor.stride()[:-1]):
        raise ValueError("CuTe sparse attention strides must preserve 16-byte alignment")
    if any(tensor.data_ptr() % 16 for tensor in (query, key, value, output)):
        raise ValueError("CuTe sparse attention pointers must be 16-byte aligned")
    rows = int(query.shape[0])
    if rows % 64 or key.shape[0] % 64:
        raise ValueError("CuTe sparse attention rows must be a multiple of 64")
    blocks = rows // 64
    if (
        block_indices.ndim != 3
        or tuple(block_indices.shape[:2]) != (query.shape[1], blocks)
        or block_indices.shape[2] < 1
        or block_counts.shape != (query.shape[1], blocks)
        or valid_sizes.shape != (key.shape[0] // 64,)
    ):
        raise ValueError("CuTe sparse attention metadata does not match Q/K/V")
    if any(
        tensor.dtype != torch.int32
        or tensor.device != query.device
        or not tensor.is_contiguous()
        or tensor.data_ptr() % 16
        for tensor in (block_indices, block_counts, valid_sizes)
    ):
        raise ValueError("CuTe sparse attention metadata must be aligned contiguous int32")

    if lse is not None and (
        lse.shape != (query.shape[1], rows)
        or lse.dtype != torch.float32
        or lse.device != query.device
        or not lse.is_contiguous()
    ):
        raise ValueError("CuTe sparse attention LSE must be contiguous FP32 [heads, query rows]")


def block_sparse_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    block_indices: torch.Tensor,
    block_counts: torch.Tensor,
    valid_sizes: torch.Tensor,
    *,
    lse: torch.Tensor | None = None,
    allow_empty_blocks: bool = False,
) -> torch.Tensor:
    """Evaluate selected key blocks for independent query and key extents.

    Block IDs address K/V tiles; counts and output address query tiles. Optional
    LSE stores the natural logarithm of each fine-attention normalizer. Empty
    key partitions produce zero output and negative-infinite LSE.
    """

    if not available(query.device):
        raise RuntimeError("CuTe sparse video attention is unavailable") from _IMPORT_ERROR
    _validate(
        query,
        key,
        value,
        output,
        block_indices,
        block_counts,
        valid_sizes,
        lse,
    )
    with torch.cuda.device(query.device):
        q_bhsd = query.unsqueeze(0).transpose(1, 2)
        k_bhsd = key.unsqueeze(0).transpose(1, 2)
        v_bhsd = value.unsqueeze(0).transpose(1, 2)
        o_bhsd = output.unsqueeze(0).transpose(1, 2)
        indices = block_indices.unsqueeze(0)
        counts = block_counts.unsqueeze(0)
        lse_bhq = None if lse is None else lse.unsqueeze(0)
        if _requires_int64_kv_strides is None:
            raise RuntimeError("CuTe sparse video attention stride validation is unavailable")
        use_int64_kv_strides = bool(_requires_int64_kv_strides(k_bhsd, v_bhsd))
        cache_key = _compile_key(
            q_bhsd,
            k_bhsd,
            v_bhsd,
            o_bhsd,
            indices,
            counts,
            valid_sizes,
            use_int64_kv_strides,
            lse_bhq,
            allow_empty_blocks,
        )
        executor = _EXECUTORS.get(cache_key)
        if executor is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "CuTe sparse video attention was not compiled before graph capture"
                )
            executor = _compile(
                q_bhsd,
                k_bhsd,
                v_bhsd,
                o_bhsd,
                indices,
                counts,
                valid_sizes,
                use_int64_kv_strides,
                lse_bhq,
                allow_empty_blocks,
            )
            _EXECUTORS[cache_key] = executor
        if _cuda_driver is None:
            raise RuntimeError("CUDA driver bindings are unavailable") from _IMPORT_ERROR
        executor(
            q_bhsd.detach(),
            k_bhsd.detach(),
            v_bhsd.detach(),
            o_bhsd.detach(),
            None if lse_bhq is None else lse_bhq.detach(),
            _SOFTMAX_SCALE,
            indices.detach(),
            valid_sizes.detach(),
            0,
            counts.detach(),
            None,
            _cuda_driver.CUstream(torch.cuda.current_stream().cuda_stream),
        )
    return o_bhsd


__all__ = [
    "available",
    "block_sparse_attention",
    "import_error",
    "should_use",
]
