"""KV-cache quantization helpers.

These helpers intentionally provide a dequantized correctness floor for cache
reads. Paged attention kernels need a separate scale-aware storage contract
before they can consume quantized pages directly.
"""
from __future__ import annotations

import torch

__all__ = [
    'FP8_MAX',
    'SCALE_EPS',
    'fp8_quantize',
    'fp8_scale_from',
    'resolve_kv_store_dtype',
    'is_fp8_kv_dtype',
    'kv_store_dtype_name',
    'kv_store_itemsize',
    'kv_cache_bytes_per_token',
    'scale_for_fp8_block',
    'quantize_fp8_block',
    'dequantize_fp8_block',
]

FP8_MAX = 448.0
SCALE_EPS = 1.0e-12


_KV_STORE_NAME_TO_DTYPE: dict[str, torch.dtype] = {
    "float8_e4m3fn": torch.float8_e4m3fn,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}

_KV_STORE_DTYPE_TO_NAME: dict[torch.dtype, str] = {
    dtype: name for name, dtype in _KV_STORE_NAME_TO_DTYPE.items()
}
_KV_STORE_DTYPE_ITEMSIZE = {
    torch.float8_e4m3fn: 1,
    torch.bfloat16: 2,
    torch.float16: 2,
    torch.float32: 4,
}


def fp8_quantize(tensor: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return (tensor / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)


def fp8_scale_from(tensor: torch.Tensor, *, dim: int | None) -> torch.Tensor:
    if dim is None:
        return tensor.abs().amax().clamp_min(SCALE_EPS) / FP8_MAX
    return tensor.abs().amax(dim=dim, keepdim=True).clamp_min(SCALE_EPS) / FP8_MAX


def resolve_kv_store_dtype(
    compute_dtype: torch.dtype,
    store_dtype: torch.dtype | str | None,
) -> torch.dtype:
    if store_dtype is None:
        return compute_dtype
    if isinstance(store_dtype, torch.dtype):
        return store_dtype
    name = str(store_dtype)
    resolved = _KV_STORE_NAME_TO_DTYPE.get(name)
    if resolved is not None:
        return resolved
    raise ValueError(f"unsupported KV cache store dtype {store_dtype!r}")


def is_fp8_kv_dtype(dtype: torch.dtype) -> bool:
    return dtype == torch.float8_e4m3fn


def kv_store_dtype_name(dtype: torch.dtype) -> str:
    name = _KV_STORE_DTYPE_TO_NAME.get(dtype)
    if name is not None:
        return name
    return str(dtype).replace("torch.", "")


def kv_store_itemsize(dtype: torch.dtype) -> int:
    size = _KV_STORE_DTYPE_ITEMSIZE.get(dtype)
    if size is not None:
        return int(size)
    return int(torch.empty((), dtype=dtype).element_size())


def kv_cache_bytes_per_token(
    *,
    num_kv_heads: int,
    head_dim: int,
    num_layers: int,
    compute_dtype: torch.dtype,
    store_dtype: torch.dtype | str | None,
) -> int:
    resolved = resolve_kv_store_dtype(compute_dtype, store_dtype)
    return int(int(num_kv_heads) * int(head_dim) * 2 * int(num_layers) * kv_store_itemsize(resolved))


def scale_for_fp8_block(block: torch.Tensor) -> torch.Tensor:
    scale = fp8_scale_from(block.to(torch.float32), dim=None)
    return scale.reshape(1, 1, 1).to(dtype=torch.float32)


def quantize_fp8_block(block: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return fp8_quantize(block.to(torch.float32), scale)


def dequantize_fp8_block(block: torch.Tensor, scale: torch.Tensor, *, dtype: torch.dtype) -> torch.Tensor:
    out = block.to(torch.float32) * scale.to(torch.float32)
    if dtype in {torch.float16, torch.bfloat16, torch.float32, torch.float64}:
        return out.to(dtype=dtype)
    return out
