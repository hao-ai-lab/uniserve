# SPDX-License-Identifier: Apache-2.0
"""Fixed-width fused operations for the MiniMax H3 video decoder."""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch.nn import functional as F

__all__ = [
    "qk_rmsnorm_partial_rope_",
    "scaled_residual_",
    "scaled_residual_layernorm",
    "scaled_residual_rmsnorm_",
    "value_first_swiglu",
    "video_rmsnorm",
]

_WIDTH = 2048
_HEAD_DIM = 64
_ROTARY_DIM = 48
_WIDTH_TL = tl.constexpr(2048)
_HEAD_DIM_TL = tl.constexpr(64)
_ROTARY_DIM_TL = tl.constexpr(48)


@triton.jit
def _video_rmsnorm_kernel(
    hidden_ptr,
    weight_ptr,
    output_ptr,
    eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    columns = tl.arange(0, BLOCK)
    hidden = tl.load(hidden_ptr + row * _WIDTH_TL + columns).to(tl.float32)
    weight = tl.load(weight_ptr + columns).to(tl.float32)
    inverse_rms = tl.rsqrt(tl.sum(hidden * hidden, axis=0) / _WIDTH_TL + eps)
    tl.store(output_ptr + row * _WIDTH_TL + columns, hidden * inverse_rms * weight)


@triton.jit
def _scaled_residual_rmsnorm_kernel(
    hidden_ptr,
    update_ptr,
    scale_ptr,
    weight_ptr,
    output_ptr,
    eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    columns = tl.arange(0, BLOCK)
    offsets = row * _WIDTH_TL + columns
    hidden = tl.load(hidden_ptr + offsets).to(tl.float32)
    update = tl.load(update_ptr + offsets).to(tl.float32)
    scale = tl.load(scale_ptr + columns).to(tl.float32)
    residual = hidden + update * scale
    inverse_rms = tl.rsqrt(tl.sum(residual * residual, axis=0) / _WIDTH_TL + eps)
    weight = tl.load(weight_ptr + columns).to(tl.float32)
    tl.store(hidden_ptr + offsets, residual)
    tl.store(output_ptr + offsets, residual * inverse_rms * weight)


@triton.jit
def _scaled_residual_kernel(
    hidden_ptr,
    update_ptr,
    scale_ptr,
    elements,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < elements
    columns = offsets % _WIDTH_TL
    hidden = tl.load(hidden_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    update = tl.load(update_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    scale = tl.load(scale_ptr + columns, mask=mask, other=0.0).to(tl.float32)
    tl.store(hidden_ptr + offsets, hidden + update * scale, mask=mask)


@triton.jit
def _scaled_residual_layernorm_kernel(
    hidden_ptr,
    update_ptr,
    scale_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    columns = tl.arange(0, BLOCK)
    offsets = row * _WIDTH_TL + columns
    hidden = tl.load(hidden_ptr + offsets).to(tl.float32)
    update = tl.load(update_ptr + offsets).to(tl.float32)
    scale = tl.load(scale_ptr + columns).to(tl.float32)
    residual = hidden + update * scale
    mean = tl.sum(residual, axis=0) / _WIDTH_TL
    centered = residual - mean
    inverse_std = tl.rsqrt(tl.sum(centered * centered, axis=0) / _WIDTH_TL + eps)
    weight = tl.load(weight_ptr + columns).to(tl.float32)
    bias = tl.load(bias_ptr + columns).to(tl.float32)
    tl.store(output_ptr + offsets, centered * inverse_std * weight + bias)


@triton.jit
def _qk_rmsnorm_partial_rope_kernel(
    query_ptr,
    key_ptr,
    cosine_ptr,
    sine_ptr,
    rows: tl.constexpr,
    heads: tl.constexpr,
    row_stride: tl.constexpr,
    head_stride: tl.constexpr,
    rotary_row_stride: tl.constexpr,
    ROW_BLOCK: tl.constexpr,
):
    row = tl.program_id(0) * ROW_BLOCK + tl.arange(0, ROW_BLOCK)
    head = tl.program_id(1)
    columns = tl.arange(0, _HEAD_DIM_TL)
    valid = row[:, None] < rows
    offsets = row[:, None] * row_stride + head * head_stride + columns[None, :]
    query = tl.load(query_ptr + offsets, mask=valid, other=0.0).to(tl.float32)
    key = tl.load(key_ptr + offsets, mask=valid, other=0.0).to(tl.float32)
    query_rstd = tl.rsqrt(tl.sum(query * query, axis=1) / _HEAD_DIM_TL + 1e-5)
    key_rstd = tl.rsqrt(tl.sum(key * key, axis=1) / _HEAD_DIM_TL + 1e-5)
    query = (query * query_rstd[:, None]).to(query_ptr.dtype.element_ty).to(tl.float32)
    key = (key * key_rstd[:, None]).to(key_ptr.dtype.element_ty).to(tl.float32)

    half_rotary: tl.constexpr = _ROTARY_DIM_TL // 2
    partner_columns = tl.where(
        columns < half_rotary,
        columns + half_rotary,
        columns - half_rotary,
    )
    partner_columns = tl.where(columns < _ROTARY_DIM_TL, partner_columns, columns)
    partner_offsets = (
        row[:, None] * row_stride + head * head_stride + partner_columns[None, :]
    )
    query_partner = tl.load(query_ptr + partner_offsets, mask=valid, other=0.0).to(tl.float32)
    key_partner = tl.load(key_ptr + partner_offsets, mask=valid, other=0.0).to(tl.float32)
    query_partner = (query_partner * query_rstd[:, None]).to(query_ptr.dtype.element_ty).to(
        tl.float32
    )
    key_partner = (key_partner * key_rstd[:, None]).to(key_ptr.dtype.element_ty).to(
        tl.float32
    )

    rotary_mask = valid & (columns[None, :] < _ROTARY_DIM_TL)
    rotary_offsets = row[:, None] * rotary_row_stride + columns[None, :]
    cosine = tl.load(cosine_ptr + rotary_offsets, mask=rotary_mask, other=1.0).to(tl.float32)
    sine = tl.load(sine_ptr + rotary_offsets, mask=rotary_mask, other=0.0).to(tl.float32)
    sign = tl.where(columns[None, :] < half_rotary, -1.0, 1.0)
    query_rotated = query * cosine + sign * query_partner * sine
    key_rotated = key * cosine + sign * key_partner * sine
    tl.store(query_ptr + offsets, tl.where(rotary_mask, query_rotated, query), mask=valid)
    tl.store(key_ptr + offsets, tl.where(rotary_mask, key_rotated, key), mask=valid)


@triton.jit
def _value_first_swiglu_kernel(
    value_gate_ptr,
    output_ptr,
    elements,
    width: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < elements
    row = offsets // width
    column = offsets - row * width
    value = tl.load(value_gate_ptr + row * (2 * width) + column, mask=mask, other=0.0).to(
        tl.float32
    )
    gate = tl.load(
        value_gate_ptr + row * (2 * width) + width + column,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    tl.store(output_ptr + offsets, value * gate / (1.0 + tl.exp(-gate)), mask=mask)


def video_rmsnorm(hidden: torch.Tensor, weight: torch.Tensor, *, eps: float) -> torch.Tensor:
    if not hidden.is_cuda:
        normalized = hidden.float() * torch.rsqrt(hidden.float().pow(2).mean(-1, keepdim=True) + eps)
        return (normalized * weight.float()).to(hidden.dtype)
    if hidden.shape[-1] != _WIDTH:
        raise ValueError("H3 video RMSNorm requires width 2048")
    output_dtype = torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled("cuda") else hidden.dtype
    output = torch.empty_like(hidden, dtype=output_dtype)
    rows = hidden.numel() // _WIDTH
    _video_rmsnorm_kernel[(rows,)](
        hidden,
        weight,
        output,
        eps,
        BLOCK=_WIDTH,
        num_warps=8,
    )
    return output


def scaled_residual_rmsnorm_(
    hidden: torch.Tensor,
    update: torch.Tensor,
    scale: torch.Tensor,
    weight: torch.Tensor,
    *,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not hidden.is_cuda:
        hidden.add_(update.float() * scale.float())
        return hidden, video_rmsnorm(hidden, weight, eps=eps)
    output_dtype = torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled("cuda") else update.dtype
    output = torch.empty_like(hidden, dtype=output_dtype)
    rows = hidden.numel() // _WIDTH
    _scaled_residual_rmsnorm_kernel[(rows,)](
        hidden,
        update,
        scale,
        weight,
        output,
        eps,
        BLOCK=_WIDTH,
        num_warps=8,
    )
    return hidden, output


def scaled_residual_(
    hidden: torch.Tensor,
    update: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    if not hidden.is_cuda:
        hidden.add_(update.float() * scale.float())
        return hidden
    elements = hidden.numel()
    _scaled_residual_kernel[(triton.cdiv(elements, 1024),)](
        hidden,
        update,
        scale,
        elements,
        BLOCK=1024,
        num_warps=4,
    )
    return hidden


def scaled_residual_layernorm(
    hidden: torch.Tensor,
    update: torch.Tensor,
    scale: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    *,
    eps: float,
) -> torch.Tensor:
    if not hidden.is_cuda:
        residual = hidden + update.float() * scale.float()
        return F.layer_norm(residual, (_WIDTH,), weight, bias, eps)
    output_dtype = torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled("cuda") else update.dtype
    output = torch.empty_like(hidden, dtype=output_dtype)
    rows = hidden.numel() // _WIDTH
    _scaled_residual_layernorm_kernel[(rows,)](
        hidden,
        update,
        scale,
        weight,
        bias,
        output,
        eps,
        BLOCK=_WIDTH,
        num_warps=8,
    )
    return output


def qk_rmsnorm_partial_rope_(
    query: torch.Tensor,
    key: torch.Tensor,
    cosine: torch.Tensor,
    sine: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not query.is_cuda:
        dtype = query.dtype
        query = query.float() * torch.rsqrt(query.float().pow(2).mean(-1, keepdim=True) + 1e-5)
        key = key.float() * torch.rsqrt(key.float().pow(2).mean(-1, keepdim=True) + 1e-5)
        query = query.to(dtype)
        key = key.to(dtype)
        first, second = query[..., :_ROTARY_DIM].chunk(2, dim=-1)
        query_rotary = torch.cat((-second, first), dim=-1)
        first, second = key[..., :_ROTARY_DIM].chunk(2, dim=-1)
        key_rotary = torch.cat((-second, first), dim=-1)
        query[..., :_ROTARY_DIM] = query[..., :_ROTARY_DIM] * cosine + query_rotary * sine
        key[..., :_ROTARY_DIM] = key[..., :_ROTARY_DIM] * cosine + key_rotary * sine
        return query, key
    if query.shape != key.shape or query.shape[-1] != _HEAD_DIM:
        raise ValueError("H3 video Q/K fusion requires matching head dimension 64")
    rows = query.numel() // (int(query.shape[-2]) * _HEAD_DIM)
    heads = int(query.shape[-2])
    row_block = 8
    _qk_rmsnorm_partial_rope_kernel[(triton.cdiv(rows, row_block), heads)](
        query,
        key,
        cosine,
        sine,
        rows,
        heads,
        int(query.stride(-3)),
        int(query.stride(-2)),
        int(cosine.stride(-3)),
        ROW_BLOCK=row_block,
        num_warps=4,
    )
    return query, key


def value_first_swiglu(value_gate: torch.Tensor) -> torch.Tensor:
    value, gate = value_gate.chunk(2, dim=-1)
    if not value_gate.is_cuda:
        return value * F.silu(gate)
    width = int(value.shape[-1])
    output = torch.empty_like(value)
    elements = output.numel()
    _value_first_swiglu_kernel[(triton.cdiv(elements, 1024),)](
        value_gate,
        output,
        elements,
        width,
        BLOCK=1024,
        num_warps=4,
    )
    return output
