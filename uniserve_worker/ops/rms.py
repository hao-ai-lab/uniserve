"""RMSNorm, fused-add RMSNorm, and QK RMSNorm providers."""
from __future__ import annotations

from functools import lru_cache

import torch

from ..backends.triton import triton_available
from .core import Dispatcher, Operator
from .qk_plan import QKNormRopePlan
from .requests import (
    AddRmsNormReq,
    MultiAxisQKNormReq,
    QKNormReq,
    QKNormRequest,
    RmsNormReq,
)

try:  # pragma: no cover - availability depends on the serving environment.
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover
    triton = None
    tl = None

_MAX_FUSED_NORM_HIDDEN = 8192
_NORM_WIDE_BLOCK = 2048
_NORM_WIDE_WARPS = 8
_NORM_NARROW_WARPS = 4
_SGL_ALIGNMENT_BYTES = 16


@lru_cache(maxsize=1)
def _sgl_rmsnorm_kernel():
    try:  # pragma: no cover - optional SGLang kernel package.
        from sgl_kernel import rmsnorm
    except Exception:
        return None
    return rmsnorm


@lru_cache(maxsize=1)
def _sgl_fused_add_rmsnorm_kernel():
    try:  # pragma: no cover - optional SGLang kernel package.
        from sgl_kernel import fused_add_rmsnorm
    except Exception:
        return None
    return fused_add_rmsnorm


if triton is not None:

    @triton.jit
    def _rms_norm_kernel(
        x_ptr, w_ptr, y_ptr, n_cols: tl.constexpr, eps: tl.constexpr, block: tl.constexpr
    ):
        row = tl.program_id(0)
        offs = tl.arange(0, block)
        mask = offs < n_cols
        x = tl.load(x_ptr + row * n_cols + offs, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        var = tl.sum(x * x, axis=0) / n_cols
        y = x * tl.rsqrt(var + eps) * w
        tl.store(y_ptr + row * n_cols + offs, y, mask=mask)

    @triton.jit
    def _add_rms_norm_kernel(
        x_ptr,
        r_ptr,
        w_ptr,
        y_ptr,
        c_ptr,
        n_cols: tl.constexpr,
        eps: tl.constexpr,
        block: tl.constexpr,
    ):
        row = tl.program_id(0)
        offs = tl.arange(0, block)
        mask = offs < n_cols
        x = tl.load(x_ptr + row * n_cols + offs, mask=mask, other=0.0).to(tl.float32)
        r = tl.load(r_ptr + row * n_cols + offs, mask=mask, other=0.0).to(tl.float32)
        c = x + r
        w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        var = tl.sum(c * c, axis=0) / n_cols
        y = c * tl.rsqrt(var + eps) * w
        tl.store(c_ptr + row * n_cols + offs, c, mask=mask)
        tl.store(y_ptr + row * n_cols + offs, y, mask=mask)

    @triton.jit
    def _qk_rms_norm_kernel(
        q_ptr,
        k_ptr,
        qw_ptr,
        kw_ptr,
        q_out_ptr,
        k_out_ptr,
        q_rows: tl.constexpr,
        k_rows: tl.constexpr,
        q_heads: tl.constexpr,
        k_heads: tl.constexpr,
        q_stride_0: tl.constexpr,
        q_stride_1: tl.constexpr,
        q_stride_2: tl.constexpr,
        k_stride_0: tl.constexpr,
        k_stride_1: tl.constexpr,
        k_stride_2: tl.constexpr,
        n_cols: tl.constexpr,
        q_eps: tl.constexpr,
        k_eps: tl.constexpr,
        block: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs = tl.arange(0, block)
        col_mask = offs < n_cols

        q_mask = (pid < q_rows) & col_mask
        q_token = pid // q_heads
        q_head = pid - q_token * q_heads
        q = tl.load(
            q_ptr + q_token * q_stride_0 + q_head * q_stride_1 + offs * q_stride_2,
            mask=q_mask,
            other=0.0,
        ).to(tl.float32)
        qw = tl.load(qw_ptr + offs, mask=col_mask, other=0.0).to(tl.float32)
        q_var = tl.sum(q * q, axis=0) / n_cols
        q_out = q * tl.rsqrt(q_var + q_eps) * qw
        tl.store(q_out_ptr + pid * n_cols + offs, q_out, mask=q_mask)

        k_pid = pid - q_rows
        k_mask = (k_pid >= 0) & (k_pid < k_rows) & col_mask
        k_token = k_pid // k_heads
        k_head = k_pid - k_token * k_heads
        k = tl.load(
            k_ptr + k_token * k_stride_0 + k_head * k_stride_1 + offs * k_stride_2,
            mask=k_mask,
            other=0.0,
        ).to(tl.float32)
        kw = tl.load(kw_ptr + offs, mask=col_mask, other=0.0).to(tl.float32)
        k_var = tl.sum(k * k, axis=0) / n_cols
        k_out = k * tl.rsqrt(k_var + k_eps) * kw
        tl.store(k_out_ptr + k_pid * n_cols + offs, k_out, mask=k_mask)


def _norm_inputs_eligible(hidden_states: torch.Tensor, weight: torch.Tensor) -> bool:
    return not (
        not hidden_states.is_cuda
        or not weight.is_cuda
        or torch.is_grad_enabled()
        or not hidden_states.is_contiguous()
        or not weight.is_contiguous()
        or hidden_states.shape[-1] != weight.numel()
        or not 0 < int(weight.numel()) <= _MAX_FUSED_NORM_HIDDEN
    )


def _triton_norm_available(device: torch.device) -> bool:
    return triton is not None and triton_available(device)


def _sgl_rms_norm_input(hidden_states: torch.Tensor) -> torch.Tensor | None:
    if not hidden_states.is_contiguous():
        if int(hidden_states.stride(-1)) != 1:
            return None
        return hidden_states.contiguous()
    return hidden_states


def _norm_launch_config(hidden_size: int) -> tuple[int, int]:
    block = triton.next_power_of_2(hidden_size)
    num_warps = _NORM_WIDE_WARPS if block >= _NORM_WIDE_BLOCK else _NORM_NARROW_WARPS
    return block, num_warps


def _reshape_norm_rows(tensor: torch.Tensor, hidden_size: int) -> torch.Tensor:
    return tensor.reshape(tensor.numel() // int(hidden_size), int(hidden_size))


def eager_rms_norm(hidden_states: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    in_dtype = hidden_states.dtype
    x = hidden_states.to(torch.float32)
    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    return weight * x.to(in_dtype)


def qk_rms_norm_eligible(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
) -> bool:
    if (
        triton is None
        or not q.is_cuda
        or not k.is_cuda
        or not q_weight.is_cuda
        or not k_weight.is_cuda
        or not triton_available(q.device)
        or torch.is_grad_enabled()
        or q.device != k.device
        or q.dtype != k.dtype
        or q.ndim != 3
        or k.ndim != 3
        or q.shape[-1] != k.shape[-1]
        or q.shape[-1] != q_weight.numel()
        or k.shape[-1] != k_weight.numel()
        or not q_weight.is_contiguous()
        or not k_weight.is_contiguous()
        or int(q.stride(-1)) != 1
        or int(k.stride(-1)) != 1
    ):
        return False
    head_dim = int(q.shape[-1])
    if head_dim <= 0 or head_dim > 1024:
        return False
    return (
        int(q.shape[0]) > 0
        and int(k.shape[0]) > 0
        and int(q.shape[1]) > 0
        and int(k.shape[1]) > 0
    )


def run_qk_rms_norm(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    q_eps: float,
    k_eps: float,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    if not qk_rms_norm_eligible(q, k, q_weight, k_weight):
        return None
    head_dim = int(q.shape[-1])
    q_heads = int(q.shape[1])
    k_heads = int(k.shape[1])
    q_tokens = int(q.shape[0])
    k_tokens = int(k.shape[0])
    q_out = torch.empty_like(q, memory_format=torch.contiguous_format)
    k_out = torch.empty_like(k, memory_format=torch.contiguous_format)
    block, num_warps = _norm_launch_config(head_dim)
    q_rows = q_tokens * q_heads
    k_rows = k_tokens * k_heads
    _qk_rms_norm_kernel[(q_rows + k_rows,)](
        q,
        k,
        q_weight,
        k_weight,
        q_out,
        k_out,
        q_rows,
        k_rows,
        q_heads,
        k_heads,
        int(q.stride(0)),
        int(q.stride(1)),
        int(q.stride(2)),
        int(k.stride(0)),
        int(k.stride(1)),
        int(k.stride(2)),
        head_dim,
        float(q_eps),
        float(k_eps),
        block,
        num_warps=num_warps,
    )
    return q_out, k_out


class SglRmsNorm(Operator):
    def __init__(self) -> None:
        super().__init__("sgl_kernel", "rms_norm")

    def can_run(self, req: RmsNormReq) -> bool:
        if _sgl_rmsnorm_kernel() is None:
            return False
        coerced = _sgl_rms_norm_input(req.hidden_states)
        if coerced is None:
            return False
        if coerced.dtype not in {torch.float16, torch.bfloat16}:
            return False
        if req.weight.dtype != coerced.dtype:
            return False
        return _norm_inputs_eligible(coerced, req.weight)

    def run(self, req: RmsNormReq) -> torch.Tensor:
        original_shape = tuple(req.hidden_states.shape)
        coerced = _sgl_rms_norm_input(req.hidden_states)
        if coerced is None:
            raise RuntimeError("sgl rms_norm became ineligible")
        hidden_size = int(req.weight.numel())
        rows = _reshape_norm_rows(coerced, hidden_size)
        out = torch.empty_like(rows)
        kernel = _sgl_rmsnorm_kernel()
        if kernel is None:
            raise RuntimeError("sgl rmsnorm kernel unavailable")
        kernel(rows, req.weight, float(req.eps), out=out)
        return out.reshape(original_shape)


class TritonRmsNorm(Operator):
    def __init__(self) -> None:
        super().__init__("triton", "rms_norm")

    def can_run(self, req: RmsNormReq) -> bool:
        return _norm_inputs_eligible(req.hidden_states, req.weight) and _triton_norm_available(
            req.hidden_states.device
        )

    def run(self, req: RmsNormReq) -> torch.Tensor:
        hidden_size = int(req.weight.numel())
        rows = req.hidden_states.numel() // hidden_size
        out = torch.empty_like(req.hidden_states)
        block, num_warps = _norm_launch_config(hidden_size)
        _rms_norm_kernel[(rows,)](
            req.hidden_states,
            req.weight,
            out,
            hidden_size,
            float(req.eps),
            block,
            num_warps=num_warps,
        )
        return out


class EagerRmsNorm(Operator):
    def __init__(self) -> None:
        super().__init__("eager", "rms_norm")

    def can_run(self, req: RmsNormReq) -> bool:
        del req
        return True

    def run(self, req: RmsNormReq) -> torch.Tensor:
        return eager_rms_norm(req.hidden_states, req.weight, req.eps)


class SglAddRmsNorm(Operator):
    def __init__(self) -> None:
        super().__init__("sgl_kernel", "add_rms_norm")

    def can_run(self, req: AddRmsNormReq) -> bool:
        if not req.in_place or _sgl_fused_add_rmsnorm_kernel() is None:
            return False
        return not (
            req.hidden_states.shape != req.residual.shape
            or req.hidden_states.dtype != req.residual.dtype
            or not req.residual.is_contiguous()
            or not _norm_inputs_eligible(req.hidden_states, req.weight)
        )

    def run(self, req: AddRmsNormReq) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_size = int(req.weight.numel())
        kernel = _sgl_fused_add_rmsnorm_kernel()
        if kernel is None:
            raise RuntimeError("sgl fused_add_rmsnorm kernel unavailable")
        kernel(
            _reshape_norm_rows(req.hidden_states, hidden_size),
            _reshape_norm_rows(req.residual, hidden_size),
            req.weight,
            float(req.eps),
        )
        return req.hidden_states, req.residual


class TritonAddRmsNorm(Operator):
    def __init__(self) -> None:
        super().__init__("triton", "add_rms_norm")

    def can_run(self, req: AddRmsNormReq) -> bool:
        return not (
            req.hidden_states.shape != req.residual.shape
            or req.hidden_states.dtype != req.residual.dtype
            or not req.residual.is_contiguous()
            or not _norm_inputs_eligible(req.hidden_states, req.weight)
            or not _triton_norm_available(req.hidden_states.device)
        )

    def run(self, req: AddRmsNormReq) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_size = int(req.weight.numel())
        rows = req.hidden_states.numel() // hidden_size
        normed = torch.empty_like(req.hidden_states)
        combined = torch.empty_like(req.hidden_states)
        block, num_warps = _norm_launch_config(hidden_size)
        _add_rms_norm_kernel[(rows,)](
            req.hidden_states,
            req.residual,
            req.weight,
            normed,
            combined,
            hidden_size,
            float(req.eps),
            block,
            num_warps=num_warps,
        )
        return normed, combined


class EagerAddRmsNorm(Operator):
    def __init__(self) -> None:
        super().__init__("eager", "add_rms_norm")

    def can_run(self, req: AddRmsNormReq) -> bool:
        del req
        return True

    def run(self, req: AddRmsNormReq) -> tuple[torch.Tensor, torch.Tensor]:
        combined = req.hidden_states + req.residual
        return eager_rms_norm(combined, req.weight, req.eps), combined


class TritonQKNorm(Operator):
    def __init__(self) -> None:
        super().__init__("triton", "qk_norm")

    def can_run(self, req: QKNormRequest) -> bool:
        if isinstance(req, MultiAxisQKNormReq):
            return False
        return qk_rms_norm_eligible(req.q, req.k, req.q_weight, req.k_weight)

    def run(self, req: QKNormRequest) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(req, MultiAxisQKNormReq):
            raise RuntimeError("triton qk_norm only handles single-axis requests")
        out = run_qk_rms_norm(req.q, req.k, req.q_weight, req.k_weight, req.eps, req.eps)
        if out is None:
            raise RuntimeError("triton qk_norm became ineligible")
        return out


class EagerQKNorm(Operator):
    def __init__(self) -> None:
        super().__init__("eager", "qk_norm")

    def can_run(self, req: QKNormRequest) -> bool:
        del req
        return True

    def run(self, req: QKNormRequest) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(req, MultiAxisQKNormReq):
            return self._run_multi_axis(req)
        return (
            eager_rms_norm(req.q, req.q_weight, req.eps),
            eager_rms_norm(req.k, req.k_weight, req.eps),
        )

    def _run_multi_axis(self, req: MultiAxisQKNormReq) -> tuple[torch.Tensor, torch.Tensor]:
        groups = QKNormRopePlan.from_norm_request(req)
        q_parts = req.q.split(req.axis_dims, dim=-1)
        k_parts = req.k.split(req.axis_dims, dim=-1)
        out_q = []
        out_k = []
        for group in groups:
            q_group = torch.cat(q_parts[group.start : group.end], dim=-1)
            k_group = torch.cat(k_parts[group.start : group.end], dim=-1)
            q_normed = eager_rms_norm(q_group, req.q_weights[group.start], req.eps)
            k_normed = eager_rms_norm(k_group, req.k_weights[group.start], req.eps)
            out_q.extend(q_normed.split(req.axis_dims[group.start : group.end], dim=-1))
            out_k.extend(k_normed.split(req.axis_dims[group.start : group.end], dim=-1))
        return torch.cat(out_q, dim=-1), torch.cat(out_k, dim=-1)


@lru_cache(maxsize=1)
def rms_norm_dispatcher() -> Dispatcher[RmsNormReq, torch.Tensor]:
    return Dispatcher(
        "rms_norm",
        [SglRmsNorm(), TritonRmsNorm(), EagerRmsNorm()],
        env_override="UNISERVE_RMS_NORM_PROVIDER",
    )


@lru_cache(maxsize=1)
def add_rms_norm_dispatcher() -> Dispatcher[AddRmsNormReq, tuple[torch.Tensor, torch.Tensor]]:
    return Dispatcher(
        "add_rms_norm",
        [SglAddRmsNorm(), TritonAddRmsNorm(), EagerAddRmsNorm()],
        env_override="UNISERVE_ADD_RMS_NORM_PROVIDER",
    )


@lru_cache(maxsize=1)
def qk_norm_dispatcher() -> Dispatcher[QKNormRequest, tuple[torch.Tensor, torch.Tensor]]:
    return Dispatcher(
        "qk_norm",
        [TritonQKNorm(), EagerQKNorm()],
        env_override="UNISERVE_QK_NORM_PROVIDER",
    )
