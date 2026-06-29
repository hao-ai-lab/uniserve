"""Audited normalization primitives shared by all model ports."""
from __future__ import annotations

import torch
import torch.nn as nn

from ..foundation.triton_compat import triton_device_supported, triton_fused_layers_enabled

__all__ = [
    'RMSNorm',
    'try_triton_qk_rms_norm',
    'can_run_triton_sensenova_qk_rms_norm_3d',
    'try_triton_sensenova_qk_rms_norm_3d',
]

try:  # pragma: no cover - availability depends on the serving environment.
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover
    triton = None
    tl = None


# Largest hidden width the fused norm kernels support; fixed by the kernel build.
_MAX_FUSED_NORM_HIDDEN = 8192

# Warp-count selection for the Triton norm kernel, fixed by the kernel build: the
# wider block (>= _NORM_WIDE_BLOCK) is launched with _NORM_WIDE_WARPS warps,
# otherwise _NORM_NARROW_WARPS.
_NORM_WIDE_BLOCK = 2048
_NORM_WIDE_WARPS = 8
_NORM_NARROW_WARPS = 4


if triton is not None:

    @triton.jit
    def _sensenova_square_pair(ptr, base, stride: tl.constexpr, start: tl.constexpr, active, offset: tl.constexpr):
        x0 = tl.load(ptr + base + (start + offset) * stride, mask=active, other=0.0).to(tl.float32)
        x1 = tl.load(ptr + base + (start + 32 + offset) * stride, mask=active, other=0.0).to(tl.float32)
        return ((x0 * x0).to(tl.float32) + (x1 * x1).to(tl.float32)).to(tl.float32)

    @triton.jit
    def _sensenova_add_rn(a, b):
        return (a + b).to(tl.float32)

    @triton.jit
    def _sensenova_mean_square_64(ptr, base, stride: tl.constexpr, start: tl.constexpr, active):
        p0 = _sensenova_square_pair(ptr, base, stride, start, active, 0)
        p1 = _sensenova_square_pair(ptr, base, stride, start, active, 1)
        p2 = _sensenova_square_pair(ptr, base, stride, start, active, 2)
        p3 = _sensenova_square_pair(ptr, base, stride, start, active, 3)
        p4 = _sensenova_square_pair(ptr, base, stride, start, active, 4)
        p5 = _sensenova_square_pair(ptr, base, stride, start, active, 5)
        p6 = _sensenova_square_pair(ptr, base, stride, start, active, 6)
        p7 = _sensenova_square_pair(ptr, base, stride, start, active, 7)
        p8 = _sensenova_square_pair(ptr, base, stride, start, active, 8)
        p9 = _sensenova_square_pair(ptr, base, stride, start, active, 9)
        p10 = _sensenova_square_pair(ptr, base, stride, start, active, 10)
        p11 = _sensenova_square_pair(ptr, base, stride, start, active, 11)
        p12 = _sensenova_square_pair(ptr, base, stride, start, active, 12)
        p13 = _sensenova_square_pair(ptr, base, stride, start, active, 13)
        p14 = _sensenova_square_pair(ptr, base, stride, start, active, 14)
        p15 = _sensenova_square_pair(ptr, base, stride, start, active, 15)
        p16 = _sensenova_square_pair(ptr, base, stride, start, active, 16)
        p17 = _sensenova_square_pair(ptr, base, stride, start, active, 17)
        p18 = _sensenova_square_pair(ptr, base, stride, start, active, 18)
        p19 = _sensenova_square_pair(ptr, base, stride, start, active, 19)
        p20 = _sensenova_square_pair(ptr, base, stride, start, active, 20)
        p21 = _sensenova_square_pair(ptr, base, stride, start, active, 21)
        p22 = _sensenova_square_pair(ptr, base, stride, start, active, 22)
        p23 = _sensenova_square_pair(ptr, base, stride, start, active, 23)
        p24 = _sensenova_square_pair(ptr, base, stride, start, active, 24)
        p25 = _sensenova_square_pair(ptr, base, stride, start, active, 25)
        p26 = _sensenova_square_pair(ptr, base, stride, start, active, 26)
        p27 = _sensenova_square_pair(ptr, base, stride, start, active, 27)
        p28 = _sensenova_square_pair(ptr, base, stride, start, active, 28)
        p29 = _sensenova_square_pair(ptr, base, stride, start, active, 29)
        p30 = _sensenova_square_pair(ptr, base, stride, start, active, 30)
        p31 = _sensenova_square_pair(ptr, base, stride, start, active, 31)
        s0 = _sensenova_add_rn(p0, p1)
        s1 = _sensenova_add_rn(p2, p3)
        s2 = _sensenova_add_rn(p4, p5)
        s3 = _sensenova_add_rn(p6, p7)
        s4 = _sensenova_add_rn(p8, p9)
        s5 = _sensenova_add_rn(p10, p11)
        s6 = _sensenova_add_rn(p12, p13)
        s7 = _sensenova_add_rn(p14, p15)
        s8 = _sensenova_add_rn(p16, p17)
        s9 = _sensenova_add_rn(p18, p19)
        s10 = _sensenova_add_rn(p20, p21)
        s11 = _sensenova_add_rn(p22, p23)
        s12 = _sensenova_add_rn(p24, p25)
        s13 = _sensenova_add_rn(p26, p27)
        s14 = _sensenova_add_rn(p28, p29)
        s15 = _sensenova_add_rn(p30, p31)
        t0 = _sensenova_add_rn(s0, s1)
        t1 = _sensenova_add_rn(s2, s3)
        t2 = _sensenova_add_rn(s4, s5)
        t3 = _sensenova_add_rn(s6, s7)
        t4 = _sensenova_add_rn(s8, s9)
        t5 = _sensenova_add_rn(s10, s11)
        t6 = _sensenova_add_rn(s12, s13)
        t7 = _sensenova_add_rn(s14, s15)
        u0 = _sensenova_add_rn(t0, t1)
        u1 = _sensenova_add_rn(t2, t3)
        u2 = _sensenova_add_rn(t4, t5)
        u3 = _sensenova_add_rn(t6, t7)
        v0 = _sensenova_add_rn(u0, u1)
        v1 = _sensenova_add_rn(u2, u3)
        return _sensenova_add_rn(v0, v1) / 64.0

    @triton.jit
    def _sensenova_mean_square(ptr, base, stride: tl.constexpr, start: tl.constexpr, active, n_cols: tl.constexpr, block: tl.constexpr):
        if n_cols == 64:
            return _sensenova_mean_square_64(ptr, base, stride, start, active)
        offs = tl.arange(0, block)
        mask = active & (offs < n_cols)
        x = tl.load(ptr + base + (start + offs) * stride, mask=mask, other=0.0).to(tl.float32)
        return tl.sum(x * x, axis=0) / n_cols

    @triton.jit
    def _rms_norm_kernel(x_ptr, w_ptr, y_ptr, n_cols: tl.constexpr, eps: tl.constexpr, block: tl.constexpr):
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

    @triton.jit
    def _sensenova_qk_rms_norm_3d_kernel(
        q_ptr,
        k_ptr,
        qw_t_ptr,
        qw_hw_ptr,
        kw_t_ptr,
        kw_hw_ptr,
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
        dim: tl.constexpr,
        t_dim: tl.constexpr,
        hw_dim: tl.constexpr,
        q_eps: tl.constexpr,
        k_eps: tl.constexpr,
        block: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs = tl.arange(0, block)
        t_col = offs < t_dim
        hw_col = offs < hw_dim

        q_active = pid < q_rows
        q_token = pid // q_heads
        q_head = pid - q_token * q_heads
        q_base = q_token * q_stride_0 + q_head * q_stride_1
        q_t_var = _sensenova_mean_square(q_ptr, q_base, q_stride_2, 0, q_active, t_dim, block)
        q_t_inv = tl.rsqrt(q_t_var + q_eps)
        q_t = tl.load(q_ptr + q_base + offs * q_stride_2, mask=q_active & t_col, other=0.0).to(tl.float32)
        q_t_norm = (q_t * q_t_inv).to(q_out_ptr.dtype.element_ty).to(tl.float32)
        q_t_weight = tl.load(qw_t_ptr + offs, mask=t_col, other=0.0).to(tl.float32)
        q_t_out = (q_t_norm * q_t_weight).to(q_out_ptr.dtype.element_ty)
        tl.store(q_out_ptr + pid * dim + offs, q_t_out, mask=q_active & t_col)

        q_hw_var = _sensenova_mean_square(q_ptr, q_base, q_stride_2, t_dim, q_active, t_dim, block)
        q_hw_inv = tl.rsqrt(q_hw_var + q_eps)
        q_h = tl.load(q_ptr + q_base + (t_dim + offs) * q_stride_2, mask=q_active & hw_col, other=0.0).to(tl.float32)
        q_w = tl.load(q_ptr + q_base + (t_dim + hw_dim + offs) * q_stride_2, mask=q_active & hw_col, other=0.0).to(tl.float32)
        q_h_norm = (q_h * q_hw_inv).to(q_out_ptr.dtype.element_ty).to(tl.float32)
        q_w_norm = (q_w * q_hw_inv).to(q_out_ptr.dtype.element_ty).to(tl.float32)
        q_h_weight = tl.load(qw_hw_ptr + offs, mask=hw_col, other=0.0).to(tl.float32)
        q_w_weight = tl.load(qw_hw_ptr + hw_dim + offs, mask=hw_col, other=0.0).to(tl.float32)
        q_h_out = (q_h_norm * q_h_weight).to(q_out_ptr.dtype.element_ty)
        q_w_out = (q_w_norm * q_w_weight).to(q_out_ptr.dtype.element_ty)
        tl.store(q_out_ptr + pid * dim + t_dim + offs, q_h_out, mask=q_active & hw_col)
        tl.store(q_out_ptr + pid * dim + t_dim + hw_dim + offs, q_w_out, mask=q_active & hw_col)

        k_pid = pid - q_rows
        k_active = (k_pid >= 0) & (k_pid < k_rows)
        k_token = k_pid // k_heads
        k_head = k_pid - k_token * k_heads
        k_base = k_token * k_stride_0 + k_head * k_stride_1
        k_t_var = _sensenova_mean_square(k_ptr, k_base, k_stride_2, 0, k_active, t_dim, block)
        k_t_inv = tl.rsqrt(k_t_var + k_eps)
        k_t = tl.load(k_ptr + k_base + offs * k_stride_2, mask=k_active & t_col, other=0.0).to(tl.float32)
        k_t_norm = (k_t * k_t_inv).to(k_out_ptr.dtype.element_ty).to(tl.float32)
        k_t_weight = tl.load(kw_t_ptr + offs, mask=t_col, other=0.0).to(tl.float32)
        k_t_out = (k_t_norm * k_t_weight).to(k_out_ptr.dtype.element_ty)
        tl.store(k_out_ptr + k_pid * dim + offs, k_t_out, mask=k_active & t_col)

        k_hw_var = _sensenova_mean_square(k_ptr, k_base, k_stride_2, t_dim, k_active, t_dim, block)
        k_hw_inv = tl.rsqrt(k_hw_var + k_eps)
        k_h = tl.load(k_ptr + k_base + (t_dim + offs) * k_stride_2, mask=k_active & hw_col, other=0.0).to(tl.float32)
        k_w = tl.load(k_ptr + k_base + (t_dim + hw_dim + offs) * k_stride_2, mask=k_active & hw_col, other=0.0).to(tl.float32)
        k_h_norm = (k_h * k_hw_inv).to(k_out_ptr.dtype.element_ty).to(tl.float32)
        k_w_norm = (k_w * k_hw_inv).to(k_out_ptr.dtype.element_ty).to(tl.float32)
        k_h_weight = tl.load(kw_hw_ptr + offs, mask=hw_col, other=0.0).to(tl.float32)
        k_w_weight = tl.load(kw_hw_ptr + hw_dim + offs, mask=hw_col, other=0.0).to(tl.float32)
        k_h_out = (k_h_norm * k_h_weight).to(k_out_ptr.dtype.element_ty)
        k_w_out = (k_w_norm * k_w_weight).to(k_out_ptr.dtype.element_ty)
        tl.store(k_out_ptr + k_pid * dim + t_dim + offs, k_h_out, mask=k_active & hw_col)
        tl.store(k_out_ptr + k_pid * dim + t_dim + hw_dim + offs, k_w_out, mask=k_active & hw_col)


class RMSNorm(nn.Module):
    """RMSNorm with fp32 variance accumulation.

    Matches the local RMSNorm variants used by current model ports. Uses fp32
    for the variance reduction because low-precision norm drift is visible in
    long interleaved generations.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    @property
    def eps(self) -> float:
        return self.variance_epsilon

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return _rms_norm(hidden_states, self.weight, self.variance_epsilon)

    def forward_with_residual(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        *,
        in_place: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``RMSNorm(hidden_states + residual)`` and the summed residual."""

        return _rms_norm(
            hidden_states,
            self.weight,
            self.variance_epsilon,
            residual=residual,
            in_place=in_place,
        )

    def extra_repr(self) -> str:
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


def _norm_inputs_eligible(hidden_states: torch.Tensor, weight: torch.Tensor) -> bool:
    """Preconditions shared by the fused norm backends (sgl and triton)."""

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
    return (
        triton is not None
        and triton_fused_layers_enabled()
        and triton_device_supported(device)
    )


def _sgl_rms_norm_input(hidden_states: torch.Tensor) -> torch.Tensor | None:
    """Coerce ``hidden_states`` for the sgl rmsnorm kernel, or ``None`` if it
    cannot be made contiguous in place (non-unit last stride)."""

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



class _TritonRmsNorm:
    def is_eligible(self, hidden_states: torch.Tensor, weight: torch.Tensor, eps: float) -> bool:
        return _norm_inputs_eligible(hidden_states, weight) and _triton_norm_available(
            hidden_states.device
        )

    def run(self, hidden_states: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
        hidden_size = int(weight.numel())
        rows = hidden_states.numel() // hidden_size
        out = torch.empty_like(hidden_states)
        block, num_warps = _norm_launch_config(hidden_size)
        _rms_norm_kernel[(rows,)](
            hidden_states,
            weight,
            out,
            hidden_size,
            float(eps),
            block,
            num_warps=num_warps,
        )
        return out


class _EagerRmsNorm:
    def is_eligible(self, hidden_states: torch.Tensor, weight: torch.Tensor, eps: float) -> bool:
        return True

    def run(self, hidden_states: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
        in_dtype = hidden_states.dtype
        x = hidden_states.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + eps)
        return weight * x.to(in_dtype)


_AddRmsResult = tuple[torch.Tensor, torch.Tensor]



class _TritonAddRmsNorm:
    def is_eligible(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
    ) -> bool:
        return not (
            hidden_states.shape != residual.shape
            or hidden_states.dtype != residual.dtype
            or not residual.is_contiguous()
            or not _norm_inputs_eligible(hidden_states, weight)
            or not _triton_norm_available(hidden_states.device)
        )

    def run(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
    ) -> _AddRmsResult:
        hidden_size = int(weight.numel())
        rows = hidden_states.numel() // hidden_size
        normed = torch.empty_like(hidden_states)
        combined = torch.empty_like(hidden_states)
        block, num_warps = _norm_launch_config(hidden_size)
        _add_rms_norm_kernel[(rows,)](
            hidden_states,
            residual,
            weight,
            normed,
            combined,
            hidden_size,
            float(eps),
            block,
            num_warps=num_warps,
        )
        return normed, combined


class _EagerAddRmsNorm:
    def is_eligible(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
    ) -> bool:
        return True

    def run(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
    ) -> _AddRmsResult:
        combined = hidden_states + residual
        return _rms_norm(combined, weight, eps), combined


def _rms_norm(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    *,
    residual: torch.Tensor | None = None,
    in_place: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Dispatch RMSNorm to a fused backend when eligible, else the eager path.

    Backend priority is decided once per router: sgl preferred over triton over
    eager. With a residual the sgl fused-add path is only a candidate when
    ``in_place`` and the triton fused-add path is a candidate in either case;
    when neither applies the eager fallback sums and norms through the
    non-residual dispatch. The eager kernel is the terminal fallback and always
    works without optional kernels installed.
    """

    from uniserve_worker import ops

    if residual is None:
        return ops.rms_norm(hidden_states, weight, eps)
    return ops.add_rms_norm(hidden_states, residual, weight, eps, in_place=in_place)


def try_triton_qk_rms_norm(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    q_eps: float,
    k_eps: float,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    if (
        triton is None
        or not triton_fused_layers_enabled()
        or not q.is_cuda
        or not k.is_cuda
        or not q_weight.is_cuda
        or not k_weight.is_cuda
        or not triton_device_supported(q.device)
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
        return None
    head_dim = int(q.shape[-1])
    if head_dim <= 0 or head_dim > 1024:
        return None
    q_heads = int(q.shape[1])
    k_heads = int(k.shape[1])
    q_tokens = int(q.shape[0])
    k_tokens = int(k.shape[0])
    if q_tokens <= 0 or k_tokens <= 0 or q_heads <= 0 or k_heads <= 0:
        return None
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


def try_triton_sensenova_qk_rms_norm_3d(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight_t: torch.Tensor,
    q_weight_hw: torch.Tensor,
    k_weight_t: torch.Tensor,
    k_weight_hw: torch.Tensor,
    q_eps: float,
    k_eps: float,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    if not _sensenova_qk_rms_norm_3d_is_eligible(
        q,
        k,
        q_weight_t,
        q_weight_hw,
        k_weight_t,
        k_weight_hw,
    ):
        return None
    q_tokens = int(q.shape[0])
    k_tokens = int(k.shape[0])
    q_heads = int(q.shape[1])
    k_heads = int(k.shape[1])
    dim = int(q.shape[-1])
    t_dim = dim // 2
    hw_dim = dim // 4
    q_out = torch.empty_like(q, memory_format=torch.contiguous_format)
    k_out = torch.empty_like(k, memory_format=torch.contiguous_format)
    q_rows = q_tokens * q_heads
    k_rows = k_tokens * k_heads
    _sensenova_qk_rms_norm_3d_kernel[(q_rows + k_rows,)](
        q,
        k,
        q_weight_t,
        q_weight_hw,
        k_weight_t,
        k_weight_hw,
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
        dim,
        t_dim,
        hw_dim,
        float(q_eps),
        float(k_eps),
        triton.next_power_of_2(t_dim),
        num_warps=4,
    )
    return q_out, k_out


def can_run_triton_sensenova_qk_rms_norm_3d(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight_t: torch.Tensor,
    q_weight_hw: torch.Tensor,
    k_weight_t: torch.Tensor,
    k_weight_hw: torch.Tensor,
) -> bool:
    return _sensenova_qk_rms_norm_3d_is_eligible(
        q,
        k,
        q_weight_t,
        q_weight_hw,
        k_weight_t,
        k_weight_hw,
    )


def _sensenova_qk_rms_norm_3d_is_eligible(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight_t: torch.Tensor,
    q_weight_hw: torch.Tensor,
    k_weight_t: torch.Tensor,
    k_weight_hw: torch.Tensor,
) -> bool:
    tensors = (q, k, q_weight_t, q_weight_hw, k_weight_t, k_weight_hw)
    if triton is None or not triton_fused_layers_enabled() or torch.is_grad_enabled():
        return False
    if not all(t.is_cuda and t.device == q.device for t in tensors):
        return False
    if not triton_device_supported(q.device):
        return False
    if q.dtype != k.dtype or q.ndim != 3 or k.ndim != 3:
        return False
    if int(q.stride(-1)) != 1 or int(k.stride(-1)) != 1:
        return False
    if not all(t.is_contiguous() for t in tensors[2:]):
        return False
    if int(q.shape[0]) <= 0 or int(k.shape[0]) <= 0 or int(q.shape[1]) <= 0 or int(k.shape[1]) <= 0:
        return False
    if int(q.shape[0]) != int(k.shape[0]) or int(q.shape[-1]) != int(k.shape[-1]):
        return False
    dim = int(q.shape[-1])
    if dim <= 0 or dim % 8 != 0 or dim > 1024:
        return False
    t_dim = dim // 2
    if int(q_weight_t.numel()) != t_dim or int(k_weight_t.numel()) != t_dim:
        return False
    if int(q_weight_hw.numel()) != t_dim or int(k_weight_hw.numel()) != t_dim:
        return False
    return True
