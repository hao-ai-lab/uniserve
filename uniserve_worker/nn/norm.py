"""Audited normalization primitives shared by all model ports."""

from __future__ import annotations

from typing import overload

import torch
import torch.nn as nn

from ..foundation.triton_compat import triton_device_supported, triton_fused_layers_enabled

__all__ = [
    "RMSNorm",
    "try_triton_qk_rms_norm",
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


class RMSNorm(nn.Module):
    """RMSNorm with fp32 variance accumulation.

    Matches the local RMSNorm variants used by current model ports. Uses fp32
    for the variance reduction because low-precision norm drift is visible in
    long composed generations.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self._select_forward_kernel()

    def _select_forward_kernel(self) -> None:
        from uniserve_worker.ops.providers import select_rms_norm_kernel

        self._forward_kernel = select_rms_norm_kernel(
            self.weight,
            self.variance_epsilon,
        )

    def _apply(self, fn, recurse: bool = True):
        module = super()._apply(fn, recurse=recurse)
        self._select_forward_kernel()
        return module

    @property
    def eps(self) -> float:
        return self.variance_epsilon

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self._forward_kernel(hidden_states, self.weight, self.variance_epsilon)

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
    return triton is not None and triton_fused_layers_enabled() and triton_device_supported(device)


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


@overload
def _rms_norm(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    *,
    residual: None = None,
    in_place: bool = False,
) -> torch.Tensor: ...


@overload
def _rms_norm(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    *,
    residual: torch.Tensor,
    in_place: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]: ...


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
    if not can_run_triton_qk_rms_norm(q, k, q_weight, k_weight, q_eps, k_eps):
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


def can_run_triton_qk_rms_norm(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    q_eps: float,
    k_eps: float,
) -> bool:
    del q_eps, k_eps
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
        return False
    head_dim = int(q.shape[-1])
    if head_dim <= 0 or head_dim > 1024:
        return False
    q_heads = int(q.shape[1])
    k_heads = int(k.shape[1])
    q_tokens = int(q.shape[0])
    k_tokens = int(k.shape[0])
    if q_tokens <= 0 or k_tokens <= 0 or q_heads <= 0 or k_heads <= 0:
        return False
    return True
