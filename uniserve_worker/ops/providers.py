"""Provider packs backed by existing in-tree kernels."""

from __future__ import annotations

from functools import lru_cache
from importlib import import_module

import torch

from ..backends.triton import triton_available
from ..execution.forward_batch import PagedDecodePlan
from .core import Dispatcher
from .requests import (
    AddRmsNormReq,
    AttentionRegime,
    AttentionReq,
    PackedRopeReq,
    QKNormReq,
    QKNormRopeReq,
    RmsNormReq,
    SiluAndMulReq,
)

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


@lru_cache(maxsize=1)
def _sgl_silu_and_mul_kernel():
    try:  # pragma: no cover - optional SGLang kernel package.
        from sgl_kernel import silu_and_mul
    except Exception:
        return None
    return silu_and_mul


class _ProviderBase:
    def __init__(self, name: str, operator: str) -> None:
        self.name = name
        self.operator = operator

class _RmsNormKernelProvider(_ProviderBase):
    def __init__(self, name: str, kernel) -> None:
        super().__init__(name, "rms_norm")
        self._kernel = kernel

    def can_run(self, req: RmsNormReq) -> bool:
        return self._kernel.is_eligible(req.hidden_states, req.weight, req.eps)

    def run(self, req: RmsNormReq):
        return self._kernel.run(req.hidden_states, req.weight, req.eps)


class _AddRmsNormKernelProvider(_ProviderBase):
    def __init__(self, name: str, kernel) -> None:
        super().__init__(name, "add_rms_norm")
        self._kernel = kernel

    def can_run(self, req: AddRmsNormReq) -> bool:
        if self.name == "sgl_kernel" and not req.in_place:
            return False
        return self._kernel.is_eligible(req.hidden_states, req.residual, req.weight, req.eps)

    def run(self, req: AddRmsNormReq):
        return self._kernel.run(req.hidden_states, req.residual, req.weight, req.eps)


class _SiluAndMulKernelProvider(_ProviderBase):
    def __init__(self, name: str, kernel) -> None:
        super().__init__(name, "silu_and_mul")
        self._kernel = kernel

    def can_run(self, req: SiluAndMulReq) -> bool:
        return self._kernel.is_eligible(req.x, None)

    def run(self, req: SiluAndMulReq):
        return self._kernel.run(req.x, None)


class _PackedRopeKernelProvider(_ProviderBase):
    def __init__(self, name: str, kernel) -> None:
        super().__init__(name, "rope")
        self._kernel = kernel

    def can_run(self, req: PackedRopeReq) -> bool:
        return self._kernel.is_eligible(req.x, req.cos, req.sin)

    def run(self, req: PackedRopeReq):
        return self._kernel.run(req.x, req.cos, req.sin)


class _SglRmsNormKernel:
    def is_eligible(self, hidden_states: torch.Tensor, weight: torch.Tensor, eps: float) -> bool:
        if _sgl_rmsnorm_kernel() is None:
            return False
        n = import_module("uniserve_worker.nn.norm")
        coerced = n._sgl_rms_norm_input(hidden_states)
        if coerced is None:
            return False
        if coerced.dtype not in {torch.float16, torch.bfloat16}:
            return False
        if weight.dtype != coerced.dtype:
            return False
        return n._norm_inputs_eligible(coerced, weight)

    def run(self, hidden_states: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
        n = import_module("uniserve_worker.nn.norm")
        original_shape = tuple(hidden_states.shape)
        coerced = n._sgl_rms_norm_input(hidden_states)
        if coerced is None:
            raise RuntimeError("sgl rms_norm became ineligible")
        hidden_size = int(weight.numel())
        rows = n._reshape_norm_rows(coerced, hidden_size)
        out = torch.empty_like(rows)
        kernel = _sgl_rmsnorm_kernel()
        if kernel is None:
            raise RuntimeError("sgl rmsnorm kernel unavailable")
        kernel(rows, weight, float(eps), out=out)
        return out.reshape(original_shape)


class _SglAddRmsNormInPlaceKernel:
    def is_eligible(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
    ) -> bool:
        if _sgl_fused_add_rmsnorm_kernel() is None:
            return False
        n = import_module("uniserve_worker.nn.norm")
        return not (
            hidden_states.shape != residual.shape
            or hidden_states.dtype != residual.dtype
            or not residual.is_contiguous()
            or not n._norm_inputs_eligible(hidden_states, weight)
        )

    def run(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
    ):
        n = import_module("uniserve_worker.nn.norm")
        hidden_size = int(weight.numel())
        kernel = _sgl_fused_add_rmsnorm_kernel()
        if kernel is None:
            raise RuntimeError("sgl fused_add_rmsnorm kernel unavailable")
        kernel(
            n._reshape_norm_rows(hidden_states, hidden_size),
            n._reshape_norm_rows(residual, hidden_size),
            weight,
            float(eps),
        )
        return hidden_states, residual


class _SglSiluAndMulKernel:
    def is_eligible(self, x: torch.Tensor, y: torch.Tensor | None) -> bool:
        if _sgl_silu_and_mul_kernel() is None:
            return False
        a = import_module("uniserve_worker.nn.activation")
        if not a._act_inputs_eligible(x, y):
            return False
        return int(x.shape[-1]) * int(x.element_size()) % _SGL_ALIGNMENT_BYTES == 0

    def run(self, x: torch.Tensor, y: torch.Tensor | None) -> torch.Tensor:
        kernel = _sgl_silu_and_mul_kernel()
        if kernel is None:
            raise RuntimeError("sgl silu_and_mul kernel unavailable")
        return kernel(x)


@lru_cache(maxsize=1)
def rms_norm_dispatcher():
    n = import_module("uniserve_worker.nn.norm")

    return Dispatcher(
        "rms_norm",
        [
            _RmsNormKernelProvider("sgl_kernel", _SglRmsNormKernel()),
            _RmsNormKernelProvider("triton", n._TritonRmsNorm()),
            _RmsNormKernelProvider("eager", n._EagerRmsNorm()),
        ],
        env_override="UNISERVE_RMS_NORM_PROVIDER",
    )


def select_rms_norm_kernel(
    weight: torch.Tensor,
    eps: float,
    *,
    override: str | None = None,
):
    """Bind the best RMSNorm kernel for a placed module parameter."""

    providers = rms_norm_dispatcher().ordered(override)
    eager = next(provider for provider in providers if provider.name == "eager")
    prototype = torch.empty(
        (1, int(weight.numel())),
        dtype=weight.dtype,
        device=weight.device,
    )
    with torch.inference_mode():
        selected = next(
            provider
            for provider in providers
            if provider.can_run(RmsNormReq(prototype, weight, float(eps)))
        )
    selected_kernel = selected._kernel
    eager_kernel = eager._kernel
    if selected is eager:
        return eager_kernel.run

    def run(hidden_states: torch.Tensor, current_weight: torch.Tensor, current_eps: float):
        if selected_kernel.is_eligible(hidden_states, current_weight, current_eps):
            return selected_kernel.run(hidden_states, current_weight, current_eps)
        return eager_kernel.run(hidden_states, current_weight, current_eps)

    return run


@lru_cache(maxsize=1)
def add_rms_norm_dispatcher():
    n = import_module("uniserve_worker.nn.norm")

    return Dispatcher(
        "add_rms_norm",
        [
            _AddRmsNormKernelProvider("sgl_kernel", _SglAddRmsNormInPlaceKernel()),
            _AddRmsNormKernelProvider("triton", n._TritonAddRmsNorm()),
            _AddRmsNormKernelProvider("eager", n._EagerAddRmsNorm()),
        ],
        env_override="UNISERVE_ADD_RMS_NORM_PROVIDER",
    )


@lru_cache(maxsize=1)
def silu_and_mul_dispatcher():
    a = import_module("uniserve_worker.nn.activation")

    return Dispatcher(
        "silu_and_mul",
        [
            _SiluAndMulKernelProvider("triton", a._TritonSiluAndMul()),
            _SiluAndMulKernelProvider("sgl_kernel", _SglSiluAndMulKernel()),
            _SiluAndMulKernelProvider("eager", a._EagerSiluAndMul()),
        ],
        env_override="UNISERVE_SILU_AND_MUL_PROVIDER",
    )


@lru_cache(maxsize=1)
def rope_dispatcher():
    r = import_module("uniserve_worker.nn.rope")

    return Dispatcher(
        "rope",
        [
            _PackedRopeKernelProvider("triton", r._TritonPackedRope()),
            _PackedRopeKernelProvider("eager", r._EagerPackedRope()),
        ],
        env_override="UNISERVE_ROPE_PROVIDER",
    )


class _TritonQKNormProvider:
    name = "triton"
    operator = "qk_norm"

    def can_run(self, req: QKNormReq) -> bool:
        if req.axis_dims is not None:
            return False
        if isinstance(req.q_weight, tuple) or isinstance(req.k_weight, tuple):
            return False
        can_run_triton_qk_rms_norm = import_module(
            "uniserve_worker.nn.norm"
        ).can_run_triton_qk_rms_norm

        return bool(
            can_run_triton_qk_rms_norm(req.q, req.k, req.q_weight, req.k_weight, req.eps, req.eps)
        )

    def run(self, req: QKNormReq):
        if req.axis_dims is not None:
            raise RuntimeError("triton qk_norm only handles single-axis requests without axis_dims")
        if isinstance(req.q_weight, tuple) or isinstance(req.k_weight, tuple):
            raise RuntimeError("triton qk_norm only handles single-axis requests without axis_dims")
        try_triton_qk_rms_norm = import_module("uniserve_worker.nn.norm").try_triton_qk_rms_norm

        out = try_triton_qk_rms_norm(req.q, req.k, req.q_weight, req.k_weight, req.eps, req.eps)
        if out is None:
            raise RuntimeError("triton qk_norm became ineligible")
        return out

    @staticmethod
    def _same_token_layout(q: torch.Tensor, k: torch.Tensor) -> bool:
        if q.ndim != k.ndim:
            return False
        if q.ndim == 3:
            return int(q.shape[0]) == int(k.shape[0])
        if q.ndim == 4:
            return int(q.shape[0]) == int(k.shape[0]) and int(q.shape[2]) == int(k.shape[2])
        return False


class _EagerQKNormProvider:
    name = "eager"
    operator = "qk_norm"

    def can_run(self, req: QKNormReq) -> bool:
        return True

    def run(self, req: QKNormReq):
        def rms(x, w):
            dtype = x.dtype
            y = x.to(torch.float32)
            y = y * torch.rsqrt(y.pow(2).mean(dim=-1, keepdim=True) + req.eps)
            return (y.to(dtype) * w).to(dtype)

        if req.axis_dims is not None:
            if not isinstance(req.q_weight, tuple) or not isinstance(req.k_weight, tuple):
                raise RuntimeError("multi-axis qk_norm requires tuple weights")
            if len(req.q_weight) != len(req.axis_dims) or len(req.k_weight) != len(req.axis_dims):
                raise RuntimeError("multi-axis qk_norm weight/axis mismatch")
            q_parts = req.q.split(req.axis_dims, dim=-1)
            k_parts = req.k.split(req.axis_dims, dim=-1)
            out_q = []
            out_k = []
            axis = 0
            while axis < len(req.axis_dims):
                group_end = _EagerQKNormRopeProvider._shared_norm_group_end(req, axis)
                q_group = torch.cat(q_parts[axis:group_end], dim=-1)
                k_group = torch.cat(k_parts[axis:group_end], dim=-1)
                q_normed = rms(q_group, req.q_weight[axis])
                k_normed = rms(k_group, req.k_weight[axis])
                out_q.extend(q_normed.split(req.axis_dims[axis:group_end], dim=-1))
                out_k.extend(k_normed.split(req.axis_dims[axis:group_end], dim=-1))
                axis = group_end
            return torch.cat(out_q, dim=-1), torch.cat(out_k, dim=-1)

        return rms(req.q, req.q_weight), rms(req.k, req.k_weight)


@lru_cache(maxsize=1)
def qk_norm_dispatcher():
    return Dispatcher(
        "qk_norm",
        [_TritonQKNormProvider(), _EagerQKNormProvider()],
        env_override="UNISERVE_QK_NORM_PROVIDER",
    )


class _TritonQKNormRopeProvider:
    name = "triton"
    operator = "qk_norm_rope"

    def can_run(self, req: QKNormRopeReq) -> bool:
        if req.axis_dims is not None:
            return self._can_run_multi_axis(req)
        if isinstance(req.q_weight, tuple) or isinstance(req.cos, tuple):
            return False
        if req.position_ids is not None or req.unsqueeze_dim != 1:
            return False
        can_run_triton_qk_rms_norm_rope = import_module(
            "uniserve_worker.nn.rope"
        ).can_run_triton_qk_rms_norm_rope

        return bool(
            can_run_triton_qk_rms_norm_rope(
                req.q,
                req.k,
                req.q_weight,
                req.k_weight,
                req.cos,
                req.sin,
                req.eps,
                req.eps,
            )
        )

    def run(self, req: QKNormRopeReq):
        if req.axis_dims is not None:
            return self._run_multi_axis(req)
        if isinstance(req.q_weight, tuple) or isinstance(req.cos, tuple):
            raise RuntimeError(
                "triton qk_norm_rope only handles single-axis requests without axis_dims"
            )
        try_triton_qk_rms_norm_rope = import_module(
            "uniserve_worker.nn.rope"
        ).try_triton_qk_rms_norm_rope

        out = try_triton_qk_rms_norm_rope(
            req.q, req.k, req.q_weight, req.k_weight, req.cos, req.sin, req.eps, req.eps
        )
        if out is None:
            raise RuntimeError("triton qk_norm_rope became ineligible")
        return out

    def _try_identity_tail_fused(
        self, req: QKNormRopeReq
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """One-launch path for multi-axis calls whose tail axes are identity.

        Applies when the caller declared every axis after the first as
        zero-position (``identity_axes``), the tail axes share one norm weight
        (a single shared-norm group), and the 3-D fused kernel accepts the
        tensors. Falls back to the general multi-axis pipeline otherwise. The
        fused kernel preserves each group's exact rounding order, so this is a
        launch-count optimization, not a numerics change.
        """
        axis_dims = req.axis_dims
        if req.identity_axes is None or axis_dims is None or len(axis_dims) < 2:
            return None
        if not all(
            isinstance(value, tuple) for value in (req.q_weight, req.k_weight, req.cos, req.sin)
        ):
            return None
        if tuple(req.identity_axes) != tuple(range(1, len(axis_dims))):
            return None
        if _EagerQKNormRopeProvider._shared_norm_group_end(req, 0) != 1:
            return None
        if _EagerQKNormRopeProvider._shared_norm_group_end(req, 1) != len(axis_dims):
            return None
        if req.q.ndim != 3 or req.k.ndim != 3:
            return None
        rope = import_module("uniserve_worker.nn.rope")
        rope_dim = int(axis_dims[0])
        tokens = int(req.q.shape[0])
        cos0, sin0 = req.cos[0], req.sin[0]
        if not (self._can_repeat_rope(cos0, tokens) and self._can_repeat_rope(sin0, tokens)):
            return None
        cos = self._align_rope_table(cos0, tokens)
        sin = self._align_rope_table(sin0, tokens)
        return rope.try_triton_qk_split_rms_norm_rope(
            req.q,
            req.k,
            req.q_weight[0],
            req.q_weight[1],
            req.k_weight[0],
            req.k_weight[1],
            cos,
            sin,
            req.eps,
            req.eps,
            rope_dim=rope_dim,
        )

    def _try_rotated_tail_fused(
        self, req: QKNormRopeReq
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """One-launch path for 3-axis calls whose tail axes share one norm.

        Applies to the SenseNova spatial-token layout: axis 0 is its own
        norm+RoPE group and axes 1..2 share a single norm weight while each
        keeps its own (non-identity) rotation table. The fused kernel
        reproduces the general pipeline's per-element arithmetic and rounding
        order exactly (see try_triton_qk_multi_axis_rms_norm_rope), so this is
        a launch-count/traffic optimization, not a numerics change. Anything
        that does not match falls back to the general multi-axis pipeline.
        """
        axis_dims = req.axis_dims
        if axis_dims is None or len(axis_dims) != 3:
            return None
        if not all(
            isinstance(value, tuple) for value in (req.q_weight, req.k_weight, req.cos, req.sin)
        ):
            return None
        if req.q.ndim != 3 or req.k.ndim != 3:
            return None
        if _EagerQKNormRopeProvider._shared_norm_group_end(req, 0) != 1:
            return None
        if _EagerQKNormRopeProvider._shared_norm_group_end(req, 1) != len(axis_dims):
            return None
        tokens = int(req.q.shape[0])
        cos_tables: list[torch.Tensor] = []
        sin_tables: list[torch.Tensor] = []
        for axis in range(3):
            cos_a, sin_a = req.cos[axis], req.sin[axis]
            if not (self._can_repeat_rope(cos_a, tokens) and self._can_repeat_rope(sin_a, tokens)):
                return None
            cos_tables.append(self._align_rope_table(cos_a, tokens))
            sin_tables.append(self._align_rope_table(sin_a, tokens))
        rope = import_module("uniserve_worker.nn.rope")
        return rope.try_triton_qk_multi_axis_rms_norm_rope(
            req.q,
            req.k,
            req.q_weight[0],
            req.q_weight[1],
            req.k_weight[0],
            req.k_weight[1],
            (cos_tables[0], cos_tables[1], cos_tables[2]),
            (sin_tables[0], sin_tables[1], sin_tables[2]),
            req.eps,
            req.eps,
            axis_dims=tuple(int(v) for v in axis_dims),
        )

    @staticmethod
    def _flatten_heads(x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...], bool]:
        if x.ndim == 3:
            return x, tuple(x.shape), False
        if x.ndim != 4:
            raise RuntimeError("triton multi-axis qk_norm_rope requires 3D or 4D q/k tensors")
        # Multi-axis callers may pass [batch, heads, seq, dim]. Existing Triton
        # norm/RoPE kernels operate on [tokens, heads, dim], so flatten
        # batch*seq tokens.
        flat = (
            x.permute(0, 2, 1, 3)
            .contiguous()
            .reshape(x.shape[0] * x.shape[2], x.shape[1], x.shape[3])
        )
        return flat, tuple(x.shape), True

    @staticmethod
    def _axis_group(
        x: torch.Tensor,
        axis_dims: tuple[int, ...],
        start_axis: int,
        end_axis: int,
    ) -> torch.Tensor:
        start = sum(int(dim) for dim in axis_dims[:start_axis])
        width = sum(int(dim) for dim in axis_dims[start_axis:end_axis])
        return x[..., start : start + width]

    def _flatten_axis_group(
        self,
        x: torch.Tensor,
        axis_dims: tuple[int, ...],
        start_axis: int,
        end_axis: int,
    ) -> tuple[torch.Tensor, tuple[int, ...], bool]:
        return self._flatten_heads(self._axis_group(x, axis_dims, start_axis, end_axis))

    @staticmethod
    def _unflatten_heads(
        x: torch.Tensor, shape: tuple[int, ...], was_flattened: bool
    ) -> torch.Tensor:
        if not was_flattened:
            return x
        batch, heads, seq_len, dim = shape
        return x.reshape(batch, seq_len, heads, dim).permute(0, 2, 1, 3).contiguous()

    @staticmethod
    def _can_repeat_rope(cos: torch.Tensor, tokens: int) -> bool:
        return (
            cos.ndim == 2
            and int(cos.shape[0]) > 0
            and (tokens % int(cos.shape[0]) == 0 or int(cos.shape[0]) % tokens == 0)
        )

    @staticmethod
    def _align_rope_table(table: torch.Tensor, tokens: int) -> torch.Tensor:
        rows = int(table.shape[0])
        if rows == tokens:
            return table.contiguous()
        if rows < tokens and tokens % rows == 0:
            return table.repeat(tokens // rows, 1).contiguous()
        if rows > tokens and rows % tokens == 0:
            return table[:tokens].contiguous()
        raise RuntimeError(f"cannot align RoPE table length {rows} to token count {tokens}")

    def _can_run_multi_axis(self, req: QKNormRopeReq) -> bool:
        if req.position_ids is not None or req.unsqueeze_dim != 1:
            return False
        try:
            plan = import_module(
                "uniserve_worker.ops.qk_norm_rope_plan"
            ).QKNormRopePlan.from_request(req)
        except RuntimeError:
            return False
        if req.q.ndim not in (3, 4) or req.k.ndim not in (3, 4):
            return False
        if not _TritonQKNormProvider._same_token_layout(req.q, req.k):
            return False
        if not (req.q.is_cuda and req.k.is_cuda):
            return False
        norm_mod = import_module("uniserve_worker.nn.norm")
        rope_mod = import_module("uniserve_worker.nn.rope")
        tokens = self._flattened_token_count(req.q)
        if tokens <= 0:
            return False
        if not self._qk_common_eligible(req, norm_mod):
            return False
        for group in plan.groups:
            if group.single_axis:
                cos = plan.cos_tables[group.start]
                sin = plan.sin_tables[group.start]
                if not self._can_repeat_rope(cos, tokens):
                    return False
                if not self._rope_table_shape_matches(cos, sin, tokens, group.dim):
                    return False
                if not self._qk_norm_rope_group_eligible(
                    req,
                    norm_mod,
                    group.dim,
                    plan.q_weights[group.start],
                    plan.k_weights[group.start],
                    cos,
                    sin,
                ):
                    return False
            else:
                if not self._qk_norm_group_eligible(
                    req,
                    norm_mod,
                    group.dim,
                    plan.q_weights[group.start],
                    plan.k_weights[group.start],
                ):
                    return False
            for local_axis in range(group.start, group.end):
                cos = plan.cos_tables[local_axis]
                sin = plan.sin_tables[local_axis]
                axis_dim = int(plan.axis_dims[local_axis])
                if not self._can_repeat_rope(cos, tokens):
                    return False
                if not self._rope_table_shape_matches(cos, sin, tokens, axis_dim):
                    return False
                if not self._packed_rope_axis_eligible(req, rope_mod, axis_dim, cos, sin):
                    return False
        return True

    @staticmethod
    def _flattened_token_count(x: torch.Tensor) -> int:
        if x.ndim == 3:
            return int(x.shape[0])
        if x.ndim == 4:
            return int(x.shape[0]) * int(x.shape[2])
        return 0

    @staticmethod
    def _rope_table_shape_matches(
        cos: torch.Tensor, sin: torch.Tensor, tokens: int, axis_dim: int
    ) -> bool:
        return (
            cos.ndim == 2
            and sin.shape == cos.shape
            and int(cos.shape[-1]) * 2 == int(axis_dim)
            and _TritonQKNormRopeProvider._can_repeat_rope(cos, int(tokens))
        )

    @staticmethod
    def _qk_common_eligible(req: QKNormRopeReq, norm_mod) -> bool:
        q_heads = int(req.q.shape[1]) if req.q.ndim in (3, 4) else 0
        k_heads = int(req.k.shape[1]) if req.k.ndim in (3, 4) else 0
        return not (
            norm_mod.triton is None
            or torch.is_grad_enabled()
            or not triton_available(req.q.device)
            or req.q.device != req.k.device
            or req.q.dtype != req.k.dtype
            or q_heads <= 0
            or k_heads <= 0
        )

    @staticmethod
    def _qk_norm_group_eligible(
        req: QKNormRopeReq,
        norm_mod,
        group_dim: int,
        q_weight: torch.Tensor,
        k_weight: torch.Tensor,
    ) -> bool:
        del norm_mod
        group_dim = int(group_dim)
        return not (
            group_dim <= 0
            or group_dim > 1024
            or not q_weight.is_cuda
            or not k_weight.is_cuda
            or q_weight.device != req.q.device
            or k_weight.device != req.q.device
            or int(q_weight.numel()) != group_dim
            or int(k_weight.numel()) != group_dim
            or not q_weight.is_contiguous()
            or not k_weight.is_contiguous()
        )

    @staticmethod
    def _qk_norm_rope_group_eligible(
        req: QKNormRopeReq,
        norm_mod,
        group_dim: int,
        q_weight: torch.Tensor,
        k_weight: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> bool:
        if not _TritonQKNormRopeProvider._qk_norm_group_eligible(
            req, norm_mod, group_dim, q_weight, k_weight
        ):
            return False
        group_dim = int(group_dim)
        return not (
            group_dim % 2 != 0
            or cos.device != req.q.device
            or sin.device != req.q.device
            or not cos.is_cuda
            or not sin.is_cuda
        )

    @staticmethod
    def _packed_rope_axis_eligible(
        req: QKNormRopeReq,
        rope_mod,
        axis_dim: int,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> bool:
        axis_dim = int(axis_dim)
        heads = int(req.q.shape[1]) if req.q.ndim in (3, 4) else 0
        return not (
            rope_mod.triton is None
            or torch.is_grad_enabled()
            or not triton_available(req.q.device)
            or axis_dim <= 0
            or axis_dim % 2 != 0
            or heads <= 0
            or not cos.is_cuda
            or not sin.is_cuda
            or cos.device != req.q.device
            or sin.device != req.q.device
        )

    def _run_multi_axis(self, req: QKNormRopeReq):
        fused = self._try_identity_tail_fused(req)
        if fused is not None:
            return fused
        fused = self._try_rotated_tail_fused(req)
        if fused is not None:
            return fused
        out_q = []
        out_k = []
        plan = import_module("uniserve_worker.ops.qk_norm_rope_plan").QKNormRopePlan.from_request(
            req
        )
        try_triton_qk_rms_norm = import_module("uniserve_worker.nn.norm").try_triton_qk_rms_norm
        try_triton_qk_rms_norm_rope = import_module(
            "uniserve_worker.nn.rope"
        ).try_triton_qk_rms_norm_rope
        packed_rope = import_module("uniserve_worker.nn.rope")._TritonPackedRope()
        for group in plan.groups:
            q_group, q_shape, q_was_flattened = self._flatten_axis_group(
                req.q, plan.axis_dims, group.start, group.end
            )
            k_group, k_shape, k_was_flattened = self._flatten_axis_group(
                req.k, plan.axis_dims, group.start, group.end
            )
            if group.single_axis:
                cos = plan.cos_tables[group.start]
                sin = plan.sin_tables[group.start]
                cos_flat = self._align_rope_table(cos, int(q_group.shape[0]))
                sin_flat = self._align_rope_table(sin, int(q_group.shape[0]))
                out = try_triton_qk_rms_norm_rope(
                    q_group,
                    k_group,
                    plan.q_weights[group.start],
                    plan.k_weights[group.start],
                    cos_flat,
                    sin_flat,
                    req.eps,
                    req.eps,
                )
                if out is None:
                    raise RuntimeError("triton multi-axis qk_norm_rope became ineligible")
                q_rot, k_rot = out
                out_q.append(self._unflatten_heads(q_rot, q_shape, q_was_flattened))
                out_k.append(self._unflatten_heads(k_rot, k_shape, k_was_flattened))
            else:
                out = try_triton_qk_rms_norm(
                    q_group,
                    k_group,
                    plan.q_weights[group.start],
                    plan.k_weights[group.start],
                    req.eps,
                    req.eps,
                )
                if out is None:
                    raise RuntimeError("triton multi-axis qk_norm became ineligible")
                q_normed, k_normed = out
                q_normed_parts = q_normed.split(plan.axis_dims[group.start : group.end], dim=-1)
                k_normed_parts = k_normed.split(plan.axis_dims[group.start : group.end], dim=-1)
                for local, (q_normed_part, k_normed_part) in enumerate(
                    zip(q_normed_parts, k_normed_parts, strict=True)
                ):
                    axis = group.start + local
                    cos = plan.cos_tables[axis]
                    sin = plan.sin_tables[axis]
                    q_normed_part = q_normed_part.contiguous()
                    k_normed_part = k_normed_part.contiguous()
                    cos_flat = self._align_rope_table(cos, int(q_normed_part.shape[0]))
                    sin_flat = self._align_rope_table(sin, int(q_normed_part.shape[0]))
                    q_rot = packed_rope.run(q_normed_part, cos_flat, sin_flat)
                    k_rot = packed_rope.run(k_normed_part, cos_flat, sin_flat)
                    axis_dim = int(plan.axis_dims[axis])
                    q_part_shape = (*q_shape[:-1], axis_dim)
                    k_part_shape = (*k_shape[:-1], axis_dim)
                    out_q.append(self._unflatten_heads(q_rot, q_part_shape, q_was_flattened))
                    out_k.append(self._unflatten_heads(k_rot, k_part_shape, k_was_flattened))
        return torch.cat(out_q, dim=-1), torch.cat(out_k, dim=-1)


class _EagerQKNormRopeProvider:
    name = "eager"
    operator = "qk_norm_rope"

    def can_run(self, req: QKNormRopeReq) -> bool:
        return True

    @staticmethod
    def _shared_norm_group_end(req: QKNormReq | QKNormRopeReq, start: int) -> int:
        return import_module(
            "uniserve_worker.ops.qk_norm_rope_plan"
        ).QKNormRopePlan.shared_norm_group_end(req, start)

    @staticmethod
    def _apply_rope_axis(
        q_normed, k_normed, cos, sin, *, apply_rotary_pos_emb, apply_rotary_emb, unsqueeze_dim
    ):
        if q_normed.ndim == 4 and cos.ndim == 2:
            dim = int(q_normed.shape[-1])
            if dim % 2:
                return apply_rotary_pos_emb(q_normed, k_normed, cos, sin, None, unsqueeze_dim)
            batch = int(q_normed.shape[0])
            tokens = int(q_normed.shape[2])
            rows = int(cos.shape[0])
            if rows == batch * tokens:
                cos = cos.contiguous().view(batch, tokens, cos.shape[1])
                sin = sin.contiguous().view(batch, tokens, sin.shape[1])
                table_shape = (batch, 1, tokens)
            else:
                cos = _TritonQKNormRopeProvider._align_rope_table(cos, tokens)
                sin = _TritonQKNormRopeProvider._align_rope_table(sin, tokens)
                table_shape = (1, 1, tokens)
            if int(cos.shape[-1]) >= dim:
                cos_view = cos[..., :dim].view(*table_shape, dim)
                sin_view = sin[..., :dim].view(*table_shape, dim)
                half = dim // 2

                def rotate_full(x):
                    out = torch.empty_like(x)
                    out[..., :half] = (
                        x[..., :half] * cos_view[..., :half]
                        - x[..., half:dim] * sin_view[..., :half]
                    )
                    out[..., half:dim] = (
                        x[..., half:dim] * cos_view[..., half:dim]
                        + x[..., :half] * sin_view[..., half:dim]
                    )
                    if dim % 2:
                        out[..., -1:] = x[..., -1:]
                    return out

                return rotate_full(q_normed), rotate_full(k_normed)
            cos_view = cos.view(*table_shape, cos.shape[-1])
            sin_view = sin.view(*table_shape, sin.shape[-1])
            half = dim // 2

            def rotate_packed(x):
                out = torch.empty_like(x)
                out[..., :half] = x[..., :half] * cos_view - x[..., half:] * sin_view
                out[..., half:] = x[..., half:] * cos_view + x[..., :half] * sin_view
                return out

            return rotate_packed(q_normed), rotate_packed(k_normed)
        if int(cos.shape[-1]) >= int(q_normed.shape[-1]):
            return apply_rotary_pos_emb(q_normed, k_normed, cos, sin, None, unsqueeze_dim)
        return apply_rotary_emb(q_normed, cos, sin), apply_rotary_emb(k_normed, cos, sin)

    def _run_multi_axis(self, req: QKNormRopeReq, *, apply_rotary_pos_emb, apply_rotary_emb):
        plan = import_module("uniserve_worker.ops.qk_norm_rope_plan").QKNormRopePlan.from_request(
            req
        )
        q_parts = req.q.split(plan.axis_dims, dim=-1)
        k_parts = req.k.split(plan.axis_dims, dim=-1)
        out_q = []
        out_k = []
        eager_norm = _EagerQKNormProvider()
        for group in plan.groups:
            q_group = torch.cat(q_parts[group.start : group.end], dim=-1)
            k_group = torch.cat(k_parts[group.start : group.end], dim=-1)
            q_normed_group, k_normed_group = eager_norm.run(
                QKNormReq(
                    q_group,
                    k_group,
                    plan.q_weights[group.start],
                    plan.k_weights[group.start],
                    req.eps,
                )
            )
            q_normed_parts = q_normed_group.split(plan.axis_dims[group.start : group.end], dim=-1)
            k_normed_parts = k_normed_group.split(plan.axis_dims[group.start : group.end], dim=-1)
            for local, (q_normed, k_normed) in enumerate(
                zip(q_normed_parts, k_normed_parts, strict=True)
            ):
                axis = group.start + local
                q_rot, k_rot = self._apply_rope_axis(
                    q_normed,
                    k_normed,
                    plan.cos_tables[axis],
                    plan.sin_tables[axis],
                    apply_rotary_pos_emb=apply_rotary_pos_emb,
                    apply_rotary_emb=apply_rotary_emb,
                    unsqueeze_dim=req.unsqueeze_dim,
                )
                out_q.append(q_rot)
                out_k.append(k_rot)
        return torch.cat(out_q, dim=-1), torch.cat(out_k, dim=-1)

    def run(self, req: QKNormRopeReq):
        rope_mod = import_module("uniserve_worker.nn.rope")
        apply_rotary_pos_emb = rope_mod.apply_rotary_pos_emb
        apply_rotary_emb = rope_mod.apply_rotary_emb

        if req.axis_dims is not None:
            return self._run_multi_axis(
                req, apply_rotary_pos_emb=apply_rotary_pos_emb, apply_rotary_emb=apply_rotary_emb
            )

        q, k = _EagerQKNormProvider().run(
            QKNormReq(req.q, req.k, req.q_weight, req.k_weight, req.eps)
        )
        return self._apply_rope_axis(
            q,
            k,
            req.cos,
            req.sin,
            apply_rotary_pos_emb=apply_rotary_pos_emb,
            apply_rotary_emb=apply_rotary_emb,
            unsqueeze_dim=req.unsqueeze_dim,
        )


@lru_cache(maxsize=1)
def qk_norm_rope_dispatcher():
    return Dispatcher(
        "qk_norm_rope",
        [_TritonQKNormRopeProvider(), _EagerQKNormRopeProvider()],
        env_override="UNISERVE_QK_NORM_ROPE_PROVIDER",
    )


class _AttentionBackendProvider:
    operator = "attention"

    def __init__(self, backend) -> None:
        self.backend = backend
        self.name = backend.name

    def capabilities(self):
        return self.backend.capabilities()

    def display_name(self, req: AttentionReq) -> str:
        if (
            req.block_table is not None
            and req.cu_seqlens_q is not None
            and req.cu_seqlens_k is not None
        ):
            return f"{self.name}_paged_varlen"
        return self.name

    @staticmethod
    def _head_dim_supported(caps, head_dim: int) -> bool:
        return int(head_dim) >= int(getattr(caps, "min_head_dim", 1) or 1)

    @staticmethod
    def _trunk_geometry_supported(
        caps, req: AttentionReq, *, backend_name: str | None = None
    ) -> bool:
        supports = getattr(caps, "supports_trunk_geometry", None)
        if backend_name == "fa4_cute" and not getattr(caps, "trunk_geometries", frozenset()):
            return False
        if not callable(supports):
            return True
        k_dim = int((req.current_k if req.current_k is not None else req.k).shape[-1])
        v_dim = int((req.current_v if req.current_v is not None else req.v).shape[-1])
        return bool(supports(int(req.q.shape[-1]), k_dim, v_dim))

    @staticmethod
    def _paged_storage_supported(caps, req: AttentionReq) -> bool:
        view_block_size = getattr(req.kv_cache, "block_size", None)
        if view_block_size is not None:
            if not bool(getattr(req.kv_cache, "supports_paged_attention_storage", True)):
                return False
            block_size = int(view_block_size or 0)
        elif req.block_table is not None:
            if not isinstance(req.k, torch.Tensor) or req.k.ndim != 4:
                return False
            block_size = int(req.k.shape[1])
        else:
            return True
        if block_size <= 0:
            return False
        multiple = int(getattr(caps, "paged_block_size_multiple", 1) or 1)
        return block_size % max(1, multiple) == 0

    @staticmethod
    def _is_one_token_decode(req: AttentionReq) -> bool:
        if req.q.ndim == 3:
            plan = getattr(req.ctx, "attention", None) if req.ctx is not None else None
            query_lens = getattr(plan, "query_lens_cpu", ()) or ()
            if isinstance(plan, PagedDecodePlan) and len(query_lens) == int(req.q.shape[0]):
                return all(int(length) == 1 for length in query_lens)
            return int(req.q.shape[0]) == 1
        if req.q.ndim == 4:
            return int(req.q.shape[2]) == 1
        return False

    @staticmethod
    def _dense_layout_supported(backend_name: str, req: AttentionReq) -> bool:
        if req.q.ndim != req.k.ndim or req.q.ndim != req.v.ndim:
            return False
        if backend_name == "flashinfer":
            return req.q.ndim == 3
        if backend_name in {"flash_attn", "fa4_cute"}:
            return req.q.ndim == 4
        if backend_name in {"torch_sdpa", "sgl_kernel"}:
            return req.q.ndim in {3, 4}
        # Context/fake providers used by tests may implement a narrower contract;
        # without declared layout caps, let their own method be the authority.
        return req.q.ndim in {3, 4}

    def can_run(self, req: AttentionReq) -> bool:
        caps = self.backend.capabilities()
        if not bool(getattr(caps, "available", True)):
            return False
        if (
            self.name in {"sgl_kernel", "flashinfer", "flash_attn", "fa4_cute"}
            and req.q.device.type != "cuda"
        ):
            return False
        if self.name == "trtllm_mha":
            if req.q.device.type != "cuda":
                return False
            major, minor = torch.cuda.get_device_capability(req.q.device)
            if (int(major), int(minor)) < (10, 0):
                return False
        if not self._head_dim_supported(caps, req.q.shape[-1]):
            return False
        if not self._trunk_geometry_supported(
            caps, req, backend_name=getattr(self.backend, "name", self.name)
        ):
            return False
        if req.regime is AttentionRegime.VISIBLE_END:
            return bool(getattr(caps, "visible_end", False)) and req.visible_end is not None
        if (
            req.regime in {AttentionRegime.EXTEND, AttentionRegime.MIXED}
            and req.cu_seqlens_q is not None
        ):
            if req.block_table is not None:
                return (
                    bool(getattr(caps, "varlen_attention", False))
                    and bool(getattr(caps, "varlen_paged_kv", False))
                    and self._paged_storage_supported(caps, req)
                )
            if bool(getattr(caps, "requires_paged_varlen", False)):
                return False
            return bool(getattr(caps, "varlen_attention", False))
        if (
            req.regime is AttentionRegime.DECODE
            or req.block_table is not None
            or req.kv_cache is not None
        ):
            if bool(getattr(caps, "paged_decode_only", False)) and not self._is_one_token_decode(
                req
            ):
                return False
            return (
                bool(getattr(caps, "paged_kv", False))
                and req.block_table is not None
                and req.cache_seqlens is not None
                and self._paged_storage_supported(caps, req)
            )
        if bool(getattr(caps, "paged_decode_only", False)):
            return False
        if req.attn_mask is not None and self.name != "torch_sdpa":
            return False
        return self._dense_layout_supported(self.name, req)

    def run(self, req: AttentionReq):
        if req.regime is AttentionRegime.VISIBLE_END:
            return self.backend.forward_visible_end(
                req.q,
                req.k,
                req.v,
                visible_end=req.visible_end,
                cu_seqlens_q=req.cu_seqlens_q,
                cu_seqlens_k=req.cu_seqlens_k,
                page_table=req.page_table,
                seqused_k=req.seqused_k,
                max_seqlen_q=req.max_seqlen_q,
                max_seqlen_k=req.max_seqlen_k,
                scale=req.scale,
                use_prefix_bounds=req.use_prefix_bounds,
                fully_visible=req.fully_visible,
                context=req.ctx,
            )
        if (
            req.cu_seqlens_q is not None
            and req.cu_seqlens_k is not None
            and hasattr(self.backend, "forward_varlen")
        ):
            return self.backend.forward_varlen(
                req.q,
                req.k,
                req.v,
                cu_seqlens_q=req.cu_seqlens_q,
                cu_seqlens_k=req.cu_seqlens_k,
                max_seqlen_q=int(req.max_seqlen_q or 0),
                max_seqlen_k=int(req.max_seqlen_k or 0),
                causal=req.causal,
                scale=req.scale,
                block_table=req.block_table,
                context=req.ctx,
            )
        if (
            req.block_table is not None
            and req.cache_seqlens is not None
            and hasattr(self.backend, "forward_paged")
        ):
            return self.backend.forward_paged(
                req.q,
                req.k,
                req.v,
                block_table=req.block_table,
                cache_seqlens=req.cache_seqlens,
                k=req.current_k,
                v=req.current_v,
                causal=req.causal,
                scale=req.scale,
                context=req.ctx,
            )
        return self.backend.forward(
            req.q,
            req.k,
            req.v,
            causal=req.causal,
            scale=req.scale,
            attn_mask=req.attn_mask,
            context=req.ctx,
        )


def can_run_attention(selection, req: AttentionReq) -> bool:
    return any(_AttentionBackendProvider(backend).can_run(req) for backend in selection.providers)


def run_attention(selection, req: AttentionReq) -> torch.Tensor:
    for backend in selection.providers:
        provider = _AttentionBackendProvider(backend)
        if provider.can_run(req):
            return provider.run(req)
    names = tuple(backend.name for backend in selection.providers)
    raise RuntimeError(f"no provisioned attention backend can execute this request: {names!r}")
