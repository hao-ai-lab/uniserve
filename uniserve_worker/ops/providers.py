"""Provider packs backed by existing in-tree kernels."""
from __future__ import annotations

from functools import lru_cache
from importlib import import_module
import torch
import torch.nn.functional as F

from .core import Capabilities, CommDispatcher, Dispatcher, Handoff
from .requests import (
    AddRmsNormReq,
    AttentionRegime,
    AttentionReq,
    PackedRopeReq,
    QKNormReq,
    QKNormRopeReq,
    RmsNormReq,
    SiluAndMulReq,
    TpAllReduceReq,
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


@lru_cache(maxsize=1)
def weak_ref_tensor_provider():
    try:  # pragma: no cover - optional SGLang kernel package.
        from sgl_kernel import weak_ref_tensor  # type: ignore

        return weak_ref_tensor
    except Exception:
        try:  # pragma: no cover - optional NPU runtime.
            from torch_npu._C import _weak_ref_tensor as weak_ref_tensor  # type: ignore

            return weak_ref_tensor
        except Exception:
            return None


class _ProviderBase:
    def __init__(self, name: str, operator: str) -> None:
        self.name = name
        self.operator = operator

    def capabilities(self) -> Capabilities:
        return Capabilities(tags=frozenset({self.name, self.operator}))


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
            _SiluAndMulKernelProvider("sgl_kernel", _SglSiluAndMulKernel()),
            _SiluAndMulKernelProvider("triton", a._TritonSiluAndMul()),
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

    def capabilities(self) -> Capabilities:
        return Capabilities(tags=frozenset({"triton", "qk_norm"}))

    def can_run(self, req: QKNormReq) -> bool:
        if req.axis_dims is not None:
            return self._can_run_sensenova_3d(req)
        if isinstance(req.q_weight, tuple) or isinstance(req.k_weight, tuple):
            return False
        try_triton_qk_rms_norm = import_module("uniserve_worker.nn.norm").try_triton_qk_rms_norm

        return try_triton_qk_rms_norm(req.q, req.k, req.q_weight, req.k_weight, req.eps, req.eps) is not None

    def run(self, req: QKNormReq):
        if req.axis_dims is not None:
            out = self._try_run_sensenova_3d(req)
            if out is None:
                raise RuntimeError("triton SenseNova qk_norm became ineligible")
            return out
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

    def _sensenova_3d_inputs(self, req: QKNormReq):
        if req.axis_dims is None or req.q.ndim not in (3, 4) or req.k.ndim not in (3, 4):
            return None
        if not self._same_token_layout(req.q, req.k):
            return None
        dim = int(req.q.shape[-1])
        if int(req.k.shape[-1]) != dim:
            return None
        if tuple(req.axis_dims) != (dim // 2, dim // 4, dim // 4):
            return None
        if not (
            isinstance(req.q_weight, tuple)
            and isinstance(req.k_weight, tuple)
            and len(req.q_weight) == len(req.k_weight) == 3
        ):
            return None
        if req.q_weight[1] is not req.q_weight[2] or req.k_weight[1] is not req.k_weight[2]:
            return None
        q_flat, q_shape, q_was_flattened = _TritonQKNormRopeProvider._flatten_heads(req.q)
        k_flat, k_shape, k_was_flattened = _TritonQKNormRopeProvider._flatten_heads(req.k)
        if int(q_flat.shape[0]) != int(k_flat.shape[0]):
            return None
        return q_flat, k_flat, q_shape, k_shape, q_was_flattened, k_was_flattened

    def _can_run_sensenova_3d(self, req: QKNormReq) -> bool:
        inputs = self._sensenova_3d_inputs(req)
        if inputs is None:
            return False
        q_flat, k_flat, _q_shape, _k_shape, _q_flattened, _k_flattened = inputs
        can_run = import_module("uniserve_worker.nn.norm").can_run_triton_sensenova_qk_rms_norm_3d
        return bool(
            can_run(
                q_flat,
                k_flat,
                req.q_weight[0],
                req.q_weight[1],
                req.k_weight[0],
                req.k_weight[1],
            )
        )

    def _try_run_sensenova_3d(self, req: QKNormReq):
        inputs = self._sensenova_3d_inputs(req)
        if inputs is None:
            return None
        q_flat, k_flat, q_shape, k_shape, q_was_flattened, k_was_flattened = inputs
        try_triton_sensenova = import_module("uniserve_worker.nn.norm").try_triton_sensenova_qk_rms_norm_3d
        out = try_triton_sensenova(
            q_flat,
            k_flat,
            req.q_weight[0],
            req.q_weight[1],
            req.k_weight[0],
            req.k_weight[1],
            req.eps,
            req.eps,
        )
        if out is None:
            return None
        q_out, k_out = out
        return (
            _TritonQKNormRopeProvider._unflatten_heads(q_out, q_shape, q_was_flattened),
            _TritonQKNormRopeProvider._unflatten_heads(k_out, k_shape, k_was_flattened),
        )


class _EagerQKNormProvider:
    name = "eager"
    operator = "qk_norm"

    def capabilities(self) -> Capabilities:
        return Capabilities(tags=frozenset({"eager", "qk_norm"}))

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

    def capabilities(self) -> Capabilities:
        return Capabilities(tags=frozenset({"triton", "qk_norm_rope"}))

    def can_run(self, req: QKNormRopeReq) -> bool:
        if req.axis_dims is not None:
            return self._can_run_multi_axis(req)
        if isinstance(req.q_weight, tuple) or isinstance(req.cos, tuple):
            return False
        if req.position_ids is not None or req.unsqueeze_dim != 1:
            return False
        try_triton_qk_rms_norm_rope = import_module("uniserve_worker.nn.rope").try_triton_qk_rms_norm_rope

        return try_triton_qk_rms_norm_rope(req.q, req.k, req.q_weight, req.k_weight, req.cos, req.sin, req.eps, req.eps) is not None

    def run(self, req: QKNormRopeReq):
        if req.axis_dims is not None:
            return self._run_multi_axis(req)
        if isinstance(req.q_weight, tuple) or isinstance(req.cos, tuple):
            raise RuntimeError("triton qk_norm_rope only handles single-axis requests without axis_dims")
        try_triton_qk_rms_norm_rope = import_module("uniserve_worker.nn.rope").try_triton_qk_rms_norm_rope

        out = try_triton_qk_rms_norm_rope(req.q, req.k, req.q_weight, req.k_weight, req.cos, req.sin, req.eps, req.eps)
        if out is None:
            raise RuntimeError("triton qk_norm_rope became ineligible")
        return out

    @staticmethod
    def _flatten_heads(x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...], bool]:
        if x.ndim == 3:
            return x, tuple(x.shape), False
        if x.ndim != 4:
            raise RuntimeError("triton multi-axis qk_norm_rope requires 3D or 4D q/k tensors")
        # SenseNova passes [batch, heads, seq, dim]. Existing Triton norm/RoPE
        # kernels operate on [tokens, heads, dim], so flatten batch*seq tokens.
        flat = x.permute(0, 2, 1, 3).contiguous().reshape(x.shape[0] * x.shape[2], x.shape[1], x.shape[3])
        return flat, tuple(x.shape), True

    @staticmethod
    def _unflatten_heads(x: torch.Tensor, shape: tuple[int, ...], was_flattened: bool) -> torch.Tensor:
        if not was_flattened:
            return x
        batch, heads, seq_len, dim = shape
        return x.reshape(batch, seq_len, heads, dim).permute(0, 2, 1, 3).contiguous()

    @staticmethod
    def _can_repeat_rope(cos: torch.Tensor, tokens: int) -> bool:
        return cos.ndim == 2 and int(cos.shape[0]) > 0 and (tokens % int(cos.shape[0]) == 0 or int(cos.shape[0]) % tokens == 0)

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
            _EagerQKNormRopeProvider._validate_multi_axis(req)
        except RuntimeError:
            return False
        if req.q.ndim not in (3, 4) or req.k.ndim not in (3, 4):
            return False
        if not _TritonQKNormProvider._same_token_layout(req.q, req.k):
            return False
        if not (req.q.is_cuda and req.k.is_cuda):
            return False
        if self._can_run_sensenova_3d(req):
            return True
        try_triton_qk_rms_norm = import_module("uniserve_worker.nn.norm").try_triton_qk_rms_norm
        try_triton_qk_rms_norm_rope = import_module("uniserve_worker.nn.rope").try_triton_qk_rms_norm_rope
        packed_rope = import_module("uniserve_worker.nn.rope")._TritonPackedRope()
        q_parts = req.q.split(req.axis_dims, dim=-1)
        k_parts = req.k.split(req.axis_dims, dim=-1)
        axis = 0
        while axis < len(req.axis_dims):
            group_end = _EagerQKNormRopeProvider._shared_norm_group_end(req, axis)
            q_group, _, _ = self._flatten_heads(torch.cat(q_parts[axis:group_end], dim=-1))
            k_group, _, _ = self._flatten_heads(torch.cat(k_parts[axis:group_end], dim=-1))
            if group_end == axis + 1:
                cos = req.cos[axis]
                sin = req.sin[axis]
                if not self._can_repeat_rope(cos, int(q_group.shape[0])):
                    return False
                cos_flat = self._align_rope_table(cos, int(q_group.shape[0]))
                sin_flat = self._align_rope_table(sin, int(q_group.shape[0]))
                out = try_triton_qk_rms_norm_rope(
                    q_group,
                    k_group,
                    req.q_weight[axis],
                    req.k_weight[axis],
                    cos_flat,
                    sin_flat,
                    req.eps,
                    req.eps,
                )
                if out is None:
                    return False
            elif try_triton_qk_rms_norm(
                q_group,
                k_group,
                req.q_weight[axis],
                req.k_weight[axis],
                req.eps,
                req.eps,
            ) is None:
                return False
            for local_axis in range(axis, group_end):
                q_flat, _, _ = self._flatten_heads(q_parts[local_axis])
                cos = req.cos[local_axis]
                sin = req.sin[local_axis]
                if not self._can_repeat_rope(cos, int(q_flat.shape[0])):
                    return False
                cos_flat = self._align_rope_table(cos, int(q_flat.shape[0]))
                sin_flat = self._align_rope_table(sin, int(q_flat.shape[0]))
                if not packed_rope.is_eligible(q_flat, cos_flat, sin_flat):
                    return False
            axis = group_end
        return True

    def _sensenova_3d_inputs(self, req: QKNormRopeReq):
        if req.axis_dims is None or tuple(req.axis_dims) != (
            int(req.q.shape[-1]) // 2,
            int(req.q.shape[-1]) // 4,
            int(req.q.shape[-1]) // 4,
        ):
            return None
        if not (
            isinstance(req.q_weight, tuple)
            and isinstance(req.k_weight, tuple)
            and isinstance(req.cos, tuple)
            and isinstance(req.sin, tuple)
            and len(req.q_weight) == len(req.k_weight) == len(req.cos) == len(req.sin) == 3
        ):
            return None
        if req.q_weight[1] is not req.q_weight[2] or req.k_weight[1] is not req.k_weight[2]:
            return None
        q_flat, q_shape, q_was_flattened = self._flatten_heads(req.q)
        k_flat, k_shape, k_was_flattened = self._flatten_heads(req.k)
        tokens = int(q_flat.shape[0])
        if int(k_flat.shape[0]) != tokens:
            return None
        try:
            cos = tuple(self._align_rope_table(table, tokens) for table in req.cos)
            sin = tuple(self._align_rope_table(table, tokens) for table in req.sin)
        except RuntimeError:
            return None
        return q_flat, k_flat, q_shape, k_shape, q_was_flattened, k_was_flattened, cos, sin

    def _can_run_sensenova_3d(self, req: QKNormRopeReq) -> bool:
        inputs = self._sensenova_3d_inputs(req)
        if inputs is None:
            return False
        q_flat, k_flat, _q_shape, _k_shape, _q_flattened, _k_flattened, cos, sin = inputs
        can_run = import_module(
            "uniserve_worker.nn.rope"
        ).can_run_triton_sensenova_qk_rms_norm_rope_3d
        return bool(
            can_run(
                q_flat,
                k_flat,
                req.q_weight[0],
                req.q_weight[1],
                req.k_weight[0],
                req.k_weight[1],
                cos[0],
                sin[0],
                cos[1],
                sin[1],
                cos[2],
                sin[2],
            )
        )

    def _try_run_sensenova_3d(self, req: QKNormRopeReq):
        inputs = self._sensenova_3d_inputs(req)
        if inputs is None:
            return None
        q_flat, k_flat, q_shape, k_shape, q_was_flattened, k_was_flattened, cos, sin = inputs
        try_triton_sensenova = import_module(
            "uniserve_worker.nn.rope"
        ).try_triton_sensenova_qk_rms_norm_rope_3d
        out = try_triton_sensenova(
            q_flat,
            k_flat,
            req.q_weight[0],
            req.q_weight[1],
            req.k_weight[0],
            req.k_weight[1],
            cos[0],
            sin[0],
            cos[1],
            sin[1],
            cos[2],
            sin[2],
            req.eps,
            req.eps,
        )
        if out is None:
            return None
        q_out, k_out = out
        return (
            self._unflatten_heads(q_out, q_shape, q_was_flattened),
            self._unflatten_heads(k_out, k_shape, k_was_flattened),
        )

    def _run_multi_axis(self, req: QKNormRopeReq):
        sensenova_3d = self._try_run_sensenova_3d(req)
        if sensenova_3d is not None:
            return sensenova_3d
        q_parts = req.q.split(req.axis_dims, dim=-1)
        k_parts = req.k.split(req.axis_dims, dim=-1)
        out_q = []
        out_k = []
        try_triton_qk_rms_norm = import_module("uniserve_worker.nn.norm").try_triton_qk_rms_norm
        try_triton_qk_rms_norm_rope = import_module("uniserve_worker.nn.rope").try_triton_qk_rms_norm_rope
        packed_rope = import_module("uniserve_worker.nn.rope")._TritonPackedRope()
        axis = 0
        while axis < len(req.axis_dims):
            group_end = _EagerQKNormRopeProvider._shared_norm_group_end(req, axis)
            q_group, q_shape, q_was_flattened = self._flatten_heads(torch.cat(q_parts[axis:group_end], dim=-1))
            k_group, k_shape, k_was_flattened = self._flatten_heads(torch.cat(k_parts[axis:group_end], dim=-1))
            if group_end == axis + 1:
                cos = req.cos[axis]
                sin = req.sin[axis]
                cos_flat = self._align_rope_table(cos, int(q_group.shape[0]))
                sin_flat = self._align_rope_table(sin, int(q_group.shape[0]))
                out = try_triton_qk_rms_norm_rope(
                    q_group,
                    k_group,
                    req.q_weight[axis],
                    req.k_weight[axis],
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
                    req.q_weight[axis],
                    req.k_weight[axis],
                    req.eps,
                    req.eps,
                )
                if out is None:
                    raise RuntimeError("triton multi-axis qk_norm became ineligible")
                q_normed, k_normed = out
                q_normed_parts = q_normed.split(req.axis_dims[axis:group_end], dim=-1)
                k_normed_parts = k_normed.split(req.axis_dims[axis:group_end], dim=-1)
                for local, (q_normed_part, k_normed_part) in enumerate(zip(q_normed_parts, k_normed_parts, strict=True)):
                    cos = req.cos[axis + local]
                    sin = req.sin[axis + local]
                    cos_flat = self._align_rope_table(cos, int(q_normed_part.shape[0]))
                    sin_flat = self._align_rope_table(sin, int(q_normed_part.shape[0]))
                    q_rot = packed_rope.run(q_normed_part, cos_flat, sin_flat)
                    k_rot = packed_rope.run(k_normed_part, cos_flat, sin_flat)
                    part_shape = (*q_shape[:-1], int(req.axis_dims[axis + local]))
                    out_q.append(self._unflatten_heads(q_rot, part_shape, q_was_flattened))
                    out_k.append(self._unflatten_heads(k_rot, part_shape, k_was_flattened))
            axis = group_end
        return torch.cat(out_q, dim=-1), torch.cat(out_k, dim=-1)


class _EagerQKNormRopeProvider:
    name = "eager"
    operator = "qk_norm_rope"

    def capabilities(self) -> Capabilities:
        return Capabilities(tags=frozenset({"eager", "qk_norm_rope"}))

    def can_run(self, req: QKNormRopeReq) -> bool:
        return True

    @staticmethod
    def _validate_multi_axis(req: QKNormRopeReq) -> None:
        if not isinstance(req.q_weight, tuple) or not isinstance(req.k_weight, tuple):
            raise RuntimeError("multi-axis qk_norm_rope requires tuple weights")
        if not isinstance(req.cos, tuple) or not isinstance(req.sin, tuple):
            raise RuntimeError("multi-axis qk_norm_rope requires tuple cos/sin")
        if len(req.axis_dims) != len(req.cos) or len(req.cos) != len(req.sin):
            raise RuntimeError("multi-axis qk_norm_rope axis/cos/sin mismatch")
        if len(req.q_weight) != len(req.axis_dims) or len(req.k_weight) != len(req.axis_dims):
            raise RuntimeError("multi-axis qk_norm_rope weight/axis mismatch")

    @staticmethod
    def _shared_norm_group_end(req: QKNormRopeReq, start: int) -> int:
        q_weight = req.q_weight[start]
        k_weight = req.k_weight[start]
        group_end = start + 1
        group_dim = int(req.axis_dims[start])
        while (
            group_end < len(req.axis_dims)
            and req.q_weight[group_end] is q_weight
            and req.k_weight[group_end] is k_weight
            and group_dim < int(q_weight.shape[-1])
        ):
            group_dim += int(req.axis_dims[group_end])
            group_end += 1
        return group_end if group_dim == int(q_weight.shape[-1]) else start + 1

    @staticmethod
    def _apply_axis_rope(q_normed, k_normed, cos, sin, *, apply_rotary_pos_emb, apply_rotary_emb, unsqueeze_dim):
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
                    out[..., :half] = x[..., :half] * cos_view[..., :half] - x[..., half:dim] * sin_view[..., :half]
                    out[..., half:dim] = x[..., half:dim] * cos_view[..., half:dim] + x[..., :half] * sin_view[..., half:dim]
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
        self._validate_multi_axis(req)
        q_parts = req.q.split(req.axis_dims, dim=-1)
        k_parts = req.k.split(req.axis_dims, dim=-1)
        out_q = []
        out_k = []
        eager_norm = _EagerQKNormProvider()
        axis = 0
        while axis < len(req.axis_dims):
            group_end = self._shared_norm_group_end(req, axis)
            q_group = torch.cat(q_parts[axis:group_end], dim=-1)
            k_group = torch.cat(k_parts[axis:group_end], dim=-1)
            q_normed_group, k_normed_group = eager_norm.run(
                QKNormReq(q_group, k_group, req.q_weight[axis], req.k_weight[axis], req.eps)
            )
            q_normed_parts = q_normed_group.split(req.axis_dims[axis:group_end], dim=-1)
            k_normed_parts = k_normed_group.split(req.axis_dims[axis:group_end], dim=-1)
            for local, (q_normed, k_normed) in enumerate(zip(q_normed_parts, k_normed_parts, strict=True)):
                q_rot, k_rot = self._apply_axis_rope(
                    q_normed,
                    k_normed,
                    req.cos[axis + local],
                    req.sin[axis + local],
                    apply_rotary_pos_emb=apply_rotary_pos_emb,
                    apply_rotary_emb=apply_rotary_emb,
                    unsqueeze_dim=req.unsqueeze_dim,
                )
                out_q.append(q_rot)
                out_k.append(k_rot)
            axis = group_end
        return torch.cat(out_q, dim=-1), torch.cat(out_k, dim=-1)

    def run(self, req: QKNormRopeReq):
        rope_mod = import_module("uniserve_worker.nn.rope")
        apply_rotary_pos_emb = rope_mod.apply_rotary_pos_emb
        apply_rotary_emb = rope_mod.apply_rotary_emb

        if req.axis_dims is not None:
            return self._run_multi_axis(req, apply_rotary_pos_emb=apply_rotary_pos_emb, apply_rotary_emb=apply_rotary_emb)

        q, k = _EagerQKNormProvider().run(QKNormReq(req.q, req.k, req.q_weight, req.k_weight, req.eps))
        return apply_rotary_pos_emb(q, k, req.cos, req.sin, req.position_ids, req.unsqueeze_dim)


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

    def capabilities(self) -> Capabilities:
        caps = self.backend.capabilities()
        return Capabilities(
            tags=frozenset({self.name, "attention"}),
            attrs={
                "paged_kv": bool(getattr(caps, "paged_kv", False)),
                "varlen_attention": bool(getattr(caps, "varlen_attention", False)),
                "varlen_paged_kv": bool(getattr(caps, "varlen_paged_kv", False)),
                "visible_end": bool(getattr(caps, "visible_end", False)),
                "trunk_geometries": getattr(caps, "trunk_geometries", frozenset()),
                "paged_block_size_multiple": int(getattr(caps, "paged_block_size_multiple", 1) or 1),
                "min_head_dim": int(getattr(caps, "min_head_dim", 1) or 1),
                "paged_decode_only": bool(getattr(caps, "paged_decode_only", False)),
            },
        )

    def display_name(self, req: AttentionReq) -> str:
        if req.block_table is not None and req.cu_seqlens_q is not None and req.cu_seqlens_k is not None:
            return f"{self.name}_paged_varlen"
        return self.name

    @staticmethod
    def _head_dim_supported(caps, head_dim: int) -> bool:
        return int(head_dim) >= int(getattr(caps, "min_head_dim", 1) or 1)

    @staticmethod
    def _trunk_geometry_supported(caps, req: AttentionReq, *, backend_name: str | None = None) -> bool:
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
        pool = getattr(req.kv_cache, "pool", None)
        if pool is not None and not bool(getattr(pool, "supports_paged_attention_storage", True)):
            return False
        if pool is None:
            return True
        block_size = int(getattr(pool, "block_size", 0) or 0)
        if block_size <= 0:
            return False
        multiple = int(getattr(caps, "paged_block_size_multiple", 1) or 1)
        return block_size % max(1, multiple) == 0

    @staticmethod
    def _is_one_token_decode(req: AttentionReq) -> bool:
        if req.q.ndim == 3:
            metadata = getattr(req.ctx, "attention_metadata", None) if req.ctx is not None else None
            query_lens = getattr(metadata, "query_lens_cpu", ()) or ()
            if getattr(metadata, "mode", None) == "decode" and len(query_lens) == int(req.q.shape[0]):
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
        if self.name in {"sgl_kernel", "flashinfer", "flash_attn", "fa4_cute"} and req.q.device.type != "cuda":
            return False
        if not self._head_dim_supported(caps, req.q.shape[-1]):
            return False
        if not self._trunk_geometry_supported(caps, req, backend_name=getattr(self.backend, "name", self.name)):
            return False
        if req.regime is AttentionRegime.VISIBLE_END:
            return bool(getattr(caps, "visible_end", False)) and req.visible_end is not None
        if req.regime in {AttentionRegime.EXTEND, AttentionRegime.MIXED} and req.cu_seqlens_q is not None:
            if req.block_table is not None:
                return (
                    bool(getattr(caps, "varlen_attention", False))
                    and bool(getattr(caps, "varlen_paged_kv", False))
                    and self._paged_storage_supported(caps, req)
                )
            return bool(getattr(caps, "varlen_attention", False))
        if req.regime is AttentionRegime.DECODE or req.block_table is not None or req.kv_cache is not None:
            if bool(getattr(caps, "paged_decode_only", False)) and not self._is_one_token_decode(req):
                return False
            return (
                bool(getattr(caps, "paged_kv", False))
                and req.block_table is not None
                and req.cache_seqlens is not None
                and self._paged_storage_supported(caps, req)
            )
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
            )
        if req.cu_seqlens_q is not None and req.cu_seqlens_k is not None and hasattr(self.backend, "forward_varlen"):
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
            )
        if req.block_table is not None and req.cache_seqlens is not None and hasattr(self.backend, "forward_paged"):
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
            )
        return self.backend.forward(req.q, req.k, req.v, causal=req.causal, scale=req.scale, attn_mask=req.attn_mask)


class _ContextAttentionProvider(_AttentionBackendProvider):
    name = "context"

    def __init__(self) -> None:
        self.backend = None

    def _backend(self, req: AttentionReq):
        if req.backend is not None:
            return req.backend
        return getattr(req.ctx, "attention_backend", None) if req.ctx is not None else None

    def display_name(self, req: AttentionReq) -> str:
        backend = self._backend(req)
        name = str(getattr(backend, "name", self.name))
        if req.block_table is not None and req.cu_seqlens_q is not None and req.cu_seqlens_k is not None:
            return f"{name}_paged_varlen"
        return name

    def capabilities(self) -> Capabilities:
        return Capabilities(tags=frozenset({"context", "attention"}))

    def can_run(self, req: AttentionReq) -> bool:
        backend = self._backend(req)
        if backend is None:
            return False
        return _AttentionBackendProvider(backend).can_run(req)

    def run(self, req: AttentionReq):
        backend = self._backend(req)
        if backend is None:
            raise RuntimeError("context attention backend disappeared")
        return _AttentionBackendProvider(backend).run(req)


@lru_cache(maxsize=1)
def attention_dispatcher():
    attention_pkg = import_module("uniserve_worker.backends.attention")
    get_attention_backend = attention_pkg.get_attention_backend
    init_attention_backends = attention_pkg.init_attention_backends

    init_attention_backends()
    names = ("sgl_kernel", "flashinfer", "flash_attn", "fa4_cute", "torch_sdpa")
    providers = [_ContextAttentionProvider()]
    for name in names:
        try:
            providers.append(_AttentionBackendProvider(get_attention_backend(name)))
        except Exception:
            continue
    if not any(provider.name == "torch_sdpa" for provider in providers):
        providers.append(_AttentionBackendProvider(get_attention_backend("torch_sdpa")))
    return Dispatcher(
        "attention",
        providers,
        env_override="UNISERVE_ATTENTION_PROVIDER",
        fallback_names=("torch_sdpa",),
    )


class _StandardTpAllReduceProvider:
    name = "standard"
    operator = "tp_all_reduce"

    def capabilities(self) -> Capabilities:
        return Capabilities(tags=frozenset({self.name, self.operator}))

    def can_dispatch(self, req: TpAllReduceReq, *, mesh=None) -> bool:
        del mesh
        return getattr(req.axis, "transport", None) is not None

    def dispatch(self, req: TpAllReduceReq, *, mesh=None) -> Handoff:
        del mesh
        transport = getattr(req.axis, "transport", None)
        if transport is None:
            raise RuntimeError("tp_all_reduce requires a transport-bound mesh axis")
        return Handoff(
            format="tensor",
            payload=transport.all_reduce(req.tensor, req.op),
            metadata={"axis": getattr(req.axis, "name", "tp"), "op": req.op},
        )

    def can_combine(self, handoff: Handoff, *, mesh=None) -> bool:
        del mesh
        return handoff.format == "tensor"

    def combine(self, handoff: Handoff, *, mesh=None):
        del mesh
        return handoff.payload


@lru_cache(maxsize=1)
def tp_all_reduce_dispatcher():
    return CommDispatcher("tp_all_reduce", [_StandardTpAllReduceProvider()])
