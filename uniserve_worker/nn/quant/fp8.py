"""FP8 linear quantization method with a dequantized correctness floor."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...foundation.errors import model_execution_error
from ..placement import place_partitioned_tensor
from .base import QuantizeMethodBase
from .kv_cache import fp8_quantize, fp8_scale_from
from .load_state import (
    Fp8LoadPhase,
    copy_load_state,
    fp8_load_phase,
    init_fp8_phase,
    set_fp8_scale_loaded,
    set_fp8_weight_loaded_offline,
    set_optional_checkpoint,
    set_skip_serving_cast,
    set_weight_loader,
)

__all__ = [
    'W8A8Fp8LinearMethod',
]

# ``torch._scaled_mm`` requires the contraction (K) dim aligned to this; fixed by
# the fp8 kernel/ABI build.
_FP8_BLOCK_ALIGNMENT = 16


class W8A8Fp8LinearMethod(QuantizeMethodBase):
    """Per-channel FP8 weights and per-token dynamic FP8 activations.

    The CUDA fast path uses ``torch._scaled_mm`` when the runtime shape is
    supported.  All other cases use an explicit dequantized matmul, which keeps
    the method correct on CPU, unsupported shapes, and older CUDA stacks.
    """

    is_quantized = True

    def create_weights(
        self,
        module: nn.Module,
        *,
        input_size: int,
        output_size: int,
        bias: bool,
        **_: object,
    ) -> None:
        module.register_parameter(
            "weight",
            nn.Parameter(torch.empty(int(output_size), int(input_size)), requires_grad=False),
        )
        module.register_parameter(
            "bias",
            nn.Parameter(torch.empty(int(output_size)), requires_grad=False) if bias else None,
        )
        module.register_parameter(
            "weight_scale",
            nn.Parameter(torch.ones(int(output_size), 1, dtype=torch.float32), requires_grad=False),
        )
        set_weight_loader(module.weight, _fp8_weight_loader(module))
        set_weight_loader(module.weight_scale, _fp8_scale_loader(module))
        set_optional_checkpoint(module.weight_scale, True)
        set_skip_serving_cast(module.weight_scale, True)
        init_fp8_phase(module)

    def process_weights_after_loading(self, module: nn.Module) -> None:
        weight = getattr(module, "weight")
        if weight.dtype == torch.float8_e4m3fn:
            # An offline fp8 checkpoint weight can only be finalized once its
            # matching scale has been loaded; the load lifecycle must have
            # reached SCALE_LOADED before this finalize step runs.
            if fp8_load_phase(module) is not Fp8LoadPhase.SCALE_LOADED:
                raise model_execution_error(
                    "FP8 checkpoint weight requires a loaded weight_scale tensor"
                )
            weight.data = weight.data.contiguous()
            module.weight_scale.data = _canonical_scale(module.weight_scale.data, weight.shape[0]).to(
                device=weight.device,
                dtype=torch.float32,
            )
            return

        dense = weight.detach().to(torch.float32)
        scale = fp8_scale_from(dense, dim=1)
        quantized = fp8_quantize(dense, scale)
        fp8_weight = nn.Parameter(quantized.contiguous(), requires_grad=False)
        copy_load_state(weight, fp8_weight)
        set_skip_serving_cast(fp8_weight, True)
        module.weight = fp8_weight
        module.weight_scale.data = scale.to(device=fp8_weight.device, dtype=torch.float32)
        set_fp8_scale_loaded(module, True)

    def apply(self, module: nn.Module, x: torch.Tensor) -> torch.Tensor:
        weight = getattr(module, "weight")
        if weight.dtype != torch.float8_e4m3fn:
            return F.linear(x, weight, getattr(module, "bias", None))
        out = _apply_fp8_linear(x, weight, module.weight_scale, getattr(module, "bias", None))
        return out


def _apply_fp8_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    original_shape = x.shape[:-1]
    x_2d = x.reshape(-1, x.shape[-1])
    if _can_use_scaled_mm(x_2d, weight):
        out = _apply_scaled_mm(x_2d, weight, weight_scale, x.dtype)
    else:
        out = _apply_dequantized(x_2d, weight, weight_scale, x.dtype)
    if bias is not None:
        out = out + bias.to(device=out.device, dtype=out.dtype)
    return out.reshape(*original_shape, weight.shape[0])


def _apply_scaled_mm(
    x_2d: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    x_float = x_2d.to(torch.float32)
    act_scale = fp8_scale_from(x_float, dim=1)
    x_fp8 = fp8_quantize(x_float, act_scale)
    scale_b = _canonical_scale(weight_scale, weight.shape[0]).t().contiguous()
    return torch._scaled_mm(  # type: ignore[attr-defined]
        x_fp8,
        weight.t(),
        scale_a=act_scale,
        scale_b=scale_b,
        out_dtype=_scaled_mm_output_dtype(out_dtype),
    )


def _apply_dequantized(
    x_2d: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    dequant_weight = weight.to(torch.float32) * _canonical_scale(weight_scale, weight.shape[0])
    out = F.linear(x_2d.to(torch.float32), dequant_weight)
    if out_dtype in {torch.float16, torch.bfloat16, torch.float32}:
        return out.to(dtype=out_dtype)
    return out


def _can_use_scaled_mm(x_2d: torch.Tensor, weight: torch.Tensor) -> bool:
    return (
        x_2d.is_cuda
        and weight.is_cuda
        and hasattr(torch, "_scaled_mm")
        and x_2d.dtype in {torch.float16, torch.bfloat16}
        and x_2d.ndim == 2
        and weight.ndim == 2
        and int(x_2d.shape[1]) == int(weight.shape[1])
        and int(x_2d.shape[1]) % _FP8_BLOCK_ALIGNMENT == 0
    )


def _scaled_mm_output_dtype(dtype: torch.dtype) -> torch.dtype:
    if dtype in {torch.float16, torch.bfloat16}:
        return dtype
    return torch.bfloat16


def _canonical_scale(scale: torch.Tensor, output_size: int) -> torch.Tensor:
    if scale.ndim == 1:
        scale = scale.reshape(-1, 1)
    if scale.shape == (1, int(output_size)):
        scale = scale.t()
    if scale.shape != (int(output_size), 1):
        raise ValueError(f"FP8 weight_scale shape {tuple(scale.shape)} != ({int(output_size)}, 1)")
    return scale.to(torch.float32)


def _fp8_weight_loader(module: nn.Module):
    def load(param: nn.Parameter, loaded_weight: torch.Tensor, *, shard_id=None) -> None:
        if loaded_weight.dtype == torch.float8_e4m3fn:
            if param.dtype != torch.float8_e4m3fn:
                fp8_param = nn.Parameter(
                    torch.empty_like(param, dtype=torch.float8_e4m3fn),
                    requires_grad=False,
                )
                copy_load_state(param, fp8_param)
                module.weight = fp8_param
                param = fp8_param
            place_partitioned_tensor(param, param.data, loaded_weight, shard_id=shard_id)
            set_fp8_weight_loaded_offline(module, True)
            return
        place_partitioned_tensor(param, param.data, loaded_weight.to(dtype=param.dtype), shard_id=shard_id)
        set_fp8_weight_loaded_offline(module, False)

    return load


def _fp8_scale_loader(module: nn.Module):
    def load(param: nn.Parameter, loaded_scale: torch.Tensor, *, shard_id=None) -> None:
        target = param.data
        canonical = loaded_scale
        if canonical.ndim == 1:
            canonical = canonical.reshape(-1, 1)
        place_partitioned_tensor(
            param, target, canonical.to(device=target.device, dtype=target.dtype), shard_id=shard_id
        )
        set_fp8_scale_loaded(module, True)

    return load
