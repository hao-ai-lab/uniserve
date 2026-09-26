"""FlashInfer CUTLASS grouped expert kernels over resident expert weights.

The kernels consume the ``FusedMoE`` resident layout directly: the up rows
of each expert precede its gate rows in ``up_gate``, and NVFP4 block scales
are stored in the 128x4 swizzled layout. BF16 weights and NVFP4 W4A4 weights
with static activation scales are supported, with SwiGLU or tanh-GELU
gating. Routing weights multiply after the down projection, and the
per-token combination happens inside the kernel.
"""

from __future__ import annotations

import torch

from uniserve.quantization import QuantizedTensor, ScaleLayout
from uniserve.tensors import BufferConfig

from . import Backend as _Backend
from . import Operator as _Operator


def _activation(name):
    from flashinfer.fused_moe import ActivationType

    return ActivationType.Swiglu if name == "silu" else ActivationType.GegluTanh


def _nvfp4(module) -> bool:
    return isinstance(module.up_gate.weight, QuantizedTensor)


class _Cutlass(_Operator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        from flashinfer.fused_moe import cutlass_fused_moe

        self._kernel = cutlass_fused_moe
        module = self.module
        self._activation_type = _activation(module.activation)
        up_gate, down = module.up_gate.weight, module.down.weight
        if not _nvfp4(module):
            self._fc1, self._fc2, self._scales = up_gate, down, []
            return

        # W4A4: values are packed E2M1 pairs; the kernel reads them as int64
        # words. Block scales keep their swizzled bytes, viewed as the
        # kernel's [E, rows, K/64] int32 words. Per-expert dequantization
        # factors fold the static activation scale into the weight tensor
        # scale; the activation global scales are their reciprocals.
        experts, rows, width = up_gate.shape
        fields13, fields2 = up_gate.buffers(), down.buffers()
        input13 = module.up_gate.input_quantizer.calibrated_scale
        input2 = module.down.input_quantizer.calibrated_scale
        device = up_gate.device
        self._fc1 = fields13["values"].view(torch.long)
        self._fc2 = fields2["values"].view(torch.long)
        self._scales = [
            torch.tensor(1.0 / input13, dtype=torch.float32, device=device),
            fields13["block_scale"]
            .view(torch.int32)
            .reshape(experts, rows, width // 64),
            fields13["tensor_scale"] * input13,
            torch.tensor(1.0 / input2, dtype=torch.float32, device=device),
            fields2["block_scale"]
            .view(torch.int32)
            .reshape(experts, down.shape[1], down.shape[2] // 64),
            fields2["tensor_scale"] * input2,
        ]

    def __call__(self, hidden, topk_ids, topk_weights):
        self._validate(hidden, topk_ids, topk_weights)
        output = torch.empty_like(hidden)
        if not hidden.shape[0]:
            return output
        self._kernel(
            input=hidden,
            token_selected_experts=topk_ids,
            token_final_scales=topk_weights,
            fc1_expert_weights=self._fc1,
            fc2_expert_weights=self._fc2,
            output_dtype=hidden.dtype,
            quant_scales=self._scales,
            output=output,
            activation_type=self._activation_type,
            tune_max_num_tokens=self.size.num_tokens,
            workspace_buffer=self.workspace["scratch"],
        )
        return output


class Backend(_Backend):
    name = "cutlass"
    operator_class = _Cutlass

    def supports(self, module) -> bool:
        weight = module.up_gate.weight
        if weight.device.type != "cuda":
            return False
        if torch.cuda.get_device_capability(weight.device)[0] != 10:
            return False
        if isinstance(weight, QuantizedTensor):
            down = module.down.weight
            # The kernel indexes each expert's swizzled scales as whole
            # 128-row tiles of 32-bit words over K/64, which the stacked
            # swizzled layout provides when rows and K are so aligned.
            return (
                weight.quantizer.format == "nvfp4"
                and weight.scale_layout is ScaleLayout.SWIZZLED_128X4
                and isinstance(down, QuantizedTensor)
                and down.quantizer.format == "nvfp4"
                and down.scale_layout is ScaleLayout.SWIZZLED_128X4
                and weight.shape[1] % 128 == 0
                and down.shape[1] % 128 == 0
                and weight.shape[2] % 64 == 0
                and down.shape[2] % 64 == 0
                and module.up_gate.input_quantizer is not None
                and module.down.input_quantizer is not None
            )
        return weight.dtype in {torch.bfloat16, torch.float16}

    def workspace_buffers(self, *, module, size):
        from flashinfer.fused_moe import cutlass_fused_moe_workspace_size

        weight = module.up_gate.weight
        nbytes = cutlass_fused_moe_workspace_size(
            max(1, size.num_tokens),
            module.hidden_size,
            module.down.weight.shape[-1],
            module.num_experts,
            module.top_k,
            x_dtype=torch.bfloat16 if _nvfp4(module) else weight.dtype,
            weight_dtype=torch.long if _nvfp4(module) else weight.dtype,
            output_dtype=torch.bfloat16 if _nvfp4(module) else weight.dtype,
            activation_type=_activation(module.activation),
            device=weight.device,
        )
        return {"scratch": BufferConfig((nbytes,), torch.uint8)}
