"""FlashInfer CUTLASS grouped expert kernels over resident dense experts.

The kernels consume the ``FusedMoE`` resident layout directly: the up rows
of each expert precede its gate rows in ``up_gate``. BF16 and FP16 weights
are supported, with SwiGLU or tanh-GELU gating. Routing weights multiply
after the down projection, and the per-token combination happens inside the
kernel. Encoded experts are not served here: the CUTLASS NVFP4 path stores
its FC1 output and gated product in BF16 before encoding the FC2 input,
which the NVFP4 reference keeps in FP32; ``cutedsl`` and ``trtllm`` serve
NVFP4 experts.
"""

from __future__ import annotations

import torch

from uniserve.quantization import QuantizedTensor
from uniserve.tensors import BufferConfig

from . import Backend as _Backend
from . import Operator as _Operator


def _activation(name):
    from flashinfer.fused_moe import ActivationType

    return ActivationType.Swiglu if name == "silu" else ActivationType.GegluTanh


class _Cutlass(_Operator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        from flashinfer.fused_moe import cutlass_fused_moe

        self._kernel = cutlass_fused_moe
        self._activation_type = _activation(self.module.activation)

    def __call__(self, hidden, topk_ids, topk_weights):
        self._validate(hidden, topk_ids, topk_weights)
        output = torch.empty_like(hidden)
        if not hidden.shape[0]:
            return output
        module = self.module
        self._kernel(
            input=hidden,
            token_selected_experts=topk_ids,
            token_final_scales=topk_weights,
            fc1_expert_weights=module.up_gate.weight,
            fc2_expert_weights=module.down.weight,
            output_dtype=hidden.dtype,
            quant_scales=[],
            output=output,
            activation_type=self._activation_type,
            tune_max_num_tokens=self.size.num_tokens,
            workspace_buffer=self.workspace["scratch"],
        )
        return output


class Backend(_Backend):
    name = "cutlass"
    operator_class = _Cutlass

    def unsupported(self, module) -> str | None:
        weight = module.up_gate.weight
        if weight.device.type != "cuda":
            return "the kernels run on CUDA devices"
        if torch.cuda.get_device_capability(weight.device)[0] != 10:
            return "the kernels are built for SM100-class devices"
        if isinstance(weight, QuantizedTensor) or isinstance(
            module.down.weight, QuantizedTensor
        ):
            return "encoded experts are served by the NVFP4 expert providers"
        if weight.dtype not in {torch.bfloat16, torch.float16}:
            return f"expert weight dtype {weight.dtype} is not BF16 or FP16"
        return None

    def workspace_buffers(self, *, module, size):
        from flashinfer.fused_moe import cutlass_fused_moe_workspace_size

        weight = module.up_gate.weight
        nbytes = cutlass_fused_moe_workspace_size(
            max(1, size.num_tokens),
            module.hidden_size,
            module.down.weight.shape[-1],
            module.num_experts,
            module.top_k,
            x_dtype=weight.dtype,
            weight_dtype=weight.dtype,
            output_dtype=weight.dtype,
            activation_type=_activation(module.activation),
            device=weight.device,
        )
        return {"scratch": BufferConfig((nbytes,), torch.uint8)}
