"""TensorRT-LLM generated (trtllm-gen) grouped kernels for NVFP4 experts.

The provider serves W4A4 NVFP4 experts whose activations use the static
calibrated scales of ``up_gate.input_quantizer`` and
``down.input_quantizer``, with SiLU (``Swiglu``) or tanh-approximated GELU
(``Geglu``) gating, over routing the model has already computed. One call
encodes the hidden states to NVFP4 with FlashInfer's ``fp4_quantize`` and
launches ``trtllm_fp4_block_scale_routed_moe``, which permutes the routed
tokens, runs the gated FC1 GEMM with FP32 accumulation, encodes the gated
product to NVFP4 inside its epilogue, runs FC2, and combines each token's
routes with its FP32 route weights into the BF16 output.

The kernels read expert weights in TensorRT-LLM's shuffled row order:
``up_gate`` with its up and gate halves interleaved and shuffled
(``RowOrder.INTERLEAVED_SHUFFLED_128``, so each gated output channel reads
its up row as the linear input and its gate row as the activated one), and
``down`` shuffled (``RowOrder.SHUFFLED_128``); block scales stay 128x4
swizzled over each expert's rows. Preparing the provider places a module's
weights in that order once, replacing its parameters with the rearranged
encoding; the logical weights are unchanged, so every provider and the
portable reference decode them identically, and only one copy stays
resident. The intermediate width is read unpadded.

Every buffer of a call comes from PyTorch's caching allocator. FlashInfer's
public calls take no caller workspace and allocate the encoded hidden
states and their block scales, the routing permutation and per-expert tile
metadata, the FC1 and FC2 intermediates and both GEMM work areas
themselves; the provider allocates the output the caller receives. Eager
calls return them to the allocator; a CUDA graph capture draws them from
the graph's private pool, which keeps them at fixed addresses for every
replay. Nothing is context-owned, so ``workspace_buffers`` is empty. Kernel
selection for a token count happens on the host at call time from
FlashInfer's tactic cache; capture must follow an eager call at the
captured token count, as for every prepared operator.
"""

from __future__ import annotations

import torch

from uniserve.quantization import RowOrder

from . import NVFP4Backend
from . import Operator as _Operator


class _TrtllmGen(_Operator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        from flashinfer import fp4_quantize
        from flashinfer.fused_moe import (
            ActivationType,
            trtllm_fp4_block_scale_routed_moe,
        )
        from flashinfer.tllm_enums import RoutingMethodType

        self._quantize = fp4_quantize
        self._kernel = trtllm_fp4_block_scale_routed_moe
        module = self.module
        self._activation = (
            ActivationType.Swiglu
            if module.activation == "silu"
            else ActivationType.Geglu
        ).value
        # Routing arrives precomputed; the kernel reads its ids and FP32
        # weights as given. No routing method applies a transform to them.
        self._routing = RoutingMethodType.TopK.value

        up_gate, down = module.up_gate.weight, module.down.weight
        experts, rows, hidden = up_gate.shape
        intermediate = down.shape[2]
        fields13, fields2 = up_gate.buffers(), down.buffers()
        self._fc1 = fields13["values"]
        self._fc2 = fields2["values"]
        # Each expert's swizzled scales occupy whole 128x4 tiles, so the
        # stacked bytes are the [E, rows, K / 16] E4M3 tensors the kernels
        # index per expert.
        self._fc1_scale = (
            fields13["block_scale"]
            .view(torch.float8_e4m3fn)
            .reshape(experts, rows, hidden // 16)
        )
        self._fc2_scale = (
            fields2["block_scale"]
            .view(torch.float8_e4m3fn)
            .reshape(experts, hidden, intermediate // 16)
        )

        # FP32 scale arithmetic on the device, as the kernels consume it. A
        # static activation scale s encodes x / s; the input encoding takes
        # the global scale 1 / s13. FC1 results carry the product of the
        # weight tensor scale and s13 (alpha13): the gate is dequantized
        # with alpha13 before its nonlinearity, and the linear half with
        # alpha13 / s2, which also encodes the gated product for FC2. FC2
        # results are dequantized with the down tensor scale times s2.
        device = up_gate.device
        input13 = torch.tensor(
            [module.up_gate.input_quantizer.calibrated_scale],
            dtype=torch.float32,
            device=device,
        )
        input2 = torch.tensor(
            [module.down.input_quantizer.calibrated_scale],
            dtype=torch.float32,
            device=device,
        )
        alpha13 = fields13["tensor_scale"] * input13
        self._input_scale = input13.reciprocal()
        self._gate_scale = alpha13
        self._linear_scale = alpha13 * input2.reciprocal()
        self._down_scale = fields2["tensor_scale"] * input2
        self._experts, self._intermediate = experts, intermediate

    def _validate(self, hidden, topk_ids, topk_weights):
        super()._validate(hidden, topk_ids, topk_weights)
        if hidden.dtype != torch.bfloat16:
            raise ValueError(
                "trtllm-gen NVFP4 experts read and write BF16 hidden states"
            )

    def __call__(self, hidden, topk_ids, topk_weights):
        self._validate(hidden, topk_ids, topk_weights)
        output = torch.empty_like(hidden)
        tokens = hidden.shape[0]
        if not tokens:
            return output

        # The kernels read the linear token-major [T, H / 16] block scales.
        values, scales = self._quantize(
            hidden.contiguous(),
            global_scale=self._input_scale,
            is_sf_swizzled_layout=False,
        )
        scales = scales.view(torch.float8_e4m3fn).reshape(
            tokens, hidden.shape[1] // 16
        )
        self._kernel(
            topk_ids=(topk_ids.contiguous(), topk_weights.contiguous()),
            routing_bias=None,
            hidden_states=values,
            hidden_states_scale=scales,
            gemm1_weights=self._fc1,
            gemm1_weights_scale=self._fc1_scale,
            gemm1_bias=None,
            gemm1_alpha=None,
            gemm1_beta=None,
            gemm1_clamp_limit=None,
            gemm2_weights=self._fc2,
            gemm2_weights_scale=self._fc2_scale,
            gemm2_bias=None,
            output1_scale_scalar=self._linear_scale,
            output1_scale_gate_scalar=self._gate_scale,
            output2_scale_scalar=self._down_scale,
            num_experts=self.module.num_experts,
            top_k=topk_ids.shape[1],
            n_group=None,
            topk_group=None,
            intermediate_size=self._intermediate,
            local_expert_offset=self.module.expert_slice.start,
            local_num_experts=self._experts,
            routed_scaling_factor=None,
            routing_method_type=self._routing,
            activation_type=self._activation,
            output=output,
            tune_max_num_tokens=self.size.num_tokens,
        )
        return output

    def close(self) -> None:
        super().close()
        # Release the borrowed weight fields and derived scales, so a closed
        # operator never keeps a replaced encoding resident.
        self._fc1 = self._fc2 = self._fc1_scale = self._fc2_scale = None
        self._input_scale = self._gate_scale = None
        self._linear_scale = self._down_scale = None


class Backend(NVFP4Backend):
    name = "trtllm"
    operator_class = _TrtllmGen
    up_gate_order = RowOrder.INTERLEAVED_SHUFFLED_128
    down_order = RowOrder.SHUFFLED_128

    def kernels_unsupported(self, device) -> str | None:
        if torch.cuda.get_device_capability(device)[0] != 10:
            return "the kernels are built for SM100-class devices"
        return None

    def invalid_expert(self, module) -> int:
        # The routed kernels mark an absent route with -1.
        return -1
