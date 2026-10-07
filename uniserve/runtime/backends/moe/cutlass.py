"""FlashInfer CUTLASS grouped expert kernels over resident dense experts.

The kernels consume the ``FusedMoE`` resident layout directly: the up rows
of each expert precede its gate rows in ``up_gate``. BF16 and FP16 weights
are supported, with SwiGLU or tanh-GELU gating. Routing weights multiply
after the down projection, and the per-token combination happens inside the
kernel. Encoded experts are not served here: the CUTLASS NVFP4 path stores
its FC1 output and gated product in BF16 before encoding the FC2 input,
which the NVFP4 reference keeps in FP32; ``cutedsl`` and ``trtllm`` serve
NVFP4 experts.

The provider uses only FlashInfer's separate combination
(``use_fused_finalize=False``): after FC2 a final pass sums every token's
routes in a fixed order, weighted by their FP32 route weights, so repeated
calls of the same inputs, eager or replayed, give bitwise identical outputs.
The runner's configuration list then excludes the GEMM2 epilogues that
combine routes while storing them (the finalize fusion); with those
selected, replays of the same inputs returned outputs other than the eager
call's.

Each call runs the GEMM1 and GEMM2 configurations measured best for its
token count on this device and FlashInfer version (``tactics.json``, see
``tactics``), and FlashInfer's default configurations for shapes the table
does not cover. The measurement admits only configurations whose outputs
equal the defaults' bit for bit, so the table changes device time, not
results.
"""

from __future__ import annotations

import torch

from uniserve.quantization import QuantizedTensor
from uniserve.tensors import BufferConfig

from . import Backend as _Backend
from . import CombiningOperator
from .tactics import Tactics


def _activation(name):
    from flashinfer.fused_moe import ActivationType

    return ActivationType.Swiglu if name == "silu" else ActivationType.GegluTanh


class _Cutlass(CombiningOperator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        from flashinfer.fused_moe import cutlass_fused_moe

        self._kernel = cutlass_fused_moe
        self._activation_type = _activation(self.module.activation)
        self._tactics = Tactics(Backend.name, self.module)

    def __call__(
        self, hidden, topk_ids, topk_weights, *, combine=True, tactic=None
    ):
        """Evaluate the routed experts; ``tactic`` overrides the table.

        A tactic is ``[gemm1, gemm2]``: indices into the runner's combined
        configuration list, ``-1`` for a GEMM's default. The kernels store
        combined rows, returned uncombined as one-route ``Routes``.
        """
        self._validate(hidden, topk_ids, topk_weights)
        output = torch.empty_like(hidden)
        tokens = hidden.shape[0]
        if not tokens:
            return self._result(output, combine)
        if tactic is None:
            tactic = self._tactics.select(tokens)
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
            use_fused_finalize=False,
            tune_max_num_tokens=self.size.num_tokens,
            profile_ids=tactic,
            workspace_buffer=self.workspace["scratch"],
            # Expert-parallel ranks hold the contiguous global experts of
            # their group rank; ids of other ranks' experts are skipped.
            ep_size=module.expert_group.size,
            ep_rank=module.expert_group.rank,
        )
        return self._result(output, combine)

    def tactic_space(self) -> list[list[int]]:
        """GEMM1 and GEMM2 configuration indices, each default (-1) first.

        GEMM2 indices follow GEMM1's in the runner's combined list.
        """
        from flashinfer.fused_moe.core import get_cutlass_fused_moe_module

        # The runner FlashInfer caches for this provider's calls (the same
        # dtypes and separate combination) owns the list the calls index.
        module = self.module
        weight = module.up_gate.weight
        major, minor = torch.cuda.get_device_capability(weight.device)
        with torch.cuda.device(weight.device):
            runner = (
                get_cutlass_fused_moe_module(f"{major * 10 + minor}")
                .MoERunner(
                    x_dtype=weight.dtype,
                    weight_dtype=weight.dtype,
                    output_dtype=weight.dtype,
                    top_k=module.top_k,
                    tp_size=1,
                    tp_rank=0,
                    ep_size=1,
                    ep_rank=0,
                    cluster_size=1,
                    cluster_rank=0,
                    enable_alltoall=False,
                    use_deepseek_fp8_block_scale=False,
                    use_w4_group_scaling=False,
                    use_mxfp8_act_scaling=False,
                    min_latency_mode=False,
                    enable_pdl=True,
                    activation_type=_activation(module.activation),
                    use_packed_weights=False,
                    use_fused_finalize=False,
                    use_wfp4afp8_humming=False,
                )
                .fused_moe_runner
            )
        gemm1 = runner.get_gemm1_tactic_count()
        gemm2 = runner.get_gemm2_tactic_count()
        return [[-1, *range(gemm1)], [-1, *range(gemm1, gemm1 + gemm2)]]


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
            ep_size=module.expert_group.size,
            ep_rank=module.expert_group.rank,
            device=weight.device,
        )
        return {"scratch": BufferConfig((nbytes,), torch.uint8)}
