"""FlashInfer CuTeDSL grouped kernels for NVFP4 experts on SM100.

The provider serves W4A4 NVFP4 experts whose activations use the static
calibrated scales of ``up_gate.input_quantizer`` and
``down.input_quantizer``, with SiLU (``Swiglu``) or tanh-approximated GELU
(``GegluTanh``) gating, over routing the model has already computed. One
call encodes the hidden states to NVFP4 with FlashInfer's ``fp4_quantize``
and runs ``CuteDslMoEWrapper``: a routing sort groups the (token, route)
pairs by expert into 128-row tiles; the FC1 GEMM gathers each tile's token
rows, accumulates in FP32, applies the gating nonlinearity to the
dequantized FP32 products and encodes the gated product to NVFP4 in its
epilogue; the FC2 GEMM dequantizes in FP32 and stores each (token, route)
row in BF16; a final pass sums every token's routes in route order in FP32,
each weighted by its FP32 route weight, and rounds the sum to BF16 once.

The provider uses only this two-stage combination (FlashInfer's
``use_fused_finalize=False``). FlashInfer's fused alternative adds each
route into the token's BF16 output row with an atomic reduction from
whichever tile computed it, so the order of the BF16 additions, and with it
the output, varies between calls of the same inputs. The two-stage
combination fixes the order: repeated calls of the same inputs, eager or
replayed, give bitwise identical outputs.

The kernels read expert weights with ``up_gate`` in
``RowOrder.INTERLEAVED_64`` (each 128-row block holds 64 up rows, which the
epilogue reads as the linear half, then the matching 64 gate rows) and
``down`` in linear row order, with 128x4-swizzled block scales that the
kernels index per expert through FlashInfer's MMA view of the same bytes.
Preparing the provider places a module's weights in that order once,
replacing its parameters with the rearranged encoding; the logical weights
are unchanged, so every provider and the portable reference decode them
identically, and only one copy stays resident.

Every per-call buffer (the encoded hidden states and their scales, routing
tables, the encoded FC1 output and its scales, the ``[T * K, H]`` BF16
route rows and the returned output) comes from PyTorch's caching
allocator, and from the graph's private pool under CUDA graph capture, so
``workspace_buffers`` is empty. Every kernel runs on the caller's current
stream; the operator owns no stream or event. Each call runs the tactic
(routing tile size and each GEMM's MMA tile, cluster shape and raster
order) measured best for its token count on this device and FlashInfer
version (``tactics.json``, see ``tactics``), and FlashInfer's default
tactic for shapes the table does not cover. The measurement admits only
tactics whose outputs equal the default's bit for bit. The first eager
call of a tactic compiles its kernels, so capture must follow an eager
call at the captured token count, as for every prepared operator.
"""

from __future__ import annotations

import torch

from uniserve.quantization import RowOrder

from . import NVFP4Backend
from . import Operator as _Operator
from .tactics import Tactics


def _tuple(value):
    """A tactic decoded from JSON, with its nested lists as tuples."""
    return tuple(map(_tuple, value)) if isinstance(value, list) else value


class _CuteDsl(_Operator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        from flashinfer import fp4_quantize
        from flashinfer.cute_dsl.utils import convert_sf_to_mma_layout
        from flashinfer.fused_moe import ActivationType, CuteDslMoEWrapper

        self._quantize = fp4_quantize
        module = self.module
        up_gate, down = module.up_gate.weight, module.down.weight
        experts, rows, hidden = up_gate.shape
        intermediate = down.shape[2]
        fields13, fields2 = up_gate.buffers(), down.buffers()

        # Values are the stacked [E, rows, K / 2] E2M1 bytes. The kernels
        # read block scales through a 6-D per-expert view of the swizzled
        # bytes; every expert fills whole 128x4 tiles, so the stacked
        # swizzle is the per-expert swizzle that view describes.
        self._fc1 = fields13["values"]
        self._fc2 = fields2["values"]
        self._fc1_scale = convert_sf_to_mma_layout(
            fields13["block_scale"], m=rows, k=hidden, num_groups=experts
        )
        self._fc2_scale = convert_sf_to_mma_layout(
            fields2["block_scale"], m=hidden, k=intermediate, num_groups=experts
        )

        # FP32 scale arithmetic on the device, as the kernels consume it. A
        # static activation scale s encodes x / s; the input encoding takes
        # the global scale 1 / s13. FC1 results are dequantized with the
        # per-expert product of the weight tensor scale and s13 before the
        # nonlinearity; the epilogue encodes the gated product with 1 / s2,
        # and FC2 results are dequantized with the down tensor scale times
        # s2 before the route weight.
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
        self._input_scale = input13.reciprocal()
        self._fc1_alpha = fields13["tensor_scale"] * input13
        self._fc2_input_scale = input2.reciprocal()
        self._fc2_alpha = fields2["tensor_scale"] * input2

        # Default SwiGLU parameters (alpha 1, beta 0, no limit) give
        # silu(gate) * up. The two-stage finalize runs on the caller's
        # stream alone; use_cuda_graph would only create the auxiliary
        # stream and events of the fused finalize's overlapped output
        # zeroing, which this combination never runs.
        self._kernel = CuteDslMoEWrapper(
            num_experts=experts,
            top_k=module.top_k,
            hidden_size=hidden,
            intermediate_size=intermediate,
            use_cuda_graph=False,
            output_dtype=torch.bfloat16,
            device=device,
            activation_type=(
                ActivationType.Swiglu
                if module.activation == "silu"
                else ActivationType.GegluTanh
            ).value,
            use_fused_finalize=False,
            quant_mode="w4a4",
        )
        self._tactics = Tactics(Backend.name, module)

    def _validate(self, hidden, topk_ids, topk_weights):
        super()._validate(hidden, topk_ids, topk_weights)
        if hidden.dtype != torch.bfloat16:
            raise ValueError(
                "CuTeDSL NVFP4 experts read and write BF16 hidden states"
            )

    def __call__(self, hidden, topk_ids, topk_weights, *, tactic=None):
        """Evaluate the routed experts; ``tactic`` overrides the table.

        A tactic is ``[choice]``: ``None`` for FlashInfer's default or one
        complete ``(tile_size, gemm1, gemm2)`` tactic of the wrapper.
        """
        self._validate(hidden, topk_ids, topk_weights)
        tokens = hidden.shape[0]
        if not tokens:
            return torch.empty_like(hidden)
        if tactic is None:
            tactic = self._tactics.select(tokens)
        choice = None if tactic is None else _tuple(tactic[0])

        # The FC1 gather reads the linear token-major [T, H / 16] scales.
        values, scales = self._quantize(
            hidden.contiguous(),
            global_scale=self._input_scale,
            is_sf_swizzled_layout=False,
        )
        scales = scales.view(torch.float8_e4m3fn).reshape(
            tokens, hidden.shape[1] // 16
        )
        return self._kernel.run(
            values,
            scales,
            topk_ids.contiguous(),
            topk_weights.contiguous(),
            self._fc1,
            self._fc1_scale,
            self._fc1_alpha,
            self._fc2_input_scale,
            self._fc2,
            self._fc2_scale,
            self._fc2_alpha,
            tactic=choice,
        )

    def tactic_space(self) -> list[list]:
        """One dimension: FlashInfer's default, then every valid tactic."""
        return [[None, *self._kernel.get_valid_tactics()]]

    def close(self) -> None:
        super().close()
        # Release the borrowed weight fields and derived scales, so a closed
        # operator never keeps a replaced encoding resident.
        self._kernel = None
        self._fc1 = self._fc2 = self._fc1_scale = self._fc2_scale = None
        self._input_scale = self._fc1_alpha = None
        self._fc2_input_scale = self._fc2_alpha = None


class Backend(NVFP4Backend):
    name = "cutedsl"
    operator_class = _CuteDsl
    up_gate_order = RowOrder.INTERLEAVED_64
    down_order = RowOrder.LINEAR

    def kernels_unsupported(self, device) -> str | None:
        # The Blackwell kernels cover SM100 and SM103; SM107 runs other
        # kernels that these experts are not validated with.
        if torch.cuda.get_device_capability(device) not in {(10, 0), (10, 3)}:
            return "the kernels are built for SM100 and SM103 devices"
        from flashinfer.cute_dsl.availability import is_cute_dsl_available

        if not is_cute_dsl_available():
            return "the CuTe DSL is not installed"
        return None
