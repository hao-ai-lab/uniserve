"""FlashInfer CuTeDSL grouped kernels for NVFP4 experts on SM100.

The provider serves W4A4 NVFP4 experts whose activations use the static
calibrated scales of ``up_gate.input_quantizer`` and
``down.input_quantizer``, with SiLU (``Swiglu``) or tanh-approximated GELU
(``GegluTanh``) gating, over routing the model has already computed. One
call reads hidden states in that NVFP4 input encoding, as a caller stored
them or as FlashInfer's ``fp4_quantize`` encodes BF16 hidden states (see
``NVFP4Operator``), and runs FlashInfer's CuTeDSL stages: a routing sort
groups the (token, route) pairs by expert into tiles; the FC1 GEMM gathers
each tile's token rows, accumulates in FP32, applies the gating
nonlinearity to the dequantized FP32 products and encodes the gated product
to NVFP4 in its epilogue; the FC2 GEMM dequantizes in FP32 and stores route
``k`` of token ``t``, unweighted, in BF16 at row ``t * K + k``. Those rows
and the route weights are the call's ``Routes``; a combined call evaluates
their value with FlashInfer's ``moe_unpermute``, which sums every token's
routes in route order in FP32, each weighted by its FP32 route weight, and
rounds the sum to BF16 once.

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

Every per-call buffer (the hidden states' encoding when the call encodes
them, routing tables, the encoded FC1 output and its scales, the BF16 route
rows ``[T * K, H]`` and a combined output) comes from PyTorch's caching
allocator, and from the graph's private pool under CUDA graph capture, so
``workspace_buffers`` is empty. Every kernel runs on the caller's current
stream; the operator owns no stream or event. Each call runs the tactic
(routing tile size and each GEMM's MMA tile, cluster shape and raster order)
measured best for its token count on this device and FlashInfer version
(``tactics.json``, see ``tactics``), and FlashInfer's default tactic for
shapes the table does not cover. The measurement admits only tactics whose
outputs equal the default's bit for bit. The first eager call of a tactic
compiles its kernels, so capture must follow an eager call at the captured
token count, as for every prepared operator.
"""

from __future__ import annotations

import torch

from uniserve.nn.functional import Routes
from uniserve.quantization import RowOrder

from . import NVFP4Backend, NVFP4Operator
from .tactics import Tactics


def _tuple(value):
    """A tactic decoded from JSON, with its nested lists as tuples."""
    return tuple(map(_tuple, value)) if isinstance(value, list) else value


class _CuteDsl(NVFP4Operator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        from flashinfer.cute_dsl.utils import convert_sf_to_mma_layout
        from flashinfer.fused_moe import ActivationType

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
        # s2.
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
        self._fc1_alpha = fields13["tensor_scale"] * input13
        self._fc2_input_scale = input2.reciprocal()
        self._fc2_alpha = fields2["tensor_scale"] * input2

        # Default SwiGLU parameters (alpha 1, beta 0, no limit) give
        # silu(gate) * up.
        self._activation = (
            ActivationType.Swiglu
            if module.activation == "silu"
            else ActivationType.GegluTanh
        )
        # ``experts`` counts the resident experts; an expert-parallel rank
        # holds the global experts ``expert_slice`` and routing names global
        # ids, which the sort skips outside that range.
        self._experts, self._hidden = experts, hidden
        self._tactics = Tactics(Backend.name, module)

    def _validate(self, hidden, topk_ids, topk_weights):
        super()._validate(hidden, topk_ids, topk_weights)
        if hidden.dtype != torch.bfloat16:
            raise ValueError(
                "CuTeDSL NVFP4 experts read and write BF16 hidden states"
            )

    def __call__(
        self, hidden, topk_ids, topk_weights, *, combine=True, tactic=None
    ):
        """Evaluate the routed experts; ``tactic`` overrides the table.

        A tactic is ``[choice]``: ``None`` for FlashInfer's default or one
        complete ``(tile_size, gemm1, gemm2)`` tactic of its list.
        """
        from flashinfer.fused_moe.cute_dsl.blockscaled_contiguous_gather_grouped_gemm_act_fusion import (  # noqa: E501
            blockscaled_contiguous_gather_grouped_gemm_act_fusion_nvfp4,
        )
        from flashinfer.fused_moe.cute_dsl.blockscaled_contiguous_grouped_gemm_finalize_fusion import (  # noqa: E501
            blockscaled_contiguous_grouped_gemm_finalize_fusion_nvfp4,
        )
        from flashinfer.fused_moe.cute_dsl.moe_utils import (
            moe_sort,
            moe_unpermute,
        )
        from flashinfer.fused_moe.cute_dsl.tuner import (
            _extract_tactic_params,
            _get_default_tactic,
        )

        self._validate(hidden, topk_ids, topk_weights)
        module = self.module
        if not combine and self._experts != module.num_experts:
            # Routes to other ranks' experts have no rows here; their
            # exchange combines the ranks' partial sums instead.
            raise ValueError(
                "uncombined routes require every expert to be resident"
            )
        tokens, top_k = topk_ids.shape
        topk_ids = topk_ids.contiguous()
        topk_weights = topk_weights.contiguous()
        if not tokens:
            empty = Routes(
                torch.empty(
                    (0, top_k, self._hidden),
                    dtype=torch.bfloat16,
                    device=hidden.device,
                ),
                topk_weights,
            )
            return empty.combine() if combine else empty
        if tactic is None:
            tactic = self._tactics.select(tokens)
        choice = None if tactic is None else _tuple(tactic[0])
        params = _extract_tactic_params(
            _get_default_tactic() if choice is None else choice
        )
        values, scales = self._encoded(hidden)

        # FlashInfer's CuTeDSL pipeline with its deterministic two-stage
        # combination (flashinfer/fused_moe/cute_dsl/fused_moe.py:283-417,
        # use_fused_finalize=False): the sort groups (token, route) pairs
        # into tile_size-row expert tiles; GEMM1 gathers their token rows
        # and stores the gated product in NVFP4; GEMM2 stores route k of
        # token t, unweighted, at row t * K + k.
        (
            tile_idx_to_expert_idx,
            tile_idx_to_mn_limit,
            expanded_idx_to_permuted_idx,
            permuted_idx_to_expanded_idx,
            _,
            num_non_exiting_tiles,
        ) = moe_sort(
            token_selected_experts=topk_ids,
            token_final_scales=topk_weights,
            num_experts=module.num_experts,
            top_k=top_k,
            local_expert_offset=module.expert_slice.start,
            num_local_experts=self._experts,
            tile_tokens_dim=params["tile_size"],
        )
        intermediate, intermediate_scale = (
            blockscaled_contiguous_gather_grouped_gemm_act_fusion_nvfp4(
                a=values,
                b=self._fc1,
                a_scale=scales,
                b_scale=self._fc1_scale,
                alpha=self._fc1_alpha,
                tile_idx_to_expert_idx=tile_idx_to_expert_idx,
                tile_idx_to_mn_limit=tile_idx_to_mn_limit,
                token_id_mapping=permuted_idx_to_expanded_idx,
                num_non_exiting_tiles=num_non_exiting_tiles,
                out=None,
                out_scale=None,
                global_scale=self._fc2_input_scale,
                a_per_token_scale=None,
                c_dtype="float4_e2m1fn",
                topk=top_k,
                mma_tiler_mn=params["gemm1_mma_tiler_mn"],
                cluster_shape_mn=params["gemm1_cluster_shape_mn"],
                enable_pdl=True,
                activation_type=self._activation.value,
                # SwiGLU and tanh-GELU both gate the up half.
                gated=True,
            )
        )
        # GEMM1 stores NVFP4 with its block scales.
        assert intermediate_scale is not None
        rows = torch.empty(
            (tokens * top_k, self._hidden),
            dtype=torch.bfloat16,
            device=hidden.device,
        )
        blockscaled_contiguous_grouped_gemm_finalize_fusion_nvfp4(
            a=intermediate,
            b=self._fc2,
            a_scale=intermediate_scale,
            b_scale=self._fc2_scale,
            alpha=self._fc2_alpha,
            tile_idx_to_expert_idx=tile_idx_to_expert_idx,
            num_non_exiting_tiles=num_non_exiting_tiles,
            tile_idx_to_mn_limit=tile_idx_to_mn_limit,
            permuted_idx_to_expanded_idx=permuted_idx_to_expanded_idx,
            token_final_scales=topk_weights,
            out=rows,
            a_per_token_scale=None,
            mma_tiler_mn=params["gemm2_mma_tiler_mn"],
            cluster_shape_mn=params["gemm2_cluster_shape_mn"],
            enable_pdl=True,
            use_fused_finalize=False,
        )
        routes = Routes(rows.view(tokens, top_k, self._hidden), topk_weights)
        if not combine:
            return routes

        # TensorRT-LLM's moeUnpermuteKernel evaluates the routes' value.
        output = torch.empty(
            (tokens, self._hidden), dtype=torch.bfloat16, device=hidden.device
        )
        moe_unpermute(
            permuted_input=rows,
            output=output,
            expanded_idx_to_permuted_idx=expanded_idx_to_permuted_idx,
            topk_scales=topk_weights,
            num_tokens=tokens,
            top_k=top_k,
            input_is_expanded=True,
            enable_pdl=True,
        )
        return output

    def tactic_space(self) -> list[list]:
        """One dimension: FlashInfer's default, then every valid tactic."""
        from flashinfer.fused_moe.cute_dsl.tuner import _get_arch_tactics

        with torch.cuda.device(self.module.up_gate.weight.device):
            return [[None, *_get_arch_tactics()]]

    def close(self) -> None:
        super().close()
        # Release the borrowed weight fields and derived scales, so a closed
        # operator never keeps a replaced encoding resident.
        self._fc1 = self._fc2 = self._fc1_scale = self._fc2_scale = None
        self._fc1_alpha = None
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
