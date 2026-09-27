"""CuTe DSL grouped expert kernels over FlashInfer's MoE sort on SM100.

The provider serves two expert representations, over routing the model has
already computed, with SiLU or tanh-approximated GELU gating:

* W4A4 NVFP4 experts whose activations use the static calibrated scales of
  ``up_gate.input_quantizer`` and ``down.input_quantizer``, with
  FlashInfer's CuTeDSL kernels. One call reads hidden states in that NVFP4
  input encoding, as a caller stored them or as FlashInfer's
  ``fp4_quantize`` encodes BF16 hidden states (see ``NVFP4Operator``).
* BF16 experts with UniServe's BF16 kernels (``uniserve_kernels.experts``),
  reading BF16 hidden states.

Both run the same stages (FlashInfer's pipeline,
flashinfer/fused_moe/cute_dsl/fused_moe.py:283-417,
``use_fused_finalize=False``): a routing sort groups the (token, route)
pairs by expert into tiles; the FC1 GEMM gathers each tile's token rows,
accumulates in FP32 and applies the gating nonlinearity to the FP32
products (NVFP4: dequantized, then encoded to NVFP4; BF16: rounded to BF16
once); the FC2 GEMM accumulates in FP32 and stores route ``k`` of token
``t``, unweighted, in BF16 at row ``t * K + k``. Those rows and the route
weights are the call's ``Routes``; a combined call evaluates their value
with FlashInfer's ``moe_unpermute``, which sums every token's routes in
route order in FP32, each weighted by its FP32 route weight, and rounds the
sum to BF16 once.

The provider uses only this two-stage combination. FlashInfer's fused
alternative adds each route into the token's BF16 output row with an atomic
reduction from whichever tile computed it, so the order of the BF16
additions, and with it the output, varies between calls of the same inputs.
The two-stage combination fixes the order: repeated calls of the same
inputs, eager or replayed, give bitwise identical outputs.

The NVFP4 kernels read expert weights with ``up_gate`` in
``RowOrder.INTERLEAVED_64`` (each 128-row block holds 64 up rows, which the
epilogue reads as the linear half, then the matching 64 gate rows) and
``down`` in linear row order, with 128x4-swizzled block scales that the
kernels index per expert through FlashInfer's MMA view of the same bytes.
Preparing the provider places a module's weights in that order once,
replacing its parameters with the rearranged encoding; the logical weights
are unchanged, so every provider and the portable reference decode them
identically, and only one copy stays resident. The BF16 kernels read the
resident linear-order weights as they are: FC1's weight loads visit each
expert's up and gate rows in the same 64-row interleave through a strided
view, so preparation moves nothing.

An expert-parallel rank holds the global experts ``expert_slice``; the sort
takes the routes to those experts and skips the others, whose rows only a
combined call leaves out (its sum is this rank's partial sum).

Every per-call buffer (the hidden states' encoding when the call encodes
them, routing tables, the FC1 output and its scales, the BF16 route rows
``[T * K, H]`` and a combined output) comes from PyTorch's caching
allocator, and from the graph's private pool under CUDA graph capture, so
``workspace_buffers`` is empty. Every kernel runs on the caller's current
stream; the operator owns no stream or event. Each call runs the tactic
(routing tile size and each GEMM's MMA tile and cluster shape, and for
NVFP4 the raster order) measured best for its token count on this device
and FlashInfer version (``tactics.json``, see ``tactics``), and the default
tactic for shapes the table does not cover. The measurement admits only
tactics whose outputs equal the default's bit for bit. The first eager call
of a tactic compiles its kernels, so capture must follow an eager call at
the captured token count, as for every prepared operator.
"""

from __future__ import annotations

from typing import NamedTuple

import torch

from uniserve.nn.functional import Routes
from uniserve.quantization import QuantizedTensor, RowOrder

from . import Backend as _Backend
from . import NVFP4Backend, NVFP4Operator, Operator
from .tactics import Tactics

NAME = "cutedsl"


def _tuple(value):
    """A tactic decoded from JSON, with its nested lists as tuples."""
    return tuple(map(_tuple, value)) if isinstance(value, list) else value


class _SortedExperts(Operator):
    """The shared stages: sort, grouped GEMMs to route rows, combination.

    A subclass supplies the routing tile of a tactic (``_tile``) and the
    grouped GEMMs that store every sorted route's row (``_grouped``).
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        module = self.module
        # ``experts`` counts the resident experts; an expert-parallel rank
        # holds the global experts ``expert_slice`` and routing names global
        # ids, which the sort skips outside that range.
        self._experts = module.up_gate.weight.shape[0]
        self._hidden = module.hidden_size
        self._tactics = Tactics(NAME, module)

    def _tile(self, choice) -> int:
        """The routing tile size of tactic ``choice`` (None: the default)."""
        raise NotImplementedError

    def _grouped(self, hidden, topk_weights, sort, rows, choice) -> None:
        """Store every sorted route's unweighted row into ``rows``.

        ``sort`` holds ``moe_sort``'s tile expert ids, tile row limits,
        permuted-to-expanded row map and non-exiting tile count.
        """
        raise NotImplementedError

    def __call__(
        self, hidden, topk_ids, topk_weights, *, combine=True, tactic=None
    ):
        """Evaluate the routed experts; ``tactic`` overrides the table.

        A tactic is ``[choice]``: ``None`` for the default or one complete
        tactic of ``tactic_space``.
        """
        from flashinfer.fused_moe.cute_dsl.moe_utils import (
            moe_sort,
            moe_unpermute,
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
            tile_tokens_dim=self._tile(choice),
        )
        rows = torch.empty(
            (tokens * top_k, self._hidden),
            dtype=torch.bfloat16,
            device=hidden.device,
        )
        self._grouped(
            hidden,
            topk_weights,
            (
                tile_idx_to_expert_idx,
                tile_idx_to_mn_limit,
                permuted_idx_to_expanded_idx,
                num_non_exiting_tiles,
            ),
            rows,
            choice,
        )
        routes = Routes(rows.view(tokens, top_k, self._hidden), topk_weights)
        if not combine:
            return routes

        # TensorRT-LLM's moeUnpermuteKernel evaluates the routes' value;
        # routes to experts the sort skipped (permuted index -1) are left
        # out of the sum.
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


class _NVFP4Weights(NamedTuple):
    """Borrowed NVFP4 weight fields and the scales the kernels consume."""

    fc1: torch.Tensor
    fc2: torch.Tensor
    fc1_scale: torch.Tensor
    fc2_scale: torch.Tensor
    fc1_alpha: torch.Tensor
    fc2_input_scale: torch.Tensor
    fc2_alpha: torch.Tensor


class _NVFP4Experts(_SortedExperts, NVFP4Operator):
    """FlashInfer's CuTeDSL W4A4 NVFP4 kernels."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        from flashinfer.cute_dsl.utils import convert_sf_to_mma_layout
        from flashinfer.fused_moe import ActivationType

        module = self.module
        up_gate, down = module.up_gate.weight, module.down.weight
        experts, rows, hidden = up_gate.shape
        intermediate = down.shape[2]
        fields13, fields2 = up_gate.buffers(), down.buffers()

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
        # Values are the stacked [E, rows, K / 2] E2M1 bytes. The kernels
        # read block scales through a 6-D per-expert view of the swizzled
        # bytes; every expert fills whole 128x4 tiles, so the stacked
        # swizzle is the per-expert swizzle that view describes.
        self._weights: _NVFP4Weights | None = _NVFP4Weights(
            fc1=fields13["values"],
            fc2=fields2["values"],
            fc1_scale=convert_sf_to_mma_layout(
                fields13["block_scale"], m=rows, k=hidden, num_groups=experts
            ),
            fc2_scale=convert_sf_to_mma_layout(
                fields2["block_scale"],
                m=hidden,
                k=intermediate,
                num_groups=experts,
            ),
            fc1_alpha=fields13["tensor_scale"] * input13,
            fc2_input_scale=input2.reciprocal(),
            fc2_alpha=fields2["tensor_scale"] * input2,
        )

        # Default SwiGLU parameters (alpha 1, beta 0, no limit) give
        # silu(gate) * up.
        self._activation = (
            ActivationType.Swiglu
            if module.activation == "silu"
            else ActivationType.GegluTanh
        )

    def _validate(self, hidden, topk_ids, topk_weights):
        super()._validate(hidden, topk_ids, topk_weights)
        if hidden.dtype != torch.bfloat16:
            raise ValueError(
                "CuTeDSL NVFP4 experts read and write BF16 hidden states"
            )

    @staticmethod
    def _params(choice):
        from flashinfer.fused_moe.cute_dsl.tuner import (
            _extract_tactic_params,
            _get_default_tactic,
        )

        return _extract_tactic_params(
            _get_default_tactic() if choice is None else choice
        )

    def _tile(self, choice) -> int:
        return self._params(choice)["tile_size"]

    def _grouped(self, hidden, topk_weights, sort, rows, choice) -> None:
        from flashinfer.fused_moe.cute_dsl.blockscaled_contiguous_gather_grouped_gemm_act_fusion import (  # noqa: E501
            blockscaled_contiguous_gather_grouped_gemm_act_fusion_nvfp4,
        )
        from flashinfer.fused_moe.cute_dsl.blockscaled_contiguous_grouped_gemm_finalize_fusion import (  # noqa: E501
            blockscaled_contiguous_grouped_gemm_finalize_fusion_nvfp4,
        )

        (
            tile_idx_to_expert_idx,
            tile_idx_to_mn_limit,
            permuted_idx_to_expanded_idx,
            num_non_exiting_tiles,
        ) = sort
        params = self._params(choice)
        top_k = self.module.top_k
        values, scales = self._encoded(hidden)
        weights = self._weights
        assert weights is not None

        # GEMM1 stores the gated product in NVFP4 with its block scales;
        # GEMM2 stores the unweighted route rows (the route weights are
        # the combination's).
        intermediate, intermediate_scale = (
            blockscaled_contiguous_gather_grouped_gemm_act_fusion_nvfp4(
                a=values,
                b=weights.fc1,
                a_scale=scales,
                b_scale=weights.fc1_scale,
                alpha=weights.fc1_alpha,
                tile_idx_to_expert_idx=tile_idx_to_expert_idx,
                tile_idx_to_mn_limit=tile_idx_to_mn_limit,
                token_id_mapping=permuted_idx_to_expanded_idx,
                num_non_exiting_tiles=num_non_exiting_tiles,
                out=None,
                out_scale=None,
                global_scale=weights.fc2_input_scale,
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
        assert intermediate_scale is not None
        blockscaled_contiguous_grouped_gemm_finalize_fusion_nvfp4(
            a=intermediate,
            b=weights.fc2,
            a_scale=intermediate_scale,
            b_scale=weights.fc2_scale,
            alpha=weights.fc2_alpha,
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

    def tactic_space(self) -> list[list]:
        """One dimension: FlashInfer's default, then every valid tactic."""
        from flashinfer.fused_moe.cute_dsl.tuner import _get_arch_tactics

        with torch.cuda.device(self.module.up_gate.weight.device):
            return [[None, *_get_arch_tactics()]]

    def close(self) -> None:
        super().close()
        # Release the borrowed weight fields and derived scales, so a closed
        # operator never keeps a replaced encoding resident.
        self._weights = None


class _BF16Experts(_SortedExperts):
    """UniServe's BF16 gather (FC1) and route (FC2) grouped GEMMs."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        from uniserve_kernels import experts

        self._kernels = experts
        # Borrowed resident weights: [E, 2I, H] with each expert's up rows
        # followed by its gate rows, and [E, H, I].
        self._weights: tuple[torch.Tensor, torch.Tensor] | None = (
            self.module.up_gate.weight,
            self.module.down.weight,
        )
        self._default = (
            experts.DEFAULT_GATHER_TACTIC,
            experts.DEFAULT_ROUTE_TACTIC,
        )

    def _validate(self, hidden, topk_ids, topk_weights):
        super()._validate(hidden, topk_ids, topk_weights)
        if hidden.dtype != torch.bfloat16:
            raise ValueError(
                "CuTeDSL BF16 experts read and write BF16 hidden states"
            )

    def _tile(self, choice) -> int:
        (gather, _) = self._default if choice is None else choice
        return gather[0][0]

    def _grouped(self, hidden, topk_weights, sort, rows, choice) -> None:
        (
            tile_idx_to_expert_idx,
            tile_idx_to_mn_limit,
            permuted_idx_to_expanded_idx,
            num_non_exiting_tiles,
        ) = sort
        (gather, route) = self._default if choice is None else choice
        assert self._weights is not None
        (fc1, fc2) = self._weights

        # FC1 stores the gated product of every permuted row in BF16; rows
        # past a tile's limit hold unspecified values that no route row
        # reads.
        gated = torch.empty(
            (permuted_idx_to_expanded_idx.shape[0], fc2.shape[2]),
            dtype=torch.bfloat16,
            device=hidden.device,
        )
        self._kernels.gather_gemm(
            hidden.contiguous(),
            fc1,
            tile_idx_to_expert_idx,
            tile_idx_to_mn_limit,
            permuted_idx_to_expanded_idx,
            num_non_exiting_tiles,
            gated,
            top_k=self.module.top_k,
            activation=self.module.activation,
            tactic=gather,
        )
        self._kernels.route_gemm(
            gated,
            fc2,
            tile_idx_to_expert_idx,
            tile_idx_to_mn_limit,
            permuted_idx_to_expanded_idx,
            num_non_exiting_tiles,
            rows,
            tactic=route,
        )

    def tactic_space(self) -> list[list]:
        """One dimension: the default, then every other FC1 and FC2 pair.

        Both GEMMs of a pair share the routing tile, their MMA tile M.
        """
        experts = self._kernels
        pairs = [
            (gather, route)
            for gather in experts.GATHER_TACTICS
            for route in experts.ROUTE_TACTICS
            if gather[0][0] == route[0][0] and (gather, route) != self._default
        ]
        return [[None, *pairs]]

    def close(self) -> None:
        super().close()
        self._weights = None


class _NVFP4Backend(NVFP4Backend):
    name = NAME
    operator_class = _NVFP4Experts
    up_gate_order = RowOrder.INTERLEAVED_64
    down_order = RowOrder.LINEAR

    def kernels_unsupported(self, device) -> str | None:
        # The Blackwell kernels cover SM100 and SM103; SM107 runs other
        # kernels that these experts are not validated with.
        if torch.cuda.get_device_capability(device) not in {(10, 0), (10, 3)}:
            return "the kernels are built for SM100 and SM103 devices"
        return _cute_dsl_unavailable()


def _cute_dsl_unavailable() -> str | None:
    from flashinfer.cute_dsl.availability import is_cute_dsl_available

    if not is_cute_dsl_available():
        return "the CuTe DSL is not installed"
    return None


def _bf16_unsupported(module) -> str | None:
    """Why the BF16 kernels cannot evaluate ``module``, or ``None``.

    The reasons state the capability boundary, so a kernel record naming
    another provider for dense experts says why this one did not serve them.
    """
    up_gate, down = module.up_gate.weight, module.down.weight
    if up_gate.device.type != "cuda":
        return "the kernels run on CUDA devices"
    if up_gate.dtype != torch.bfloat16 or down.dtype != torch.bfloat16:
        return (
            f"{str(up_gate.dtype).removeprefix('torch.')} experts: the "
            "kernels serve BF16 and NVFP4 experts"
        )
    major, minor = torch.cuda.get_device_capability(up_gate.device)
    if (major, minor) != (10, 0):
        return f"SM{major}{minor}: the BF16 kernels are validated on SM100 only"
    reason = _cute_dsl_unavailable()
    if reason is not None:
        return reason
    if not (up_gate.is_contiguous() and down.is_contiguous()):
        return "expert weights must be contiguous"

    # FC1 reads K = H and pairs 64-row up and gate blocks, so H and I are
    # multiples of 64; FC2 reads K = I and stores 16-byte aligned rows.
    hidden, intermediate = module.hidden_size, down.shape[2]
    if hidden % 64 or intermediate % 64:
        return (
            f"hidden width {hidden} and intermediate width {intermediate}: "
            "the BF16 kernels need multiples of 64"
        )
    if module.group.size > 1:
        return "tensor-parallel expert shards are not validated"
    return None


class Backend(_Backend):
    """NVFP4 experts through FlashInfer's kernels, BF16 through UniServe's."""

    name = NAME

    def unsupported(self, module) -> str | None:
        if _nvfp4(module):
            return _NVFP4Backend().unsupported(module)
        return _bf16_unsupported(module)

    def prepare(self, *, module, size, workspace) -> Operator:
        if _nvfp4(module):
            return _NVFP4Backend().prepare(
                module=module, size=size, workspace=workspace
            )
        return _BF16Experts(module=module, size=size, workspace=workspace)


def _nvfp4(module) -> bool:
    """Whether ``module`` holds encoded experts, which the NVFP4 path owns."""
    return any(
        isinstance(linear.weight, QuantizedTensor)
        for linear in (module.up_gate, module.down)
    )
