"""Native grouped-expert kernels evaluate the routed expert equation.

Every value is positive where a test bounds relative error, so each rounding
step bounds the relative error of every output independently of
cancellation.
"""

import pytest
import torch
from torch.nn import functional as F

from uniserve.model import TextSize
from uniserve.nn import functional
from uniserve.nn.moe import FusedMoE
from uniserve.quantization import Quantizer, ScaleLayout
from uniserve.runtime import CUDAStream, ExecutionContext
from uniserve.runtime.cuda_graph import CUDAGraph

pytestmark = [pytest.mark.integration, pytest.mark.gpu]

EXPERTS, HIDDEN, INTERMEDIATE, TOP_K, TOKENS = 8, 256, 128, 2, 40
DEVICE = torch.device("cuda:0")
E4M3_ONE = 0x38  # the E4M3 byte encoding 1.0


def _gamma(roundings):
    """Relative bound of ``roundings`` BF16 unit roundoffs (Higham gamma)."""
    unit = 2**-8
    return roundings * unit / (1 - roundings * unit)


def _routes(generator, tokens=TOKENS):
    """Distinct experts per token and positive FP32 route weights."""
    ids = torch.stack(
        [
            torch.randperm(EXPERTS, generator=generator)[:TOP_K]
            for _ in range(tokens)
        ]
    ).to(torch.int32)
    weights = torch.rand(tokens, TOP_K, generator=generator) + 0.25
    return ids.to(DEVICE), weights.to(DEVICE)


def _reference(hidden, up_gate, down, ids, weights, activation):
    """The routed equation in FP64 over already-decoded operands."""
    hidden, up_gate, down = hidden.double(), up_gate.double(), down.double()
    result = torch.zeros_like(hidden)
    for slot in range(TOP_K):
        expert = ids[:, slot].long()
        projected = torch.einsum("th,toh->to", hidden, up_gate[expert])
        up, gate = projected.split(INTERMEDIATE, dim=-1)
        activated = (
            F.silu(gate)
            if activation == "silu"
            else F.gelu(gate, approximate="tanh")
        ) * up
        result += weights[:, slot, None].double() * torch.einsum(
            "ti,thi->th", activated, down[expert]
        )
    return result


def _replay_matches(module, hidden, ids, weights, expected, *, rtol):
    """Evaluate eagerly, then replay a capture over rewritten routing."""
    stream = CUDAStream.external(torch.cuda.Stream(device=DEVICE))
    stream.wait(torch.cuda.current_stream(DEVICE))
    generator = torch.Generator().manual_seed(47)
    with stream, ExecutionContext(module, stream=stream) as context:
        context.prepare(TextSize(TOKENS, 1))
        with context.activate():
            actual = module(hidden, ids, weights)
        torch.testing.assert_close(
            actual.double(), expected(hidden, ids, weights), rtol=rtol, atol=0
        )

        with CUDAGraph(context=context) as graph:
            graph.capture(lambda: module(hidden, ids, weights))
            replacement_ids, replacement_weights = _routes(generator)
            ids.copy_(replacement_ids)
            weights.copy_(replacement_weights)
            hidden.copy_(hidden.flip(0))
            actual = graph.replay()
            torch.testing.assert_close(
                actual.double(),
                expected(hidden, ids, weights),
                rtol=rtol,
                atol=0,
            )
    torch.cuda.synchronize(DEVICE)


@pytest.mark.parametrize("activation", ["silu", "gelu_tanh"])
@torch.inference_mode()
def test_bf16_experts_match_the_routed_equation(activation):
    generator = torch.Generator().manual_seed(41)
    module = FusedMoE(
        EXPERTS,
        HIDDEN,
        INTERMEDIATE,
        top_k=TOP_K,
        activation=activation,
        device=DEVICE,
        dtype=torch.bfloat16,
    )
    # Positive operands scaled so each projection stays near unit size.
    for parameter, fan_in in (
        (module.up_gate.weight, HIDDEN),
        (module.down.weight, INTERMEDIATE),
    ):
        parameter.copy_(
            (torch.rand(parameter.shape, generator=generator) + 0.125) / fan_in
        )
    hidden = (torch.rand(TOKENS, HIDDEN, generator=generator) + 0.125).to(
        DEVICE, torch.bfloat16
    )
    ids, weights = _routes(generator)

    def expected(hidden, ids, weights):
        return _reference(
            hidden,
            module.up_gate.weight,
            module.down.weight,
            ids,
            weights,
            activation,
        )

    # BF16 roundings of the activated intermediate, each expert's projection
    # and the combined output, plus the kernel's approximate activation,
    # whose relative error is below one BF16 unit on positive inputs.
    _replay_matches(module, hidden, ids, weights, expected, rtol=_gamma(4))


def _nvfp4(codes, block_scale, tensor_scale):
    """Encode E2M1 ``codes [E, rows, K]`` with per-expert scales.

    ``block_scale`` holds one E4M3 byte per expert for all its blocks, or
    one byte per ``[E, rows, K / 16]`` block.
    """
    experts, rows, width = codes.shape
    codes, block_scale = codes.to(DEVICE), block_scale.to(DEVICE)
    values = (codes[..., 0::2] | (codes[..., 1::2] << 4)).to(torch.uint8)
    if block_scale.ndim == 1:
        block_scale = block_scale[:, None, None]
    scales = block_scale.to(torch.uint8).expand(experts, rows, width // 16)
    return (
        Quantizer("nvfp4")
        .from_tensors(
            {
                "values": values.contiguous(),
                "block_scale": scales.reshape(
                    experts * rows, width // 16
                ).contiguous(),
                "tensor_scale": tensor_scale.float().to(DEVICE),
            },
            shape=(experts, rows, width),
            dtype=torch.bfloat16,
        )
        .repack(scale_layout=ScaleLayout.SWIZZLED_128X4)
    )


def _exact_nvfp4(activation, generator):
    """Return NVFP4 experts and hidden states every encoding keeps exact.

    Hidden blocks contain E2M1 magnitudes with a maximum of six, so the unit
    activation scale encodes them exactly. Each up row selects one hidden
    column; each gate row reads columns holding 6 and 2 with weight 4, so
    every gate is 32, where SiLU and tanh-GELU equal the identity in FP32.
    The intermediate ``32 * x`` then encodes exactly at block scale 32.
    Per-expert power-of-two tensor scales are compensated by block scales,
    so a kernel reading another expert's scale changes the result.
    """
    module = FusedMoE(
        EXPERTS,
        HIDDEN,
        INTERMEDIATE,
        top_k=TOP_K,
        activation=activation,
        device=DEVICE,
        dtype=torch.bfloat16,
    )

    # Positive E2M1 codes 1..7 encode 0.5, 1, 1.5, 2, 3, 4 and 6.
    hidden_codes = torch.randint(1, 8, (TOKENS, HIDDEN), generator=generator)
    hidden_codes[:, ::16] = 7
    hidden_codes[:, 1] = 4
    magnitudes = torch.tensor((0, 0.5, 1, 1.5, 2, 3, 4, 6))
    hidden = magnitudes[hidden_codes].to(DEVICE, torch.bfloat16)

    up_gate_codes = torch.zeros(
        EXPERTS, 2 * INTERMEDIATE, HIDDEN, dtype=torch.long
    )
    rows = torch.arange(INTERMEDIATE)
    for expert in range(EXPERTS):
        # The first channel of every intermediate block reads a six.
        columns = torch.randint(0, HIDDEN, (INTERMEDIATE,), generator=generator)
        columns[::16] = 0
        up_gate_codes[expert, rows, columns] = 2
    up_gate_codes[:, INTERMEDIATE:, :2] = 6
    shifts = torch.arange(EXPERTS) % 3
    module.up_gate.weight = torch.nn.Parameter(
        _nvfp4(
            up_gate_codes,
            E4M3_ONE + 8 * shifts,
            torch.exp2(-shifts.float()),
        ),
        requires_grad=False,
    )
    down_codes = torch.randint(
        0, 8, (EXPERTS, HIDDEN, INTERMEDIATE), generator=generator
    )
    module.down.weight = torch.nn.Parameter(
        _nvfp4(
            down_codes,
            torch.full((EXPERTS,), E4M3_ONE - 8 * 3),
            torch.exp2(-(torch.arange(EXPERTS) % 2).float()),
        ),
        requires_grad=False,
    )
    module.up_gate.input_quantizer = Quantizer("nvfp4", calibrated_scale=1.0)
    module.down.input_quantizer = Quantizer("nvfp4", calibrated_scale=1.0)
    return module, hidden


@pytest.mark.parametrize("activation", ["silu", "gelu_tanh"])
@torch.inference_mode()
def test_nvfp4_experts_apply_per_expert_scales_and_static_activations(
    activation,
):
    """Operands are exact in every encoding the kernel applies."""
    generator = torch.Generator().manual_seed(43)
    module, hidden = _exact_nvfp4(activation, generator)
    ids, weights = _routes(generator)

    up_gate = module.up_gate.weight.dequantize(dtype=torch.float32)
    down = module.down.weight.dequantize(dtype=torch.float32)

    def expected(hidden, ids, weights):
        return _reference(hidden, up_gate, down, ids, weights, activation)

    # Every expert projection is exact in FP32. BF16 roundings of each
    # expert's projection and of the combined output, and the FP32 route
    # products and sums, remain.
    _replay_matches(module, hidden, ids, weights, expected, rtol=_gamma(3))


@torch.inference_mode()
def test_preparing_nvfp4_experts_keeps_one_copy_of_the_logical_weights():
    """A kernel's physical placement changes neither values nor footprint."""
    module, _ = _exact_nvfp4("gelu_tanh", torch.Generator().manual_seed(53))
    logical = {
        name: getattr(module, name).weight.dequantize()
        for name in ("up_gate", "down")
    }
    # The smaller projection's encoded bytes: a retained second copy of
    # either projection would exceed it.
    smallest = min(
        sum(
            field.nbytes
            for field in getattr(module, name).weight.buffers().values()
        )
        for name in ("up_gate", "down")
    )
    torch.cuda.synchronize(DEVICE)
    resident = torch.cuda.memory_allocated(DEVICE)

    with ExecutionContext(module) as context:
        context.prepare(TextSize(TOKENS, 1))
        torch.cuda.synchronize(DEVICE)
        assert torch.cuda.memory_allocated(DEVICE) - resident < smallest
        for name, value in logical.items():
            assert torch.equal(getattr(module, name).weight.dequantize(), value)

        # A second preparation borrows the placed weights as they are.
        placed = module.up_gate.weight
        context.prepare(TextSize(2 * TOKENS, 1))
        assert module.up_gate.weight is placed


@pytest.mark.parametrize("activation", ["silu", "gelu_tanh"])
@torch.inference_mode()
def test_nvfp4_gating_applies_the_declared_nonlinearity(activation):
    """Each probe channel is the only nonzero input of its FC2 block.

    Token t routes only to expert t, whose channel 16m has up value 6 and
    gate ``g_m``; the other channels are zero. The FC2 input encoding then
    stores ``act(g_m) * 6`` as six times its E4M3 block scale, and the
    diagonal down projection copies it to output column 16m. The gates
    include -3.75 and -3.375, where erf GELU departs from the tanh
    approximation by more than a quarter, and span SiLU's and GELU's
    curvature elsewhere.
    """
    gates = torch.tensor((-3.75, -3.375, -1.5, -0.75, 0.375, 0.75, 1.5, 3.0))
    experts = tokens = 8
    module = FusedMoE(
        experts,
        HIDDEN,
        INTERMEDIATE,
        top_k=1,
        activation=activation,
        device=DEVICE,
        dtype=torch.bfloat16,
    )

    probes = torch.arange(0, INTERMEDIATE, 16)
    codes = torch.zeros(experts, 2 * INTERMEDIATE, HIDDEN, dtype=torch.long)
    block_scale = torch.full(
        (experts, 2 * INTERMEDIATE, HIDDEN // 16), E4M3_ONE, dtype=torch.long
    )
    # Up rows read hidden column 0, which holds 6, with weight one. Gate
    # rows read it with weight +-0.5 and the block scale |g| / 3, which
    # E4M3 represents exactly for these gates.
    codes[:, probes, 0] = 2
    codes[:, INTERMEDIATE + probes, 0] = torch.where(gates > 0, 1, 9)
    block_scale[:, INTERMEDIATE + probes, 0] = (
        (gates.abs() / 3).to(torch.float8_e4m3fn).view(torch.uint8).long()
    )
    module.up_gate.weight = torch.nn.Parameter(
        _nvfp4(codes, block_scale, torch.ones(experts)), requires_grad=False
    )
    diagonal = torch.zeros(experts, HIDDEN, INTERMEDIATE, dtype=torch.long)
    diagonal[:, torch.arange(INTERMEDIATE), torch.arange(INTERMEDIATE)] = 2
    module.down.weight = torch.nn.Parameter(
        _nvfp4(diagonal, torch.full((experts,), E4M3_ONE), torch.ones(experts)),
        requires_grad=False,
    )
    # The FC2 input scale keeps every probe's block scale a normal E4M3.
    module.up_gate.input_quantizer = Quantizer("nvfp4", calibrated_scale=1.0)
    module.down.input_quantizer = Quantizer("nvfp4", calibrated_scale=2.0**-7)

    hidden = torch.zeros(tokens, HIDDEN, device=DEVICE, dtype=torch.bfloat16)
    hidden[:, 0] = 6.0
    ids = torch.arange(tokens, dtype=torch.int32, device=DEVICE)[:, None]
    weights = torch.ones(tokens, 1, device=DEVICE)
    # The portable reference encodes the FC2 input with exact FP32 division
    # and the exact nonlinearity.
    expected = functional.fused_moe(
        hidden.float(),
        module.up_gate.weight,
        module.down.weight,
        ids,
        weights,
        activation=activation,
        input_quantizers=(
            module.up_gate.input_quantizer,
            module.down.input_quantizer,
        ),
    )

    with ExecutionContext(module) as context:
        context.prepare(TextSize(tokens, 1))
        with context.activate():
            actual = module(hidden, ids, weights)
    # The kernel's approximate nonlinearity and reciprocal may round a
    # block scale near an E4M3 midpoint to the neighboring value: one E4M3
    # step, at most 2**-3 of the value, plus the BF16 output rounding.
    torch.testing.assert_close(
        actual.float(), expected, rtol=2**-3 + _gamma(1), atol=0
    )


def _uncalibrated_nvfp4():
    """NVFP4 experts whose activations have no static encoding scale."""
    module = FusedMoE(
        EXPERTS,
        HIDDEN,
        INTERMEDIATE,
        top_k=TOP_K,
        activation="silu",
        device=DEVICE,
        dtype=torch.bfloat16,
    )
    for linear, rows, width in (
        (module.up_gate, 2 * INTERMEDIATE, HIDDEN),
        (module.down, HIDDEN, INTERMEDIATE),
    ):
        codes = torch.ones(EXPERTS, rows, width, dtype=torch.long)
        linear.weight = torch.nn.Parameter(
            _nvfp4(
                codes,
                torch.full((EXPERTS,), E4M3_ONE),
                torch.ones(EXPERTS),
            ),
            requires_grad=False,
        )
        linear.input_quantizer = Quantizer("nvfp4")
    return module


def _unaligned_nvfp4():
    """Calibrated NVFP4 experts whose intermediate width is 96."""
    intermediate = 96
    module = FusedMoE(
        EXPERTS,
        HIDDEN,
        intermediate,
        top_k=TOP_K,
        activation="gelu_tanh",
        device=DEVICE,
        dtype=torch.bfloat16,
    )
    for linear, rows, width in (
        (module.up_gate, 2 * intermediate, HIDDEN),
        (module.down, HIDDEN, intermediate),
    ):
        codes = torch.ones(EXPERTS, rows, width, dtype=torch.long)
        linear.weight = torch.nn.Parameter(
            _nvfp4(
                codes,
                torch.full((EXPERTS,), E4M3_ONE),
                torch.ones(EXPERTS),
            ),
            requires_grad=False,
        )
        linear.input_quantizer = Quantizer("nvfp4", calibrated_scale=1.0)
    return module


@pytest.mark.parametrize(
    "build",
    [
        lambda: FusedMoE(
            EXPERTS,
            HIDDEN,
            INTERMEDIATE,
            top_k=TOP_K,
            activation="silu",
            device=DEVICE,
            dtype=torch.float32,
        ),
        _uncalibrated_nvfp4,
        _unaligned_nvfp4,
    ],
    ids=["fp32", "uncalibrated-nvfp4", "unaligned-nvfp4"],
)
@torch.inference_mode()
def test_representations_without_a_native_kernel_fail_at_preparation(build):
    module = build()
    with pytest.raises(ValueError, match="no native expert kernel covers"):
        with ExecutionContext(module) as context:
            context.prepare(TextSize(TOKENS, 1))


@torch.inference_mode()
def test_cutlass_does_not_serve_nvfp4_experts():
    module, _ = _exact_nvfp4("silu", torch.Generator().manual_seed(59))
    with pytest.raises(ValueError, match="does not support"):
        with ExecutionContext(module, moe="cutlass") as context:
            context.prepare(TextSize(TOKENS, 1))
