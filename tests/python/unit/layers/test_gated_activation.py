"""Gated activations: formulas, layouts, FP8 rows and CUDA coverage."""

from __future__ import annotations

import pytest
import torch
from torch.nn import functional as F

from uniserve.nn.functional import (
    Rounding,
    gelu_and_mul,
    silu_and_mul,
    swiglu,
    swiglu_nvfp4,
)
from uniserve.quantization import Quantizer

pytestmark = pytest.mark.unit

# BF16 and FP16 round the FP32 product once; FP32 keeps the formula's own
# rounding. Bounds follow the repository's gated-activation tests.
_TOLERANCES = {
    torch.bfloat16: 2e-2,
    torch.float16: 2e-3,
    torch.float32: 1e-5,
}


def _gate(activation):
    if activation == "silu":
        return lambda x, **options: silu_and_mul(
            x, **options, rounding=Rounding.ONCE
        )
    return lambda x, **options: gelu_and_mul(
        x,
        approximate="tanh" if activation == "gelu_tanh" else "none",
        **options,
    )


def _reference(x: torch.Tensor, activation: str) -> torch.Tensor:
    gate, value = x.double().chunk(2, dim=-1)
    if activation == "silu":
        return F.silu(gate) * value
    approximate = "tanh" if activation == "gelu_tanh" else "none"
    return F.gelu(gate, approximate=approximate) * value


@pytest.fixture(params=("cpu", pytest.param("cuda", marks=pytest.mark.gpu)))
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    return request.param


@pytest.mark.parametrize("activation", ("silu", "gelu", "gelu_tanh"))
@pytest.mark.parametrize(
    "dtype", (torch.bfloat16, torch.float16, torch.float32)
)
# 2112 and 4304 are DiffusionGemma's dense and vision MLP widths; 1 and 1025
# leave partial column blocks.
@pytest.mark.parametrize("width", (1, 1025, 2112, 4304))
def test_packed_gating_matches_the_fp32_formula(
    device, activation, dtype, width
):
    generator = torch.Generator(device=device).manual_seed(width)
    # Gates span both GELU tails and the SiLU saturation range.
    x = (
        torch.randn((3, 5, 2 * width), generator=generator, device=device).mul_(
            4
        )
    ).to(dtype)

    with torch.inference_mode():
        actual = _gate(activation)(x)

    tolerance = _TOLERANCES[dtype]
    assert actual.shape == (3, 5, width) and actual.dtype == dtype
    torch.testing.assert_close(
        actual.double(),
        _reference(x, activation),
        rtol=tolerance,
        atol=tolerance,
    )


@pytest.mark.parametrize("activation", ("silu", "gelu", "gelu_tanh"))
@pytest.mark.parametrize("rounding", (Rounding.ONCE, Rounding.STEPWISE))
def test_separate_gate_and_value_equal_their_packed_rows(
    device, activation, rounding
):
    """Two separately stored projections gate exactly as their packed rows."""
    if activation != "silu" and rounding is Rounding.STEPWISE:
        pytest.skip("GELU gating rounds once")
    generator = torch.Generator(device=device).manual_seed(17)
    packed = (
        torch.randn((7, 2 * 1025), generator=generator, device=device)
        .mul_(4)
        .to(torch.bfloat16)
    )
    # Distinct allocations, as two projections write their own outputs.
    gate = packed[:, :1025].clone()
    value = packed[:, 1025:].clone()

    def gating(x):
        if activation == "silu":
            return silu_and_mul(x, rounding=rounding)
        return gelu_and_mul(
            x, approximate="tanh" if activation == "gelu_tanh" else "none"
        )

    with torch.inference_mode():
        assert torch.equal(gating((gate, value)), gating(packed))


def test_a_gate_and_value_pair_must_agree():
    gate = torch.zeros((4, 8), dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="one shape, dtype and device"):
        silu_and_mul((gate, gate[:, :4]), rounding=Rounding.ONCE)
    with pytest.raises(ValueError, match="one shape, dtype and device"):
        silu_and_mul((gate, gate.float()), rounding=Rounding.ONCE)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cuda_separate_operands_encode_their_packed_fp8_rows():
    if torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("FP8 execution requires compute capability 9 or newer")
    generator = torch.Generator(device="cuda").manual_seed(23)
    packed = torch.randn((9, 2 * 2112), generator=generator, device="cuda").to(
        torch.bfloat16
    )
    gate, value = packed[:, :2112].clone(), packed[:, 2112:].clone()
    encodings = [
        Quantizer("fp8", axis=0).empty(
            (9, 2112), dtype=torch.bfloat16, device="cuda"
        )
        for _ in range(2)
    ]

    with torch.inference_mode():
        silu_and_mul(packed, out=encodings[0], rounding=Rounding.STEPWISE)
        silu_and_mul(
            (gate, value), out=encodings[1], rounding=Rounding.STEPWISE
        )

    packed_buffers, separate_buffers = (e.buffers() for e in encodings)
    assert torch.equal(
        separate_buffers["values"].view(torch.uint8),
        packed_buffers["values"].view(torch.uint8),
    )
    assert torch.equal(separate_buffers["scale"], packed_buffers["scale"])


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("activation", ("silu", "gelu_tanh"))
def test_cuda_gating_reads_strided_rows_and_writes_caller_storage(activation):
    generator = torch.Generator(device="cuda").manual_seed(5)
    # Packed gate/up rows borrowed from a wider projection output.
    wide = torch.randn((64, 3 * 2112), generator=generator, device="cuda").to(
        torch.bfloat16
    )
    x = wide[:, : 2 * 2112]
    out = torch.empty((64, 2 * 2112), dtype=torch.bfloat16, device="cuda")[
        :, ::2
    ]

    with torch.inference_mode():
        contiguous = _gate(activation)(x.contiguous())
        strided = _gate(activation)(x)
        written = _gate(activation)(x, out=out)

    assert torch.equal(strided, contiguous)
    assert written is out
    assert torch.equal(out, contiguous)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("activation", ("silu", "gelu", "gelu_tanh"))
def test_cuda_gating_encodes_row_scaled_fp8(activation):
    if torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("FP8 execution requires compute capability 9 or newer")
    generator = torch.Generator(device="cuda").manual_seed(11)
    x = torch.randn((9, 2 * 2112), generator=generator, device="cuda").to(
        torch.bfloat16
    )
    encoded = Quantizer("fp8", axis=0).empty(
        (9, 2112), dtype=torch.bfloat16, device="cuda"
    )

    with torch.inference_mode():
        _gate(activation)(x, out=encoded)

    buffers = encoded.buffers()
    reference = _reference(x, activation)
    decoded = buffers["values"].double() * buffers["scale"].double()
    # E4M3 nearest rounding has unit roundoff 2^-4 and half a subnormal
    # spacing of 2^-10 of each row scale.
    bound = reference.abs() / 16 + buffers["scale"].double() / 1024
    assert (buffers["scale"] > 0).all()
    assert ((decoded - reference).abs() <= bound).all()
    # One scale per row from the row's own absolute maximum.
    torch.testing.assert_close(
        buffers["scale"].double(),
        reference.abs().amax(-1, keepdim=True) / 448,
        rtol=2e-2,
        atol=0,
    )


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cuda_gating_runs_in_grad_mode_without_grad_operands():
    x = torch.randn((4, 2 * 2112), device="cuda", dtype=torch.bfloat16)

    with torch.inference_mode():
        expected = gelu_and_mul(x, approximate="tanh")
    actual = gelu_and_mul(x, approximate="tanh")

    assert torch.equal(actual, expected)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize(
    ("case", "reason"),
    (
        ("records_autograd", "autograd"),
        ("strided_channels", "unit-strided"),
        ("fp8_rank_three", "rank-2"),
        ("fp8_too_wide", "FP8 row width"),
    ),
)
def test_cuda_gating_raises_without_a_kernel(case, reason):
    x = torch.randn((4, 2 * 64), device="cuda", dtype=torch.bfloat16)
    out = None
    if case == "records_autograd":
        x.requires_grad_()
    elif case == "strided_channels":
        x = torch.randn((4, 2 * 64, 2), device="cuda", dtype=torch.bfloat16)[
            ..., 0
        ]
    elif case == "fp8_rank_three":
        x = x.reshape(2, 2, 2 * 64)
        out = Quantizer("fp8", axis=0).empty(
            (2, 2, 64), dtype=torch.bfloat16, device="cuda"
        )
    else:
        x = torch.zeros((2, 2 * 32769), device="cuda", dtype=torch.bfloat16)
        out = Quantizer("fp8", axis=0).empty(
            (2, 32769), dtype=torch.bfloat16, device="cuda"
        )

    with pytest.raises(ValueError, match=f"gelu_and_mul.*{reason}"):
        gelu_and_mul(x, approximate="tanh", out=out)


def _assert_same_encoding(encoded, expected):
    """Compare every buffer of two NVFP4 encodings bit for bit."""
    from uniserve.quantization import QuantizedTensor, ScaleLayout

    assert isinstance(encoded, QuantizedTensor)
    assert encoded.scale_layout is ScaleLayout.LINEAR
    reference = expected.buffers()
    for name, field in encoded.buffers().items():
        if field.dtype != torch.float32:
            field, want = (
                field.view(torch.uint8),
                reference[name].view(torch.uint8),
            )
        else:
            want = reference[name]
        assert torch.equal(field, want), name


def _nvfp4_operands(width, *, generator):
    """Separate strided gate and value rows with two edge K16 blocks.

    Row 0's first block gates to zero and row 1's second block saturates its
    E4M3 block scale under the test's calibrated scale.
    """
    rows = (
        torch.randn((300, 2, width), generator=generator, device="cuda").to(
            torch.bfloat16
        )
        * 3
    )
    gate, value = rows.unbind(1)
    gate[0, :16] = 0
    value[1, 16:32] = 1e4
    return gate, value


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize(
    ("activation", "rounding"),
    (
        ("silu", Rounding.STEPWISE),
        ("silu", Rounding.ONCE),
        ("gelu", Rounding.ONCE),
        ("gelu_tanh", Rounding.ONCE),
    ),
)
# 8192 is the H3 VAE decoder's MLP width and 14336 the H3 DiT's; 400 leaves
# a partial column tile.
@pytest.mark.parametrize("width", (400, 8192, 14336))
def test_cuda_gating_encodes_calibrated_nvfp4(activation, rounding, width):
    """Calibrated NVFP4 gating stores the quantizer's encoding of its rows.

    Values, block scales and tensor scale equal, bit for bit, what the
    calibrated quantizer stores for the unencoded gating of the same rows.
    """
    if torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("NVFP4 encoding requires compute capability 10 or newer")
    generator = torch.Generator(device="cuda").manual_seed(width)
    gate, value = _nvfp4_operands(width, generator=generator)
    quantizer = Quantizer("nvfp4", calibrated_scale=0.0123)

    def gating(**options):
        if activation == "silu":
            return silu_and_mul((gate, value), rounding=rounding, **options)
        approximate = "tanh" if activation == "gelu_tanh" else "none"
        return gelu_and_mul((gate, value), approximate=approximate, **options)

    with torch.inference_mode():
        plain = gating()
        encoded = gating(
            out=quantizer.empty(
                (300, width), dtype=torch.bfloat16, device="cuda"
            )
        )
        _assert_same_encoding(encoded, quantizer.quantize(plain))


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("biased", (False, True))
@pytest.mark.parametrize("width", (400, 8192))
def test_cuda_swiglu_encodes_calibrated_nvfp4(biased, width):
    """Calibrated NVFP4 SwiGLU stores the quantizer's encoding of swiglu."""
    if torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("NVFP4 encoding requires compute capability 10 or newer")
    generator = torch.Generator(device="cuda").manual_seed(width + biased)
    gate, value = _nvfp4_operands(width, generator=generator)
    biases = {
        name: torch.randn(width, generator=generator, device="cuda").to(
            torch.bfloat16
        )
        if biased
        else None
        for name in ("value_bias", "gate_bias")
    }
    quantizer = Quantizer("nvfp4", calibrated_scale=0.0123)

    with torch.inference_mode():
        plain = swiglu(value, gate, **biases)
        encoded = swiglu_nvfp4(
            value,
            gate,
            out=quantizer.empty(
                (300, width), dtype=torch.bfloat16, device="cuda"
            ),
            **biases,
        )
        _assert_same_encoding(encoded, quantizer.quantize(plain))
