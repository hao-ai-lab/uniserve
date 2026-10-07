"""Packed gated activations: formulas, layouts, FP8 rows and CUDA coverage."""

from __future__ import annotations

import pytest
import torch
from torch.nn import functional as F

from uniserve.nn.functional import gelu_and_mul, silu_and_mul
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
        return lambda x, **options: silu_and_mul(x, **options)
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
