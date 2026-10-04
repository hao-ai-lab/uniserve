"""Logit softcapping: ``tanh(x / cap) * cap`` in FP32.

``functional.softcap`` and a softcapped ``VocabParallelHead`` define their
values as the tensor expression ``torch.tanh(x.float() / cap) * cap``. The
CUDA launch is elementwise, so comparing it with that expression over every
16-bit input value covers every logit a 16-bit head projects.
"""

from __future__ import annotations

import pytest
import torch

from uniserve.nn.functional import softcap
from uniserve.nn.linear import VocabParallelHead

pytestmark = pytest.mark.unit


def _expected(x: torch.Tensor, cap: float) -> torch.Tensor:
    return torch.tanh(x.float() / cap) * cap


def _every_value(dtype: torch.dtype, device) -> torch.Tensor:
    """Every non-NaN value of a 16-bit floating dtype."""
    patterns = torch.arange(-(2**15), 2**15, dtype=torch.int32)
    values = patterns.to(torch.int16).view(dtype).to(device)
    return values[~torch.isnan(values)]


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
@pytest.mark.parametrize("cap", (30.0, 0.37))
def test_cuda_softcap_equals_the_tensor_expression_for_every_input(dtype, cap):
    """Subnormal quotients included, since the expression keeps them."""
    x = _every_value(dtype, "cuda")

    assert torch.equal(softcap(x, cap), _expected(x, cap))
    rounded = softcap(x, cap, dtype=dtype)
    assert rounded.dtype == dtype
    assert torch.equal(rounded, _expected(x, cap).to(dtype))


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@torch.inference_mode()
def test_cuda_softcapped_head_writes_capped_fp32_logits():
    """Into a fresh tensor or a caller's contiguous or strided output."""
    torch.manual_seed(5)
    head = VocabParallelHead(
        64, 1000, softcap=30.0, device="cuda", dtype=torch.bfloat16
    )
    head.weight.normal_(0.0, 1.0)
    hidden = torch.randn(7, 64, device="cuda").to(torch.bfloat16)
    expected = _expected(torch.nn.functional.linear(hidden, head.weight), 30.0)

    assert torch.equal(head(hidden), expected)
    out = torch.empty_like(expected)
    assert head(hidden, out=out) is out and torch.equal(out, expected)
    strided = torch.empty(expected.shape[1], 7, device="cuda").t()
    head(hidden, out=strided)
    assert torch.equal(strided, expected)


def test_cpu_softcap_bounds_values_and_rounds_once():
    x = torch.linspace(-200.0, 200.0, 4001).to(torch.bfloat16)

    capped = softcap(x, 30.0)
    assert capped.dtype == torch.float32
    assert torch.equal(capped, _expected(x, 30.0))
    assert capped.abs().max() <= 30.0
    assert torch.equal(
        softcap(x, 30.0, dtype=torch.bfloat16),
        _expected(x, 30.0).to(torch.bfloat16),
    )
