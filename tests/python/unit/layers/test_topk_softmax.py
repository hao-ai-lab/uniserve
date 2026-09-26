"""Softmax top-k expert routing.

``topk_softmax`` returns each token's k most probable experts under the FP32
softmax of its router scores, renormalized and scaled per expert when asked.
On CUDA the kernel reproduces the tensor composition bit for bit, including
the order in which ``torch.topk`` lists equal probabilities.
"""

from __future__ import annotations

import pytest
import torch

from uniserve.nn.functional import topk_softmax

pytestmark = pytest.mark.unit


def _composition(scores, k, renormalize, scale):
    probabilities = torch.softmax(scores, dim=-1, dtype=torch.float32)
    weights, ids = torch.topk(probabilities, k, dim=-1)
    if renormalize:
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(
            torch.finfo(weights.dtype).eps
        )
    if scale is not None:
        weights = weights * scale[ids]
    return ids.to(torch.int32), weights


def test_routing_takes_the_most_probable_experts_in_order():
    scores = torch.tensor([[0.0, 3.0, 1.0, 2.0], [2.0, 2.0, -1.0, 0.0]])
    scale = torch.tensor([1.0, 2.0, 3.0, 4.0])

    ids, weights = topk_softmax(scores, 2, renormalize=True, scale=scale)

    probabilities = torch.softmax(scores, dim=-1)
    assert ids.dtype is torch.int32 and weights.dtype is torch.float32
    assert ids[0].tolist() == [1, 3]
    assert set(ids[1].tolist()) == {0, 1}
    first = probabilities[0, [1, 3]]
    torch.testing.assert_close(
        weights[0], first / first.sum() * scale[[1, 3]], rtol=0, atol=1e-7
    )
    torch.testing.assert_close(
        weights[1], torch.tensor([0.5, 0.5]) * scale[ids[1].long()]
    )


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize(
    "dtype", (torch.bfloat16, torch.float16, torch.float32)
)
@pytest.mark.parametrize(("experts", "k"), ((128, 8), (6, 2), (60, 5)))
@pytest.mark.parametrize("renormalize", (True, False))
def test_cuda_routing_reproduces_the_composition_bit_for_bit(
    dtype, experts, k, renormalize
):
    """BF16 scores tie often, which exercises torch.topk's order of ties."""
    torch.manual_seed(13)
    scores = (torch.randn((777, experts), device="cuda") * 4).to(dtype)
    scale = (torch.rand((experts,), device="cuda") + 0.5).to(torch.bfloat16)

    for expert_scale in (None, scale):
        ids, weights = topk_softmax(
            scores, k, renormalize=renormalize, scale=expert_scale
        )
        expected_ids, expected_weights = _composition(
            scores, k, renormalize, expert_scale
        )

        assert torch.equal(ids, expected_ids)
        assert torch.equal(weights, expected_weights)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cuda_routing_without_a_kernel_raises():
    scores = torch.randn((4, 512), device="cuda")

    with pytest.raises(ValueError, match="topk_softmax has no CUDA kernel"):
        topk_softmax(scores, 8, renormalize=True)
