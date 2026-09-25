"""Exactness and capture checks for exact top-k sampling."""

from __future__ import annotations

import pytest
import torch

from uniserve.sampling import sample_top_k

pytestmark = pytest.mark.unit


def _inputs(device: torch.device) -> tuple[torch.Tensor, ...]:
    logits = torch.tensor(
        [
            [0.2, 1.4, -0.6, 2.1, 0.7, 1.0],
            [1.6, -0.4, 0.8, 1.2, 0.1, 2.0],
        ],
        dtype=torch.bfloat16,
        device=device,
    )
    draws = torch.tensor([0.25, 0.76], dtype=torch.float32, device=device)
    parameters = torch.tensor(
        [
            [0.7, 0.84, 0.08],
            [0.0, 0.92, 0.0],
        ],
        dtype=torch.float32,
        device=device,
    )
    return logits, draws, parameters


def test_top_k_provider_preserves_filtered_categorical_and_greedy_selection():
    inputs = _inputs(torch.device("cpu"))

    logits, draws, parameters = inputs
    tokens, valid = sample_top_k(
        logits,
        draws,
        parameters,
        4,
    )

    assert valid.tolist() == [True, True]
    # Row 0 retains IDs 1, 3, 5 with probabilities about .234, .635, .131.
    # Draw .25 selects ID 3; row 1 selects its maximum at zero temperature.
    assert tokens.tolist() == [3, 5]


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_top_k_provider_is_capture_eligible_and_matches_eager_tokens() -> None:
    inputs = _inputs(torch.device("cuda"))
    expected = torch.tensor([3, 5], device="cuda")
    logits, draws, parameters = inputs
    sample_top_k(logits, draws, parameters, 4)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()

    with torch.cuda.graph(graph):
        captured, valid = sample_top_k(logits, draws, parameters, 4)
    graph.replay()
    torch.cuda.synchronize()

    assert bool(valid.all())
    assert torch.equal(captured, expected)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_top_k_provider_accepts_every_serving_wave_row_count() -> None:
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(42)

    for row_count in range(1, 17):
        logits = torch.randn(
            (row_count, 257),
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        )
        draws = torch.rand(
            (row_count,),
            dtype=torch.float32,
            device=device,
            generator=generator,
        ).clamp_(1e-7, 1.0 - 1e-7)
        parameters = torch.tensor(
            [0.0, 1.0, 0.0],
            dtype=torch.float32,
            device=device,
        ).expand(row_count, 3)

        tokens, valid = sample_top_k(
            logits,
            draws,
            parameters,
            1,
        )

        assert bool(valid.all())
        assert torch.equal(tokens, torch.argmax(logits, dim=-1))


@pytest.mark.parametrize(
    "device",
    (
        "cpu",
        pytest.param(
            "cuda",
            marks=(
                pytest.mark.gpu,
                pytest.mark.skipif(
                    not torch.cuda.is_available(), reason="CUDA is required"
                ),
            ),
        ),
    ),
)
@pytest.mark.parametrize(
    "parameters",
    (
        pytest.param([1.0, 0.9, 0.0], id="top-p"),
        pytest.param([1.0, 1.0, 0.5], id="min-p"),
    ),
)
def test_top_k_zero_draw_skips_candidates_removed_by_filters(
    device, parameters
) -> None:
    # The top-3 candidates are IDs 1, 2, and 0 with probabilities about .517,
    # .468, and .016. Top-p .9 and min-p .5 each drop ID 0, the lowest-ID
    # candidate, so a draw of exactly 0.0 must select ID 1.
    logits = torch.tensor(
        [[0.5, 4.0, 3.9, 0.0, -1.0, -2.0]],
        dtype=torch.float32,
        device=device,
    )
    draws = torch.zeros((1,), dtype=torch.float32, device=device)

    tokens, valid = sample_top_k(
        logits,
        draws,
        torch.tensor([parameters], dtype=torch.float32, device=device),
        3,
    )

    assert valid.tolist() == [True]
    assert tokens.tolist() == [1]
