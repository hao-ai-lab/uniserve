"""Exactness and capture checks for the sampling provider."""

from __future__ import annotations

import pytest
import torch
from uniserve_kernel.sampling import sample_top_k

from uniserve_worker.foundation.triton_compat import ensure_blackwell_ptxas

pytestmark = pytest.mark.unit


def _reference(
    logits: torch.Tensor,
    draws: torch.Tensor,
    penalty_token_ids: torch.Tensor,
    penalty_counts: torch.Tensor,
    parameters: torch.Tensor,
    top_k: int,
) -> torch.Tensor:
    work = logits.float().clone()
    for row in range(work.shape[0]):
        repetition, frequency, presence = parameters[row, 3:].tolist()
        for token_id, count in zip(
            penalty_token_ids[row].tolist(),
            penalty_counts[row].tolist(),
            strict=True,
        ):
            if count <= 0:
                continue
            value = work[row, token_id]
            value = value / repetition if value > 0 else value * repetition
            work[row, token_id] = value - frequency * count - presence
    work /= torch.where(parameters[:, :1] > 0, parameters[:, :1], 1)
    candidate_values, candidate_ids = torch.topk(work, top_k, dim=-1, sorted=True)
    for row in range(work.shape[0]):
        min_p = float(parameters[row, 2])
        if min_p > 0:
            candidate_values[
                row,
                candidate_values[row] < candidate_values[row, 0] + torch.log(parameters[row, 2]),
            ] = float("-inf")
        cumulative = torch.softmax(candidate_values[row], dim=-1).cumsum(dim=-1)
        drop = cumulative > parameters[row, 1]
        drop[1:] = drop[:-1].clone()
        drop[0] = False
        candidate_values[row, drop] = float("-inf")
    candidate_probabilities = torch.softmax(candidate_values, dim=-1)
    token_order = torch.argsort(candidate_ids, dim=-1)
    cumulative = candidate_probabilities.gather(1, token_order).cumsum(dim=-1)
    sampled_order = (cumulative < draws.unsqueeze(1)).sum(dim=-1).clamp_max(top_k - 1)
    sampled = token_order.gather(1, sampled_order.unsqueeze(1))[:, 0]
    selected = torch.where(parameters[:, 0] > 0, sampled, torch.zeros_like(sampled))
    return candidate_ids.gather(1, selected.unsqueeze(1))[:, 0]


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
    penalty_token_ids = torch.tensor(
        [
            [1, 3],
            [2, 5],
        ],
        dtype=torch.long,
        device=device,
    )
    penalty_counts = torch.tensor([[2.0, 1.0], [3.0, 0.0]], device=device)
    parameters = torch.tensor(
        [
            [0.7, 0.84, 0.08, 1.1, 0.2, 0.1],
            [0.0, 0.92, 0.0, 1.2, 0.1, 0.05],
        ],
        dtype=torch.float32,
        device=device,
    )
    return logits, draws, penalty_token_ids, penalty_counts, parameters


def test_top_k_provider_matches_the_full_expression() -> None:
    inputs = _inputs(torch.device("cpu"))

    tokens, valid = sample_top_k(*inputs, 4)

    assert valid.tolist() == [True, True]
    assert torch.equal(tokens, _reference(*inputs, 4))


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_top_k_provider_is_capture_eligible_and_matches_eager_tokens() -> None:
    assert ensure_blackwell_ptxas()
    inputs = _inputs(torch.device("cuda"))
    expected = _reference(*inputs, 4)
    sample_top_k(*inputs, 4)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()

    with torch.cuda.graph(graph):
        captured, valid = sample_top_k(*inputs, 4)
    graph.replay()
    torch.cuda.synchronize()

    assert bool(valid.all())
    assert torch.equal(captured, expected)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_top_k_provider_accepts_every_serving_wave_row_count() -> None:
    assert ensure_blackwell_ptxas()
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
        penalty_token_ids = torch.empty((row_count, 0), dtype=torch.long, device=device)
        penalty_counts = torch.empty((row_count, 0), dtype=torch.float32, device=device)
        parameters = torch.tensor(
            [0.0, 1.0, 0.0, 1.0, 0.0, 0.0],
            dtype=torch.float32,
            device=device,
        ).expand(row_count, 6)

        tokens, valid = sample_top_k(
            logits,
            draws,
            penalty_token_ids,
            penalty_counts,
            parameters,
            1,
        )

        assert bool(valid.all())
        assert torch.equal(tokens, torch.argmax(logits, dim=-1))
