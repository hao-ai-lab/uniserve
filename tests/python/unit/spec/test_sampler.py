"""Speculative decoding sampler conformance."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from uniserve_worker.spec import speculative_sample_target_only

pytestmark = pytest.mark.unit


def _logits_from_probs(rows: list[list[float]]) -> torch.Tensor:
    return torch.log(torch.tensor(rows, dtype=torch.float32).clamp_min(1e-8))


def test_target_only_accepts_draft_prefix_and_samples_bonus():
    logits = _logits_from_probs(
        [
            [0.10, 0.80, 0.10, 0.00],
            [0.10, 0.20, 0.65, 0.05],
            [0.10, 0.20, 0.25, 0.45],
        ]
    )

    got = speculative_sample_target_only(
        logits,
        [1, 2],
        {"temperature": 1.0, "top_k": 0, "top_p": 1.0},
        uniform_samples=torch.tensor([0.20, 0.60]),
        uniform_sample_for_final=torch.tensor([0.95]),
    )

    assert got.num_accepted_tokens == 2
    assert got.sampled_token_id == 3
    torch.testing.assert_close(got.sampled_token_device.cpu(), torch.tensor([3]))


def test_target_only_rejection_samples_residual_without_rejected_candidate():
    logits = _logits_from_probs(
        [
            [0.50, 0.20, 0.30],
            [0.20, 0.20, 0.60],
        ]
    )

    got = speculative_sample_target_only(
        logits,
        [1],
        {"temperature": 1.0, "top_k": 0, "top_p": 1.0},
        uniform_samples=torch.tensor([0.90]),
        uniform_sample_for_final=torch.tensor([0.90]),
    )

    assert got.num_accepted_tokens == 0
    assert got.sampled_token_id == 2


def test_target_only_threshold_single_accepts_high_probability_candidate():
    logits = _logits_from_probs(
        [
            [0.30, 0.40, 0.30],
            [0.10, 0.80, 0.10],
        ]
    )

    got = speculative_sample_target_only(
        logits,
        [1],
        {"temperature": 1.0, "top_k": 0, "top_p": 1.0},
        threshold_single=0.35,
        uniform_samples=torch.tensor([0.99]),
        uniform_sample_for_final=torch.tensor([0.50]),
    )

    assert got.num_accepted_tokens == 1
    assert got.sampled_token_id == 1


def test_target_only_uses_sglang_relaxed_penalties_for_all_verify_rows():
    logits = torch.tensor(
        [
            [-20.0, 20.0, -20.0],
            [-20.0, 20.0, 19.0],
        ]
    )

    got = speculative_sample_target_only(
        logits,
        [1],
        {
            "temperature": 1.0,
            "top_k": 0,
            "top_p": 1.0,
            "repetition_penalty": 2.0,
        },
        recent=[],
        uniform_samples=torch.tensor([0.0]),
        uniform_sample_for_final=torch.tensor([0.1]),
    )

    assert got.num_accepted_tokens == 1
    assert got.sampled_token_id == 1


def test_qwen3_spec_row_uses_sglang_target_only_path_for_stochastic_verify():
    logits = torch.tensor(
        [
            [-40.0, 40.0, -40.0],
            [-40.0, -40.0, 40.0],
        ]
    )
    state = SimpleNamespace(
        sampling={"temperature": 1.0, "top_k": 0, "top_p": 1.0},
        kv_length=None,
    )

    def set_kv_length(length: int, *, lane: str) -> None:
        state.kv_length = (length, lane)

    state.set_kv_length = set_kv_length
    from uniserve_worker.contracts.forward_stats import ForwardStats
    from uniserve_worker.execution.engine import _verify_spec_row

    stats = ForwardStats()

    got = _verify_spec_row(
        logits,
        {"req_id": 7, "pos_range": [10, 11], "recent_tokens": []},
        (1,),
        state,
        stats=stats,
    )

    assert got["sampled_token_id"] == 2
    assert got["num_accepted_tokens"] == 1
    assert got["sampled_token_device"].cpu().tolist() == [2]
    assert got["sampled_position_device"].cpu().tolist() == [12]
    # KV-length advance is system-owned; the verify path records the text lane.
    assert state.kv_length == (12, "text")
    assert stats.spec_verify_path_counts == {"sglang_target_only": 1}
