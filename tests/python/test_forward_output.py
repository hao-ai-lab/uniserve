"""Owned model outputs preserve logical values.

Values stay preserved after producer storage is reused.
"""

import torch

from uniserve_worker.model_executor.output import ExecutionOutput
from uniserve_worker.protocol.output import ForwardStats


def test_forward_output_clone_preserves_ragged_shapes_dtypes_and_owned_values():
    values = (
        torch.arange(12, dtype=torch.float32).view(3, 4).t(),
        torch.arange(6, dtype=torch.float32).view(2, 3),
        torch.arange(3, dtype=torch.bfloat16),
        torch.empty(0, 3, dtype=torch.float32),
        torch.tensor(5.0),
    )
    output = ExecutionOutput(values)
    expected = tuple(value.clone() for value in values)
    retained = []
    for _ in range(3):
        retained.append(output.clone())
        for value in values:
            value.add_(1)
    for iteration, copied in enumerate(retained):
        for actual, original in zip(copied.values, expected, strict=True):
            torch.testing.assert_close(
                actual, original + iteration, rtol=0, atol=0
            )
    retained[0].values[0].fill_(-1)
    torch.testing.assert_close(
        retained[0].values[1], expected[1], rtol=0, atol=0
    )
    torch.testing.assert_close(
        retained[1].values[0], expected[0] + 1, rtol=0, atol=0
    )


def test_forward_output_clone_preserves_empty_output():
    assert ExecutionOutput(()).clone().values == ()


def test_combined_microbatches_preserve_rows_and_accumulate_observations():
    first = ExecutionOutput(
        (torch.arange(6).reshape(2, 3),),
        stats=ForwardStats(
            mode_counts={"prefill": 1},
            mode_tokens={"prefill": 2},
            cuda_graph_replays=1,
            cuda_graph_unpadded_tokens=2,
            cuda_graph_padded_tokens=2,
        ),
    )
    second = ExecutionOutput(
        (torch.ones(1, 3), torch.zeros(3, 3)),
        stats=ForwardStats(
            mode_counts={"prefill": 1},
            mode_tokens={"prefill": 4},
            cuda_graph_replays=1,
            cuda_graph_unpadded_tokens=4,
            cuda_graph_padded_tokens=4,
        ),
    )
    combined = ExecutionOutput.combine(iter((first, second)))
    saved = combined.clone()
    first.values[0].fill_(-1)
    second.values[1].fill_(3)

    for actual, expected in zip(
        saved.values,
        (torch.arange(6).reshape(2, 3), torch.ones(1, 3), torch.zeros(3, 3)),
        strict=True,
    ):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert saved.stats.mode_tokens == {"prefill": 6}
    assert saved.stats.mode_counts == {"prefill": 2}
    assert saved.stats.cuda_graph_replays == 2
    assert saved.stats.cuda_graph_unpadded_tokens == 6
    assert saved.stats.cuda_graph_padded_tokens == 6
    assert first.stats.mode_tokens == {"prefill": 2}
    assert ExecutionOutput.combine(()).values == ()
