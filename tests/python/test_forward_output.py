"""Owned model outputs preserve logical values after producer storage is reused."""

import torch

from uniserve_worker.execution.batch import ExecutionOutput


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
            torch.testing.assert_close(actual, original + iteration, rtol=0, atol=0)
    retained[0].values[0].fill_(-1)
    torch.testing.assert_close(retained[0].values[1], expected[1], rtol=0, atol=0)
    torch.testing.assert_close(retained[1].values[0], expected[0] + 1, rtol=0, atol=0)


def test_forward_output_clone_preserves_empty_output():
    assert ExecutionOutput(()).clone().values == ()
