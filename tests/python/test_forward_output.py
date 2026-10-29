"""Owned model outputs preserve logical values.

Values stay preserved after producer storage is reused.
"""

import pytest
import torch

from uniserve_worker.model_executor.output import ExecutionOutput


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


def _reused_after_free(*, record: bool) -> bool:
    """Free a lane-allocated output that the caller still reads; report reuse.

    The caller's stream is held busy before its read, so the read is still
    pending when the output is freed and the lane allocates again.
    """
    lane, caller = torch.cuda.Stream(), torch.cuda.current_stream()
    with torch.cuda.stream(lane):
        output = ExecutionOutput((torch.ones(1 << 20, device="cuda"),))
    caller.wait_stream(lane)
    if record:
        output.record_stream(caller)
    address = output.values[0].data_ptr()
    torch.cuda._sleep(1 << 30)
    total = output.values[0].sum()
    del output
    with torch.cuda.stream(lane):
        reused = torch.empty(1 << 20, device="cuda")
    result = reused.data_ptr() == address
    torch.cuda.synchronize()
    assert total.item() == 1 << 20
    return result


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_forward_output_recorded_on_its_reader_outlives_that_streams_reads():
    assert _reused_after_free(record=False)
    assert not _reused_after_free(record=True)
