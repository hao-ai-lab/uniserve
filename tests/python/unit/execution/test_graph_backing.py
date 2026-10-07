"""Graphs replayed one at a time borrow their fixed I/O from one Scratch."""

import pytest
import torch

from uniserve.runtime import Scratch
from uniserve_worker.model_executor.cuda_graph import (
    stage_inputs,
    stage_outputs,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def scratch():
    owner = Scratch()
    yield owner
    owner.close()


def test_staged_inputs_keep_values_aliases_and_broadcasts(scratch):
    shared = torch.arange(6.0).reshape(2, 3)
    broadcast = torch.ones(1, 4).expand(3, 4)
    value = ((shared, broadcast), {"again": shared, "step": 2})
    staged = stage_inputs(value, scratch.view)

    (first, expanded), named = staged
    assert torch.equal(first, shared) and first.data_ptr() != shared.data_ptr()
    assert named["again"] is first and named["step"] == 2
    assert expanded.stride() == broadcast.stride()
    assert torch.equal(expanded, broadcast)

    # A later, smaller call's inputs land on the same bytes.
    later = stage_inputs(((torch.full((2, 2), 9.0),), {}), scratch.view)
    assert later[0][0].data_ptr() == first.data_ptr()
    assert torch.equal(later[0][0], torch.full((2, 2), 9.0))


def test_staged_outputs_write_results_into_shared_backing(scratch):
    def call(inputs):
        return {"video": inputs * 2, "count": 3}

    inputs = torch.arange(5.0)
    wrapped = stage_outputs(call, call(inputs), scratch.view)
    result = wrapped(inputs + 1)
    assert result["count"] == 3
    assert torch.equal(result["video"], (inputs + 1) * 2)

    # A second, smaller call's outputs view the same bytes as the first's.
    other = stage_outputs(call, call(torch.zeros(2)), scratch.view)
    smaller = other(torch.ones(2))
    assert smaller["video"].data_ptr() == result["video"].data_ptr()
    assert torch.equal(smaller["video"], torch.full((2,), 2.0))


def test_staged_outputs_reject_a_changed_output_structure(scratch):
    calls = iter(({"video": torch.zeros(2)}, (torch.zeros(2),)))
    wrapped = stage_outputs(
        lambda inputs: next(calls), {"video": torch.zeros(2)}, scratch.view
    )
    wrapped(None)
    with pytest.raises(ValueError, match="output structure changed"):
        wrapped(None)
