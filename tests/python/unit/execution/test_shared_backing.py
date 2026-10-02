"""Graphs replayed one at a time share their input and output backing."""

import pytest
import torch

from uniserve_worker.model_executor.cuda_graph import SharedBacking

pytestmark = pytest.mark.unit


def test_views_pack_from_the_arena_start_and_refuse_larger_calls():
    backing = SharedBacking()
    first = backing.views(
        "inputs", (torch.zeros(3, 5), torch.zeros(7, dtype=torch.int64))
    )
    assert first is not None
    arena = first[0].untyped_storage().data_ptr()
    assert first[0].data_ptr() == arena
    # Each tensor starts on an aligned offset after the previous one.
    assert first[1].data_ptr() - arena == SharedBacking.ALIGNMENT
    assert (first[0].shape, first[1].dtype) == ((3, 5), torch.int64)

    # A smaller call reuses the same bytes; a larger one does not fit.
    second = backing.views("inputs", (torch.zeros(2, 2),))
    assert second is not None and second[0].data_ptr() == arena
    assert backing.views("inputs", (torch.zeros(1024),)) is None
    # Roles have separate arenas.
    outputs = backing.views("outputs", (torch.zeros(2),))
    assert outputs is not None and outputs[0].data_ptr() != arena


def test_staged_inputs_keep_values_aliases_and_broadcasts():
    backing = SharedBacking()
    shared = torch.arange(6.0).reshape(2, 3)
    broadcast = torch.ones(1, 4).expand(3, 4)
    value = ((shared, broadcast), {"again": shared, "step": 2})
    staged = backing.stage_inputs(value)

    (first, expanded), named = staged
    assert torch.equal(first, shared) and first.data_ptr() != shared.data_ptr()
    assert named["again"] is first and named["step"] == 2
    assert expanded.stride() == broadcast.stride()
    assert torch.equal(expanded, broadcast)

    # A later graph's inputs land on the same arena bytes.
    later = backing.stage_inputs(((torch.full((2, 3), 9.0),), {}))
    assert later[0][0].data_ptr() == first.data_ptr()


def test_staging_falls_back_to_private_backing_when_inputs_do_not_fit():
    backing = SharedBacking()
    small = backing.stage_inputs((torch.zeros(4),))
    large = torch.arange(1024.0)
    staged = backing.stage_inputs((large,))
    assert torch.equal(staged[0], large)
    assert staged[0].data_ptr() != small[0].data_ptr()


def test_output_call_writes_results_into_the_shared_arena():
    backing = SharedBacking()

    def call(inputs):
        return {"video": inputs * 2, "count": 3}

    inputs = torch.arange(5.0)
    wrapped = backing.output_call(call, call(inputs))
    result = wrapped(inputs + 1)
    assert result["count"] == 3
    assert torch.equal(result["video"], (inputs + 1) * 2)

    # A second call's outputs view the same bytes as the first's.
    other = backing.output_call(call, call(torch.zeros(2)))
    assert (
        other(torch.ones(2))["video"].data_ptr() == result["video"].data_ptr()
    )

    # Outputs larger than the arena keep the call unwrapped.
    assert backing.output_call(call, call(torch.zeros(1024))) is call
