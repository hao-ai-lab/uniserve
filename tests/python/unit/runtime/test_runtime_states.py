"""Request-slot continuation state behavior."""

from __future__ import annotations

import pytest
import torch

from uniserve_worker.runtime.runtime_states import RuntimeStates

pytestmark = pytest.mark.unit


def _states() -> RuntimeStates:
    return RuntimeStates(
        request_pool_size=3,
        vocab_size=8,
        continuation_width=1,
        device="cpu",
    )


def test_release_and_reuse_start_from_canonical_request_state():
    states = _states()
    states.reset((2,), valid_cache_lengths=(5,), logical_lengths=(7,), sampling_positions=(9,))
    states.future_input_tokens[2, 0] = 6
    states.penalty_counts[2, 6] = 4
    states.predicates[2] = True
    states.selected_points[2] = 3

    states.release((2,))
    states.reset((2,))

    assert states.valid_cache_lengths[2].item() == 0
    assert states.logical_lengths[2].item() == 0
    assert states.sampling_positions[2].item() == 0
    assert states.future_input_tokens[2].tolist() == [1]
    assert states.penalty_counts[2].count_nonzero() == 0
    assert not states.predicates[2]
    assert states.selected_points[2].item() == 0


def test_request_mutations_never_change_the_padding_sentinel_row():
    states = _states()

    states.reset((1, 3), valid_cache_lengths=(4, 6), logical_lengths=(5, 7))
    states.publish_sampling(
        torch.tensor([1, 3]),
        tokens=torch.tensor([2, 4]),
        predicates=torch.tensor([True, False]),
        selected_points=torch.tensor([1, 2]),
        logical_lengths=torch.tensor([6, 9]),
        sampling_positions=torch.tensor([8, 10]),
    )

    assert states.valid_cache_lengths[0].item() == 0
    assert states.logical_lengths[0].item() == 0
    assert states.sampling_positions[0].item() == 0
    assert states.future_input_tokens[0].tolist() == [1]
    assert states.penalty_counts[0].count_nonzero() == 0
    assert not states.predicates[0]
    assert states.selected_points[0].item() == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cuda_host_admission_initializes_complete_request_rows():
    device = torch.device("cuda", torch.cuda.current_device())
    states = RuntimeStates(
        request_pool_size=3,
        vocab_size=8,
        continuation_width=2,
        device=device,
    )
    states.future_input_tokens[1, :].fill_(9)
    states.future_input_tokens[3, :].fill_(9)
    states.penalty_counts[1, :].fill_(4)
    states.penalty_counts[3, :].fill_(4)
    states.predicates[(1, 3),] = True
    states.selected_points[(1, 3),] = 7

    states.reset(
        (1, 3),
        valid_cache_lengths=(4, 6),
        logical_lengths=(5, 7),
        sampling_positions=(8, 10),
    )
    torch.cuda.synchronize(device)

    torch.testing.assert_close(
        states.future_input_tokens[(1, 3),].cpu(), torch.ones((2, 2), dtype=torch.int64)
    )
    assert states.penalty_counts[(1, 3),].count_nonzero().item() == 0
    assert states.predicates[(1, 3),].count_nonzero().item() == 0
    assert states.selected_points[(1, 3),].count_nonzero().item() == 0
    torch.testing.assert_close(
        states.valid_cache_lengths[(1, 3),].cpu(), torch.tensor((4, 6), dtype=torch.int32)
    )
    torch.testing.assert_close(
        states.logical_lengths[(1, 3),].cpu(), torch.tensor((5, 7), dtype=torch.int32)
    )
    torch.testing.assert_close(
        states.sampling_positions[(1, 3),].cpu(), torch.tensor((8, 10), dtype=torch.int64)
    )


@pytest.mark.parametrize("indices", [(0,), (4,), (1, 1)])
def test_request_mutations_reject_invalid_slot_sets(indices: tuple[int, ...]):
    with pytest.raises(ValueError):
        _states().reset(indices)
