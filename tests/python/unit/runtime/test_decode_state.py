"""Request continuation coordinates remain coherent as live batches change."""

from itertools import accumulate

import pytest
import torch

from uniserve_worker.model_executor._decode_inputs import (
    gather_request_decode_inputs,
)
from uniserve_worker.storage.decode_state import DecodeState

pytestmark = [pytest.mark.unit, pytest.mark.gpu]


@pytest.mark.parametrize("device", ("cpu", "cuda:0"))
@pytest.mark.parametrize("stride", (1, 2))
def test_tagged_decode_counts_only_active_tokens(device, stride):
    state = DecodeState(
        request_pool_size=3, vocab_size=8, continuation_width=2, device=device
    )
    slots = (3, 1)
    indices = torch.tensor([3, 0, 1, 0], device=device)[::2]
    tokens = torch.tensor([5 | (1 << 31), 0, 7 | (1 << 31), 0], device=device)[
        ::2
    ]
    predicates = torch.tensor([True, True, False, True], device=device)[::2]
    if stride == 1:
        indices, tokens, predicates = (
            value.contiguous() for value in (indices, tokens, predicates)
        )
    state.apply_tokens(
        slots,
        device_slots=indices,
        tokens=tokens,
        predicates=predicates,
        valid=torch.tensor([True, True], device=device),
        active=torch.tensor([True, False], device=device),
        penalty_bases=[state.penalty_counts[slot] for slot in slots],
    )

    assert state.future_input_tokens[:, 0].tolist() == [1, 7, 1, 5]
    assert state.valid_cache_lengths.tolist() == [0, 1, 0, 1]
    assert state.predicates.tolist() == [False, False, False, True]
    expected = torch.zeros_like(state.penalty_counts)
    expected[3, 5] = 1
    torch.testing.assert_close(state.penalty_counts, expected)


def test_device_indices_cannot_reset_a_cpu_states_padding_slot():
    state = DecodeState(
        request_pool_size=1, vocab_size=8, continuation_width=1, device="cpu"
    )
    indices = torch.tensor([0], device="cuda:0")
    with pytest.raises(ValueError, match="outside runtime-state capacity"):
        state.reset(indices, valid_cache_lengths=[7])
    assert state.valid_cache_lengths.tolist() == [0, 0]


@pytest.mark.parametrize("capacity", (17, 257))
@pytest.mark.parametrize("axes", (1, 3))
def test_decode_export_and_inputs_follow_live_request_coordinates(
    capacity, axes
):
    device = torch.device("cuda:0")
    state = DecodeState(
        request_pool_size=capacity,
        vocab_size=capacity + 1,
        continuation_width=1,
        device=device,
    )
    slots = list(range(1, capacity + 1))
    state.reset(
        slots,
        valid_cache_lengths=[0] * capacity,
        logical_lengths=[1] * capacity,
        sampling_positions=[16] * capacity,
    )
    expected_tokens = [1] * (capacity + 1)
    expected_lengths = [0] * (capacity + 1)
    expected_positions = [0] + [1] * capacity
    expected_sampling = [0] + [16] * capacity
    expected_predicates = [False] * (capacity + 1)
    # Table 0 belongs to a full-attention group and table 1 to a windowed
    # group whose readers need two history tokens; both use two-token pages.
    # Unit ``(table, slot, column)`` is numbered in row-major order.
    units = torch.arange(
        2 * (capacity + 1) * 5, dtype=torch.int32, device=device
    ).reshape(2, capacity + 1, 5)
    starts = torch.zeros((2, capacity + 1), dtype=torch.int32, device=device)
    shapes = torch.tensor(
        [[0, 2, -1], [1, 2, 2]], dtype=torch.int32, device=device
    )
    slot_indices = torch.zeros(capacity, dtype=torch.int64, device=device)
    tables = torch.full((2, capacity, 5), -1, dtype=torch.int32, device=device)
    start_pages = torch.full(
        (2, capacity), -1, dtype=torch.int32, device=device
    )
    outputs = {
        name: torch.empty(capacity, dtype=dtype, device=device)
        for name, dtype in (
            ("input_ids", torch.int64),
            ("cache_lengths", torch.int32),
            ("query_lengths", torch.int32),
        )
    }
    outputs["write_indices"] = torch.empty(
        (2, capacity), dtype=torch.int64, device=device
    )
    outputs["positions"] = torch.full(
        (axes, capacity), -9, dtype=torch.int64, device=device
    )
    for name in ("query_offsets", "prefix_offsets"):
        outputs[name] = torch.empty(
            capacity + 1, dtype=torch.int32, device=device
        )

    # Exercise a singleton, a partial batch and the non-power-of-two capacity,
    # then shrink again while changing the input width.
    for count, width in ((1, 1), (3, 3), (capacity, 5), (1, 3)):
        live = slots[::-1][:count]
        indices = torch.tensor(live, dtype=torch.int64, device=device)
        tokens = torch.tensor(
            [slot | (1 << 31) for slot in live], device=device
        )
        predicates = torch.tensor(
            [slot % 2 == 0 for slot in live], device=device
        )
        state.apply_tokens(
            live,
            device_slots=indices,
            tokens=tokens,
            predicates=predicates,
            valid=torch.ones(count, dtype=torch.bool, device=device),
            active=torch.ones(count, dtype=torch.bool, device=device),
            penalty_bases=[None] * count,
        )
        for slot in live:
            expected_tokens[slot] = slot
            expected_lengths[slot] += 1
            expected_positions[slot] += 1
            expected_sampling[slot] += 1
            expected_predicates[slot] = slot % 2 == 0
        assert state.future_input_tokens[:, 0].tolist() == expected_tokens
        assert state.valid_cache_lengths.tolist() == expected_lengths
        assert state.logical_lengths.tolist() == expected_positions
        assert state.sampling_positions.tolist() == expected_sampling
        assert state.predicates.tolist() == expected_predicates

        # The windowed group has retired every page before the one its
        # readers' windows reach, so its first gathered page is its first
        # installed page.
        first = [max(length - 2, 0) // 2 for length in expected_lengths]
        starts[1].copy_(torch.tensor(first, dtype=torch.int32))

        slot_indices[:count].copy_(indices)
        gather_request_decode_inputs(
            request_pool_indices=slot_indices,
            request_unit_tables=units,
            request_start_pages=starts,
            table_shapes=shapes,
            request_cache_lengths=state.valid_cache_lengths,
            request_tokens=state.future_input_tokens[:, 0],
            request_positions=state.logical_lengths,
            block_tables=tables,
            start_pages=start_pages,
            rows=count,
            columns=width,
            **outputs,
        )
        padding = capacity - count
        assert slot_indices.tolist() == live + [0] * padding
        assert outputs["input_ids"].tolist() == live + [1] * padding
        assert outputs["positions"].tolist() == [
            [expected_positions[s] for s in live] + [0] * padding
        ] + [[0] * capacity] * (axes - 1)
        assert (
            outputs["cache_lengths"].tolist()
            == [expected_lengths[s] for s in live] + [0] * padding
        )
        assert outputs["query_lengths"].tolist() == [1] * capacity
        for table in range(2):
            assert (
                tables[table, :, :width].tolist()
                == units[table, live, :width].tolist() + [[0] * width] * padding
            )
        assert start_pages.tolist() == [
            [0] * capacity,
            [first[s] for s in live] + [0] * padding,
        ]
        assert outputs["query_offsets"].tolist() == list(range(capacity + 1))
        assert outputs["prefix_offsets"].tolist() == list(
            accumulate(
                [expected_lengths[s] for s in live] + [0] * padding, initial=0
            )
        )
        # Each token writes the unit of its page, counted from the table's
        # first installed page, at its offset in the page.
        writes = [
            [
                int(units[table, s, length // 2 - start]) * 2 + length % 2
                for s in live
                for length, start in (
                    (expected_lengths[s], first[s] if table else 0),
                )
            ]
            + [-1] * padding
            for table in range(2)
        ]
        assert outputs["write_indices"].tolist() == writes
