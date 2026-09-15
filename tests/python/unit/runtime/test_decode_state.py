"""Request continuation coordinates remain coherent as live batches change."""

from itertools import accumulate

import pytest
import torch

from uniserve_worker.ops.staging import gather_request_decode_inputs
from uniserve_worker.runtime.decode_state import DecodeState

pytestmark = [pytest.mark.unit, pytest.mark.gpu]


@pytest.mark.parametrize("capacity", (17, 257))
@pytest.mark.parametrize("axes", (1, 3))
def test_decode_publication_and_staging_follow_live_request_coordinates(capacity, axes):
    device = torch.device("cuda:0")
    state = DecodeState(
        request_pool_size=capacity, vocab_size=capacity + 1, continuation_width=1, device=device
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
    pages = torch.arange(2 * (capacity + 1) * 5, dtype=torch.int32, device=device).reshape(
        2, capacity + 1, 5
    )
    staging_slots = torch.zeros(capacity, dtype=torch.int64, device=device)
    table_storage = torch.full((capacity, 5), -1, dtype=torch.int32, device=device)
    outputs = {
        name: torch.empty(capacity, dtype=dtype, device=device)
        for name, dtype in (
            ("input_ids", torch.int64),
            ("cache_lengths", torch.int32),
            ("query_lengths", torch.int32),
            ("write_indices", torch.int64),
        )
    }
    outputs["positions"] = torch.full((axes, capacity), -9, dtype=torch.int64, device=device)
    for name in ("query_offsets", "prefix_offsets"):
        outputs[name] = torch.empty(capacity + 1, dtype=torch.int32, device=device)

    # Exercise a singleton, a partial batch and the non-power-of-two capacity,
    # then shrink again while changing the page horizon and selected KV group.
    for count, width, group in ((1, 1, 0), (3, 3, 1), (capacity, 5, 0), (1, 3, 1)):
        live = slots[::-1][:count]
        indices = torch.tensor(live, dtype=torch.int64, device=device)
        tokens = torch.tensor([slot | (1 << 31) for slot in live], device=device)
        predicates = torch.tensor([slot % 2 == 0 for slot in live], device=device)
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

        staging_slots[:count].copy_(indices)
        tables = table_storage[:, :width]
        gather_request_decode_inputs(
            request_pool_indices=staging_slots,
            request_page_tables=pages,
            request_cache_lengths=state.valid_cache_lengths,
            request_tokens=state.future_input_tokens[:, 0],
            request_positions=state.logical_lengths,
            block_tables=tables,
            rows=count,
            group_id=group,
            page_size=2,
            **outputs,
        )
        padding = capacity - count
        assert staging_slots.tolist() == live + [0] * padding
        assert outputs["input_ids"].tolist() == live + [1] * padding
        assert (
            outputs["positions"].tolist()
            == [[expected_positions[s] for s in live] + [0] * padding]
            + [[0] * capacity] * (axes - 1)
        )
        assert (
            outputs["cache_lengths"].tolist() == [expected_lengths[s] for s in live] + [0] * padding
        )
        assert outputs["query_lengths"].tolist() == [1] * capacity
        assert tables.tolist() == pages[group, live, :width].tolist() + [[0] * width] * padding
        assert outputs["query_offsets"].tolist() == list(range(capacity + 1))
        assert (
            outputs["prefix_offsets"].tolist()
            == list(accumulate([expected_lengths[s] for s in live] + [0] * padding, initial=0))
        )
        assert (
            outputs["write_indices"].tolist()
            == [(group * (capacity + 1) + s) * 10 + expected_lengths[s] for s in live]
            + [-1] * padding
        )
