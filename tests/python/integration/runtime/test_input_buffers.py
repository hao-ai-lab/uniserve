"""Model input staging preserves columns supplied by host producers.

Columns supplied by device producers are preserved too.
"""

import pytest
import torch

from uniserve.nn.attention import PagedInput
from uniserve_worker.model_executor.input_batch import TokenRow
from uniserve_worker.model_executor.input_buffers import (
    TokenBufferConfig,
    TokenBuffers,
)
from uniserve_worker.protocol.call import ForwardMode
from uniserve_worker.sampling.metadata import TokenSelection
from uniserve_worker.storage.decode_state import DecodeState

pytestmark = [pytest.mark.integration, pytest.mark.gpu]


@pytest.mark.parametrize("devices", (("cpu", "cuda:0"), ("cuda:0", "cpu")))
@pytest.mark.parametrize("position_dtype", (torch.int32, torch.int64))
def test_token_positions_preserve_row_order_across_source_devices(
    devices, position_dtype
):
    # A host-created prefix may share a call with a device continuation. Both
    # remain ordinary numerical rows, regardless of where their values
    # originate.
    rows = tuple(
        TokenRow(
            forward_mode=ForwardMode.PREFILL,
            token_ids=torch.tensor([token], dtype=torch.long, device=device),
            positions=torch.tensor(
                [position], dtype=position_dtype, device=device
            ),
            selection=TokenSelection.LAST_LOGITS,
            request_pool_idx=index + 1,
        )
        for index, (device, token, position) in enumerate(
            zip(devices, (5, 9), (17, 23), strict=True)
        )
    )
    attention = PagedInput.from_blocks(
        blocks=((0,), (1,)),
        query_lengths=(1, 1),
        prefix_lengths=(0, 0),
        block_size=4,
        causal=True,
        device="cpu",
    )
    buffers = TokenBuffers(
        config=TokenBufferConfig(
            max_rows=2,
            max_tokens=2,
            max_text_tokens=2,
            max_blocks_per_row=1,
            hidden_size=0,
        ),
        device="cuda:0",
    )
    try:
        batch = buffers.prepare_inputs(
            rows, forward_mode=ForwardMode.PREFILL, attention=attention
        )
        assert (
            batch.inputs.input_ids is not None
            and batch.inputs.positions is not None
        )
        assert (
            batch.inputs.input_ids.dtype
            == batch.inputs.positions.dtype
            == torch.int64
        )
        assert batch.inputs.input_ids.cpu().tolist() == [5, 9]
        assert batch.inputs.positions.cpu().tolist() == [17, 23]
    finally:
        torch.cuda.synchronize()
        buffers.close()


@pytest.mark.parametrize("continuation_first", (False, True))
def test_mixed_forward_reads_current_continuation_in_row_order(
    continuation_first,
):
    states = DecodeState(
        request_pool_size=2,
        vocab_size=32,
        continuation_width=1,
        device="cuda:0",
    )
    prefix = TokenRow(
        forward_mode=ForwardMode.PREFILL,
        token_ids=torch.tensor([5, 7]),
        positions=torch.tensor([17, 18]),
        selection=TokenSelection.LAST_LOGITS,
        request_pool_idx=1,
    )
    continuation = TokenRow(
        forward_mode=ForwardMode.DECODE,
        selection=TokenSelection.LAST_LOGITS,
        request_pool_idx=2,
        request_indexed_decode=True,
    )
    rows = (
        (continuation, prefix) if continuation_first else (prefix, continuation)
    )
    query_lens = (1, 2) if continuation_first else (2, 1)
    attention = PagedInput.from_blocks(
        blocks=((0,), (1,)),
        query_lengths=query_lens,
        prefix_lengths=(0, 0),
        block_size=4,
        causal=True,
        device="cpu",
    )
    buffers = TokenBuffers(
        config=TokenBufferConfig(
            max_rows=2,
            max_tokens=3,
            max_text_tokens=3,
            max_blocks_per_row=1,
            hidden_size=0,
        ),
        device="cuda:0",
    )
    enabled = torch.ones(1, dtype=torch.bool, device="cuda:0")
    try:
        # Reusing the same row must read the latest committed continuation,
        # including when its packed token offset changes with row ordering.
        for token, position in ((9, 23), (13, 29)):
            states.apply_tokens(
                (2,),
                tokens=torch.tensor([token], device="cuda:0"),
                predicates=enabled,
                valid=enabled,
                active=enabled,
                penalty_bases=(None,),
                logical_position=position,
                sampling_position=position,
            )
            batch = buffers.prepare_inputs(
                rows,
                forward_mode=ForwardMode.PREFILL,
                attention=attention,
                states=states,
            )
            expected_tokens = (
                [token, 5, 7] if continuation_first else [5, 7, token]
            )
            expected_positions = (
                [position, 17, 18] if continuation_first else [17, 18, position]
            )
            assert (
                batch.inputs.input_ids is not None
                and batch.inputs.positions is not None
            )
            assert batch.inputs.input_ids.cpu().tolist() == expected_tokens
            assert batch.inputs.positions.cpu().tolist() == expected_positions
    finally:
        torch.cuda.synchronize()
        buffers.close()


def test_indexed_decode_stages_live_tokens_cache_addresses_and_finish_controls():  # noqa: E501
    from tests.python.fixtures.cache import mha_pool

    pool = mha_pool(
        num_layers=1,
        num_kv_heads=1,
        head_dim=8,
        dtype=torch.float32,
        total_layers=1,
        total_kv_heads=1,
        num_pages=4,
        page_size=4,
        device="cuda:0",
        request_pool_size=2,
        max_blocks_per_request=2,
    )
    states = DecodeState(
        request_pool_size=2,
        vocab_size=32,
        continuation_width=1,
        device="cuda:0",
    )
    buffers = TokenBuffers(
        config=TokenBufferConfig(
            max_rows=2,
            max_tokens=2,
            max_text_tokens=2,
            max_blocks_per_row=2,
            hidden_size=0,
        ),
        device="cuda:0",
    )
    slots = torch.tensor([1, 2], device="cuda:0")
    enabled = torch.ones(2, dtype=torch.bool, device="cuda:0")
    try:
        pool.block_tables.install(((1, 0, (1, 3), 8), (2, 0, (2,), 4)))
        for iteration, order in enumerate(((1, 2), (2, 1))):
            lengths = (5 + iteration, 2 + iteration)
            pool.block_tables.set_verified(
                slots, torch.tensor(lengths, device="cuda:0")
            )
            for slot, token in ((1, 7 + iteration), (2, 11 + iteration)):
                states.apply_tokens(
                    (slot,),
                    tokens=torch.tensor([token], device="cuda:0"),
                    predicates=enabled[:1],
                    valid=enabled[:1],
                    active=enabled[:1],
                    penalty_bases=(None,),
                    logical_position=23 + iteration,
                    sampling_position=23 + iteration,
                )
            rows = tuple(
                TokenRow(
                    forward_mode=ForwardMode.DECODE,
                    request_pool_idx=slot,
                    seq_len=lengths[slot - 1],
                    write_kv=True,
                    request_indexed_decode=True,
                    selection=TokenSelection.LAST_LOGITS,
                    decode_predicate=enabled[:1],
                    decode_predicate_tagged=True,
                    decode_force_finish=iteration == 0 and slot == 1,
                )
                for slot in order
            )
            batch = buffers.prepare_inputs(
                rows,
                forward_mode=ForwardMode.DECODE,
                cache=pool,
                tables=pool.block_tables,
                states=states,
            )
            assert batch.request_pool_indices.cpu().tolist() == list(order)
            assert batch.inputs.input_ids.cpu().tolist() == [
                (7 if slot == 1 else 11) + iteration for slot in order
            ]
            assert batch.inputs.positions.cpu().tolist() == [23 + iteration] * 2
            attention = batch.inputs.attention
            assert attention.prefixes.values.cpu().tolist() == [
                lengths[slot - 1] for slot in order
            ]
            assert attention.queries.offsets.cpu().tolist() == [0, 1, 2]
            assert attention.write_indices.cpu().tolist() == [
                (13 if slot == 1 else 10) + iteration for slot in order
            ]
            assert batch.decode_force_finish.cpu().tolist() == [
                iteration == 0 and slot == 1 for slot in order
            ]
    finally:
        torch.cuda.synchronize()
        buffers.close()
        pool.close()
