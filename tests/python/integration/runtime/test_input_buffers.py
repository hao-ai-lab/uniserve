"""Model input staging preserves columns supplied by host and device producers."""

import pytest
import torch

from uniserve.attention.metadata import AttentionMetadata, AttentionMode
from uniserve.model.tensors import TokenSelection
from uniserve_worker.execution.input_buffers import InputBufferConfig, InputBuffers
from uniserve_worker.execution.rows import ForwardRow
from uniserve_worker.protocol.batch import ForwardMode
from uniserve_worker.runtime.decode_state import DecodeState

pytestmark = [pytest.mark.integration, pytest.mark.gpu]


@pytest.mark.parametrize("devices", (("cpu", "cuda:0"), ("cuda:0", "cpu")))
@pytest.mark.parametrize("position_dtype", (torch.int32, torch.int64))
def test_token_positions_preserve_row_order_across_source_devices(devices, position_dtype):
    # A host-created prefix may share a call with a device continuation. Both
    # remain ordinary numerical rows, regardless of where their values originate.
    rows = tuple(
        ForwardRow(
            forward_mode=ForwardMode.PREFILL,
            token_ids=torch.tensor([token], dtype=torch.long, device=device),
            positions=torch.tensor([position], dtype=position_dtype, device=device),
            selection=TokenSelection.LAST_LOGITS,
            request_pool_idx=index + 1,
        )
        for index, (device, token, position) in enumerate(
            zip(devices, (5, 9), (17, 23), strict=True)
        )
    )
    attention = AttentionMetadata(
        attention_mode=AttentionMode.DENSE,
        prefix_lens=torch.zeros(2, dtype=torch.int32),
        query_lens=torch.ones(2, dtype=torch.int32),
        out_cache_loc=torch.empty(0, dtype=torch.int64),
        has_cache_writes=False,
        prefix_lens_cpu=(0, 0),
        query_lens_cpu=(1, 1),
        seq_lens_cpu=(1, 1),
    )
    buffers = InputBuffers(
        config=InputBufferConfig(2, 2, 2, 1, 0),
        device="cuda:0",
    )
    try:
        batch = buffers.stage(rows, forward_mode=ForwardMode.PREFILL, attention=attention)
        assert batch.input_ids is not None and batch.positions is not None
        assert batch.input_ids.dtype == batch.positions.dtype == torch.int64
        assert batch.input_ids.cpu().tolist() == [5, 9]
        assert batch.positions.cpu().tolist() == [17, 23]
    finally:
        torch.cuda.synchronize()
        buffers.close()


@pytest.mark.parametrize("continuation_first", (False, True))
def test_mixed_forward_reads_current_continuation_in_row_order(continuation_first):
    states = DecodeState(
        request_pool_size=2,
        vocab_size=32,
        continuation_width=1,
        device="cuda:0",
    )
    prefix = ForwardRow(
        forward_mode=ForwardMode.PREFILL,
        token_ids=torch.tensor([5, 7]),
        positions=torch.tensor([17, 18]),
        selection=TokenSelection.LAST_LOGITS,
        request_pool_idx=1,
    )
    continuation = ForwardRow(
        forward_mode=ForwardMode.DECODE,
        selection=TokenSelection.LAST_LOGITS,
        request_pool_idx=2,
        request_indexed_decode=True,
    )
    rows = (continuation, prefix) if continuation_first else (prefix, continuation)
    query_lens = (1, 2) if continuation_first else (2, 1)
    attention = AttentionMetadata(
        attention_mode=AttentionMode.DENSE,
        prefix_lens=torch.zeros(2, dtype=torch.int32),
        query_lens=torch.tensor(query_lens, dtype=torch.int32),
        out_cache_loc=torch.empty(0, dtype=torch.int64),
        has_cache_writes=False,
        prefix_lens_cpu=(0, 0),
        query_lens_cpu=query_lens,
        seq_lens_cpu=query_lens,
    )
    buffers = InputBuffers(config=InputBufferConfig(2, 3, 3, 1, 0), device="cuda:0")
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
            batch = buffers.stage(
                rows, forward_mode=ForwardMode.PREFILL, attention=attention, states=states
            )
            expected_tokens = [token, 5, 7] if continuation_first else [5, 7, token]
            expected_positions = [position, 17, 18] if continuation_first else [17, 18, position]
            assert batch.input_ids is not None and batch.positions is not None
            assert batch.input_ids.cpu().tolist() == expected_tokens
            assert batch.positions.cpu().tolist() == expected_positions
    finally:
        torch.cuda.synchronize()
        buffers.close()
