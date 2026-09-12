"""Model input staging preserves columns supplied by host and device producers."""

import pytest
import torch

from uniserve_worker.execution.forward_batch import AttentionMetadata, AttentionMode, TokenSelection
from uniserve_worker.execution.input_buffers import InputBuffers, InputGeometry
from uniserve_worker.execution.rows import ForwardRow
from uniserve_worker.protocol.batch import ForwardMode

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
        geometry=InputGeometry(2, 2, 2, 1, 0),
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
