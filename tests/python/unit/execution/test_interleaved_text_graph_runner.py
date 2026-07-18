from types import SimpleNamespace

import pytest
import torch

from uniserve_worker.models.interleaved_text import (
    InterleavedTextDecodeGraphRunner,
    InterleavedTextPrefillGraphRunner,
    _InterleavedDecodeGraphPast,
    _Row,
)
from uniserve_worker.models.sensenova.model import _SenseNovaDecoderModel


class _Owner:
    pass


def _row(pool, *, block_ids=None, base_len=3):
    return _Row(
        text_cache=object(),
        past_cache=SimpleNamespace(pool=pool),
        token_id=7,
        pos=base_len,
        base_len=base_len,
        block_ids=list(block_ids or [1]),
        token_tensor=None,
    )


def test_interleaved_decode_graph_padding_uses_reserved_block_offsets():
    pool = SimpleNamespace(num_blocks=64, block_size=16, reserved_block_ids=(62, 63))
    runner = InterleavedTextDecodeGraphRunner()
    driver = SimpleNamespace(owner=_Owner())
    rows = [_row(pool, block_ids=[3], base_len=11)]

    padded = runner._pad_rows(driver, rows, 4, pool)

    assert padded[:1] == rows
    assert [row.block_ids for row in padded[1:]] == [[62, 63], [62, 63], [62, 63]]
    assert [row.base_len for row in padded[1:]] == [0, 1, 2]
    assert [row.pos for row in padded[1:]] == [0, 1, 2]
    assert [row.token_id for row in padded[1:]] == [0, 0, 0]


def test_interleaved_decode_graph_padding_requires_reserved_blocks():
    pool = SimpleNamespace(num_blocks=64, block_size=16)
    runner = InterleavedTextDecodeGraphRunner()
    driver = SimpleNamespace(owner=object())

    with pytest.raises(Exception, match="reserved KV padding blocks"):
        runner._pad_rows(driver, [_row(pool)], 2, pool)


def test_interleaved_decode_graph_fences_staging_after_graph_submission():
    calls = []
    device = torch.device("cuda:0")
    pool = SimpleNamespace(
        num_blocks=64,
        block_size=16,
        k=SimpleNamespace(device=device),
    )
    row = _row(pool)
    runner = InterleavedTextDecodeGraphRunner()
    runner._prepare = lambda driver, ops: ([row], None)
    runner._decode = SimpleNamespace(
        resolve_bucket=lambda batch: batch,
        maybe_run_host_inputs=lambda **kwargs: calls.append("submit"),
    )

    class _Stager:
        def acquire_slot(self, *, device):
            calls.append(("acquire", device))
            return "slot"

        def mark_slot_submitted(self, slot, *, device):
            calls.append(("mark", slot, device))

    runner._stager = _Stager()

    assert runner.maybe_run_batch(SimpleNamespace(owner=_Owner()), [{}]) is None
    assert calls == [
        ("acquire", device),
        "submit",
        ("mark", "slot", device),
    ]


def test_interleaved_decode_graph_forward_uses_cache_position_without_index_sidecar():
    runner = InterleavedTextDecodeGraphRunner()
    input_ids = torch.tensor([[5], [6]], dtype=torch.long)
    positions = torch.tensor([[11], [12]], dtype=torch.long)
    cache = SimpleNamespace(pool=object(), base_len=11)
    state = SimpleNamespace(batch_size=2, input_ids=input_ids, positions=positions, cache=cache)
    calls = []

    class Owner:
        def interleaved_text_forward(self, **kwargs):
            calls.append(kwargs)
            assert "indexes" not in kwargs
            assert kwargs["cache_position"].data_ptr() == positions.reshape(-1).data_ptr()
            assert kwargs["text_only_rope"] is True
            return SimpleNamespace(logits=torch.arange(6, dtype=torch.float32).view(2, 1, 3))

    logits = runner._forward(SimpleNamespace(owner=Owner()), state)

    assert calls
    assert tuple(logits.shape) == (2, 3)
    torch.testing.assert_close(logits, torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]))


def test_interleaved_decode_graph_past_accepts_one_token_per_batch_row():
    cache = SimpleNamespace(pool=object(), block_ids_by_row=[[1], [2]])
    past = _InterleavedDecodeGraphPast(cache)

    assert past.request_cache_for_update(layer_idx=0, n_tokens=1) is cache

    with pytest.raises(Exception, match="one token per row"):
        past.request_cache_for_update(layer_idx=0, n_tokens=2)


def test_interleaved_prefill_graph_selects_last_real_token_from_all_logits():
    runner = InterleavedTextPrefillGraphRunner()
    input_ids = torch.tensor([5, 6, 0, 0], dtype=torch.long)
    positions = torch.tensor([0, 1, 0, 0], dtype=torch.long)
    cache = SimpleNamespace(pool=object(), base_len=0)
    state = SimpleNamespace(
        num_tokens=4,
        batch_size=1,
        input_ids=input_ids,
        positions=positions,
        last_token_indices=torch.tensor([1], dtype=torch.long),
        cache=cache,
    )
    calls = []

    class Owner:
        def interleaved_text_forward(self, **kwargs):
            calls.append(kwargs)
            assert kwargs["return_all_logits"] is True
            return SimpleNamespace(logits=torch.arange(12, dtype=torch.float32).view(1, 4, 3))

    logits = runner._forward(SimpleNamespace(owner=Owner()), state)

    assert calls
    torch.testing.assert_close(logits, torch.tensor([[3.0, 4.0, 5.0]]))


def test_interleaved_prefill_graph_selects_packed_row_logits():
    runner = InterleavedTextPrefillGraphRunner()
    input_ids = torch.tensor([5, 6, 7, 8, 9, 10, 0, 0], dtype=torch.long)
    positions = torch.arange(8, dtype=torch.long)
    cache = SimpleNamespace(pool=object(), base_len=0)
    state = SimpleNamespace(
        num_tokens=8,
        batch_size=4,
        input_ids=input_ids,
        positions=positions,
        last_token_indices=torch.tensor([1, 4, 5, 0], dtype=torch.long),
        cache=cache,
    )

    class Owner:
        def interleaved_text_forward(self, **kwargs):
            return SimpleNamespace(logits=torch.arange(24, dtype=torch.float32).view(1, 8, 3))

    logits = runner._forward(SimpleNamespace(owner=Owner()), state)

    torch.testing.assert_close(
        logits,
        torch.tensor(
            [
                [3.0, 4.0, 5.0],
                [12.0, 13.0, 14.0],
                [15.0, 16.0, 17.0],
                [0.0, 1.0, 2.0],
            ]
        ),
    )


def test_decoder_cache_position_builds_batched_text_indexes():
    embeds = torch.empty((2, 1, 4))
    positions = torch.tensor([11, 12])

    indexes = _SenseNovaDecoderModel._indexes_from_cache_position(positions, embeds)

    assert tuple(indexes.shape) == (3, 2, 1)
    torch.testing.assert_close(indexes[0, :, 0], positions)
    assert torch.count_nonzero(indexes[1:]).item() == 0


def test_decoder_cache_position_preserves_single_batch_index_shape():
    embeds = torch.empty((1, 3, 4))
    positions = torch.tensor([4, 5, 6])

    indexes = _SenseNovaDecoderModel._indexes_from_cache_position(positions, embeds)

    assert tuple(indexes.shape) == (3, 3)
    torch.testing.assert_close(indexes[0], positions)
    assert torch.count_nonzero(indexes[1:]).item() == 0
