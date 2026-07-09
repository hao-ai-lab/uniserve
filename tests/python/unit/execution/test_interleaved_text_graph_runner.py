from types import SimpleNamespace

import pytest
import torch

from uniserve_worker.execution.forward.graph.interleaved_text import (
    InterleavedTextDecodeGraphRunner,
    _Row,
)
from uniserve_worker.models.sensenova.model import _SenseNovaDecoderModel


class _Owner:
    def interleaved_decode_graph_padding_block_id(self, pool):
        return int(pool.num_blocks) - 1


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
    pool = SimpleNamespace(num_blocks=64, block_size=16)
    runner = InterleavedTextDecodeGraphRunner()
    driver = SimpleNamespace(owner=_Owner())
    rows = [_row(pool, block_ids=[3], base_len=11)]

    padded = runner._pad_rows(driver, rows, 4, pool)

    assert padded[:1] == rows
    assert [row.block_ids for row in padded[1:]] == [[63], [63], [63]]
    assert [row.base_len for row in padded[1:]] == [0, 1, 2]
    assert [row.pos for row in padded[1:]] == [0, 1, 2]
    assert [row.token_id for row in padded[1:]] == [0, 0, 0]


def test_interleaved_decode_graph_padding_requires_reserved_block_hook():
    pool = SimpleNamespace(num_blocks=64, block_size=16)
    runner = InterleavedTextDecodeGraphRunner()
    driver = SimpleNamespace(owner=object())

    with pytest.raises(Exception, match="reserved KV padding block"):
        runner._pad_rows(driver, [_row(pool)], 2, pool)


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
