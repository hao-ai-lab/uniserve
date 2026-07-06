from types import SimpleNamespace

import pytest

from uniserve_worker.execution.interleaved_text_graph_runner import (
    InterleavedTextDecodeGraphRunner,
    _Row,
)


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
