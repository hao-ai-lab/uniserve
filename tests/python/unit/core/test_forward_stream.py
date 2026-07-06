from __future__ import annotations

import pytest
import torch

from uniserve_worker.contracts.forward_mode import ForwardMode
from uniserve_worker.execution.forward_stream import (
    ForwardPagedKVSegment,
    ForwardPagedKVView,
    ForwardStreamBuilder,
)
from uniserve_worker.runtime.kv_pool import PagedKVPool

pytestmark = pytest.mark.unit


def test_forward_stream_builds_causal_and_bidirectional_visible_end():
    builder = ForwardStreamBuilder()
    builder.add_segment(
        op_index=0,
        req_id=1,
        kind="decode_und",
        mode=ForwardMode.DECODE,
        modality="und",
        segment_class="decode",
        q_len=3,
        prefix_len=4,
        visible_policy="causal",
    )
    image_indexes = torch.tensor(
        [
            [10, 10, 10, 10],
            [0, 0, 1, 1],
            [0, 1, 0, 1],
        ]
    )
    builder.add_segment(
        op_index=1,
        req_id=2,
        kind="denoise_gen",
        mode=ForwardMode.DENOISE,
        modality="gen",
        segment_class="denoise",
        q_len=4,
        prefix_len=9,
        branch_id=2,
        visible_policy="bidirectional",
        indexes=image_indexes,
    )

    stream = builder.build()

    assert stream.cu_seqlens_q.tolist() == [0, 3, 7]
    assert stream.visible_end.tolist() == [[5, 6, 7, 0], [13, 13, 13, 13]]
    assert not stream.fully_visible
    torch.testing.assert_close(stream.indexes[:, :3], torch.tensor([[4, 5, 6], [0, 0, 0], [0, 0, 0]]))
    torch.testing.assert_close(stream.indexes[:, 3:], image_indexes)


def test_forward_stream_marks_decode_and_denoise_rows_fully_visible():
    builder = ForwardStreamBuilder()
    builder.add_segment(
        op_index=0,
        req_id=1,
        kind="decode_und",
        mode=ForwardMode.DECODE,
        modality="und",
        segment_class="decode",
        q_len=1,
        prefix_len=4,
        visible_policy="causal",
    )
    builder.add_segment(
        op_index=1,
        req_id=2,
        kind="denoise_gen",
        mode=ForwardMode.DENOISE,
        modality="gen",
        segment_class="denoise",
        q_len=3,
        prefix_len=9,
        visible_policy="bidirectional",
        indexes=torch.tensor([[10, 10, 10], [0, 0, 1], [0, 1, 0]]),
    )

    stream = builder.build()

    assert stream.visible_end.tolist() == [[5, 0, 0], [12, 12, 12]]
    assert stream.fully_visible


def test_forward_stream_keeps_cfg_branches_structurally_distinct():
    cond_indexes = torch.tensor([[20, 20], [0, 1], [0, 0]])
    tu_indexes = torch.tensor([[2, 2], [0, 1], [0, 0]])
    builder = ForwardStreamBuilder()
    builder.add_segment(
        op_index=0,
        req_id=9,
        kind="denoise_gen",
        mode=ForwardMode.DENOISE,
        modality="gen",
        segment_class="denoise",
        q_len=2,
        prefix_len=20,
        branch_id=0,
        visible_policy="bidirectional",
        indexes=cond_indexes,
    )
    builder.add_segment(
        op_index=0,
        req_id=9,
        kind="denoise_gen",
        mode=ForwardMode.DENOISE,
        modality="gen",
        segment_class="denoise",
        q_len=2,
        prefix_len=2,
        branch_id=1,
        visible_policy="bidirectional",
        indexes=tu_indexes,
    )

    stream = builder.build()

    assert stream.visible_end.tolist() == [[22, 22], [4, 4]]
    torch.testing.assert_close(stream.indexes[:, :2], cond_indexes)
    torch.testing.assert_close(stream.indexes[:, 2:], tu_indexes)


def test_forward_stream_rejects_bad_segment_shapes():
    builder = ForwardStreamBuilder()
    with pytest.raises(Exception, match="indexes"):
        builder.add_segment(
            op_index=0,
            req_id=1,
            kind="denoise_gen",
            mode=ForwardMode.DENOISE,
            modality="gen",
            segment_class="denoise",
            q_len=2,
            prefix_len=0,
            indexes=torch.zeros(3, 3, dtype=torch.long),
        )


def test_forward_paged_kv_view_appends_ragged_segments_into_one_pool():
    pool = PagedKVPool(
        num_layers=1,
        num_blocks=5,
        block_size=4,
        num_kv_heads=1,
        head_dim=2,
        device="cpu",
        dtype=torch.float32,
    )
    view = ForwardPagedKVView(
        pool,
        [
            ForwardPagedKVSegment(block_ids=(0, 1), base_len=3, q_len=2),
            ForwardPagedKVSegment(block_ids=(2, 3, 4), base_len=7, q_len=3),
        ],
    )

    assert view.block_table().tolist() == [[0, 1, 0], [2, 3, 4]]
    assert view.cache_seqlens_before().tolist() == [3, 7]
    assert view.cache_seqlens_after().tolist() == [5, 10]
    assert view.cu_seqlens_after().tolist() == [0, 5, 15]
    assert view.persistent_cache_seqlens_after().tolist() == [5, 10]
    assert view.block_table() is view.block_table()
    assert view.cache_seqlens_after() is view.cache_seqlens_after()
    assert view.cu_seqlens_after() is view.cu_seqlens_after()

    k = torch.arange(10, dtype=torch.float32).view(5, 1, 2)
    v = -k
    view.append_packed(0, k, v)

    torch.testing.assert_close(pool.k[0, 0, 3], k[0])
    torch.testing.assert_close(pool.k[0, 1, 0], k[1])
    torch.testing.assert_close(pool.v[0, 3, 3], v[2])
    torch.testing.assert_close(pool.v[0, 4, 0], v[3])
    torch.testing.assert_close(pool.v[0, 4, 1], v[4])


def test_forward_paged_kv_view_requires_one_pool():
    pool_a = PagedKVPool(1, 2, 4, 1, 2, device="cpu", dtype=torch.float32)
    pool_b = PagedKVPool(1, 2, 4, 1, 2, device="cpu", dtype=torch.float32)
    cache_a = pool_a.view([0], 0)
    cache_b = pool_b.view([0], 0)

    with pytest.raises(Exception, match="one PagedKVPool"):
        ForwardPagedKVView.from_request_caches([cache_a, cache_b], [1, 1])


def test_forward_paged_kv_view_rejects_wrong_packed_token_count():
    pool = PagedKVPool(1, 2, 4, 1, 2, device="cpu", dtype=torch.float32)
    view = ForwardPagedKVView(pool, [ForwardPagedKVSegment(block_ids=(0,), base_len=0, q_len=2)])
    with pytest.raises(Exception, match="expected 2"):
        view.append_packed(0, torch.zeros(1, 1, 2), torch.zeros(1, 1, 2))


def test_forward_paged_kv_view_skips_read_only_denoise_segments():
    pool = PagedKVPool(1, 3, 4, 1, 2, device="cpu", dtype=torch.float32)
    pool.k.fill_(123)
    pool.v.fill_(456)
    view = ForwardPagedKVView(
        pool,
        [
            ForwardPagedKVSegment(block_ids=(0,), base_len=1, q_len=2, write_kv=True),
            ForwardPagedKVSegment(
                block_ids=(1,),
                base_len=2,
                q_len=2,
                write_kv=False,
                persist_kv=False,
                branch_id=7,
            ),
        ],
    )

    assert view.cache_seqlens_before().tolist() == [1, 2]
    assert view.cache_seqlens_after().tolist() == [3, 4]
    assert view.persistent_cache_seqlens_after().tolist() == [3, 2]
    k = torch.arange(8, dtype=torch.float32).view(4, 1, 2)
    v = -k
    view.append_packed(0, k, v)

    torch.testing.assert_close(pool.k[0, 0, 1], k[0])
    torch.testing.assert_close(pool.k[0, 0, 2], k[1])
    torch.testing.assert_close(pool.v[0, 0, 1], v[0])
    torch.testing.assert_close(pool.v[0, 0, 2], v[1])
    torch.testing.assert_close(pool.k[0, 1], torch.full_like(pool.k[0, 1], 123))
    torch.testing.assert_close(pool.v[0, 1], torch.full_like(pool.v[0, 1], 456))
    assert view.segments[1].branch_id == 7
    assert not view.segments[1].persist_kv


def test_forward_paged_kv_view_rejects_persistent_segment_without_kv_write():
    pool = PagedKVPool(1, 2, 4, 1, 2, device="cpu", dtype=torch.float32)
    with pytest.raises(Exception, match="persistent"):
        ForwardPagedKVView(
            pool,
            [
                ForwardPagedKVSegment(
                    block_ids=(0,),
                    base_len=0,
                    q_len=1,
                    write_kv=False,
                    persist_kv=True,
                )
            ],
        )
