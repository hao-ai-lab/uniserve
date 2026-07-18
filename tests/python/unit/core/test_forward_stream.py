from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from uniserve_worker.contracts.forward_mode import ForwardMode
from uniserve_worker.runtime.forward_stream import (
    ForwardGraphPagedKVView,
    ForwardGraphStreamState,
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


def test_forward_stream_generated_indexes_preserve_explicit_position_start():
    builder = ForwardStreamBuilder()
    builder.add_segment(
        op_index=0,
        req_id=1,
        kind="decode_und",
        mode=ForwardMode.DECODE,
        modality="und",
        segment_class="decode",
        q_len=2,
        prefix_len=100,
        visible_policy="causal",
        index_start=7,
    )
    builder.add_segment(
        op_index=1,
        req_id=2,
        kind="decode_und",
        mode=ForwardMode.DECODE,
        modality="und",
        segment_class="decode",
        q_len=1,
        prefix_len=4,
        visible_policy="causal",
    )

    stream = builder.build()

    assert stream.visible_end.tolist() == [[101, 102], [5, 0]]
    torch.testing.assert_close(stream.indexes, torch.tensor([[7, 8, 4], [0, 0, 0], [0, 0, 0]]))


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


def test_forward_graph_stream_state_refreshes_values_without_reallocating_tensors():
    builder = ForwardStreamBuilder()
    builder.add_segment(
        op_index=0,
        req_id=1,
        kind="decode_und",
        mode=ForwardMode.DECODE,
        modality="und",
        segment_class="decode",
        q_len=1,
        prefix_len=3,
        visible_policy="causal",
    )
    builder.add_segment(
        op_index=1,
        req_id=2,
        kind="denoise_gen",
        mode=ForwardMode.DENOISE,
        modality="gen",
        segment_class="denoise",
        q_len=2,
        prefix_len=7,
        branch_id=1,
        visible_policy="bidirectional",
        indexes=torch.tensor([[7, 7], [0, 1], [1, 0]], dtype=torch.long),
    )
    initial = builder.build(device="cpu")
    state = ForwardGraphStreamState.from_stream(initial)
    cu_ptr = state.stream.cu_seqlens_q.data_ptr()
    visible_ptr = state.stream.visible_end.data_ptr()
    indexes_ptr = state.stream.indexes.data_ptr()
    und_ptr = state.stream.und_indices.data_ptr()
    gen_ptr = state.stream.gen_indices.data_ptr()

    refreshed_builder = ForwardStreamBuilder()
    refreshed_builder.add_segment(
        op_index=2,
        req_id=11,
        kind="decode_und",
        mode=ForwardMode.DECODE,
        modality="und",
        segment_class="decode",
        q_len=1,
        prefix_len=9,
        visible_policy="causal",
    )
    refreshed_builder.add_segment(
        op_index=3,
        req_id=12,
        kind="denoise_gen",
        mode=ForwardMode.DENOISE,
        modality="gen",
        segment_class="denoise",
        q_len=2,
        prefix_len=4,
        branch_id=1,
        visible_policy="bidirectional",
        indexes=torch.tensor([[4, 4], [1, 1], [0, 1]], dtype=torch.long),
    )
    refreshed = refreshed_builder.build(device="cpu")
    refreshed = replace(
        refreshed,
        und_indices=torch.tensor([2], dtype=torch.long),
        gen_indices=torch.tensor([0, 1], dtype=torch.long),
    )

    graph_stream = state.refresh(refreshed)

    assert graph_stream is state.stream
    assert graph_stream.cu_seqlens_q.data_ptr() == cu_ptr
    assert graph_stream.visible_end.data_ptr() == visible_ptr
    assert graph_stream.indexes.data_ptr() == indexes_ptr
    assert graph_stream.und_indices.data_ptr() == und_ptr
    assert graph_stream.gen_indices.data_ptr() == gen_ptr
    assert [seg.req_id for seg in graph_stream.segments] == [11, 12]
    torch.testing.assert_close(graph_stream.cu_seqlens_q, refreshed.cu_seqlens_q)
    torch.testing.assert_close(graph_stream.visible_end, refreshed.visible_end)
    torch.testing.assert_close(graph_stream.indexes, refreshed.indexes)
    torch.testing.assert_close(graph_stream.und_indices, refreshed.und_indices)
    torch.testing.assert_close(graph_stream.gen_indices, refreshed.gen_indices)


def test_forward_graph_stream_state_rejects_capacity_changes():
    builder = ForwardStreamBuilder()
    builder.add_segment(
        op_index=0,
        req_id=1,
        kind="decode_und",
        mode=ForwardMode.DECODE,
        modality="und",
        segment_class="decode",
        q_len=1,
        prefix_len=0,
        visible_policy="causal",
    )
    state = ForwardGraphStreamState.from_stream(builder.build())
    changed = ForwardStreamBuilder()
    changed.add_segment(
        op_index=0,
        req_id=1,
        kind="prefill_und",
        mode=ForwardMode.EXTEND,
        modality="und",
        segment_class="extend",
        q_len=2,
        prefix_len=0,
        visible_policy="causal",
    )

    with pytest.raises(Exception, match="capacity"):
        state.refresh(changed.build())


def test_forward_graph_stream_state_refreshes_ragged_query_distribution():
    initial_builder = ForwardStreamBuilder()
    initial_builder.add_segment(
        op_index=0,
        req_id=1,
        kind="prefill_und",
        mode=ForwardMode.EXTEND,
        modality="und",
        segment_class="extend",
        q_len=1,
        prefix_len=0,
    )
    initial_builder.add_segment(
        op_index=1,
        req_id=2,
        kind="prefill_und",
        mode=ForwardMode.EXTEND,
        modality="und",
        segment_class="extend",
        q_len=2,
        prefix_len=4,
    )
    state = ForwardGraphStreamState.from_stream(initial_builder.build())
    cu_ptr = state.stream.cu_seqlens_q.data_ptr()

    refreshed_builder = ForwardStreamBuilder()
    refreshed_builder.add_segment(
        op_index=0,
        req_id=3,
        kind="prefill_und",
        mode=ForwardMode.EXTEND,
        modality="und",
        segment_class="extend",
        q_len=2,
        prefix_len=1,
    )
    refreshed_builder.add_segment(
        op_index=1,
        req_id=4,
        kind="decode_und",
        mode=ForwardMode.DECODE,
        modality="und",
        segment_class="decode",
        q_len=1,
        prefix_len=8,
    )
    refreshed = refreshed_builder.build()

    graph_stream = state.refresh(refreshed)

    assert graph_stream.cu_seqlens_q.data_ptr() == cu_ptr
    torch.testing.assert_close(graph_stream.cu_seqlens_q, refreshed.cu_seqlens_q)


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


def test_forward_graph_paged_kv_view_refreshes_tables_without_reallocating_tensors():
    pool = PagedKVPool(
        num_layers=1,
        num_blocks=5,
        block_size=4,
        num_kv_heads=1,
        head_dim=2,
        device="cpu",
        dtype=torch.float32,
    )
    view = ForwardGraphPagedKVView(
        pool,
        [
            ForwardPagedKVSegment(block_ids=(0, 1), base_len=3, q_len=2),
            ForwardPagedKVSegment(block_ids=(2,), base_len=0, q_len=3, persist_kv=False),
        ],
    )
    block_table = view.block_table()
    cache_after = view.cache_seqlens_after()
    cu_after = view.cu_seqlens_after()
    persistent_after = view.persistent_cache_seqlens_after()
    page_ids, offsets, token_indices = view._write_plan()
    ptrs = {
        "block_table": block_table.data_ptr(),
        "cache_after": cache_after.data_ptr(),
        "cu_after": cu_after.data_ptr(),
        "persistent_after": persistent_after.data_ptr(),
        "page_ids": page_ids.data_ptr(),
        "offsets": offsets.data_ptr(),
    }

    view.refresh(
        [
            ForwardPagedKVSegment(block_ids=(3, 4), base_len=4, q_len=2),
            ForwardPagedKVSegment(block_ids=(1,), base_len=1, q_len=3, persist_kv=False),
        ]
    )
    refreshed_page_ids, refreshed_offsets, refreshed_token_indices = view._write_plan()

    assert view.block_table().data_ptr() == ptrs["block_table"]
    assert view.cache_seqlens_after().data_ptr() == ptrs["cache_after"]
    assert view.cu_seqlens_after().data_ptr() == ptrs["cu_after"]
    assert view.persistent_cache_seqlens_after().data_ptr() == ptrs["persistent_after"]
    assert refreshed_page_ids.data_ptr() == ptrs["page_ids"]
    assert refreshed_offsets.data_ptr() == ptrs["offsets"]
    assert token_indices is None
    assert refreshed_token_indices is None
    assert view.block_table().tolist() == [[3, 4], [1, 0]]
    assert view.cache_seqlens_before().tolist() == [4, 1]
    assert view.cache_seqlens_after().tolist() == [6, 4]
    assert view.cu_seqlens_after().tolist() == [0, 6, 10]
    assert view.persistent_cache_seqlens_after().tolist() == [6, 1]
    assert refreshed_page_ids.tolist() == [4, 4, 1, 1, 1]
    assert refreshed_offsets.tolist() == [0, 1, 1, 2, 3]
    assert view.max_seqlen_k() == 8

    k = torch.arange(10, dtype=torch.float32).view(5, 1, 2)
    v = -k
    view.append_packed(0, k, v)

    torch.testing.assert_close(pool.k[0, 4, 0], k[0])
    torch.testing.assert_close(pool.k[0, 4, 1], k[1])
    torch.testing.assert_close(pool.v[0, 1, 1], v[2])
    torch.testing.assert_close(pool.v[0, 1, 2], v[3])
    torch.testing.assert_close(pool.v[0, 1, 3], v[4])


def test_forward_graph_paged_kv_view_rejects_capacity_changes():
    pool = PagedKVPool(1, 2, 4, 1, 2, device="cpu", dtype=torch.float32)
    view = ForwardGraphPagedKVView(
        pool,
        [ForwardPagedKVSegment(block_ids=(0,), base_len=0, q_len=1)],
    )

    with pytest.raises(Exception, match="capacity"):
        view.refresh([ForwardPagedKVSegment(block_ids=(0,), base_len=0, q_len=2)])


def test_forward_graph_paged_kv_view_refreshes_ragged_query_distribution():
    pool = PagedKVPool(1, 8, 4, 1, 2, device="cpu", dtype=torch.float32)
    view = ForwardGraphPagedKVView(
        pool,
        [
            ForwardPagedKVSegment(block_ids=(0,), base_len=0, q_len=1),
            ForwardPagedKVSegment(block_ids=(1, 2), base_len=3, q_len=2),
        ],
        block_width_capacity=3,
    )
    cu_ptr = view.cu_seqlens_after().data_ptr()

    view.refresh(
        [
            ForwardPagedKVSegment(block_ids=(3,), base_len=1, q_len=2),
            ForwardPagedKVSegment(block_ids=(4, 5, 6), base_len=7, q_len=1),
        ]
    )

    assert view.cu_seqlens_after().data_ptr() == cu_ptr
    assert view.cache_seqlens_before().tolist() == [1, 7]
    assert view.cu_seqlens_after().tolist() == [0, 3, 11]


def test_forward_graph_paged_kv_view_refreshes_within_block_capacity():
    pool = PagedKVPool(1, 6, 4, 1, 2, device="cpu", dtype=torch.float32)
    view = ForwardGraphPagedKVView(
        pool,
        [ForwardPagedKVSegment(block_ids=(0, 1), base_len=3, q_len=1)],
        block_width_capacity=4,
    )
    table_ptr = view.block_table().data_ptr()

    view.refresh(
        [ForwardPagedKVSegment(block_ids=(1, 2, 3), base_len=8, q_len=1)]
    )

    assert view.block_table().data_ptr() == table_ptr
    assert view.block_table().tolist() == [[1, 2, 3, 0]]
    assert view.cache_seqlens_after().tolist() == [9]
    assert view.max_seqlen_k() == 16
    with pytest.raises(Exception, match="block-table width"):
        view.refresh(
            [ForwardPagedKVSegment(block_ids=(0, 1, 2, 3, 4), base_len=8, q_len=1)]
        )


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
